"""shots negatives (plan.json + addendum1.json, hashed first): the v4 spotter
fitted on clips only (baseline) and on clips + ordinary-play windows (new),
left one match out, judged on whole-match precision.

    VEO_UNSEAL=... python score.py
"""
import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "eval"))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball/events_tune"))
import eval_ap as EA  # noqa: E402

LS, V, V3, V4 = EA.LS, EA.V, EA.V3, EA.V4
GPU_WIN = ROOT / ".cache/gpu_windows"
TREE_NEW = ROOT / ".cache/gpu_neg_tree"
TREE_DEV = ROOT / ".cache/gpu_shots_windows"        # realistic/run.py's tree of the 7 dev windows
DATA = ROOT / "data/negatives/soccertrack_v2"
MATCH_MIN = 92.0
DEV_WINDOWS = list(LS.TRAIN_WINDOWS)


def check():
    for name in ("plan.json", "addendum1.json"):
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != (HERE / f"{name}.sha256").read_text().split()[0]:
            raise SystemExit(f"{name} does not match its sha256")
    return {n: (HERE / f"{n}.sha256").read_text().split()[0] for n in ("plan.json", "addendum1.json")}


def patch_info():
    sd = sys.modules[LS.labels.__module__].sd if hasattr(sys.modules[LS.labels.__module__], "sd") else None
    orig = sd.info_of

    def info_of(name):
        if (DATA / f"{name}.json").exists():
            return {**json.loads((DATA / f"{name}.json").read_text()), "dir": DATA}
        return orig(name)
    sd.info_of = info_of


def relink(w):
    """ball_grass/relink.py's link, on a collected window."""
    import score_soccernet as sc
    import tune_rules as tr
    from src import pixel_scale
    from src import soccertrack_v2 as st
    base = GPU_WIN / w / f"output_{w}"
    out = base / "metric_v3s_link_nograss.parquet"
    if out.exists():
        return out
    run = base / "shots_v3s_link"
    if not (run / "tracks_grass.parquet").exists():
        sc.swap_ball_rows(base / "events_v1", run, pd.read_parquet(base / "v3s_rows.parquet"))
        pd.read_parquet(run / "tracks_teams.parquet").assign(grass_frac=float("nan")).to_parquet(run / "tracks_grass.parquet")
    cal = st.calibration(w.split("_")[1])
    with contextlib.redirect_stdout(io.StringIO()):
        merged = sc.merged_from_cached(run, cal, select="link", top_k=4, tau=0.10,
                                       link_params=tr.LINK, ball_grass_filter=False)
    pixel_scale.to_pitch_metres(merged, lambda uv: st.image_to_pitch(uv, cal)).to_parquet(out)
    return out


def tree(w):
    base = GPU_WIN / w / f"output_{w}"
    d = TREE_NEW / w / f"output_{w}"
    d.mkdir(parents=True, exist_ok=True)
    for name, target in (("metric_v3s_link.parquet", relink(w)), ("v3s_rows.parquet", base / "v3s_rows.parquet")):
        if not (d / name).exists():
            os.symlink(target, d / name)


def window_data(names, gpu):
    LS.GPU = V.GPU = gpu
    out = {}
    for n in names:
        u = {"name": n, "match": n.split("_")[1], "span": None}
        f = V4.features_v4(u)
        out[n] = (f, LS.labels(u), u["match"])
    return out


def minutes(f):
    fr = f.index.to_numpy()
    return (fr.max() - fr.min()) / LS.FPS / 60


def fit_predict(train, cols, targets, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    base = LS.L.model()
    X = pd.concat([train[k][0][cols] for k in train])
    y = pd.concat([LS.target(train[k][0], train[k][1]) for k in train])
    clf = HistGradientBoostingClassifier(**{**base.get_params(), "random_state": seed}).fit(X, y)
    return {k: clf.predict_proba(targets[k][0][cols])[:, 1] for k in targets}


def detections(d, probs):
    p = pd.concat([pd.DataFrame({"unit": n, "match": d[n][2], "frame": d[n][0].index.to_numpy(), "p": probs[n]})
                   for n in probs], ignore_index=True)
    labs = {n: (d[n][1], d[n][2]) for n in probs}
    return EA.flags(EA.detections(p), labs)


def stats(per, ms, recall_at):
    """Pooled over matches ms: threshold where the clips' recall first reaches
    recall_at, and the whole-match figures there."""
    c = pd.concat([per[m]["clip_det"] for m in ms], ignore_index=True).sort_values("score", ascending=False)
    n_lab = sum(per[m]["n_lab"] for m in ms)
    cum = np.cumsum(c.right.to_numpy())
    hit = np.flatnonzero(cum >= recall_at * n_lab)
    if not len(hit):
        return None
    t = float(c.score.iloc[hit[0]])
    return at(per, ms, t) | {"threshold": round(t, 4)}


def at(per, ms, t):
    n_lab = sum(per[m]["n_lab"] for m in ms)
    def above(d):
        return d[d.score >= t].right.astype(bool)
    found = sum(int(above(per[m]["clip_det"]).sum()) for m in ms)
    fa_c = sum(int((~above(per[m]["clip_det"])).sum()) for m in ms)
    fa_w = sum(int((~above(per[m]["win_det"])).sum()) for m in ms)
    wmin = sum(per[m]["win_min"] for m in ms)
    rate = fa_w / wmin if wmin else float("nan")
    rest = len(ms) * MATCH_MIN - sum(per[m]["clip_min"] for m in ms)
    fa = fa_c + rate * rest
    return {"clips: recall": round(found / max(n_lab, 1), 3), "clips: precision": round(found / max(found + fa_c, 1), 3),
            "windows: false alarms per minute": round(rate, 3),
            "whole match: shots found per match": round(found / len(ms), 1),
            "whole match: false alarms per match": round(fa / len(ms), 1),
            "whole match: precision": round(found / max(found + fa, 1e-9), 3)}


def main():
    sha = check()
    from src.unseal import unsealed
    for m in V3.NEW_PLAYERS:
        if not unsealed(m, "shots negatives: scoring (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    patch_info()
    wins = json.loads((HERE / "windows.json").read_text())["windows"]
    collected = set()
    for p in (HERE / "state").glob("*.json"):
        collected |= set(json.loads(p.read_text()).get("ok", []))
    new_names = [w["name"] for w in wins if w["name"] in collected]
    missing = [w["name"] for w in wins if w["name"] not in collected]
    for n in new_names:
        tree(n)
    us = EA.units()
    dc = EA.data("v4", us)                       # sets LS.GPU to the clips' tree
    cols = LS.cols_of(dc)
    dw = {**window_data(new_names, TREE_NEW), **window_data(DEV_WINDOWS, TREE_DEV)}
    win_labels = {n: len(dw[n][1]) for n in dw if len(dw[n][1])}
    matches = sorted({x[2] for x in dc.values()})
    per = {"baseline": {}, "new": {}}
    for m in matches:
        clips_tr = {k: v for k, v in dc.items() if v[2] != m}
        wins_tr = {k: v for k, v in dw.items() if v[2] != m}
        targets = {**{k: v for k, v in dc.items() if v[2] == m}, **{k: v for k, v in dw.items() if v[2] == m}}
        for model, train in (("baseline", clips_tr), ("new", {**clips_tr, **wins_tr})):
            ps = [fit_predict(train, cols, targets, s) for s in EA.SEEDS]
            probs = {k: np.mean([p[k] for p in ps], axis=0) for k in targets}
            cd = detections(targets, {k: probs[k] for k in targets if k in dc})
            wn = [k for k in targets if k in dw]
            wd = detections(targets, {k: probs[k] for k in wn}) if wn else pd.DataFrame(columns=["score", "right"])
            per[model][m] = {"clip_det": cd, "win_det": wd, "n_lab": sum(len(dc[k][1]) for k in targets if k in dc),
                             "clip_min": sum(minutes(dc[k][0]) for k in targets if k in dc),
                             "win_min": sum(minutes(dw[k][0]) for k in wn)}
        print("match", m, "done", flush=True)
    import pickle
    (ROOT / ".cache/shots_negatives_per.pkl").write_bytes(pickle.dumps(per))   # the fits, to re-read without refitting
    res = {}
    for model in per:
        res[model] = {"recall 0.60": stats(per[model], matches, 0.60), "recall 0.80": stats(per[model], matches, 0.80),
                      "per match at the pooled recall-0.60 threshold": {
                          m: at(per[model], [m], stats(per[model], matches, 0.60)["threshold"]) for m in matches}}
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(5000):
        ms = list(rng.choice(matches, len(matches)))
        a, b = stats(per["new"], ms, 0.60), stats(per["baseline"], ms, 0.60)
        if a and b:
            diffs.append(a["whole match: precision"] - b["whole match: precision"])
    gain = res["new"]["recall 0.60"]["whole match: precision"] - res["baseline"]["recall 0.60"]["whole match: precision"]
    ci = [round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)]
    owner = {k: bool(res[k]["recall 0.80"] and res[k]["recall 0.80"]["whole match: precision"] >= 0.80) for k in res}
    out = {"sha256": sha, "windows run": len(new_names), "windows missing": missing,
           "window minutes (new, dev)": [round(sum(minutes(dw[n][0]) for n in new_names), 1),
                                         round(sum(minutes(dw[n][0]) for n in DEV_WINDOWS), 1)],
           "labelled shots inside windows": win_labels,
           "primary: whole-match precision at clip recall 0.60, new - baseline": round(gain, 3),
           "95% bootstrap over matches": ci,
           "adopt (gain >= 0.10, interval above 0)": bool(gain >= 0.10 and ci[0] > 0),
           "owner's bar (recall 0.80, whole-match precision 0.80) met": owner,
           "models": res}
    (HERE / "negatives.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "models"}, indent=1, default=str))
    for k in res:
        print(k, json.dumps({x: res[k][x] for x in ("recall 0.60", "recall 0.80")}, default=str))


if __name__ == "__main__":
    main()
