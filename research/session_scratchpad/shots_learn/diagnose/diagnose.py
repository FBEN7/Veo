"""shots diagnose (plan.json, hashed first): the errors of the v4 spotter by
our ball's state, and the spotter refitted with the annotation ball.

    VEO_UNSEAL=... python diagnose.py
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "negatives"))
import score as S  # noqa: E402  (negatives/score.py: its data, fit and evaluation, unchanged)

EA, LS, V, V3 = S.EA, S.LS, S.V, S.V3
TREE_T = ROOT / ".cache/gpu_shots_truthball"
PER_O = ROOT / ".cache/shots_negatives_per.pkl"     # negatives/score.py's fits ("new" = arm O)
BALL_S, NEAR_M, ACT_S = 0.5, 3.0, 2.0
ROWS_COLS = {"frame": "int64", "confidence": "float64", "px": "float64", "py": "float64", "crop_h": "float64"}


def check():
    out = {}
    for name in ("plan.json", "addendum1.json"):
        want = (HERE / f"{name}.sha256").read_text().split()[0]
        got = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"{name} does not match its sha256")
        out[name] = got
    return out


def actions(name):
    """All annotated actions of a unit with a position, corner origin (as our metric)."""
    ev = pd.read_csv(LS.sd.info_of(name)["dir"] / f"{name}_events.csv")
    ev = ev[ev.x.notna() & ev.y.notna()].sort_values("frame", kind="stable")
    return ev.assign(xc=ev.x + 52.5, yc=ev.y + 34.0)


def truth_unit(name, src_tree):
    """TREE_T/<name>: our metric with the ball rows replaced by the annotation
    ball; an empty v3s_rows (no candidates)."""
    d = TREE_T / name / f"output_{name}"
    out = d / "metric_v3s_link.parquet"
    if out.exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(src_tree / name / f"output_{name}" / "metric_v3s_link.parquet")
    pl = m[m.cls != "ball"]
    frames = np.sort(pl[pl.cls == "player"].frame.unique())
    ev = actions(name).drop_duplicates("frame")
    if len(ev) >= 1:   # addendum 1: held at the first / last action outside them (np.interp's ends)
        f = frames
        ball = pd.DataFrame({"frame": f, "cls": "ball", "team": "ball", "track_id": -1, "confidence": 1.0,
                             "x": np.interp(f, ev.frame, ev.xc), "y": np.interp(f, ev.frame, ev.yc)})
    else:
        ball = pd.DataFrame(columns=["frame", "cls", "x", "y", "confidence"])
    pd.concat([pl, ball], ignore_index=True).sort_values("frame", kind="stable").to_parquet(out)
    pd.DataFrame({c: pd.Series(dtype=t) for c, t in ROWS_COLS.items()}).to_parquet(d / "v3s_rows.parquet")


def our_ball(name, src_tree):
    m = pd.read_parquet(src_tree / name / f"output_{name}" / "metric_v3s_link.parquet")
    b = m[(m.cls == "ball") & m.x.notna()]
    return b.frame.to_numpy(), b.x.to_numpy(), b.y.to_numpy()


def ball_state(ob, frame, x, y):
    fr, bx, by = ob
    k = np.abs(fr - frame) <= BALL_S * LS.FPS
    if not k.any():
        return "no ball"
    j = np.flatnonzero(k)[np.argmin(np.abs(fr[k] - frame))]
    return "ball <= 3 m" if np.hypot(bx[j] - x, by[j] - y) <= NEAR_M else "ball > 3 m"


def decompose(per, ms, t, src_of):
    """arm O at threshold t: true shots by our ball, false alarms by what was there."""
    shots, fas = [], []
    for m in ms:
        for kind in ("clip_det", "win_det"):
            det = per[m][kind]
            if not len(det):
                continue
            for n, g in det.groupby("unit"):
                ev, ob = actions(n), our_ball(n, src_of[n])
                g = g[g.score >= t].sort_values("score", ascending=False, kind="stable")
                for r in g[~g.right.astype(bool)].itertuples():
                    near = ev[np.abs(ev.frame / LS.FPS - r.t) <= ACT_S]
                    if len(near):
                        a = near.iloc[int(np.argmin(np.abs(near.frame.to_numpy() / LS.FPS - r.t)))]
                        fas.append({"where": "clips" if kind == "clip_det" else "windows", "action": a.label,
                                    "ball": ball_state(ob, a.frame, a.xc, a.yc)})
                    else:
                        fas.append({"where": "clips" if kind == "clip_det" else "windows", "action": "none within 2 s",
                                    "ball": "-"})
                if kind == "clip_det":   # label-level hits, as eval_ap.flags restricted to the kept detections
                    lab = ev[ev.label.isin(LS.sd.LABELS)]
                    span = next(u for u in UNITS if u["name"] == n).get("span")
                    if span:
                        lab = lab[(lab.frame >= span[0]) & (lab.frame <= span[1])]
                    lt = lab.frame.to_numpy() / LS.FPS
                    used = np.zeros(len(lt), bool)
                    for r in g.itertuples():
                        if not len(lt):
                            break
                        dd = np.abs(lt - r.t)
                        dd[used] = np.inf
                        j = int(np.argmin(dd))
                        if dd[j] <= LS.TOL_S:
                            used[j] = True
                    for i, a in enumerate(lab.itertuples()):
                        shots.append({"found": bool(used[i]), "ball": ball_state(ob, a.frame, a.xc, a.yc)})
    sh, fa = pd.DataFrame(shots), pd.DataFrame(fas)
    return {"true shots": {"n": len(sh), "by our ball (found / missed)": {
                b: {"found": int(((sh.ball == b) & sh.found).sum()), "missed": int(((sh.ball == b) & ~sh.found).sum())}
                for b in ("no ball", "ball > 3 m", "ball <= 3 m")}},
            "false alarms": {"n": len(fa),
                             "by where": fa.groupby("where").size().to_dict(),
                             "by nearest action": fa.groupby("action").size().sort_values(ascending=False).to_dict(),
                             "by our ball at that action": fa.groupby("ball").size().to_dict()}}


UNITS = []


def main():
    sha = check()
    from src.unseal import unsealed
    for m in V3.NEW_PLAYERS:
        if not unsealed(m, "shots diagnose: M7 / M9 clips and windows (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    S.patch_info()
    per_o = pickle.loads(PER_O.read_bytes())["new"]
    wins = json.loads((S.HERE / "windows.json").read_text())["windows"]
    collected = set()
    for p in (S.HERE / "state").glob("*.json"):
        collected |= set(json.loads(p.read_text()).get("ok", []))
    new_names = [w["name"] for w in wins if w["name"] in collected]
    UNITS.extend(EA.units())
    src_of = {u["name"]: V3.NEW for u in UNITS} | {n: S.TREE_NEW for n in new_names} | {n: S.TREE_DEV for n in S.DEV_WINDOWS}
    for n, src in src_of.items():
        truth_unit(n, src)
    print("annotation-ball trees made", flush=True)
    # arm T: the same data, fit and evaluation as score.py's "new", on the annotation-ball trees
    orig = LS.st.image_to_pitch
    LS.st.image_to_pitch = lambda uv, cal: np.empty((0, 2)) if len(uv) == 0 else orig(uv, cal)
    LS.GPU = V.GPU = TREE_T
    dc = {u["name"]: (S.V4.features_v4(u), LS.labels(u), u["match"]) for u in UNITS}
    cols = LS.cols_of(dc)
    dw = S.window_data(new_names + S.DEV_WINDOWS, TREE_T)
    matches = sorted({x[2] for x in dc.values()})
    per_t = {}
    for m in matches:
        train = {**{k: v for k, v in dc.items() if v[2] != m}, **{k: v for k, v in dw.items() if v[2] != m}}
        targets = {**{k: v for k, v in dc.items() if v[2] == m}, **{k: v for k, v in dw.items() if v[2] == m}}
        ps = [S.fit_predict(train, cols, targets, s) for s in EA.SEEDS]
        probs = {k: np.mean([p[k] for p in ps], axis=0) for k in targets}
        cd = S.detections(targets, {k: probs[k] for k in targets if k in dc})
        wn = [k for k in targets if k in dw]
        wd = S.detections(targets, {k: probs[k] for k in wn}) if wn else pd.DataFrame(columns=["score", "right"])
        per_t[m] = {"clip_det": cd, "win_det": wd, "n_lab": sum(len(dc[k][1]) for k in targets if k in dc),
                    "clip_min": sum(S.minutes(dc[k][0]) for k in targets if k in dc),
                    "win_min": sum(S.minutes(dw[k][0]) for k in wn)}
        print("arm T: match", m, "done", flush=True)
    LS.st.image_to_pitch = orig
    (ROOT / ".cache/shots_diagnose_per_t.pkl").write_bytes(pickle.dumps(per_t))
    res = {}
    for arm, per in (("O (our ball)", per_o), ("T (annotation ball)", per_t)):
        res[arm] = {r: S.stats(per, matches, r) for r in (0.60, 0.80)}
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(5000):
        ms = list(rng.choice(matches, len(matches)))
        a, b = S.stats(per_t, ms, 0.80), S.stats(per_o, ms, 0.80)
        if a and b:
            diffs.append(a["whole match: precision"] - b["whole match: precision"])
    pt = res["T (annotation ball)"][0.80]["whole match: precision"]
    gain = pt - res["O (our ball)"][0.80]["whole match: precision"]
    ci = [round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)]
    reading = ("ball is the lever" if pt >= 0.50 and gain >= 0.20 and ci[0] > 0
               else "ball is not enough" if pt < 0.50 else "in between")
    t_o = res["O (our ball)"][0.80]["threshold"]
    out = {"plan sha256": sha, "units": {"clips": len(dc), "windows": len(dw)},
           "primary: whole-match precision at clip recall 0.80, T - O": round(gain, 3),
           "95% bootstrap over matches": ci, "bootstrap draws used": len(diffs), "reading": reading,
           "operating points": {a: {str(r): v for r, v in x.items()} for a, x in res.items()},
           "decomposition (arm O at its recall-0.80 threshold)": decompose(per_o, matches, t_o, src_of)}
    (HERE / "diagnose.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
