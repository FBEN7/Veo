"""The learned shot spotter of protocol.json.

Per unit (a clip's span, or a whole window) the pitch tracks of shots_dev.py
(v3s_link: marks and static points dropped, linked, release calibration), the
per-frame features of events_learn/learn.py (ball, best candidate, nearest
players, changes across t) plus goal geometry, one HistGradientBoosting
classifier for 'shot' (a SHOT or GOAL label within 0.24 s), events at the
peaks of its smoothed probability, 1.0 s apart, above a threshold set by rate
(events as many as labels) on training out-of-fold predictions.

    python learn_shots.py cv              # nested leave-one-match-out on the training units
    python learn_shots.py fit             # frozen.json + model.pkl (if the dev gate passes)
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball/events_learn"))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball/events_tune"))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/shots_goals"))
import learn as L  # noqa: E402
import score_soccernet as sc  # noqa: E402
import shots_dev as sd  # noqa: E402
from detect_ball_heatmap import read_sidecar  # noqa: E402
from src import ball_spots as bs  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
import tune_rules as tr  # noqa: E402

FPS = 25.0
GPU = ROOT / ".cache/gpu"
POS_S = 0.24
NMS_S = 1.0
TOL_S = 2.0
GOALS = np.array([[0.0, 34.0], [105.0, 34.0]])
MOUTH_HALF = 3.66 + 5.0
GATE = {"recall": 0.45, "precision": 0.40}
TRAIN_MATCHES = ["117092", "117093", "118575", "118576", "118578", "128058", "132877"]
TRAIN_WINDOWS = ["st2_117093_1st_f015000", "st2_118575_1st_f015000", "st2_118576_1st_f030000",
                 "st2_118578_1st_f030000", "st2_118578_2nd_f015000", "st2_128058_1st_f030000",
                 "st2_132877_1st_f015000"]


def units(plan: Path) -> list[dict]:
    out = [{"name": c["name"], "match": c["match"], "span": c["span_local"]}
           for c in json.loads(plan.read_text())["clips"]]
    return out


def v3s_rows(u: dict) -> pd.DataFrame:
    """The unit's v3 rows, marks and static points dropped: from Kaggle's
    metric_step.py for a clip, else computed here (cached)."""
    g = GPU / u["name"]
    made = g / f"output_{u['name']}" / "v3s_rows.parquet"
    if made.exists():
        return pd.read_parquet(made)
    cache = g / "v3s_rows.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    info = sd.info_of(u["name"])
    cal = st.calibration(str(info["match"]))
    path = g / f".cache/v3_balls_{u['name']}.parquet"
    n = len(read_sidecar(path)["frames"])
    rows = pd.read_parquet(path)
    rows = bs.drop_marks(rows, bs.spot_marks(rows, cal, n))
    keep = np.ones(len(rows), bool)
    for x, y in tr.static_points(rows, n):
        keep &= np.hypot(rows.px.to_numpy() - x, rows.py.to_numpy() - y) > 1.5 * tr.STATIC_PX
    rows = rows[keep]
    rows.to_parquet(cache)
    return rows


def goal_features(px: np.ndarray, py: np.ndarray, frames: np.ndarray) -> pd.DataFrame:
    """Distance to the nearer goal, velocity towards it, and whether the
    velocity ray meets its line within MOUTH_HALF of the centre."""
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 2
    dt = step / FPS
    vx = (pd.Series(px).shift(-1) - pd.Series(px).shift(1)).to_numpy() / (2 * dt)
    vy = (pd.Series(py).shift(-1) - pd.Series(py).shift(1)).to_numpy() / (2 * dt)
    d = np.stack([np.hypot(px - gx, py - gy) for gx, gy in GOALS])
    k = np.nanargmin(np.where(np.isnan(d), np.inf, d), axis=0)
    gx, gy = GOALS[k, 0], GOALS[k, 1]
    dist = np.hypot(px - gx, py - gy)
    ux, uy = (gx - px) / np.maximum(dist, 1e-6), (gy - py) / np.maximum(dist, 1e-6)
    v_to = vx * ux + vy * uy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (gx - px) / vx
        yy = py + t * vy
    ray = ((t > 0) & (np.abs(yy - gy) <= MOUTH_HALF)).astype(float)
    ray[~np.isfinite(t)] = np.nan
    f = pd.DataFrame({"g_dist": dist, "g_vto": v_to, "g_ray": ray}, index=frames)
    for o in (-12, -6, 6, 12):
        sh = f[["g_dist", "g_vto", "g_ray"]].shift(-o // step)
        f[[f"g_dist_{o:+d}", f"g_vto_{o:+d}", f"g_ray_{o:+d}"]] = sh.to_numpy()
    f["g_vto_max_next"] = pd.Series(v_to, index=frames)[::-1].rolling(6, min_periods=1).max()[::-1]
    return f


def features(u: dict) -> pd.DataFrame:
    """learn.py's features on the unit's GPU tracks, plus goal geometry;
    a clip keeps its span's frames only."""
    cache = GPU / u["name"] / "shot_features.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    info = sd.info_of(u["name"])
    cal = st.calibration(str(info["match"]))
    made = GPU / u["name"] / f"output_{u['name']}" / "metric_v3s_link.parquet"
    metric = (pd.read_parquet(made) if made.exists()
              else sd.metric_of({"name": u["name"], "kind": "clip"}, GPU))   # the dev windows
    frames = np.sort(metric[metric.cls == "player"].frame.unique())
    b = L._ball_series(metric, frames)
    rows = v3s_rows(u)
    rows = rows[rows.frame.isin(frames)]
    best = rows.sort_values("confidence", ascending=False).drop_duplicates("frame").set_index("frame")
    xy = st.image_to_pitch(np.c_[best.px, best.py + best.crop_h / 2], cal)
    c = pd.DataFrame({"c_max": best.confidence, "c_x": xy[:, 0], "c_y": xy[:, 1]}, index=best.index)
    c["c_n"] = rows[rows.confidence >= 0.10].groupby("frame").size()
    c = c.reindex(frames)
    px = np.where(b.x.notna(), b.x, c.c_x)
    py = np.where(b.y.notna(), b.y, c.c_y)
    pl = L._players(metric, frames, px, py)
    base = pd.concat([b[["has", "speed", "confidence"]].rename(columns={"confidence": "b_conf"}),
                      c[["c_max", "c_n"]], pl.drop(columns="near_id")], axis=1)
    base["x"], base["y"] = px, py
    base["dAB"] = base.dA - base.dB
    feats = [base.add_suffix("_0")]
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 2
    for o in L.OFFSETS:
        sh = base.shift(-o // step)
        feats.append(sh[["has", "speed", "c_max", "d1", "near_team", "n3", "dAB"]].add_suffix(f"_{o:+d}"))
    f = pd.concat(feats, axis=1)
    k = 6 // step
    nid, nt = pl.near_id, pl.near_team
    f["id_change"] = (nid.shift(k) != nid.shift(-k)).astype(float).where(nid.shift(k).notna() & nid.shift(-k).notna())
    f["team_change"] = (nt.shift(k) != nt.shift(-k)).astype(float).where(nt.shift(k).notna() & nt.shift(-k).notna())
    f["disp"] = np.hypot(pd.Series(px, index=frames).shift(-k) - pd.Series(px, index=frames).shift(k),
                         pd.Series(py, index=frames).shift(-k) - pd.Series(py, index=frames).shift(k))
    f["speed_max_next"] = b.speed[::-1].rolling(6, min_periods=1).max()[::-1].to_numpy()
    f["has_frac_1s"] = b.has.rolling(25 // step, center=True, min_periods=1).mean().to_numpy()
    f = pd.concat([f, goal_features(px, py, frames)], axis=1)
    if u.get("span"):
        a, z = u["span"]
        f = f[(f.index >= a) & (f.index <= z)]
    f.index.name = "frame"
    f.to_parquet(cache)
    return f


def labels(u: dict) -> list[float]:
    info = sd.info_of(u["name"])
    lab = pd.read_csv(info["dir"] / f"{u['name']}_events.csv")
    t = lab[lab.label.isin(sd.LABELS)].frame.to_numpy() / FPS
    if u.get("span"):
        a, z = (s / FPS for s in u["span"])
        t = t[(t >= a) & (t <= z)]
    return sorted(t.tolist())


def target(f: pd.DataFrame, lab: list[float]) -> pd.Series:
    t = f.index.to_numpy() / FPS
    return pd.Series([float(np.any(np.abs(np.asarray(lab) - x) <= POS_S)) for x in t], index=f.index)


def peaks(frames, p, thr):
    s = pd.Series(p).rolling(3, center=True, min_periods=1).mean().to_numpy()
    kept = []
    for i in np.argsort(-s):
        if s[i] < thr:
            break
        if all(abs(frames[i] - frames[j]) / FPS >= NMS_S for j in kept):
            kept.append(i)
    return sorted(kept)


def data(us):
    return {u["name"]: (features(u), labels(u), u["match"]) for u in us}


def cols_of(d):
    return [c for c in next(iter(d.values()))[0].columns if c != "near_team_0"]


def fit(d, names, cols):
    X = pd.concat([d[n][0][cols] for n in names])
    y = pd.concat([target(d[n][0], d[n][1]) for n in names])
    return L.model().fit(X, y)


def oof(d, names, cols):
    probs = {}
    for m in sorted({d[n][2] for n in names}):
        tr_ = [n for n in names if d[n][2] != m]
        clf = fit(d, tr_, cols)
        for n in names:
            if d[n][2] == m:
                probs[n] = clf.predict_proba(d[n][0][cols])[:, 1]
    return probs


def rate_threshold(d, names, probs, grid=np.arange(0.2, 0.96, 0.025)):
    n_lab = sum(len(d[n][1]) for n in names)
    n_ev = {float(t): sum(len(peaks(d[n][0].index.to_numpy(), probs[n], t)) for n in names) for t in grid}
    return min(n_ev, key=lambda t: (abs(n_ev[t] - n_lab), -t))


def score(d, names, probs, thr):
    tp = fp = fn = 0
    per = {}
    for n in names:
        fr = d[n][0].index.to_numpy()
        pred = [fr[i] / FPS for i in peaks(fr, probs[n], thr[d[n][2]] if isinstance(thr, dict) else thr)]
        m, f, x = sc.match_one_to_one(pred, d[n][1], TOL_S)
        tp, fp, fn = tp + len(m), fp + len(f), fn + len(x)
        mm = per.setdefault(d[n][2], [0, 0, 0])
        mm[0], mm[1], mm[2] = mm[0] + len(m), mm[1] + len(f), mm[2] + len(x)
    return {"tp": tp, "fp": fp, "fn": fn, "recall": tp / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1), "per match [tp, fp, fn]": per}


def nested(d, names, cols):
    probs, thr = {}, {}
    for m in sorted({d[n][2] for n in names}):
        tr_ = [n for n in names if d[n][2] != m]
        inner = oof(d, tr_, cols)
        thr[m] = rate_threshold(d, tr_, inner)
        clf = fit(d, tr_, cols)
        for n in names:
            if d[n][2] == m:
                probs[n] = clf.predict_proba(d[n][0][cols])[:, 1]
    return probs, thr


def train_units():
    us = [u for u in units(HERE / "clips_train.json")
          if (GPU / u["name"] / f"output_{u['name']}" / "metric_v3s_link.parquet").exists()]
    us += [{"name": w, "match": w.split("_")[1], "span": None} for w in TRAIN_WINDOWS
           if (GPU / w).exists()]
    return us


def main():
    cmd = sys.argv[1]
    us = train_units()
    d = data(us)
    names = list(d)
    cols = cols_of(d)
    probs, thr = nested(d, names, cols)
    res = {"units": len(names), "matches": sorted({d[n][2] for n in names}),
           "labels": sum(len(d[n][1]) for n in names), "thresholds": thr,
           "cv": score(d, names, probs, thr), "gate": GATE}
    res["gate passed"] = res["cv"]["recall"] >= GATE["recall"] and res["cv"]["precision"] >= GATE["precision"]
    print(json.dumps(res, indent=1))
    (HERE / "cv.json").write_text(json.dumps(res, indent=1))
    if cmd == "fit":
        if not res["gate passed"]:
            raise SystemExit("dev gate not passed: no model, no M5 processing")
        if set(res["matches"]) != set(TRAIN_MATCHES):
            raise SystemExit(f"not all training matches present: {res['matches']}")
        inner = oof(d, names, cols)
        t = rate_threshold(d, names, inner)
        clf = fit(d, names, cols)
        mp = HERE / "model.pkl"
        mp.write_bytes(pickle.dumps({"model": clf, "cols": cols, "threshold": t}))
        out = {"protocol sha256": (HERE / "protocol.json.sha256").read_text().strip(),
               "threshold": t, "model sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
               "cv": res, "units": names}
        (HERE / "frozen.json").write_text(json.dumps(out, indent=1))
        (HERE / "frozen.json.sha256").write_text(
            hashlib.sha256((HERE / "frozen.json").read_bytes()).hexdigest() + "\n")
        print("frozen, threshold", t)


if __name__ == "__main__":
    main()
