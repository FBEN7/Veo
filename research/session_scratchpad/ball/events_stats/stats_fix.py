"""Better distance and possession estimators, chosen on the dev windows and
checked on M1 (descriptive, like stats_eval.py, whose matching it reuses).

Distance per track: positions smoothed by a centred moving average of
`smooth_s`, then sampled every `sample_s`, steps above 40 km/h dropped. The
truth's reference is the same estimator at 1 s / 1 s on its own positions
(they sit on a 1.05 x 0.68 m grid, so they jitter too). Scored per person with a
perfect identity (detections matched to the person, chained), and in total.

Possession: the team of the nearest player within `radius_m` of the ball, at
every frame with a ball (no ball-speed gate), carried forward over frames
without one for up to `hold_s`; share of the time attributed. Truth as in
stats_eval.py (ball actions, each interval up to 6 s to the side of the first).

    python stats_fix.py tune W ...          # grid on these windows
    python stats_fix.py score CONFIG.json W ...
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/user/Veo")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from eval_soccertrack_v2 import boxes, match_feet, truth  # noqa: E402
from score_st2_actions import side_of_team  # noqa: E402
import stats_eval as se  # noqa: E402

FPS = 25.0
REF = (1.0, 1.0)
GRID_D = {"smooth_s": [0.2, 0.5, 1.0, 2.0], "sample_s": [0.08, 0.5, 1.0, 2.0]}
GRID_P = {"radius_m": [1.5, 2.5, 4.0], "hold_s": [0.0, 2.0, 5.0, 10.0]}


def path_km(df: pd.DataFrame, smooth_s: float, sample_s: float, max_gap_s: float = 0.4) -> float:
    """Distance (m) along runs of a position series (frame, x, y)."""
    df = df.sort_values("frame")
    f = df.frame.to_numpy()
    cut = np.r_[0, np.flatnonzero(np.diff(f) / FPS > max_gap_s) + 1, len(f)]
    total = 0.0
    for a, b in zip(cut[:-1], cut[1:]):
        g = df.iloc[a:b]
        if len(g) < 2:
            continue
        t = g.frame.to_numpy() / FPS
        step = max(np.median(np.diff(t)), 1e-6)
        k = max(int(round(smooth_s / step)), 1)
        x = pd.Series(g.x.to_numpy()).rolling(k, center=True, min_periods=1).mean().to_numpy()
        y = pd.Series(g.y.to_numpy()).rolling(k, center=True, min_periods=1).mean().to_numpy()
        idx = np.unique(np.searchsorted(t, np.arange(t[0], t[-1] + 1e-9, sample_s)).clip(0, len(t) - 1))
        if len(idx) < 2:
            continue
        d = np.hypot(np.diff(x[idx]), np.diff(y[idx]))
        v = d / np.diff(t[idx]) * 3.6
        total += float(d[v < 40].sum())
    return total


def window_data(w: str, ball: str = "v3s_link"):
    metric = pd.read_parquet(ROOT / f"output_{w}/tune_{ball}/metric.parquet")
    gt = pd.read_parquet(ROOT / f"data/soccertrack_v2/{w}_gt.parquet")
    stride = int(np.gcd.reduce(metric.frame.unique().astype(int))) or 1
    gt = gt[(gt.frame % stride == 0) & (gt.role == "player")]
    gtb = truth(ROOT / f"data/soccertrack_v2/{w}_gt.parquet")
    gtb = gtb[gtb.frame % stride == 0]
    det = boxes(metric)
    det = det[det.frame % stride == 0]
    person = match_feet(det, gtb)
    sides = side_of_team(det, person, gtb)
    p = metric.loc[det.index]
    pos = pd.DataFrame({"frame": p.frame, "x": p.x - se.SHIFT[0], "y": p.y - se.SHIFT[1], "person": person})
    pos = pos.dropna(subset=["person"])
    pos = pos[pos.person.isin(set(gt.track_id))]
    ev = pd.read_csv(ROOT / f"data/soccertrack_v2/{w}_events.csv").sort_values("frame")
    t = ev.frame.to_numpy() / FPS
    dur = np.minimum(np.diff(t), se.POSS_MAX_GAP_S)
    share = pd.Series(dur, index=ev.side.to_numpy()[:-1]).groupby(level=0).sum()
    return {"metric": metric, "gt": gt, "pos": pos, "sides": sides,
            "truth_left": float(share.get("left", 0.0) / share.sum())}


def distance_errors(d, smooth_s, sample_s):
    ref = {pid: path_km(g, *REF) for pid, g in d["gt"].groupby("track_id")}
    ours = {pid: path_km(g, smooth_s, sample_s) for pid, g in d["pos"].groupby("person")}
    gt_cov = {pid: path_km(d["gt"][d["gt"].track_id == pid][d["gt"].frame.isin(set(g.frame))], *REF)
              for pid, g in d["pos"].groupby("person")}
    # per person, on the frames the person was detected (coverage apart)
    e = [abs(ours[p] - gt_cov[p]) / gt_cov[p] for p in ours if gt_cov.get(p, 0) > 30]
    return {"per person median abs error": float(np.median(e)) if e else None,
            "truth km (all frames)": sum(ref.values()) / 1000,
            "ours km on matched frames": sum(ours.values()) / 1000,
            "truth km on matched frames": sum(gt_cov.values()) / 1000}


def possession_left(d, radius_m, hold_s):
    m = d["metric"]
    ball = m[m.cls == "ball"][["frame", "x", "y"]]
    pl = m[(m.cls == "player") & m.team.isin(["team_A", "team_B"])][["frame", "x", "y", "team"]]
    j = ball.merge(pl, on="frame", suffixes=("_b", ""))
    j["d"] = np.hypot(j.x - j.x_b, j.y - j.y_b)
    near = j.loc[j.groupby("frame").d.idxmin()]
    near = near[near.d <= radius_m].set_index("frame").team
    frames = np.sort(m.frame.unique())
    s = near.reindex(frames)
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 2
    lim = int(hold_s * FPS / step)
    s = s.ffill(limit=lim) if lim > 0 else s
    s = s.dropna().map(d["sides"])
    if s.empty:
        return None, 0
    return float((s == "left").mean()), int(len(s))


def tune(windows):
    data = {w: window_data(w) for w in windows}
    rows = []
    for sm, sa in itertools.product(*GRID_D.values()):
        errs = [distance_errors(data[w], sm, sa)["per person median abs error"] for w in windows]
        rows.append({"kind": "distance", "smooth_s": sm, "sample_s": sa, "mean of window medians": float(np.mean(errs)),
                     "per window": dict(zip(windows, errs))})
    for r, h in itertools.product(*GRID_P.values()):
        res = [possession_left(data[w], r, h) for w in windows]
        errs = [abs(v - data[w]["truth_left"]) for (v, _), w in zip(res, windows) if v is not None]
        rows.append({"kind": "possession", "radius_m": r, "hold_s": h, "mean abs error": float(np.mean(errs)) if errs else None,
                     "frames": [n for _, n in res]})
    best_d = min((r for r in rows if r["kind"] == "distance"), key=lambda r: r["mean of window medians"])
    best_p = min((r for r in rows if r["kind"] == "possession" and r["mean abs error"] is not None),
                 key=lambda r: r["mean abs error"])
    base_d = [r for r in rows if r["kind"] == "distance" and r["smooth_s"] == 0.2 and r["sample_s"] == 0.08][0]
    return {"windows": windows, "best distance": best_d, "baseline-like distance (0.2 s, every frame)": base_d,
            "best possession": best_p, "grid": rows}


def score(cfg, windows):
    out = []
    for w in windows:
        d = window_data(w)
        de = distance_errors(d, cfg["smooth_s"], cfg["sample_s"])
        pv, n = possession_left(d, cfg["radius_m"], cfg["hold_s"])
        out.append({"window": w, **de, "possession truth left": d["truth_left"], "possession ours left": pv,
                    "possession abs error": abs(pv - d["truth_left"]) if pv is not None else None, "frames": n})
    return out


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        res = tune(sys.argv[2:])
        Path(__file__).with_name("stats_fix_tune.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({k: v for k, v in res.items() if k != "grid"}, indent=1, default=float))
    else:
        cfg = json.loads(Path(sys.argv[2]).read_text())
        print(json.dumps(score(cfg, sys.argv[3:]), indent=1, default=float))
