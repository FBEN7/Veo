"""Physical stats and possession of the panoramic pipeline against SoccerTrack v2's
ground truth (descriptive; `src/stats.py` as the report computes them).

Per window, on the cached pitch tracks of a ball input (`output_<w>/tune_<ball>/
metric.parquet`, players in pitch metres, origin at the corner):

- **distance** (km per player): the report's `physical_stats` per track; the
  ground truth's distance per person from its own positions, smoothed the same
  way, at the same stride. Three readings:
  - total over outfield players, ours / truth;
  - per person with a perfect identity (the detections matched to the person,
    frame by frame, chained): what positions alone cost;
  - per person with our identities (each track to the person it matches most):
    what identities cost on top.
- **positions**: share of outfield person-frames with a matched detection, and
  the median distance between its pitch point and the truth's.
- **possession**: the report's `possession_proxy` (nearest player within 3 m of
  a ball slower than 20 km/h), teams mapped to sides as the actions scorer does,
  against the truth's share from the ball actions: each interval between two
  consecutive actions, up to 6 s, goes to the side of the first.

    python stats_eval.py [--ball v3s_link] WINDOW ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/user/Veo")
sys.path.insert(0, str(ROOT))
from eval_soccertrack_v2 import boxes, match_feet, truth  # noqa: E402
from score_st2_actions import side_of_team  # noqa: E402
from src import stats  # noqa: E402

FPS = 25.0
SHIFT = np.array([52.5, 34.0])     # metric tracks: corner origin; truth: centre spot
POSS_MAX_GAP_S = 6.0


def _dist(xy: np.ndarray, t: np.ndarray) -> float:
    """Distance along a path the way physical_stats measures it: 5-sample
    rolling median, steps implying >= 40 km/h dropped."""
    if len(t) < 2:
        return 0.0
    s = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}).rolling(5, min_periods=1, center=True).median()
    d = np.hypot(np.diff(s.x), np.diff(s.y))
    v = d / np.diff(t) * 3.6
    return float(d[v < stats.MAX_SPEED_KMH].sum())


def _chain_dist(df: pd.DataFrame, max_gap: int = 10) -> float:
    """Sum of path distances over runs of frames no more than max_gap apart."""
    df = df.sort_values("frame")
    f = df.frame.to_numpy()
    cut = np.r_[0, np.flatnonzero(np.diff(f) > max_gap) + 1, len(f)]
    return sum(_dist(df[["x", "y"]].to_numpy()[a:b], f[a:b] / FPS) for a, b in zip(cut[:-1], cut[1:]))


def window_stats(w: str, ball: str) -> dict:
    metric = pd.read_parquet(ROOT / f"output_{w}/tune_{ball}/metric.parquet")
    gt = pd.read_parquet(ROOT / f"data/soccertrack_v2/{w}_gt.parquet")
    stride = int(np.gcd.reduce(metric.frame.unique().astype(int))) or 1
    gt = gt[gt.frame % stride == 0]
    gtb = truth(ROOT / f"data/soccertrack_v2/{w}_gt.parquet")
    gtb = gtb[gtb.frame % stride == 0]
    det = boxes(metric)
    det = det[det.frame % stride == 0]
    person = match_feet(det, gtb)
    sides = side_of_team(det, person, gtb)
    out = {"window": w, "ball": ball, "team map": sides}

    # --- positions: coverage and error of matched detections (outfield)
    p = metric.loc[det.index]
    pos = pd.DataFrame({"frame": p.frame, "track": p.track_id, "team": p.team,
                        "x": p.x - SHIFT[0], "y": p.y - SHIFT[1], "person": person})
    g_out = gt[gt.role == "player"]
    m = pos.dropna(subset=["person"]).merge(
        g_out[["frame", "track_id", "x", "y"]].rename(columns={"track_id": "person", "x": "gx", "y": "gy"}),
        on=["frame", "person"])
    out["positions"] = {"coverage": round(len(m.drop_duplicates(["frame", "person"])) / max(len(g_out), 1), 3),
                        "median error m": round(float(np.hypot(m.x - m.gx, m.y - m.gy).median()), 2)}

    # --- distance: truth per outfield person
    gdist = {pid: _dist(g.sort_values("frame")[["x", "y"]].to_numpy(), g.sort_values("frame").frame.to_numpy() / FPS)
             for pid, g in g_out.groupby("track_id")}
    ours = stats.physical_stats(metric)
    out["distance"] = {"truth total km": round(sum(gdist.values()) / 1000, 2),
                       "ours total km (team tracks)": round(float(ours.distance_m.sum()) / 1000, 2) if len(ours) else 0.0}
    # perfect identity: the matched detections of each person, chained
    oracle = {pid: _chain_dist(g) for pid, g in m.groupby("person")}
    # our identities: each track to the person it matches most
    owner = pos.dropna(subset=["person"]).groupby("track").person.agg(lambda s: s.mode().iloc[0])
    tracked = {}
    for tid, g in pos[pos.track.isin(owner.index)].groupby("track"):
        tracked[owner[tid]] = tracked.get(owner[tid], 0.0) + _chain_dist(g)

    def err(est):
        e = [abs(est.get(pid, 0.0) - d) / d for pid, d in gdist.items() if d > 50]
        return round(float(np.median(e)), 3) if e else None
    out["distance"] |= {"per person median abs error, perfect identity": err(oracle),
                        "per person median abs error, our identities": err(tracked),
                        "persons": len(gdist)}

    # --- possession: truth from the actions, ours from possession_proxy
    ev = pd.read_csv(ROOT / f"data/soccertrack_v2/{w}_events.csv").sort_values("frame")
    t = ev.frame.to_numpy() / FPS
    dur = np.minimum(np.diff(t), POSS_MAX_GAP_S)
    share_t = pd.Series(dur, index=ev.side.to_numpy()[:-1]).groupby(level=0).sum()
    share_t = share_t / share_t.sum()
    prox = stats.possession_proxy(metric)
    ours_side = {sides.get(k): v for k, v in prox.items() if k in ("team_A", "team_B") and v is not None}
    out["possession"] = {"truth left": round(float(share_t.get("left", 0.0)), 3),
                         "ours left": ours_side.get("left"),
                         "frames used": prox.get("frames_used")}
    if ours_side.get("left") is not None:
        out["possession"]["abs error"] = round(abs(ours_side["left"] - out["possession"]["truth left"]), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ball", default="v3s_link")
    ap.add_argument("--out")
    ap.add_argument("windows", nargs="+")
    args = ap.parse_args()
    res = [window_stats(w, args.ball) for w in args.windows]
    for r in res:
        print(json.dumps(r))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
