"""Diagnostic: is the linked ball after the kick better with the hybrid
rows? shot_flight/flight.py's measures (1.5 s after each labelled shot:
linked share, flight tracked >= 5 m, toward the goal >= 3 m, kept
candidates toward the goal) on the v3b tracks (gpu_shots3) and on the
hybrid ones, per shot of shots_learn/v3/missed.parquet (M1 left out as
flight.py does).

    VEO_UNSEAL=... python flight_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from src import ball_link as bl  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.unseal import unsealed  # noqa: E402

CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
SHIFT = np.array([52.5, 34.0])
WIN = 38
TREES = {"v3b": ROOT / ".cache/gpu_shots3", "hybrid": ROOT / ".cache/gpu_shots_hybrid"}


def measure(tree: Path, w: str, g: pd.DataFrame, cal) -> list[dict]:
    out = tree / w / f"output_{w}"
    met = pd.read_parquet(out / "metric_v3s_link.parquet")
    ball = met[met.cls == "ball"].drop_duplicates("frame").set_index("frame")
    kept = pd.read_parquet(out / "v3s_rows.parquet")
    kxy = bl.ground_points(kept, cal)
    kept = kept.assign(gx=kxy[:, 0], gy=kxy[:, 1])
    ev = pd.read_csv(CLIPS / f"{w}_events.csv")
    gt = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
    rows = []
    for s in g.itertuples():
        e = ev[ev.label.isin(["SHOT", "GOAL"])]
        e = e.iloc[(e.frame - s.frame).abs().argsort()[:1]].iloc[0]
        f0 = int(e.frame) - int(e.frame) % 2
        q = gt[(gt.frame == f0) & (gt.player_id.astype(str) == str(int(e.player_id)))]
        if q.empty or not np.isfinite(q.x.iloc[0]):
            continue
        sx, sy = float(q.x.iloc[0]), float(q.y.iloc[0])
        goal = 52.5 if sx >= 0 else -52.5
        d_shooter = abs(goal - sx)
        frames = range(f0 + 2, f0 + WIN + 1, 2)
        b = ball.reindex(frames)
        bx, by = b.x.to_numpy() - SHIFT[0], b.y.to_numpy() - SHIFT[1]
        has = np.isfinite(bx)
        travel = np.nanmax(np.hypot(bx - sx, by - sy)) if has.any() else 0.0
        closer = np.nanmax(d_shooter - np.abs(goal - bx)) if has.any() else 0.0
        k = kept[(kept.frame > f0) & (kept.frame <= f0 + WIN) & (kept.frame % 2 == 0)]
        rows.append({"unit": w, "frame": int(s.frame), "match": w.split("_")[1], "hit": bool(s.hit),
                     "linked share": has.mean(), "flight tracked": bool(travel >= 5.0),
                     "toward the goal": bool(closer >= 3.0),
                     "candidates toward the goal": k[(d_shooter - np.abs(goal - k.gx)) >= 3.0].frame.nunique() / len(frames)})
    return rows


def main():
    for m in ("128057", "132831"):
        if not unsealed(m, "shots hybrid: ball after the kick, v3b against hybrid tracks"):
            raise SystemExit("VEO_UNSEAL needed")
    shots = pd.read_parquet(ROOT / "research/session_scratchpad/shots_learn/v3/missed.parquet")
    shots = shots[shots.match != "117092"]
    res = {}
    for name, tree in TREES.items():
        rows = []
        for w, g in shots.groupby("unit"):
            rows += measure(tree, w, g, st.calibration(w.split("_")[1]))
        res[name] = pd.DataFrame(rows)
    cols = ["linked share", "flight tracked", "toward the goal", "candidates toward the goal"]
    out = {"shots": len(res["v3b"])}
    for name, d in res.items():
        out[name] = {"hit (v3 spotter)": d[d.hit][cols].mean().round(3).to_dict(),
                     "missed (v3 spotter)": d[~d.hit][cols].mean().round(3).to_dict()}
    a, b = res["v3b"], res["hybrid"]
    out["toward the goal: v3b no -> hybrid yes / v3b yes -> hybrid no"] = [
        int((~a["toward the goal"] & b["toward the goal"]).sum()), int((a["toward the goal"] & ~b["toward the goal"]).sum())]
    (HERE / "flight_compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
