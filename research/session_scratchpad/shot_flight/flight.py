"""Is the ball tracked after the kick? (plan.json)

    VEO_UNSEAL=... python flight.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import ball_link as bl  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.unseal import unsealed  # noqa: E402

CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
SHIFT = np.array([52.5, 34.0])          # metric tracks: corner origin; truth: centre spot
WIN = 38                                # frames (1.5 s)
NEW_PLAYERS = {"128057", "132831"}


def main():
    for m in NEW_PLAYERS:
        if not unsealed(m, "shot_flight: ball after the kick"):
            raise SystemExit("VEO_UNSEAL needed")
    shots = pd.read_parquet(ROOT / "research/session_scratchpad/shots_learn/v3/missed.parquet")
    shots = shots[shots.match != "117092"]
    rows = []
    for w, g in shots.groupby("unit"):
        m = w.split("_")[1]
        base = ROOT / (".cache/gpu_nograss" if m in NEW_PLAYERS else ".cache/gpu") / w / f"output_{w}"
        met = pd.read_parquet(base / "metric_v3s_link_nograss.parquet")
        ball = met[met.cls == "ball"].drop_duplicates("frame").set_index("frame")
        kept = pd.read_parquet(base / "v3s_rows.parquet")
        cal = st.calibration(m)
        kxy = bl.ground_points(kept, cal)
        kept = kept.assign(gx=kxy[:, 0], gy=kxy[:, 1])
        ev = pd.read_csv(CLIPS / f"{w}_events.csv")
        gt = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
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
            k_toward = k[(d_shooter - np.abs(goal - k.gx)) >= 3.0].frame.nunique() / len(frames)
            rows.append({"unit": w, "match": m, "hit": bool(s.hit), "linked share": has.mean(),
                         "flight tracked": bool(travel >= 5.0), "toward the goal": bool(closer >= 3.0),
                         "candidates toward the goal": k_toward, "shooter to goal line m": d_shooter})
    d = pd.DataFrame(rows)
    cols = ["linked share", "flight tracked", "toward the goal", "candidates toward the goal", "shooter to goal line m"]
    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0], "shots": len(d),
           "hit": d[d.hit][cols].mean().round(3).to_dict() | {"n": int(d.hit.sum())},
           "missed": d[~d.hit][cols].mean().round(3).to_dict() | {"n": int((~d.hit).sum())},
           "per match (missed: flight tracked)": d[~d.hit].groupby("match")["flight tracked"].agg(["size", "mean"]).round(2).to_dict("index")}
    print(json.dumps(out, indent=1))
    (HERE / "flight.json").write_text(json.dumps(out, indent=1))
    d.to_parquet(HERE / "flight_rows.parquet")


if __name__ == "__main__":
    main()
