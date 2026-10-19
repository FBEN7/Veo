"""Headroom for re-ranking: on free-flight pass frames, is the top-1 a
corridor hit (3 m, >= 1 m from players), and is ANY saved candidate one?
Read-only, uses saved candidates (v1 heatmap top-8, COCO)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
from src.paths import DATA_DIR
from eval_ball_detector import frames_of, seg_dist, ground
S = Path("/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball")
def hit(c, cal, p, people):
    if not len(c): return False
    g = ground(np.atleast_2d(c), cal)[0]
    if not np.isfinite(g).all(): return False
    d, t = seg_dist(g, p["from"], p["to"])
    return d <= 3.0 and 0.05 <= t <= 0.95 and np.hypot(*(people - g).T).min() >= 1.0
for fname, det in (("heatmap_T.json", "ball_heatmap.3000.pt"), ("coco_T2.json", "coco"), ("V.json", "ball_heatmap.3000.pt"), ("V.json", "coco")):
    res = json.loads((S / fname).read_text())["results"]
    top1 = anyk = top3 = n = 0; np_ = []
    for r in res:
        w = r["window"]; info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
        cal = st.calibration(info["match"]); sets = frames_of(w)
        gt = pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{w}_gt.parquet")
        people = {f: g[["x", "y"]].to_numpy(float) for f, g in gt.groupby("frame")}
        cands = r[det]["candidates"]
        for p in sets["passes"]:
            for f in p["frames"]:
                c = np.array(cands.get(str(f), []), float).reshape(-1, 4)
                n += 1
                h = [hit(c[i], cal, p, people[f]) for i in range(len(c))]
                top1 += bool(h and h[0]); top3 += any(h[:3]); anyk += any(h)
                np_.append(len(c))
    print(f"{fname:15s} {det:22s} pass frames {n}: top-1 corridor {top1/n:.2f}, any of top-3 {top3/n:.2f}, any saved {anyk/n:.2f}, median cands {np.median(np_):.0f}")
