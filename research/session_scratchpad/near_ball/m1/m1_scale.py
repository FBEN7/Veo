import os, glob, sys
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, ".")
from src import soccertrack_v2 as s
CL = "data/shot_clips/soccertrack_v2"
cal = s.calibration("117092")
units = sorted(os.path.basename(p)[:-11] for p in glob.glob(f"{CL}/st2_117092_*_gt.parquet"))
F = []
for w in units[::2]:
    g = pd.read_parquet(f"{CL}/{w}_gt.parquet"); g = g[g.x.notna() & g.frame.isin(range(0, 300, 30))]
    run = os.path.dirname(os.path.realpath(f".cache/gpu_shots3/{w}/output_{w}/metric_v3s_link.parquet"))
    t = pd.read_parquet(f"{run}/shots_v3s_link/tracks.parquet"); t = t[(t.cls == "player") & t.frame.isin(range(0, 300, 30))]
    for f, a in g.groupby("frame"):
        b = t[t.frame == f]
        if len(b): F.append((a[["x", "y"]].to_numpy(float), np.c_[b.px, b.py]))
print(len(F), "frames")
def score(sx, sy, tx=0.0, ty=0.0):
    hit = n = 0
    for xy, v in F:
        p = s.image_to_pitch(np.c_[v[:, 0] * sx + tx, v[:, 1] * sy + ty], cal)
        d = np.linalg.norm(xy[:, None] - p[None], axis=2); d = np.where(np.isfinite(d), d, 1e6)
        r, c = linear_sum_assignment(np.minimum(d, 1e6)); hit += (d[r, c] <= 1.5).sum(); n += len(xy)
    return hit / n
best = []
for sx in np.arange(0.85, 1.16, 0.03):
    for sy in np.arange(0.40, 1.21, 0.03):
        best.append((score(sx, sy), sx, sy))
best.sort(reverse=True)
print("identity", round(score(1, 1), 3))
for b in best[:5]: print([round(float(x), 3) for x in b])
