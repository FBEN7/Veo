import os, glob, sys
import numpy as np, pandas as pd
CL = "data/shot_clips/soccertrack_v2"
units = sorted(os.path.basename(p)[:-11] for p in glob.glob(f"{CL}/st2_117092_*_gt.parquet"))
def load(w, frames):
    g = pd.read_parquet(f"{CL}/{w}_gt.parquet"); g = g[g.bx.notna() & g.frame.isin(frames)]
    run = os.path.dirname(os.path.realpath(f".cache/gpu_shots3/{w}/output_{w}/metric_v3s_link.parquet"))
    t = pd.read_parquet(f"{run}/shots_v3s_link/tracks.parquet"); t = t[(t.cls == "player") & t.frame.isin(frames)]
    return g, t
pairs = {}
for half in ("1st", "2nd"):
    T, V = [], []
    for w in [u for u in units if f"_{half}_" in u]:
        g, t = load(w, range(0, 300, 6))
        for f, a in g.groupby("frame"):
            b = t[t.frame == f]
            if b.empty: continue
            T.append(np.c_[a.bx + a.bw / 2, a.by + a.bh]); V.append(np.c_[b.px, b.py])
    pairs[half] = (T, V)
    print(half, len(T), "frames")
# grid over scales; vote offsets video - s*truth
for half, (T, V) in pairs.items():
    best = None
    for sx in np.arange(0.8, 1.8, 0.02):
        for sy in np.arange(0.8, 2.6, 0.02):
            off = np.concatenate([(v[None, :, :] - np.array([sx, sy]) * t[:, None, :]).reshape(-1, 2) for t, v in zip(T[::3], V[::3])])
            H, xe, ye = np.histogram2d(off[:, 0], off[:, 1], bins=[np.arange(-3000, 3000, 15), np.arange(-3000, 3000, 15)])
            k = H.max()
            if best is None or k > best[0]:
                i, j = np.unravel_index(H.argmax(), H.shape); best = (k, sx, sy, xe[i] + 7.5, ye[j] + 7.5)
    print(half, "best votes, sx, sy, tx, ty:", [round(float(x), 3) for x in best])
