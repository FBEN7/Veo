import os, glob
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
CL = "data/shot_clips/soccertrack_v2"
units = sorted(os.path.basename(p)[:-11] for p in glob.glob(f"{CL}/st2_117092_*_gt.parquet"))
def load(w, frames):
    g = pd.read_parquet(f"{CL}/{w}_gt.parquet"); g = g[g.bx.notna() & g.frame.isin(frames)]
    run = os.path.dirname(os.path.realpath(f".cache/gpu_shots3/{w}/output_{w}/metric_v3s_link.parquet"))
    t = pd.read_parquet(f"{run}/shots_v3s_link/tracks.parquet"); t = t[(t.cls == "player") & t.frame.isin(frames)]
    return g, t
def frames_of(rng):
    T, V = [], []
    for w in units:
        g, t = load(w, rng)
        for f, a in g.groupby("frame"):
            b = t[t.frame == f]
            if len(b): T.append((np.c_[a.bx + a.bw / 2, a.by + a.bh], a.role.to_numpy())); V.append(np.c_[b.px, b.py])
    return T, V
def feats(p, deg):
    x, y = p[:, 0] / 4000, p[:, 1] / 2000
    cols = [np.ones_like(x)] + [x ** i * y ** j for d in range(1, deg + 1) for i in range(d + 1) for j in [d - i]]
    return np.stack(cols, 1)
affine = lambda p: 1.02 * p + np.array([-37.5, -22.5])
def pairs(T, V, mapf, lim):
    P, Q = [], []
    for (t, _), v in zip(T, V):
        m = mapf(t); d = np.linalg.norm(m[:, None] - v[None], axis=2)
        r, c = linear_sum_assignment(np.where(d <= lim, d, 1e6))
        ok = d[r, c] <= lim; P.append(t[r[ok]]); Q.append(v[c[ok]])
    return np.concatenate(P), np.concatenate(Q)
def control(T, V, mapf, lim=43.0):
    hit = n = 0; res = []
    for (t, role), v in zip(T, V):
        t = t[role == "player"]; m = mapf(t); d = np.linalg.norm(m[:, None] - v[None], axis=2)
        r, c = linear_sum_assignment(np.where(d <= lim, d, 1e6)); ok = d[r, c] <= lim
        hit += ok.sum(); n += len(t); res += list(d[r, c][ok])
    return round(hit / n, 4), round(float(np.median(res)), 1)
Tf, Vf = frames_of(range(0, 300, 2)); Th, Vh = frames_of(range(1126, 1425, 2))
print("fit frames", len(Tf), "held-out frames", len(Th))
print("identity   held-out control, median px:", control(Th, Vh, lambda p: p))
print("affine 1.02 held-out:", control(Th, Vh, affine))
P, Q = pairs(Tf, Vf, affine, 30)
for deg in (1, 2, 3):
    mapf = affine
    for it in range(3):
        P, Q = pairs(Tf, Vf, mapf, 30 if it == 0 else 20)
        A = feats(P, deg); cx = np.linalg.lstsq(A, Q[:, 0], rcond=None)[0]; cy = np.linalg.lstsq(A, Q[:, 1], rcond=None)[0]
        mapf = (lambda cx, cy, deg: (lambda p: np.c_[feats(p, deg) @ cx, feats(p, deg) @ cy]))(cx, cy, deg)
    print(f"poly deg {deg}: fit pairs {len(P)}, held-out control, median px:", control(Th, Vh, mapf))

# diagnosis with the degree-3 map: nearest detection distance for each truth player, by image region
near, xs, ys = [], [], []
for (t, role), v in zip(Th, Vh):
    t = t[role == "player"]; m = mapf(t); d = np.linalg.norm(m[:, None] - v[None], axis=2)
    near += list(d.min(axis=1)); xs += list(m[:, 0]); ys += list(m[:, 1])
near, xs, ys = map(np.array, (near, xs, ys))
print("nearest detection, quantiles px:", np.round(np.quantile(near, [0.25, 0.5, 0.75, 0.9]), 1))
print("share <=43:", round(float((near <= 43).mean()), 3), "43-150:", round(float(((near > 43) & (near <= 150)).mean()), 3), ">150:", round(float((near > 150).mean()), 3))
bx = pd.cut(xs, [0, 1000, 1600, 2200, 2800, 4000]); by = pd.cut(ys, [0, 700, 850, 1000, 1300, 2000])
print(pd.crosstab(by, bx, values=(near <= 43), aggfunc="mean").round(2))
print(pd.crosstab(by, bx).astype(int))
dets = np.concatenate(Vh); print("detections y quantiles", np.round(np.quantile(dets[:, 1], [0.05, 0.5, 0.95]))); print("mapped truth y quantiles", np.round(np.quantile(ys, [0.05, 0.5, 0.95])))
