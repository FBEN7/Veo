import sys, itertools
sys.path.insert(0, "/home/user/Veo")
import numpy as np, pandas as pd
from src import ball_link as bl
K = np.array([[1e6, 0, 2048], [0, 1e6, 540], [0, 0, 1]]); s = 20.0
Hp = np.array([[s, 0, 2048 - 52.5 * s], [0, s, 540 - 34 * s], [0, 0, 1]])
cal = {"K": K, "D": np.zeros(4), "Knew": K, "Hinv": np.linalg.inv(Hp)}
rng = np.random.default_rng(1)
bad = 0
for trial in range(300):
    nf = rng.integers(6, 15)
    frames = np.arange(nf) * 2
    rows = []
    for f in frames:
        for _ in range(rng.choice([0, 1, 1, 2], p=[0.4, 0.3, 0.2, 0.1])):
            x, y = rng.uniform(-2.5, 2.5), rng.uniform(-1.5, 1.5)
            rows.append(dict(frame=int(f), cls="ball", px=2048 + x * s, py=540 + y * s,
                             crop_h=0.0, confidence=rng.uniform(0.3, 0.99)))
    if not rows:
        continue
    rows = pd.DataFrame(rows)
    players = pd.DataFrame(dict(frame=frames, cls="player", px=2048.0, py=540.0 + 100 * s, crop_h=40.0))
    c_null = float(rng.choice([0.3, 1.0, 3.0])); tau = 0.5
    out = bl.link(rows, cal, players, c_null=c_null, c_static=0.0, tau=tau)
    xy = bl.ground_points(rows, cal); sig = bl.ground_sigma(xy, cal)
    fr = rows.frame.to_numpy(); p = rows.confidence.to_numpy()
    unary = -bl._logit(p) + bl._logit(tau)
    grid = np.unique(np.r_[fr, frames]); g = np.searchsorted(grid, fr)
    def cost(sel):
        sel = sorted(sel, key=lambda i: g[i])
        c = c_null * (len(grid) - len(sel)) + unary[sel].sum()
        for a, b in zip(sel, sel[1:]):
            if g[b] - g[a] <= bl.MAX_SKIP + 1:
                c += bl.motion_cost(xy[[a]], sig[[a]], fr[[a]], xy[[b]], sig[[b]], fr[[b]], 25)[0, 0]
        return c
    opts = [[None] + list(np.flatnonzero(g == j)) for j in range(len(grid))]
    best = np.inf
    for combo in itertools.product(*opts):
        sel = [i for i in combo if i is not None]
        best = min(best, cost(sel))
    got = cost([rows.index.get_loc(i) for i in out.index])
    if abs(got - best) > 1e-9:
        bad += 1
        if bad < 4: print("mismatch", trial, got, best, c_null)
print("trials done, mismatches:", bad)
