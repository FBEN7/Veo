import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
from src.ball_heatmap import pitch_to_image, local_scale
t0 = time.time()
ext, cells, nearmin, ends = [], [], [], []
for m, h in (("118575", 1), ("128058", 1)):
    cal = st.calibration(m)
    ev = st.events(m); ev = ev[ev.half == h].sort_values("frame").reset_index(drop=True)
    g = pd.read_parquet(f"/home/user/Veo/.cache/soccertrack_v2/gsr/{m}_{st._half(h)}.parquet", columns=["frame", "player_id", "role", "x", "y"])
    gp = g[g.role.isin(("player", "goalkeeper"))].sort_values("frame")
    fr = gp.frame.to_numpy(); xy = gp[["x", "y"]].to_numpy(float)
    g = g[g.player_id.notna()].copy(); g["player_id"] = g.player_id.astype(str)
    e = ev.merge(g[["frame", "player_id", "x", "y"]], on=["frame", "player_id"], how="left")
    E = e.to_dict("records")
    for a, b in zip(E[:-1], E[1:]):
        if a["label"] != "PASS" or b["side"] != a["side"] or b["player_id"] == a["player_id"] or pd.isna(a["x"]) or pd.isna(b["x"]):
            continue
        A, B = np.array([a["x"], a["y"]]), np.array([b["x"], b["y"]]); L = np.hypot(*(B - A)); d = b["frame"] - a["frame"]
        if L < 10 or d < 20:
            continue
        for s in (0.3, 0.5, 0.7):
            lo, hi = max(s - 0.2, 0.08), min(s + 0.35, 0.95)
            pts = A + np.linspace(lo, hi, 20)[:, None] * (B - A)
            uv = pitch_to_image(pts, cal)
            ext.append(max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])) + 40)
            f = int(round(a["frame"] + s * d))
            i0, i1 = np.searchsorted(fr, f), np.searchsorted(fr, f, side="right")
            P = xy[i0:i1]
            if len(P):
                dd = np.hypot(*(P[:, None, :] - pts[None]).transpose(2, 0, 1)).min(axis=1)
                nearmin.append(int((dd < 2.5).sum()))
    print(m, h, len(ext), f"{time.time()-t0:.0f}s", flush=True)
ext = np.array(ext); nearmin = np.array(nearmin)
print("corridor-window extent px quantiles", np.round(np.quantile(ext, [.25, .5, .75, .9]), 0), "share <=240/300/360:", [round(float((ext <= x).mean()), 2) for x in (240, 300, 360)])
print("players within 2.5 m of the corridor centreline (time window) count dist:", np.bincount(nearmin)[:8], "share >=1:", round(float((nearmin >= 1).mean()), 2))
