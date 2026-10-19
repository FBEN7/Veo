"""Read-only: chance hit rate of each corridor (random pitch points at T pass
frames), and the T-only selection rule."""
import json, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
from src.ball_heatmap import pitch_to_image
from eval_ball_detector import frames_of
DATA = "/home/user/Veo/data/soccertrack_v2"
D = "/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/design_splits/"
rng = np.random.default_rng(1)
recs = []
for w in ("st2_118576_2nd_f015000", "st2_118578_2nd_f015000", "st2_128058_2nd_f015000"):
    info = json.load(open(f"{DATA}/{w}.json")); cal = st.calibration(info["match"])
    gt = pd.read_parquet(f"{DATA}/{w}_gt.parquet")
    people = {fr: q[["x", "y"]].to_numpy(float) for fr, q in gt.groupby("frame")}
    for p in frames_of(w)["passes"]:
        A, B = np.array(p["from"]), np.array(p["to"]); ab = B - A
        uvs = pitch_to_image(A + np.linspace(0, 1, 201)[:, None] * ab, cal)
        fr = p["frames"][1]
        G = np.c_[rng.uniform(-52.5, 52.5, 300), rng.uniform(-34, 34, 300)]
        uv = pitch_to_image(G, cal)
        t = (G - A) @ ab / (ab @ ab); tc = np.clip(t, 0, 1)
        perp = np.hypot(*(G - (A + tc[:, None] * ab)).T)
        px = np.hypot(*(uv[:, None, :] - uvs[None]).transpose(2, 0, 1)).min(axis=1)
        near = np.hypot(*(G[:, None, :] - people[fr][None]).transpose(2, 0, 1)).min(axis=1)
        recs.append(pd.DataFrame(dict(perp=perp, px=px, near=near, t=t)))
R = pd.concat(recs)
r = pd.read_csv(D + "agree_grid.csv")
ch = []
for row in r.itertuples():
    lo, hi = map(float, row.t.replace("-0.1", "m0.1").replace("-", " ").replace("m", "-").split())
    a = ((R.perp <= row.W) | (R.px <= row.P)) & (R.near >= row.R) & (R.t >= lo) & (R.t <= hi)
    ch.append(a.mean())
r["chance"] = np.round(ch, 4)
r.to_csv(D + "agree_grid.csv", index=False)
x = r[(r.R == 1.0) & (r.t == "0.0-1.0")]
best = x.T_kappa.max()
ok = x[x.T_kappa >= best - 0.05].sort_values("chance")
print("T max kappa (R 1.0, t 0-1):", best)
print(ok[["W", "P", "R", "t", "chance", "T_kappa", "T_prec", "T_rec", "V_kappa", "V_prec", "V_rec"]].head(10).to_string(index=False))
print(x[(x.W.isin([1.0, 2.0, 3.0, 4.0])) & (x.P.isin([0, 20]))][["W", "P", "chance", "T_kappa", "V_kappa"]].to_string(index=False))
