import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
from src.ball_heatmap import pitch_to_image
df = pd.read_csv("dev_analysis.csv")
DATA = "/home/user/Veo/data/soccertrack_v2"
B = df[df.j == "B"].drop_duplicates(["w", "f", "src"])
ff = B[(B.la == "PASS") & B.same & (B.L >= 10) & (B.dur >= 0.8) & (B.s >= 0.2) & (B.s <= 0.8)].drop_duplicates(["w", "f"]).copy()
cals = {}
px, pxv = [], []
for r in ff.itertuples():
    info = json.load(open(f"{DATA}/{r.w}.json"))
    cal = cals.setdefault(r.w, st.calibration(info["match"]))
    e = pd.read_csv(f"{DATA}/{r.w}_events.csv").sort_values("frame")
    a = e[e.frame <= r.f].iloc[-1]; b = e[e.frame > r.f].iloc[0]
    A, Bp = np.array([a.x, a.y]), np.array([b.x, b.y])
    pts = A + np.linspace(0, 1, 201)[:, None] * (Bp - A)
    uv = pitch_to_image(pts, cal)
    buv = pitch_to_image([[r.gx, r.gy]], cal)[0]
    d = uv - buv
    k = np.hypot(*d.T).argmin()
    px.append(np.hypot(*d[k])); pxv.append(-d[k][1])  # + = ball above (smaller v) the segment in image
ff["px"] = px; ff["pxv"] = pxv
ff.to_csv("ff_T_V.csv", index=False)
T = ff[ff.tdir.isin(["tiles_T", "tiles_ff"])]
print("T px quantiles", np.round(T.px.quantile([.25,.5,.75,.9,.95]).to_numpy(),1), "ball above segment share", round(float((T.pxv>0).mean()),2), "median signed", round(T.pxv.median(),1))
def cover(x, W, P, lo, hi, rp):
    ok = ((x.perp_seg <= W) | (x.px <= P)) & ((x.t - x.s) >= lo) & ((x.t - x.s) <= hi) & (x.near >= rp)
    return round(float(ok.mean()), 2)
for W in (2.0, 2.5, 3.0, 4.0):
    for P in (0, 16, 24):
        print(f"W {W} P {P}:", [(lo, hi, rp, cover(T, W, P, lo, hi, rp)) for lo, hi in ((-1, 1), (-0.2, 0.35)) for rp in (0, 0.8, 1.2)])
print("safety (negatives must be outside): perp<=6 or px<=48 covers", round(float(((T.perp_seg<=6)|(T.px<=48)).mean()),2), " perp<=8|px<=64", round(float(((T.perp_seg<=8)|(T.px<=64)).mean()),2))
print(T.sort_values("perp_seg").tail(6)[["w","f","src","L","s","t","perp_seg","px","pxv","near"]].round(2))
