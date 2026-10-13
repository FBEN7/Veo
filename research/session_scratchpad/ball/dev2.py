import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
from src.ball_heatmap import pitch_to_image, local_scale
df = pd.read_csv("dev_analysis.csv")
DATA = "/home/user/Veo/data/soccertrack_v2"
B = df[df.j == "B"].drop_duplicates(["w", "f", "src"])
ff = B[(B.la == "PASS") & B.same & (B.L >= 10) & (B.dur >= 0.8) & (B.s >= 0.2) & (B.s <= 0.8)].drop_duplicates(["w", "f"])
cals = {}
rows = []
for r in ff.itertuples():
    info = json.load(open(f"{DATA}/{r.w}.json"))
    cal = cals.setdefault(r.w, st.calibration(info["match"]))
    e = pd.read_csv(f"{DATA}/{r.w}_events.csv").sort_values("frame")
    a = e[e.frame <= r.f].iloc[-1]; b = e[e.frame > r.f].iloc[0]
    A, Bp = np.array([a.x, a.y]), np.array([b.x, b.y])
    pts = A + np.linspace(0, 1, 101)[:, None] * (Bp - A)
    uv = pitch_to_image(pts, cal)
    g = np.array([r.gx, r.gy])
    ball_uv = pitch_to_image([g], cal)[0]
    d = np.hypot(*(uv - ball_uv).T)
    k = d.argmin()
    # closest pitch point and components
    c = pts[k]
    rows.append(dict(w=r.w, f=r.f, px=d.min(), dx=g[0]-c[0], dy=g[1]-c[1], perp=r.perp_seg, sx=r.sx, sy=r.sy, gy=r.gy, L=r.L, src=r.src))
x = pd.DataFrame(rows)
q = [0.25, .5, .75, .9, .95]
print("pixel dist ball->projected segment", np.round(x.px.quantile(q).to_numpy(), 1))
print("share px<=8,12,16,24,32,48:", [round(float((x.px <= t).mean()), 2) for t in (8, 12, 16, 24, 32, 48)])
print("|dx| m", np.round(x.dx.abs().quantile(q).to_numpy(), 2), " |dy| m", np.round(x.dy.abs().quantile(q).to_numpy(), 2))
print("dy mean (signed, +=towards camera)", round(x.dy.mean(), 2), "median", round(x.dy.median(), 2))
print("perp in units of 1/sy (px depth):", np.round((x.dy.abs() * x.sy).quantile(q).to_numpy(), 1))
print("far (gy<-10) n", (x.gy < -10).sum(), "px median", x[x.gy < -10].px.median(), "perp median", x[x.gy < -10].perp.median())
print("mid n", ((x.gy >= -10) & (x.gy <= 10)).sum(), "px median", x[(x.gy >= -10) & (x.gy <= 10)].px.median(), x[(x.gy >= -10) & (x.gy <= 10)].perp.median())
print("near (gy>10) n", (x.gy > 10).sum(), "px median", x[x.gy > 10].px.median(), x[x.gy > 10].perp.median())
# within-ellipse in metres
for W in (1.5, 2.0, 2.5, 3.0):
    print(f"perp <= {W} m:", round(float((x.perp <= W).mean()), 2), "| perp<=W or px<=16:", round(float(((x.perp <= W) | (x.px <= 16)).mean()), 2))
