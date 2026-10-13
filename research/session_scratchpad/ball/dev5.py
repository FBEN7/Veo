import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
from src.ball_heatmap import pitch_to_image
df = pd.read_csv("dev_analysis.csv")
DATA = "/home/user/Veo/data/soccertrack_v2"

def line_dist(g):
    x, y = abs(g[0]), abs(g[1])
    d = [abs(y - 34) if x <= 52.5 else 99, abs(x - 52.5) if y <= 34 else 99, abs(g[0]) if y <= 34 else 99,
         abs(x - 36) if y <= 20.16 else 99, abs(y - 20.16) if 36 <= x <= 52.5 else 99,
         abs(x - 47) if y <= 9.16 else 99, abs(y - 9.16) if 47 <= x <= 52.5 else 99,
         abs(np.hypot(g[0], g[1]) - 9.15)]
    return min(d)
def spot_dist(g):
    return min(np.hypot(g[0] - a, g[1] - b) for a, b in ((0, 0), (41.5, 0), (-41.5, 0)))
cals = {}
rows = []
for r in df.itertuples():
    info = json.load(open(f"{DATA}/{r.w}.json"))
    cal = cals.setdefault(r.w, st.calibration(info["match"]))
    g = np.array([r.gx, r.gy])
    e = pd.read_csv(f"{DATA}/{r.w}_events.csv").sort_values("frame")
    a = e[e.frame <= r.f].iloc[-1]; b = e[e.frame > r.f].iloc[0]
    A, Bp = np.array([a.x, a.y]), np.array([b.x, b.y])
    if np.isfinite(A).all() and np.isfinite(Bp).all():
        pts = A + np.linspace(0, 1, 201)[:, None] * (Bp - A)
        uv = pitch_to_image(pts, cal); buv = pitch_to_image([g], cal)[0]
        px = float(np.hypot(*(uv - buv).T).min())
    else:
        px = np.nan
    rows.append(dict(px=px, line=line_dist(g), spot=spot_dist(g), outside=(abs(g[0]) > 52.5 or abs(g[1]) > 34)))
df = pd.concat([df, pd.DataFrame(rows)], axis=1)
ffm = (df.la == "PASS") & df.same & (df.L >= 10) & (df.dur >= 0.8) & (df.s >= 0.2) & (df.s <= 0.8)
df["ffm"] = ffm
df["J"] = df.j.map(lambda s: "B" if s == "B" else ("B*" if s == "B*" else "N"))
T = df[df.tdir.isin(["tiles_T", "tiles_ff"])]
# automatic corridor measure vs judge, on T free-flight tiles
x = T[T.ffm]
for W, P, rp, lo, hi in ((1.0, 0, 2.0, -9, 9), (2.5, 0, 1.0, -9, 9), (2.5, 20, 1.0, -9, 9), (2.5, 20, 1.0, -0.2, 0.35), (3.0, 24, 1.0, -0.25, 0.4), (3.0, 24, 0.8, -9, 9)):
    auto = ((x.perp_seg <= W) | (x.px <= P)) & (x.near >= rp) & ((x.t - x.s) >= lo) & ((x.t - x.s) <= hi)
    j = x.J == "B"
    tp, fp, fn, tn = int((auto & j).sum()), int((auto & ~j).sum()), int((~auto & j).sum()), int((~auto & ~j).sum())
    print(f"W{W} P{P} rp{rp} t-s[{lo},{hi}]: auto&B {tp}, auto&notB {fp}, B missed {fn}, both no {tn}; precision {tp/max(tp+fp,1):.2f} recall {tp/max(tp+fn,1):.2f}")
    print("    false autos:", x[auto & ~j][["src", "J", "near", "perp_seg", "px", "line", "spot"]].round(1).values.tolist())
print()
# failure modes of heatmap and coco top-1 when judged N, on T (all tiles)
for src in ("ball_heatmap.3000.pt", "coco"):
    n = T[(T.src == src) & (T.J != "B")].drop_duplicates(["w", "f"])
    print(src, "non-ball top-1 on T:", len(n), "| B*:", int((n.J == "B*").sum()),
          "| within 1.0 m of a player:", int((n.near <= 1.0).sum()), "| 1-2 m:", int(((n.near > 1) & (n.near <= 2)).sum()),
          "| on a line (<0.5 m) and >2 m from players:", int(((n.line <= 0.5) & (n.near > 2)).sum()),
          "| spot <1 m:", int((n.spot <= 1).sum()), "| outside pitch:", int(n.outside.sum()), "| other:",
          int(((n.near > 2) & (n.line > 0.5) & (n.spot > 1) & ~n.outside & (n.J == "N")).sum()))
