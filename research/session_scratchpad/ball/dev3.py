import numpy as np, pandas as pd
df = pd.read_csv("dev_analysis.csv")
B = df[df.j == "B"].drop_duplicates(["w", "f", "src"])
ff = B[(B.la == "PASS") & B.same & (B.L >= 10) & (B.dur >= 0.8) & (B.s >= 0.2) & (B.s <= 0.8)].drop_duplicates(["w", "f"])
T = ff[ff.tdir.isin(["tiles_T", "tiles_ff"])]
V = ff[~ff.tdir.isin(["tiles_T", "tiles_ff"])]
q = [0.25, .5, .75, .9, .95]
for name, x in (("T", T), ("V", V)):
    print(name, len(x), "perp", np.round(x.perp_seg.quantile(q).to_numpy(), 2), "recv_now", np.round(x.recv_now.quantile(q).to_numpy(), 2),
          "d_interp", np.round(x.d_interp.quantile(q).to_numpy(), 2), "t-s mean", round((x.t - x.s).mean(), 3),
          "|t-s|", np.round((x.t - x.s).abs().quantile(q).to_numpy(), 2), "near<1.5", round(float((x.near < 1.5).mean()), 2), "near<2", round(float((x.near < 2).mean()), 2))
# coverage of candidate corridor designs on T only
def cover(x, W, lo, hi, rp):
    ok = (x.perp_seg <= W) & ((x.t - x.s) >= lo) & ((x.t - x.s) <= hi) & (x.near >= rp)
    return round(float(ok.mean()), 2)
for W in (1.5, 2.0, 2.5, 3.0):
    print("W", W, [ (lo, hi, rp, cover(T, W, lo, hi, rp)) for lo, hi in ((-1, 1), (-0.2, 0.3), (-0.25, 0.4), (-0.15, 0.25)) for rp in (0, 1.0, 1.5)])
