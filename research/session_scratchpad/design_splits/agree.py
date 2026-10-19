import itertools
import numpy as np, pandas as pd
D = "/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/design_splits/"
df = pd.read_csv(D + "tiles_ff_geometry.csv")
df["B"] = df.j == "B"
# duplicates across checks (judge consistency)
det = df[df.src != "decoy"]
dup = det.groupby(["w", "f", "src"]).filter(lambda g: len(g) > 1)
print("re-judged candidates:", dup.groupby(["w", "f", "src"]).ngroups,
      "consistent:", int(dup.groupby(["w", "f", "src"]).j.nunique().eq(1).sum()))
# one row per (w,f,src): prefer the free-flight check judgement
det = det.sort_values("tdir", key=lambda s: s.isin(["tiles_ff", "tiles_ffV"]), ascending=False)
det = det.drop_duplicates(["w", "f", "src"])
dec = df[df.src == "decoy"]
print("unique detector tiles at pass frames: T", int((det.set == "T").sum()), "V", int((det.set == "V").sum()),
      "| decoys T", int((dec.set == "T").sum()), "V", int((dec.set == "V").sum()))


def auto(x, W, P, R, tlo, thi):
    return (((x.perp <= W) | (x.px <= P)) & (x.near >= R) & (x.t >= tlo) & (x.t <= thi))


def stats(x, a):
    b = x.B
    tp, fp, fn, tn = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n / n
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, acc=round(po, 3), kappa=round(kappa, 3),
                prec=round(tp / max(tp + fp, 1), 3), rec=round(tp / max(tp + fn, 1), 3))


grid = list(itertools.product((1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0), (0, 12, 20, 30),
                              (0.0, 0.5, 1.0, 1.5, 2.0), ((0.1, 0.9), (0.0, 1.0), (-0.1, 1.1))))
T = det[det.set == "T"]; V = det[det.set == "V"]
res = []
for W, P, R, (lo, hi) in grid:
    sT = stats(T, auto(T, W, P, R, lo, hi)); sV = stats(V, auto(V, W, P, R, lo, hi))
    res.append(dict(W=W, P=P, R=R, t=f"{lo}-{hi}", **{f"T_{k}": v for k, v in sT.items()},
                    **{f"V_{k}": v for k, v in sV.items()}))
r = pd.DataFrame(res)
pd.set_option("display.width", 250)
cols = ["W", "P", "R", "t", "T_tp", "T_fp", "T_fn", "T_tn", "T_kappa", "T_prec", "T_rec",
        "V_tp", "V_fp", "V_fn", "V_tn", "V_kappa", "V_prec", "V_rec"]
print("registered (W1, P0, R2, t .1-.9):")
print(r[(r.W == 1.0) & (r.P == 0) & (r.R == 2.0) & (r.t == "0.1-0.9")][cols].to_string(index=False))
print("\nbest on T by kappa (top 12):")
print(r.sort_values(["T_kappa", "T_prec"], ascending=False).head(12)[cols].to_string(index=False))
print("\nbest on T by kappa with T precision >= 0.9 (top 8):")
print(r[r.T_prec >= 0.9].sort_values("T_kappa", ascending=False).head(8)[cols].to_string(index=False))
r.to_csv(D + "agree_grid.csv", index=False)
# decoys: auto false-positive rate on random points
for W, P, R, lo, hi in ((2.5, 20, 1.0, 0.0, 1.0), (3.0, 20, 1.0, 0.0, 1.0), (3.0, 0, 1.0, 0.0, 1.0)):
    a = auto(dec, W, P, R, lo, hi)
    print("decoys auto-hit", W, P, R, int(a.sum()), "of", len(dec))
