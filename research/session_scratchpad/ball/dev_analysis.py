"""Read-only: where is a visually confirmed match ball relative to the
previous/next action segment? Uses judged tiles (B) + keys + score files."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st  # noqa: E402
from src.ball_heatmap import BALL_M, local_scale, pitch_to_image  # noqa: E402

D = Path("/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball")
DATA = Path("/home/user/Veo/data/soccertrack_v2")
SCORES = ["heatmap_T.json", "coco_T2.json", "V.json", "ffV.json", "ff_heatmap.json", "ff_coco.json"]

cands = {}
for s in SCORES:
    for r in json.loads((D / s).read_text())["results"]:
        for name, v in r.items():
            if isinstance(v, dict) and "candidates" in v:
                for f, rows in v["candidates"].items():
                    cands.setdefault((r["window"], int(f), name), rows)


def size_of(w, f, src, u, v):
    rows = cands.get((w, f, src), [])
    for row in rows:
        if abs(row[0] - u) < 0.5 and abs(row[1] - v) < 0.5:
            return row[3]
    return np.nan


infos, evs, gts, cals = {}, {}, {}, {}
out = []
for tdir in ["tiles_T", "tiles_ff", "tiles_ffV", "tiles_E4"]:
    judged = {}
    for line in (D / tdir / "judged.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        n, j = line.split()[:2]
        judged[int(n)] = j
    for k in json.loads((D / tdir / "key.json").read_text()):
        j = judged.get(k["tile"])
        if k["source"] == "decoy":
            continue
        w, f = k["window"], int(k["frame"])
        if w not in infos:
            infos[w] = json.loads((DATA / f"{w}.json").read_text())
            e = pd.read_csv(DATA / f"{w}_events.csv").sort_values("frame").reset_index(drop=True)
            evs[w] = e
            g = pd.read_parquet(DATA / f"{w}_gt.parquet")
            gts[w] = {fr: gg for fr, gg in g.groupby("frame")}
            cals[w] = st.calibration(infos[w]["match"])
        cal, e = cals[w], evs[w]
        size = size_of(w, f, k["source"], k["u"], k["v"])
        if not np.isfinite(size):
            g0 = st.image_to_pitch([[k["u"], k["v"]]], cal)[0]
            size = BALL_M * local_scale(g0, cal)[0] if np.isfinite(g0).all() else 0
        g = st.image_to_pitch([[k["u"], k["v"] + size / 2]], cal)[0]
        before, after = e[e.frame <= f], e[e.frame > f]
        if before.empty or after.empty or not np.isfinite(g).all():
            continue
        a, b = before.iloc[-1], after.iloc[0]
        A, B = np.array([a.x, a.y], float), np.array([b.x, b.y], float)
        ab = B - A
        L = float(np.hypot(*ab))
        t = float(np.dot(g - A, ab) / max(L * L, 1e-9))
        tc = float(np.clip(t, 0, 1))
        perp_seg = float(np.hypot(*(g - (A + tc * ab))))
        # signed perpendicular to the infinite line
        nrm = np.array([-ab[1], ab[0]]) / max(L, 1e-9)
        perp_signed = float(np.dot(g - A, nrm))
        s = (f - a.frame) / max(b.frame - a.frame, 1)
        interp = A + s * ab
        d_interp = float(np.hypot(*(g - interp)))
        ppl = gts[w].get(f)
        near = float(np.hypot(*(ppl[["x", "y"]].to_numpy(float) - g).T).min()) if ppl is not None else np.nan
        # receiver's position NOW vs at reception
        recv_now = np.nan
        if ppl is not None and pd.notna(b.player_id):
            rr = ppl[ppl.player_id.astype(str) == str(b.player_id)]
            if not rr.empty:
                R = rr[["x", "y"]].to_numpy(float)[0]
                # distance from the ball to segment passer(at kick)->receiver(now)
                ab2 = R - A
                t2 = float(np.clip(np.dot(g - A, ab2) / max(np.dot(ab2, ab2), 1e-9), 0, 1))
                recv_now = float(np.hypot(*(g - (A + t2 * ab2))))
        sx, sy = local_scale(g, cal)
        out.append(dict(tdir=tdir, tile=k["tile"], j=j, src=k["source"], w=w, f=f,
                        la=a.label, lb=b.label, same=(a.side == b.side and a.player_id != b.player_id),
                        L=L, dur=(b.frame - a.frame) / 25, s=s, t=t, perp_seg=perp_seg,
                        perp_signed=perp_signed, d_interp=d_interp, near=near,
                        recv_now=recv_now, gx=g[0], gy=g[1], sx=sx, sy=sy, size=size,
                        speed=L / max((b.frame - a.frame) / 25, 1e-3)))
df = pd.DataFrame(out)
df.to_csv(D / "dev_analysis.csv", index=False)
B = df[df.j == "B"].drop_duplicates(["w", "f", "src"])
print("B tiles", len(B), "unique (w,f)", B[["w", "f"]].drop_duplicates().shape[0])
ff = B[(B.la == "PASS") & B.same & (B.L >= 10) & (B.dur >= 0.8) & (B.s >= 0.2) & (B.s <= 0.8)]
ff = ff.drop_duplicates(["w", "f"])
print("free flight (completed ground PASS >=10 m, s in .2-.8):", len(ff))
q = [0.25, 0.5, 0.75, 0.9, 0.95]
for col in ["perp_seg", "d_interp", "recv_now", "near", "t", "s", "L", "dur", "sx", "sy"]:
    print(f"  {col:10s}", np.round(ff[col].quantile(q).to_numpy(), 2), "mean", round(ff[col].mean(), 2))
print("  |t-s| quantiles", np.round((ff.t - ff.s).abs().quantile(q).to_numpy(), 2))
print("  t-s mean (ball behind/ahead of const speed)", round((ff.t - ff.s).mean(), 3))
print("  share perp_seg<=1, <=1.5,<=2,<=3,<=4:", [round(float((ff.perp_seg <= x).mean()), 2) for x in (1, 1.5, 2, 3, 4)])
print("  share recv_now<=1,<=2,<=3:", [round(float((ff.recv_now <= x).mean()), 2) for x in (1, 2, 3)])
print("  share d_interp <=2.5, <=4, <=6:", [round(float((ff.d_interp <= x).mean()), 2) for x in (2.5, 4, 6)])
print("  by side of camera (gy>0 near):", ff.groupby(ff.gy > 0).perp_seg.median().round(2).to_dict())
print(ff[["tdir", "w", "f", "src", "L", "dur", "s", "t", "perp_seg", "perp_signed", "recv_now", "d_interp", "near", "sx"]].round(2).to_string())
# what about all B tiles in any gap (not only passes)
print()
print("all B by opener label:", B.la.value_counts().to_dict())
other = B[~B.index.isin(ff.index)]
print(other[["tdir", "w", "f", "la", "lb", "same", "L", "dur", "s", "perp_seg", "d_interp", "near"]].round(2).to_string())
