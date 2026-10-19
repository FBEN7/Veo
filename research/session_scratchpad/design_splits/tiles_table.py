"""Read-only: one row per judged non-decoy tile at a free-flight pass frame,
with the geometry an automatic free-flight measure would use."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
from src.ball_heatmap import BALL_M, local_scale, pitch_to_image
from eval_ball_detector import frames_of

D = Path("/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball")
OUT = Path("/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/design_splits")
DATA = Path("/home/user/Veo/data/soccertrack_v2")
SCORES = ["heatmap_T.json", "coco_T2.json", "V.json", "ffV.json", "ff_heatmap.json", "ff_coco.json"]
cands = {}
for s in SCORES:
    for r in json.loads((D / s).read_text())["results"]:
        for name, v in r.items():
            if isinstance(v, dict) and "candidates" in v:
                for f, rows in v["candidates"].items():
                    cands.setdefault((r["window"], int(f), name), rows)

rows = []
cache = {}
for tdir in ["tiles_T", "tiles_ff", "tiles_ffV", "tiles_E4"]:
    judged = {}
    for line in (D / tdir / "judged.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        n, j = line.split()[:2]
        judged[int(n)] = j
    for k in json.loads((D / tdir / "key.json").read_text()):
        w, f = k["window"], int(k["frame"])
        if w not in cache:
            info = json.loads((DATA / f"{w}.json").read_text())
            cal = st.calibration(info["match"])
            sets = frames_of(w)
            gt = pd.read_parquet(DATA / f"{w}_gt.parquet")
            people = {fr: g[["x", "y"]].to_numpy(float) for fr, g in gt.groupby("frame")}
            pid = {fr: g["player_id"].astype(str).to_numpy() for fr, g in gt.groupby("frame")}
            lab = pd.read_csv(DATA / f"{w}_events.csv").sort_values("frame").reset_index(drop=True)
            fmap = {}
            for i, p in enumerate(sets["passes"]):
                for q, fr in enumerate(p["frames"]):
                    fmap[fr] = (i, q, p)
            cache[w] = (info, cal, sets, people, pid, lab, fmap)
        info, cal, sets, people, pid, lab, fmap = cache[w]
        if f not in fmap:
            continue
        i, q, p = fmap[f]
        size = np.nan
        if k["source"] != "decoy":
            for row in cands.get((w, f, k["source"]), []):
                if abs(row[0] - k["u"]) < 0.5 and abs(row[1] - k["v"]) < 0.5:
                    size = row[3]; rank = None
                    break
        if not np.isfinite(size):
            g0 = st.image_to_pitch([[k["u"], k["v"]]], cal)[0]
            size = BALL_M * local_scale(g0, cal)[0] if np.isfinite(g0).all() else 0
        g = st.image_to_pitch([[k["u"], k["v"] + size / 2]], cal)[0]
        A, B = np.array(p["from"], float), np.array(p["to"], float)
        ab = B - A; L = float(np.hypot(*ab))
        t = float(np.dot(g - A, ab) / max(L * L, 1e-9)); tc = float(np.clip(t, 0, 1))
        perp = float(np.hypot(*(g - (A + tc * ab))))
        pts = A + np.linspace(0, 1, 201)[:, None] * ab
        uv = pitch_to_image(pts, cal)
        px = float(np.hypot(*(uv - np.array([k["u"], k["v"] + size / 2])).T).min())
        P = people.get(f)
        near = float(np.hypot(*(P - g).T).min()) if P is not None else np.nan
        # pass start/end frames
        a_fr = lab[(lab.x == p["from"][0]) & (lab.y == p["from"][1])].frame
        s = q  # index of frame within the 4
        sx, sy = local_scale(g, cal)
        rows.append(dict(tdir=tdir, set="T" if tdir in ("tiles_T", "tiles_ff") else "V",
                         tile=k["tile"], src=k["source"], j=judged.get(k["tile"]), w=w, f=f,
                         pass_i=i, q=q, L=L, t=t, perp=perp, px=px, near=near, gx=g[0], gy=g[1],
                         sx=sx, sy=sy, u=k["u"], v=k["v"]))
df = pd.DataFrame(rows)
df.to_csv(OUT / "tiles_ff_geometry.csv", index=False)
print(df.groupby(["set", "tdir", "src", "j"]).size())
