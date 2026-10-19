import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
from src.ball_heatmap import pitch_to_image
from eval_ball_detector import frames_of, ground
DATA = "/home/user/Veo/data/soccertrack_v2"
res = {}
for f in ("heatmap_T.json", "coco_T2.json"):
    for r in json.load(open(f))["results"]:
        res.setdefault(r["window"], {}).update(r)
out = {"ball_heatmap.3000.pt": [], "coco": []}
plac = {"ball_heatmap.3000.pt": [], "coco": []}
for w, r in res.items():
    info = json.load(open(f"{DATA}/{w}.json")); cal = st.calibration(info["match"])
    sets = frames_of(w)
    gt = pd.read_parquet(f"{DATA}/{w}_gt.parquet")
    people = {fr: g[["x", "y"]].to_numpy(float) for fr, g in gt.groupby("frame")}
    for p in sets["passes"]:
        A, B = np.array(p["from"]), np.array(p["to"])
        pts = A + np.linspace(0, 1, 201)[:, None] * (B - A); uvs = pitch_to_image(pts, cal)
        for name in out:
            hits = []
            for fr in p["frames"]:
                c = np.array(r[name]["candidates"].get(str(fr), []))
                if not len(c) or fr not in people:
                    hits.append(0.0); continue
                g = ground(c[:1], cal)[0]
                if not np.isfinite(g).all():
                    hits.append(0.0); continue
                ab = B - A; t = np.clip(np.dot(g - A, ab) / np.dot(ab, ab), 0, 1)
                perp = np.hypot(*(g - (A + t * ab)))
                px = np.hypot(*(uvs - c[0, :2]).T).min()
                near = np.hypot(*(people[fr] - g).T).min()
                hits.append(float((perp <= 2.5 or px <= 20) and near >= 1.0))
            out[name].append(np.mean(hits))
for k, v in out.items():
    print(k, "T auto corridor flight hit (W2.5|20px, >=1 m from players), passes", len(v), round(float(np.mean(v)), 3))
d = np.array(out["ball_heatmap.3000.pt"]) - np.array(out["coco"])
rng = np.random.default_rng(0); b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
print("diff", round(d.mean(), 3), np.round(np.percentile(b, [2.5, 97.5]), 3))
