"""Chance-hit rates of the three proposed corridor measures, with random
pitch points (uniform in the image inside the outline). Read-only."""
import json, sys
import numpy as np, pandas as pd, cv2
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
from src.paths import DATA_DIR
from src.ball_heatmap import pitch_to_image
from eval_ball_detector import frames_of, seg_dist
rng = np.random.default_rng(1)
for w in ["st2_117093_1st_f015000", "st2_132877_1st_f015000", "st2_118578_2nd_f015000"]:
    info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
    cal = st.calibration(info["match"]); sets = frames_of(w)
    gt = pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{w}_gt.parquet")
    people = {f: g[["x", "y"]].to_numpy(float) for f, g in gt.groupby("frame")}
    poly = np.asarray(info["pitch"], np.float32); lo, hi = poly.min(0), poly.max(0)
    res = {"labels 2.5m|20px": [], "protocol 4m |t-s|<=.3": [], "motion 3m": [], "registered 1m": []}
    for p in sets["passes"]:
        A, B = np.array(p["from"]), np.array(p["to"])
        seg_uv = pitch_to_image(np.linspace(A, B, 60), cal)
        for qi, f in enumerate(p["frames"]):
            s = [0.25, 0.4167, 0.5833, 0.75][qi]
            uv = rng.uniform(lo, hi, (400, 2))
            ins = np.array([cv2.pointPolygonTest(poly.reshape(-1,1,2), (float(a), float(b)), False) >= 0 for a, b in uv])
            uv = uv[ins]
            g = st.image_to_pitch(uv, cal)
            for (u, v), gg in zip(uv, g):
                if not np.isfinite(gg).all(): continue
                d, t = seg_dist(gg, A, B)
                near = np.hypot(*(people[f] - gg).T).min() if f in people else 99
                pxd = np.hypot(*(seg_uv - (u, v)).T).min()
                res["labels 2.5m|20px"].append((d <= 2.5 or pxd <= 20) and near >= 1.0)
                res["protocol 4m |t-s|<=.3"].append(d <= 4.0 and 0 <= t <= 1 and abs(t - s) <= 0.3 and near >= 1.0)
                res["motion 3m"].append(d <= 3.0 and 0.05 <= t <= 0.95 and near >= 1.0)
                res["registered 1m"].append(d <= 1.0 and 0.1 <= t <= 0.9 and near >= 2.0)
    print(w, len(sets["passes"]), {k: round(float(np.mean(v)), 4) for k, v in res.items()})
