"""How well would an automatic corridor measure agree with the blind
free-flight judgements (tiles_ff on T, tiles_ffV on V)? Read-only."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
from src.ball_heatmap import BALL_M, local_scale
from src.paths import DATA_DIR
from eval_ball_detector import frames_of, seg_dist
S = Path("/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball")
rows = []
for d in ("tiles_ff", "tiles_ffV"):
    key = json.loads((S / d / "key.json").read_text())
    judged = {}
    for line in (S / d / "judged.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip(): continue
        a, b = line.split()[:2]; judged[int(a)] = b
    cache = {}
    for k in key:
        if k["source"] == "decoy": continue
        w = k["window"]
        if w not in cache:
            info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
            gt = pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{w}_gt.parquet")
            cache[w] = (st.calibration(info["match"]), frames_of(w),
                        {f: g[["x", "y"]].to_numpy(float) for f, g in gt.groupby("frame")})
        cal, sets, people = cache[w]
        p = next((p for p in sets["passes"] if k["frame"] in p["frames"]), None)
        if p is None: continue
        g0 = st.image_to_pitch([[k["u"], k["v"]]], cal)[0]
        size = BALL_M * local_scale(g0, cal)[0] if np.isfinite(g0).all() else 0
        g = st.image_to_pitch([[k["u"], k["v"] + size / 2]], cal)[0]
        if not np.isfinite(g).all(): continue
        dd, t = seg_dist(g, p["from"], p["to"])
        near = np.hypot(*(people[k["frame"]] - g).T).min()
        rows.append({"set": d, "src": k["source"][:4], "judged": judged.get(k["tile"]),
                     "d": dd, "t": t, "near": near})
df = pd.DataFrame(rows)
df["B"] = df.judged == "B"
print(df.groupby(["set", "src"]).B.agg(["sum", "count"]))
for tol in (1.0, 2.0, 3.0, 4.0):
    for nr in (1.0, 1.5, 2.0):
        auto = (df.d <= tol) & df.t.between(0.05, 0.95) & (df.near >= nr)
        tp = (auto & df.B).sum(); fp = (auto & ~df.B).sum(); fn = (~auto & df.B).sum(); tn = (~auto & ~df.B).sum()
        po = (tp + tn) / len(df); pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / len(df) ** 2
        print(f"tol {tol} m, near>={nr}: TP {tp} FP {fp} FN {fn} TN {tn}  agree {po:.2f} kappa {(po - pe) / (1 - pe):.2f}")
print(df[df.B][["d", "near", "t"]].describe().round(2))
