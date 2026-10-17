"""The dense path (detect_ball_heatmap.py --v2 output) against
eval_ball_detector.decode's --v2 path on the same 10 frames of a training
window: rows (u, v, p_raw, size, logit) and n_bg must be identical;
confidence must be p_cal of the logit with the checkpoint's calibration."""
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/user/Veo")
import eval_ball_detector as ev  # noqa: E402
from src import ball_heatmap as bh  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402

w, ckpt, pq = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(4)
info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
cal = st.calibration(info["match"])
m = bh.build(motion=True)
m.load_state_dict(torch.load(ckpt))
m.eval()
df = pd.read_parquet(pq)
wanted = sorted(set(range(400, 420, 2)))
fn = lambda prev, cur, nxt, bg, pitch, cal: bh.detect_v2(  # noqa: E731
    m, prev, cur, nxt, bg, pitch, tau=ev.V2_TAU, top_k=ev.TOP_K, cal=cal)
cands, n_bg = ev.decode({"window": w, "info": info, "cal": cal}, {}, {"v2": fn}, wanted)
calib = bh.load_calibration(ckpt)
same, nrows = True, 0
for t in wanted:
    a = cands["v2"][t]
    b = df[df.frame == t][["px", "py", "p_raw", "crop_h", "logit"]].to_numpy()
    nrows += len(a)
    same &= a.shape == b.shape and np.array_equal(a, b)
    same &= bool((df[df.frame == t].n_bg == n_bg[t]).all())
conf_ok = np.allclose(df.confidence, bh.p_cal(df.logit, calib), rtol=0, atol=1e-12)
print(f"frames {len(wanted)}, rows {nrows} (dense {len(df)}), identical: {same}, "
      f"confidence = p_cal(logit): {conf_ok}, n_bg {sorted(set(n_bg.values()))}, "
      f"columns {list(df.columns)}")
