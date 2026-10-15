"""Protocol v5 part 2: drop the painted-spot marks (src/ball_spots.py) from a
dense v3 run: .cache/v3_balls_<w>.parquet -> .cache/v3m_balls_<w>.parquet."""
import hashlib
import sys

import pandas as pd

sys.path.insert(0, "/home/user/Veo")
from detect_ball_heatmap import read_sidecar, write_sidecar  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
import src.ball_spots as bs  # noqa: E402

w = sys.argv[1]
src = f".cache/v3_balls_{w}.parquet"
meta = read_sidecar(src)
df = pd.read_parquet(src)
marks = bs.spot_marks(df, st.calibration(meta["match"]), len(meta["frames"]))
f = bs.drop_marks(df, marks)
out = f".cache/v3m_balls_{w}.parquet"
f.to_parquet(out)
keep = {k: v for k, v in meta.items() if k not in ("frames", "parquet sha256")}
write_sidecar(out, meta["frames"], **keep, spot_marks={
    "marks": [list(map(float, m)) for m in marks],
    "code sha256": hashlib.sha256(open("src/ball_spots.py", "rb").read()).hexdigest(),
    "from": src, "from sha256": meta["parquet sha256"], "rows dropped": int(len(df) - len(f))})
print(f"{w}: {len(marks)} marks, {len(df) - len(f)} rows dropped", flush=True)
