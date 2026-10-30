"""The 'v3s_link' pitch tracks of a unit, on Kaggle where its video is (the
grass fractions read it): shots_dev.py metric_of's steps -- painted marks
dropped (src/ball_spots), static points dropped (tune_rules.static_points),
the rows swapped into output_<w>/events_v1, merged with link selection
(top_k 4, tau 0.10, tune_rules.LINK), to pitch metres with the release
calibration -- into output_<w>/metric_v3s_link.parquet, with the kept rows in
output_<w>/v3s_rows.parquet. Run from the repo root.

    python metric_step.py W
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball/events_tune"))
import score_soccernet as sc  # noqa: E402
import tune_rules as tr  # noqa: E402
from detect_ball_heatmap import read_sidecar  # noqa: E402
from src import ball_spots as bs  # noqa: E402
from src import pixel_scale  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402

w = sys.argv[1]
info = json.loads((ROOT / "data/soccertrack_v2" / f"{w}.json").read_text())
cal = st.calibration(str(info["match"]))
path = ROOT / f".cache/v3_balls_{w}.parquet"
n = len(read_sidecar(path)["frames"])
rows = pd.read_parquet(path)
rows = bs.drop_marks(rows, bs.spot_marks(rows, cal, n))
keep = np.ones(len(rows), bool)
for x, y in tr.static_points(rows, n):
    keep &= np.hypot(rows.px.to_numpy() - x, rows.py.to_numpy() - y) > 1.5 * tr.STATIC_PX
run = ROOT / f"output_{w}"
rows[keep].to_parquet(run / "v3s_rows.parquet")
dst = sc.swap_ball_rows(run / "events_v1", run / "shots_v3s_link", rows[keep])
with contextlib.redirect_stdout(io.StringIO()):
    merged = sc.merged_from_cached(dst, cal, select="link", top_k=4, tau=0.10, link_params=tr.LINK)
metric = pixel_scale.to_pitch_metres(merged, lambda uv: st.image_to_pitch(uv, cal))
metric.to_parquet(run / "metric_v3s_link.parquet")
print(f"{w}: {int(keep.sum())} of {len(rows)} rows kept, metric {len(metric)} rows", flush=True)
