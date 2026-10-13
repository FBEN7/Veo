"""Smoke test (training window 118575 1st f015000 only): the COCO ball rows
plus decoys (a resting ball at a penalty spot, p 0.8, and two random
on-pitch points a frame, p 0.5-0.7), chosen by select_single_ball and by
ball_link; grass fractions of the decoys absent (kept)."""
import contextlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/Veo")
import score_soccernet as sc  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.ball_heatmap import pitch_to_image  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402

w = "st2_118575_1st_f015000"
S = Path("/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/F")
src = Path(f"/home/user/Veo/output_{w}/events_v1")
dst = S / "link118575"
if dst.exists():
    shutil.rmtree(dst)
sc.swap_ball_rows(src, dst)
info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
cal = st.calibration(info["match"])
g = pd.read_parquet(dst / "tracks_grass.parquet")
rng = np.random.default_rng(0)
frames = np.unique(g.frame)
spot = pitch_to_image([[41.5, 0.0]], cal)[0]
rand = rng.uniform((-50, -32), (50, 32), (len(frames) * 2, 2))
uv = pitch_to_image(rand, cal)
rows = []
for i, f in enumerate(frames):
    rows.append((f, spot[0], spot[1], 0.8))
    rows += [(f, *uv[2 * i], rng.uniform(0.5, 0.7)),
             (f, *uv[2 * i + 1], rng.uniform(0.5, 0.7))]
d = pd.DataFrame(rows, columns=["frame", "px", "py", "confidence"])
d = d.assign(time_s=d.frame / 25, track_id=-1, cls="ball", crop_h=6.0,
             crop_w=np.nan, detection_method="decoy", team="ball",
             grass_frac=np.nan)[g.columns]
pd.concat([g, d], ignore_index=True).sort_values(
    "frame", kind="stable").to_parquet(dst / "tracks_grass.parquet")
print("COCO ball rows", int((g.cls == "ball").sum()), "decoys", len(d))
for select, kw in (("v1", {}), ("link", {"tau": 0.5})):
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        events, merged = sc.events_from_cached(dst, cal, select=select, **kw)
    b = merged[merged.cls == "ball"]
    print(select, f"{time.time() - t0:.1f} s", "ball rows", len(b),
          "decoy share", round((b.detection_method == "decoy").mean(), 3),
          "spot share", round(((b.px - spot[0]).abs() < 1).mean(), 3),
          "events", len(events), "hash", merged.attrs["player_hash"])
with contextlib.redirect_stdout(io.StringIO()):
    _, m0 = sc.events_from_cached(src, cal)
print("cached run's hash", m0.attrs["player_hash"])
