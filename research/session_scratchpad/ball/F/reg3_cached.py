"""Regression (3): events_from_cached reproduces cached events.json and
tracks_merged.parquet. Prints equality only."""
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/user/Veo")
import score_soccernet as sc  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402

RUNS = sys.argv[1:] or [
    "st2_118575_1st_f015000/events_v1", "st2_118576_1st_f030000/events_v1",
    "st2_128058_1st_f030000/events_v1",
    "st2_117093_1st_f015000/events_coco", "st2_132877_1st_f015000/events_coco",
    "st2_117093_1st_f015000/events_heatmap",
    "st2_132877_1st_f015000/events_heatmap"]
for run in RUNS:
    w, sub = run.split("/")
    d = Path(f"/home/user/Veo/output_{w}/{sub}")
    info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
    cal = st.calibration(info["match"])
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        events, merged = sc.events_from_cached(d, cal)
    buf = io.StringIO()
    tmp = Path("/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/F/_ev.json")
    sc.write_events(tmp, events)
    same_ev = tmp.read_text() == (d / "events.json").read_text()
    cached = pd.read_parquet(d / "tracks_merged.parquet")
    same_m = merged.reset_index(drop=True).equals(cached.reset_index(drop=True))
    print(run, "events.json identical:", same_ev, "tracks_merged identical:",
          same_m, f"[{time.time() - t0:.0f} s]", flush=True)
