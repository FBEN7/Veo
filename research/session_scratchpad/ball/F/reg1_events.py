"""Regression (1): detect_events with default params on cached training-window
runs reproduces events.json; also compares against the HEAD version of
src/events.py. Usage: python reg1_events.py [--head]"""
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/user/Veo")
from src import pixel_scale  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402

WINDOWS = ["st2_118575_1st_f015000", "st2_118576_1st_f030000",
           "st2_128058_1st_f030000"]


def dump(events):
    return json.dumps(events, indent=1, default=lambda v: v.item()
                      if hasattr(v, "item") else str(v))


def load_head():
    src = subprocess.run(["git", "-C", "/home/user/Veo", "show",
                          "HEAD:src/events.py"], capture_output=True,
                         text=True, check=True).stdout
    p = Path(__file__).parent / "_events_head.py"
    p.write_text(src.replace("from . import ball_tracking",
                             "from src import ball_tracking"))
    spec = importlib.util.spec_from_file_location("events_head", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


from src import events as ev  # noqa: E402
head = load_head()
for w in WINDOWS:
    d = Path(f"/home/user/Veo/output_{w}/events_v1")
    info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
    cal = st.calibration(info["match"])
    merged = pd.read_parquet(d / "tracks_merged.parquet")
    metric = pixel_scale.to_pitch_metres(merged, lambda uv: st.image_to_pitch(uv, cal))
    with contextlib.redirect_stdout(io.StringIO()):
        new = ev.detect_events(metric, absolute_pitch=True, ball_time_aware=True)
        new_p = ev.detect_events(metric, absolute_pitch=True, ball_time_aware=True,
                                 params={}) if "params" in ev.detect_events.__code__.co_varnames else new
        old = head.detect_events(metric, absolute_pitch=True, ball_time_aware=True)
    cached = (d / "events.json").read_text()
    print(w, "n", len(new), "== cached events.json:", dump(new) == cached,
          "== HEAD:", dump(new) == dump(old), "params={} same:", dump(new_p) == dump(new))
