import sys, io, contextlib, json
sys.path.insert(0, "/home/user/Veo")
import pandas as pd, numpy as np
import score_soccernet as sc
from src import events as ev, pixel_scale, soccertrack_v2 as st
w = "st2_118575_1st_f015000"
info = json.loads(open(f"/home/user/Veo/data/soccertrack_v2/{w}.json").read())
cal = st.calibration(info["match"])
d = f"/home/user/Veo/output_{w}/events_v1"
merged = pd.read_parquet(d + "/tracks_merged.parquet")
metric = pixel_scale.to_pitch_metres(merged, lambda uv: st.image_to_pitch(uv, cal))
m0 = metric.copy(deep=True)
with contextlib.redirect_stdout(io.StringIO()):
    e1 = ev.detect_events(metric, absolute_pitch=True, ball_time_aware=True)
print("mutated:", not metric.equals(m0), list(metric.columns) == list(m0.columns))
with contextlib.redirect_stdout(io.StringIO()):
    e2 = ev.detect_events(metric, absolute_pitch=True, ball_time_aware=True, params={"POSSESSION_RADIUS_M": 3.0})
    e3 = ev.detect_events(metric, absolute_pitch=True, ball_time_aware=True)
print(len(e1), len(e2), len(e3), json.dumps(e1, default=str) == json.dumps(e3, default=str))
print(ev.POSSESSION_RADIUS_M)
