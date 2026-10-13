"""E5: the calibrated fixed-view pipeline on a window with its ball rows
taken from COCO (as cached) or from the heatmap, from events_v1's cached
teams. Usage: python e5.py WINDOW coco|heatmap"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/user/Veo")
from score_soccernet import run_pipeline  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402

w, source = sys.argv[1], sys.argv[2]
root = Path(f"/home/user/Veo/output_{w}")
src = root / "events_v1"
d = root / f"events_{source}"
if d.exists():
    shutil.rmtree(d)
d.mkdir()
for f in ("clip.json", "profile.json", "pitch.json", "team_features.json"):
    shutil.copy(src / f, d / f)
raw = pd.read_parquet(src / "tracks.parquet")
teamed = pd.read_parquet(src / "tracks_teams.parquet")
if source == "heatmap":
    balls = pd.read_parquet(f"/home/user/Veo/.cache/heatmap_balls_{w}.parquet")
    balls = balls[raw.columns]
    raw = pd.concat([raw[raw.cls != "ball"], balls], ignore_index=True)
    teamed = pd.concat([teamed[teamed.cls != "ball"], balls.assign(team="ball")],
                       ignore_index=True)
    raw = raw.sort_values("frame", kind="stable").reset_index(drop=True)
    teamed = teamed.sort_values("frame", kind="stable").reset_index(drop=True)
raw.to_parquet(d / "tracks.parquet")
teamed.to_parquet(d / "tracks_teams.parquet")
info = json.loads(Path(f"/home/user/Veo/data/soccertrack_v2/{w}.json").read_text())
cal = st.calibration(info["match"])
events, _ = run_pipeline(f"/home/user/Veo/data/soccertrack_v2/{w}.mp4", d,
                         fixed_camera=True, model_name="yolov8s.pt", stride=2,
                         pitch=info["pitch"], pitch_margin_px=10.0,
                         to_pitch=lambda uv: st.image_to_pitch(uv, cal))
print("EVENTS", w, source, len(events),
      dict(Counter(e["event_type"] for e in events)))
