"""Run the calibrated fixed-view pipeline on a window in output_<w>/<sub>/ from its cached detections."""
import sys, json, shutil
from pathlib import Path
sys.path.insert(0, '/home/user/Veo')
from score_soccernet import run_pipeline
from src import soccertrack_v2 as st
w, sub = sys.argv[1], sys.argv[2]
src = Path(f'/home/user/Veo/output_{w}')
d = src / sub
if d.exists():
    shutil.rmtree(d)
d.mkdir()
for f in ('clip.json', 'profile.json', 'tracks.parquet'):
    shutil.copy(src / f, d / f)
info = json.loads(Path(f'/home/user/Veo/data/soccertrack_v2/{w}.json').read_text())
cal = st.calibration(info['match'])
events, _ = run_pipeline(f'/home/user/Veo/data/soccertrack_v2/{w}.mp4', d, fixed_camera=True,
                         model_name='yolov8s.pt', stride=2, pitch=info['pitch'], pitch_margin_px=10.0,
                         to_pitch=lambda uv: st.image_to_pitch(uv, cal))
from collections import Counter
print('EVENTS', w, len(events), dict(Counter(e['event_type'] for e in events)))
