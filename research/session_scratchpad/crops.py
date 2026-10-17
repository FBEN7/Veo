"""Cache the player crops team assignment samples (same frames as
team_assignment_v2._sample_frames), so variants run without the video."""
import sys, pickle, cv2, numpy as np, pandas as pd
sys.path.insert(0, '.')
from src import team_assignment_v2 as ta
w = sys.argv[1]
t = pd.read_parquet(f'output_{w}/tracks_teams.parquet')
p = t[t.cls == 'player']
wanted = ta._sample_frames(p, ta.SAMPLES_PER_TRACK)
cap = cv2.VideoCapture(f'data/soccertrack_v2/{w}.mp4')
crops = {}; idx = 0; last = max(wanted)
while idx <= last:
    ok, fr = cap.read()
    if not ok: break
    for tid, px, py, h in wanted.get(idx, ()):
        x1, x2 = int(px - 0.3 * h), int(px + 0.3 * h); y1, y2 = int(py - h), int(py)
        H, W = fr.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 - x1 >= 4 and y2 - y1 >= 8:
            # crop, and where the box's foot point sits inside it
            crops.setdefault(int(tid), []).append((idx, fr[y1:y2, x1:x2].copy(), px - x1, py - y1, h))
    idx += 1
pickle.dump(crops, open(f'{sys.argv[2]}', 'wb'))
print(w, len(crops), 'tracks', sum(map(len, crops.values())), 'crops')
