"""Per-track kit features vs GT side, for current and whole-torso variants."""
import sys, cv2, numpy as np, pandas as pd
sys.path.insert(0, '.')
from eval_soccertrack_v2 import truth, boxes, match_feet
from src import team_assignment_v2 as ta
w = sys.argv[1]
o = f'output_{w}/'
gt = truth(f'data/soccertrack_v2/{w}_gt.parquet'); gt = gt[gt.frame % 2 == 0]
t = pd.read_parquet(o + 'tracks_teams.parquet'); d = boxes(t); d['person'] = match_feet(d, gt)
side = gt.groupby('person').side.first()
main = d.dropna(subset=['person']).groupby('track_id').person.agg(lambda s: s.mode().iloc[0])
share = d.groupby('track_id').person.apply(lambda s: s.notna().mean())
players = t[t.cls == 'player']
wanted = ta._sample_frames(players, 12)
cap = cv2.VideoCapture(f'data/soccertrack_v2/{w}.mp4'); feats = {}; whole = {}; idx = 0; last = max(wanted)
def whole_feat(frame, px, py, h):
    w_ = h * 0.4; x1, x2 = int(px - w_ * .25), int(px + w_ * .25); y1, y2 = int(py - h * .75), int(py - h * .45)
    c = frame[max(0, y1):y2, max(0, x1):x2]
    if c.size < 12: return None
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    ng = ~((hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] >= 40))
    if ng.sum() >= 4: hsv = hsv[ng]
    ang = hsv[:, 0] * 2 * np.pi / 180
    return np.array([np.median(np.cos(ang)), np.median(np.sin(ang)), np.median(hsv[:, 1]) / 255, np.median(hsv[:, 2]) / 255])
while idx <= last:
    ok, fr = cap.read()
    if not ok: break
    for tid, px, py, h in wanted.get(idx, ()):
        f = ta._torso_feature(fr, px, py, h)
        if f is not None: feats.setdefault(tid, []).append(f)
        g = whole_feat(fr, px, py, h)
        if g is not None: whole.setdefault(tid, []).append(g)
    idx += 1
rows = []
for tid in feats:
    lab = side.get(main.get(tid)) if share.get(tid, 0) > 0.5 else 'nonplayer'
    rows.append(dict(tid=tid, lab=lab, team=t[t.track_id == tid].team.iloc[0],
                     **{f'cur_{k}': v for k, v in zip('cshv', np.median(feats[tid], 0))},
                     **{f'all_{k}': v for k, v in zip('cshv', np.median(whole[tid], 0))} if tid in whole else {}))
R = pd.DataFrame(rows)
print(R.groupby('lab')[[c for c in R if c.startswith(('cur', 'all'))]].median().round(2))
print(pd.crosstab(R.lab, R.team))
