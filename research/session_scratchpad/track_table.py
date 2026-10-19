"""Per-track diagnostics for team assignment on a fixed view (tuning only).

Columns: track, rows, label (GT side of the track's main person when >50%
of its rows match one, else 'nonplayer'), current team, kit features
(cos/sin hue, sat, val; current torso rule), depth inside the pitch outline
(signed px / player height: median, 10th pct), movement (px range / height).
"""
import sys, json, cv2, numpy as np, pandas as pd
sys.path.insert(0, '.')
from eval_soccertrack_v2 import truth, boxes, match_feet
from src import team_assignment_v2 as ta
w = sys.argv[1]; out = sys.argv[2]
o = f'output_{w}/'
gt = truth(f'data/soccertrack_v2/{w}_gt.parquet'); gt = gt[gt.frame % 2 == 0]
pitch = json.loads(open(f'output_{w}/pitch.json').read())['pitch']
poly = np.asarray(pitch, np.float32).reshape(-1, 1, 2)
t = pd.read_parquet(o + 'tracks_teams.parquet')
p = t[t.cls == 'player'].copy()
d = boxes(p); p['person'] = match_feet(d, gt)
side = gt.groupby('person').side.first()
p['depth'] = [cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in zip(p.px, p.py)]
wanted = ta._sample_frames(p, 24)
cap = cv2.VideoCapture(f'data/soccertrack_v2/{w}.mp4'); feats = {}; idx = 0; last = max(wanted)
while idx <= last:
    ok, fr = cap.read()
    if not ok: break
    for tid, px, py, h in wanted.get(idx, ()):
        f = ta._torso_feature(fr, px, py, h)
        if f is not None: feats.setdefault(tid, []).append(f)
    idx += 1
rows = []
for tid, g in p.groupby('track_id'):
    h = g.crop_h.median()
    m = g.person.dropna()
    share = len(m) / len(g)
    main = m.mode().iloc[0] if len(m) else None
    lab = side.get(main) if share > 0.5 else 'nonplayer'
    f = np.median(feats[tid], 0) if tid in feats else [np.nan] * 4
    rows.append(dict(track=tid, rows=len(g), label=lab, match_share=round(share, 2),
                     person=main, team=g.team.iloc[0], h=h,
                     cos=f[0], sin=f[1], sat=f[2], val=f[3],
                     depth_med=g.depth.median() / h, depth_p10=g.depth.quantile(.1) / h,
                     move=np.hypot(g.px.max() - g.px.min(), g.py.max() - g.py.min()) / h,
                     py_med=g.py.median()))
R = pd.DataFrame(rows); R.to_csv(out, index=False)
print(R.groupby('label').agg(n=('track', 'size'), rows=('rows', 'sum'),
      depth_med=('depth_med', 'median'), move=('move', 'median'),
      sat=('sat', 'median'), val=('val', 'median')).round(2))
print(pd.crosstab(R.label, R.team, values=R.rows, aggfunc='sum').fillna(0).astype(int))
