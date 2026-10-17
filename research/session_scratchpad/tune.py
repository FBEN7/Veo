"""Tuning sweep for fixed-view team assignment on one window (tuning windows only)."""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, '/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad'); sys.path.insert(0, '/home/user/Veo')
import numpy as np, pandas as pd
from src.team_assignment_v2 import track_kit_features, cluster_teams
from src.player_filter import track_depth
from variant import run_variant
from teams_lab import crop_feature

S = Path('/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad')
w = sys.argv[1]; tag = sys.argv[2]
o = Path(f'/home/user/Veo/output_{w}')
tt = pd.read_parquet(o / 'tracks_teams.parquet')
players = tt[tt.cls == 'player'].drop(columns='team')
cut = json.loads((o / 'pitch.json').read_text())
fp = S / f'feats_{tag}.pkl'
if fp.exists():
    feats = pickle.load(open(fp, 'rb'))
else:
    feats = track_kit_features(f'/home/user/Veo/data/soccertrack_v2/{w}.mp4', players)
    pickle.dump(feats, open(fp, 'wb'))
cp = S / f'crops_{tag}.pkl'
if not cp.exists():
    import subprocess
    subprocess.run([sys.executable, str(S / 'crops.py'), w, str(cp)], check=True, cwd='/home/user/Veo')
crops = pickle.load(open(cp, 'rb'))
cone = {}
for tid, items in crops.items():
    f = [crop_feature(c, fx, fy, h, 'cone_top') for _, c, fx, fy, h in items]
    f = [x for x in f if x is not None]
    if f and tid in feats:
        cone[tid] = np.median(f, axis=0)
depth = track_depth(tt, cut['pitch'])

def teams_for(feature, D):
    if feature == 'cone_top':
        fe, kf = cone, 'hsv4'          # 3-vectors passed through unchanged
    else:
        fe, kf = feats, feature
    core = None if D is None else set(depth.index[depth >= D])
    return cluster_teams(fe, kit_feature=kf, fit_tracks=core, verbose=False)

rows = []
for feature in ('hsv4', 'chroma', 'cone_top'):
    for D in (None, 0, 0.05, 0.1, 0.15, 0.2, 0.3):
        team = teams_for(feature, D)
        r = run_variant(w, team, f'{feature}_{D}')
        # pre-declared stress test: edge tracks (depth < 0.15) copied 4x
        edge = [t for t in feats if depth.get(t, 0) < 0.15]
        fe2 = dict(cone if feature == 'cone_top' else feats); d2 = depth.copy()
        nid = max(fe2) + 1000
        rng = np.random.default_rng(0)
        for k in range(4):
            for t in edge:
                if t in fe2:
                    fe2[nid] = fe2[t] + rng.normal(0, 0.01, len(fe2[t])); d2[nid] = depth[t]; nid += 1
        core2 = None if D is None else set(d2.index[d2 >= D])
        team2 = cluster_teams(fe2, kit_feature='hsv4' if feature == 'cone_top' else feature, fit_tracks=core2, verbose=False)
        # same partition of the original tracks (up to swapping team_A/team_B)
        a = pd.Series({t: team[t] for t in team}); b = pd.Series({t: team2.get(t) for t in team})
        same = (a == b).all() or (a == b.replace({'team_A': 'team_B', 'team_B': 'team_A'})).all()
        rows.append(dict(window=tag, feature=feature, D=D, stress_same=bool(same), **r))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(S / f'tune_{tag}.csv', index=False)
