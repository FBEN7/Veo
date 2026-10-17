"""Ablation on a report window (no selection: settings were frozen on tuning windows)."""
import sys, json
from pathlib import Path
sys.path.insert(0, '/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad'); sys.path.insert(0, '/home/user/Veo')
import numpy as np, pandas as pd
from src.team_assignment_v2 import cluster_teams
from variant import run_variant
w, tag = sys.argv[1], sys.argv[2]
tf = json.loads(Path(f'/home/user/Veo/output_{w}/team_features.json').read_text())
feats = {int(k): np.array(v['kit']) for k, v in tf.items()}
depth = pd.Series({int(k): v['depth'] for k, v in tf.items()})
rows = []
for feature, D in (('hsv4', None), ('hsv4', 0.15), ('chroma', None), ('chroma', 0.15)):
    core = None if D is None else set(depth.index[depth >= D])
    team = cluster_teams(feats, kit_feature=feature, fit_tracks=core, verbose=False)
    r = run_variant(w, team, f'report_{feature}_{D}')
    rows.append(dict(window=tag, feature=feature, D=D, **r)); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/report_{tag}.csv', index=False)
