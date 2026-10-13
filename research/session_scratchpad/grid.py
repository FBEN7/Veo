import sys, pickle, itertools
sys.path.insert(0, '/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad'); sys.path.insert(0, '.')
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from teams_lab import *
from eval_soccertrack_v2 import truth

def proxy(team, table, gt):
    """Row-weighted, outfield tracks only: accuracy among team_A/B, coverage."""
    role = gt.groupby('person').role.agg(lambda s: s.mode().iloc[0])
    T = table[(table.label != 'nonplayer')].copy()
    T = T[T.person.map(role) == 'player']
    T['team'] = T.track.map(team)
    named = T[T.team.isin(['team_A', 'team_B'])]
    tab = pd.crosstab(named.team, named.label, values=named.rows, aggfunc='sum').fillna(0)
    r, c = linear_sum_assignment(-tab.to_numpy()) if tab.size else ([], [])
    acc = tab.to_numpy()[r, c].sum() / max(named.rows.sum(), 1)
    cov = named.rows.sum() / max(T.rows.sum(), 1)
    N = table[table.label == 'nonplayer']
    nonp_named = N[N.track.map(team).isin(['team_A', 'team_B'])].rows.sum() / max(N.rows.sum(), 1)
    return round(acc, 3), round(cov, 3), round(acc * cov, 3), round(nonp_named, 3)

w, crops_p, table_p = sys.argv[1:4]
crops = pickle.load(open(crops_p, 'rb')); table = pd.read_csv(table_p)
gt = truth(f'data/soccertrack_v2/{w}_gt.parquet')
geom = track_geometry(w)
members = {'all': None, 'inner0.3': lambda g: g.depth_h >= 0.3,
           'inner0.3|move2': lambda g: g.depth_h >= 0.3 or g.move_h >= 2}
rows = []
for mode in ('current', 'cone_all', 'cone_top', 'lab'):
    feats = track_features(crops, mode)
    for scale, select, (mname, member) in itertools.product((True, False), ('count', 'rows'), members.items()):
        team = assign(feats, geom, scale=scale, select=select, member=member)
        a, c, ac, nn = proxy(team, table, gt)
        rows.append(dict(mode=mode, scale=scale, select=select, member=mname, acc=a, cov=c, acc_x_cov=ac, nonplayer_named=nn))
R = pd.DataFrame(rows).sort_values('acc_x_cov', ascending=False)
print(R.to_string(index=False))
