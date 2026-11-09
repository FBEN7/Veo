import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/veo/research/session_scratchpad/goals_v3')
import false_runs as R
from score import load
E, F = R.E, R.F
from src.unseal import unsealed
for m in sorted(E.SEALED): unsealed(m, "goals v3: defended ends check (development use, seal lifted 2026-11-05)")
for m in E.MATCHES:
    for half in (1, 2):
        h = load(m, half); tp = R.truth_people(m, half); tends = F.defended_ends(tp)
        tp_f = {f: g for f, g in tp.groupby('frame')}; dp_f = {f: g for f, g in h['people'].groupby('frame')}
        votes = []
        for f in sorted(dp_f)[::10]:
            t = tp_f.get(f)
            if t is None: continue
            for i, j in R.pair(dp_f[f], t).items(): votes.append((dp_f[f].team.iloc[i], t.team.iloc[j]))
        vt = pd.DataFrame(votes, columns=['d', 'g'])
        side_of = vt.groupby('d').g.agg(lambda s: s.mode().iloc[0]).to_dict()
        consistent = all(h['ends'][k] == tends[side_of[k]] for k in h['ends'])
        dm = h['people'].groupby('team').x.mean().round(1).to_dict(); tm = tp.groupby('team').x.mean().round(1).to_dict()
        # truth share vs det share correlation over the half
        s = h['s'].share; ts = E.half_gt(m, half)['s'].share
        j = pd.concat([s, ts], axis=1, keys=['d', 't']).dropna()
        print(m, half, 'side_of', side_of, 'ends consistent', consistent, 'det mean x', dm, 'truth mean x', tm, 'corr', round(j.d.corr(j.t), 2), 'mean |d-t|', round((j.d - j.t).abs().mean(), 3), flush=True)
