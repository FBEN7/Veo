import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/veo/research/session_scratchpad/goals_v3')
import false_runs as R
from score import load
E, F = R.E, R.F
from src.unseal import unsealed
for m in sorted(E.SEALED): unsealed(m, "goals v3: kick-off ends check (development use, seal lifted 2026-11-05)")
for m in E.MATCHES:
    for half in (1, 2):
        h = load(m, half); tp = R.truth_people(m, half); tends = F.defended_ends(tp)
        ts = E.half_gt(m, half)['s'].share
        p = h['people']
        for W in (10, 30):
            k = p[p.frame <= W * 25].groupby('team').x.mean()
            kends = {t: (-1 if v < k.drop(t).iloc[0] else 1) for t, v in k.items()}
            if W == 10: k10 = (kends, k.round(1).to_dict())
            else: k30 = kends
        print(m, half, 'truth share first 10 s', round(ts.loc[:250].mean(), 2), 'det ends', h['ends'], 'kick-off ends (10 s)', k10, 'same as det ends:', k10[0] == h['ends'], '30 s same:', k30 == k10[0], flush=True)
