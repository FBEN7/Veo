import sys, numpy as np
sys.path.insert(0, '/home/user/veo/research/session_scratchpad/goals_v3'); sys.argv=['x']
import false_runs as R
from score import load
E, F = R.E, R.F
from src.unseal import unsealed
for m in sorted(E.SEALED): unsealed(m, "goals v3: scan vs truth time alignment check (development use, seal lifted 2026-11-05)")
for m in E.MATCHES:
    for half in (1, 2):
        h = load(m, half); tp = R.truth_people(m, half)
        tp_f = {f: g for f, g in tp.groupby('frame')}; dp_f = {f: g for f, g in h['people'].groupby('frame')}
        fr = sorted(dp_f)[200::25][:80]
        res = {}
        for lag in range(-20, 21):
            v = []
            for f in fr:
                t = tp_f.get(f + lag * 25)
                if t is None: continue
                d = dp_f[f]
                D = np.hypot(d.x.to_numpy()[:, None] - t.x.to_numpy()[None], d.y.to_numpy()[:, None] - t.y.to_numpy()[None])
                v.append(np.median(D.min(1)))
            res[lag] = np.mean(v) if v else np.nan
        best = min(res, key=res.get)
        print(m, half, 'best lag s', best, 'dist at best', round(res[best], 2), 'at 0', round(res[0], 2), flush=True)
