import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/veo/research/session_scratchpad/goals_v3')
import score as S
E, F = S.E, S.F
from src.unseal import unsealed
for m in sorted(E.SEALED): unsealed(m, "goals v3: exploratory rescore with kick-off ends (development use, seal lifted 2026-11-05)")
def load_ko(m, half):
    h = S.load(m, half); p = h['people']
    k = p[p.frame <= 250].groupby('team').x.mean()
    ends = {t: (-1 if v < k.drop(t).iloc[0] else 1) for t, v in k.items()}
    s = pd.read_parquet(S.SCANS / f"{m}_{'1st' if half == 1 else '2nd'}.parquet")
    share = F.own_half_share(p, ends, frames=sorted(s.frame.unique()))
    d0 = np.hypot(p.x, p.y).groupby(p.frame).min().reindex(share.index)
    h.update(s=pd.DataFrame({'share': share, 'd0': d0}), ends=ends,
             side_team={'left': next(t for t, e in ends.items() if e < 0), 'right': next(t for t, e in ends.items() if e > 0)})
    return h
for name, L in (('mean-x ends (registered)', S.load), ('kick-off ends', load_ko)):
    halves = [L(m, h) for m in E.MATCHES for h in (1, 2)]
    tot = dict(hits=0, restartable=0, fr=0, team=0)
    for mo in E.MATCHES:
        cell = E.choose([h for h in halves if h['match'] != mo], S.CELLS)
        for h in halves:
            if h['match'] != mo: continue
            r = S.held(h, cell)
            tot['hits'] += r['hits']; tot['restartable'] += r['restartable']; tot['fr'] += r['false runs']; tot['team'] += r['team right']
    print(name, tot, 'per match false', tot['fr'] / 10, flush=True)
