import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/veo/research/session_scratchpad/goals_v3')
import score as S
E, F = S.E, S.F
from src.unseal import unsealed
for m in sorted(E.SEALED): unsealed(m, "goals v3: run lengths (development use, seal lifted 2026-11-05)")
res = json.load(open('goals_v3.json'))['held out']['per match']
hit_len, fr_len, fr_maxlen = [], [], []
for m in E.MATCHES:
    c = res[m]['cell']; cell = (c['T'], c['R'], c['N'])
    for half in (1, 2):
        h = S.load(m, half); rs = E.runs(h['s'], *cell)
        gf = h['goals'].frame.to_numpy(); rest = F.restartable(gf, h['actions']); mt = F.match_runs([r[0] for r in rs], gf)
        # longest raw stretch inside each merged run
        sh = h['s'].share.where(h['s'].d0 <= c['R'], -1.0) if c['R'] is not None else h['s'].share
        raw = F.raw_runs(sh, c['T'], 1)
        def longest(a, b): return max([(y - x) // 25 + 1 for x, y in raw if x >= a and y <= b] or [0])
        for gi, ri in mt.items():
            if rest[gi]: hit_len.append(((rs[ri][1] - rs[ri][0]) // 25 + 1, longest(*rs[ri])))
        for ri, r in enumerate(rs):
            if ri not in mt.values(): fr_len.append(((r[1] - r[0]) // 25 + 1, longest(*r)))
H = pd.DataFrame(hit_len, columns=['span', 'longest']); Fr = pd.DataFrame(fr_len, columns=['span', 'longest'])
print('hits', len(H), H.describe().round(1).to_string()); print('false', len(Fr), Fr.describe().round(1).to_string())
for L in (2, 3, 4, 5, 6, 8, 10, 15):
    print('longest >=', L, 'hits kept', int((H.longest >= L).sum()), 'false kept', int((Fr.longest >= L).sum()))
