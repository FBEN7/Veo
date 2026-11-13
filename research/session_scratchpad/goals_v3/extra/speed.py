import sys, json, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, '/home/user/veo/research/session_scratchpad/goals_v3')
import score as S
E, F = S.E, S.F
from src.unseal import unsealed
for m in sorted(E.SEALED): unsealed(m, "goals v3: crowd speed cue (development use, seal lifted 2026-11-05)")
res = json.load(open('goals_v3.json'))['held out']['per match']
def crowd_speed(p_f, f):
    a, b = p_f.get(f), p_f.get(f + 25)
    if a is None or b is None or not len(a) or not len(b): return np.nan
    D = np.hypot(a.x.to_numpy()[:, None] - b.x.to_numpy()[None], a.y.to_numpy()[:, None] - b.y.to_numpy()[None])
    r, c = linear_sum_assignment(D); d = D[r, c]
    return float(np.median(d[d <= 10])) if (d <= 10).any() else np.nan
rows = []
for m in E.MATCHES:
    c = res[m]['cell']; cell = (c['T'], c['R'], c['N'])
    for half in (1, 2):
        h = S.load(m, half); p_f = {f: g for f, g in h['people'].groupby('frame')}
        rs = E.runs(h['s'], *cell); gf = h['goals'].frame.to_numpy(); rest = F.restartable(gf, h['actions'])
        mt = F.match_runs([r[0] for r in rs], gf); hit = {ri for gi, ri in mt.items() if rest[gi]}
        # background: 200 random samples of the half
        rng = np.random.default_rng(0); allf = sorted(p_f)
        bg = [crowd_speed(p_f, f) for f in rng.choice(allf[60:-2], 100, replace=False)]
        for ri, (a, b) in enumerate(rs):
            v = [crowd_speed(p_f, f) for f in range(a, b + 1, 25)]
            pre = [crowd_speed(p_f, f) for f in range(a - 10 * 25, a, 25)]
            kind = 'hit' if ri in hit else ('false' if ri not in mt.values() else 'non-restartable')
            rows.append(dict(match=m, half=half, kind=kind, run=np.nanmedian(v), pre10=np.nanmedian(pre), bg=np.nanmedian(bg)))
d = pd.DataFrame(rows)
print(d.groupby('kind')[['run', 'pre10', 'bg']].describe(percentiles=[.25, .5, .75]).round(2).T.to_string())
for thr in (0.8, 1.0, 1.2, 1.5, 2.0):
    print('run speed <=', thr, 'hits kept', int(((d.kind == 'hit') & (d.run <= thr)).sum()), '/', int((d.kind == 'hit').sum()), 'false kept', int(((d.kind == 'false') & (d.run <= thr)).sum()), '/', int((d.kind == 'false').sum()))
for thr in (1.0, 1.5, 2.0, 2.5):
    print('pre-10s speed <=', thr, 'hits kept', int(((d.kind == 'hit') & (d.pre10 <= thr)).sum()), 'false kept', int(((d.kind == 'false') & (d.pre10 <= thr)).sum()))
d.to_csv('/tmp/claude-0/-home-user-veo/b69e5b60-e0fe-5920-8f15-285768d7ea57/scratchpad/speed_rows.csv', index=False)
