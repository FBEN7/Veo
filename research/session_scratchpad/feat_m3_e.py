import numpy as np, pandas as pd
exec(open('feat_m3_d.py').read().split("def V(")[0])
# Stress test built from M3 only: replicate the off-pitch bench/staff tracks (depth_med<0, nonplayer)
# k times with small jitter, so the bench outnumbers a team, as on the report window's failure mode.
bench = (d.label=='nonplayer') & (d.depth_med<0)
print('bench tracks', bench.sum(), ' left tracks', (d.label=='left').sum(), ' right tracks', (d.label=='right').sum())
rng = np.random.default_rng(1)
for k in (1, 2, 3, 4):
    extra = pd.concat([d[bench]]*(k-1)) if k>1 else d.iloc[:0]
    dd = pd.concat([d, extra], ignore_index=True)
    jit = np.zeros(len(dd)); jit[len(d):] = 1
    cc, ss_, sa, va = (dd[x].to_numpy().copy() for x in ('cos','sin','sat','val'))
    sa = np.clip(sa + jit*rng.normal(0,0.03,len(dd)),0,1); va = np.clip(va + jit*rng.normal(0,0.02,len(dd)),0,1)
    for name, X, std in (('current', np.c_[cc,ss_,sa,va], True),
                         ('chroma std', np.c_[sa*cc,sa*ss_,va], True),
                         ('cone raw', np.c_[sa*va*cc,sa*va*ss_,va], False)):
        t = run(X, std, 2.5)
        acc, cov, npl = metrics(t, dd)
        print(f'bench x{k} ({bench.sum()*k} tracks) {name:11s} acc(assigned) {acc:.3f} player cov {cov:.3f} nonplayer rows given team {npl:.2f}')
