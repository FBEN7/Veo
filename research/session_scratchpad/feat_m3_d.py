import sys, numpy as np, pandas as pd
from itertools import permutations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
exec(open('feat_m3.py').read().split("FEATS = {")[0])

def run(X, std, cut=2.5, seed=0, idx=None):
    X = X if idx is None else X[idx]
    F = StandardScaler().fit_transform(X) if std else X.copy()
    km = KMeans(3, n_init=10, random_state=seed).fit(F)
    cnt = np.bincount(km.labels_, minlength=3); big = np.argsort(cnt)[::-1][:2]
    mp = {int(big[0]): 'A', int(big[1]): 'B'}
    team = np.array([mp.get(int(l), 'o') for l in km.labels_], dtype=object)
    cent = km.cluster_centers_[sorted(mp)]
    dist = np.linalg.norm(F[:, None] - cent[None], axis=2).min(1)
    sc = np.median(dist[team != 'o']); team[(dist/sc > cut) & (team != 'o')] = 'o'
    return team

def metrics(team, sub):
    pl = (sub.label != 'nonplayer').to_numpy(); r = sub.rows.to_numpy(); lab_ = sub.label.to_numpy()
    best = None
    for a, b in permutations(['left','right']):
        mapped = np.array([{'A':a,'B':b}.get(t,'o') for t in team])
        corr = (mapped == lab_) & pl
        asg = (mapped != 'o') & pl
        acc_asg = (r*corr).sum()/max(1,(r*asg).sum())
        cov = (r*asg).sum()/(r*pl).sum()
        if best is None or acc_asg*cov > best[0]*best[1]: best = (acc_asg, cov)
    npl = ~pl
    nin = (r*npl*(team!='o')).sum()/max(1,(r*npl).sum())
    return best[0], best[1], nin

def V(name, X, std):
    out = []
    for cut in (2.0, 2.5, 3.0):
        out.append(metrics(run(X, std, cut), d))
    seeds = np.array([metrics(run(X, std, 2.5, seed=s), d) for s in range(5)])
    rng = np.random.default_rng(0); bs = []
    for _ in range(40):
        idx = rng.integers(0, len(d), len(d))
        bs.append(metrics(run(X, std, 2.5, idx=idx), d.iloc[idx]))
    bs = np.array(bs)
    print(f'{name:26s} cut2.0 acc/cov/npl {out[0][0]:.3f}/{out[0][1]:.3f}/{out[0][2]:.2f}  cut2.5 {out[1][0]:.3f}/{out[1][1]:.3f}/{out[1][2]:.2f}  cut3.0 {out[2][0]:.3f}/{out[2][1]:.3f}/{out[2][2]:.2f} | seeds acc min {seeds[:,0].min():.3f} | boot acc p10/p50 {np.percentile(bs[:,0],10):.3f}/{np.median(bs[:,0]):.3f} cov p10/p50 {np.percentile(bs[:,1],10):.3f}/{np.median(bs[:,1]):.3f} npl p90 {np.percentile(bs[:,2],90):.2f}')
    return bs

C = s*v
res = {}
res['current'] = V('current (cos,sin,s,v) std', np.c_[c, si, s, v], True)
res['chS'] = V('chroma (s cos,s sin,v) std', np.c_[s*c, s*si, v], True)
res['coR'] = V('cone (sv cos,sv sin,v) raw', np.c_[C*c, C*si, v], False)
res['coS'] = V('cone std', np.c_[C*c, C*si, v], True)
for w in (0.5, 2.0):
    V(f'cone raw, v weight {w}', np.c_[C*c, C*si, w*v], False)
V('cone raw, 2x chroma (=v w 0.5)', np.c_[2*C*c, 2*C*si, v], False)
# paired bootstrap: same resamples -> differences
