import sys, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
exec(open('feat_m3.py').read().split("def run(")[0].split("FEATS = {")[0])  # reuse loading + lab
FEATS = {
 'current': (np.c_[c, si, s, v], True),
 'chroma std': (np.c_[s*c, s*si, v], True),
 'chroma raw': (np.c_[s*c, s*si, v], False),
 'cone raw': (np.c_[s*v*c, s*v*si, v], False),
 'cone std': (np.c_[s*v*c, s*v*si, v], True),
}
pd.set_option('display.width',250)
# 1) separability without clustering: outfield players (labelled, non-GK, match_share>=0.8)
m_out = (d.label!='nonplayer') & ~d.gk & (d.match_share>=0.8)
m_bench = (d.label=='nonplayer') & (d.depth_med < 0)
for name,(X,std) in FEATS.items():
    F = StandardScaler().fit_transform(X) if std else X
    L = F[(m_out & (d.label=='left')).to_numpy()]; R = F[(m_out & (d.label=='right')).to_numpy()]
    B = F[m_bench.to_numpy()]
    cl, cr = np.median(L,0), np.median(R,0)
    within = np.median(np.r_[np.linalg.norm(L-cl,axis=1), np.linalg.norm(R-cr,axis=1)])
    between = np.linalg.norm(cl-cr)
    bench_near = np.median(np.minimum(np.linalg.norm(B-cl,axis=1), np.linalg.norm(B-cr,axis=1)))
    # nearest-centre error of outfield tracks (row weighted)
    W = d.rows[m_out].to_numpy(); y = (d.label[m_out]=='left').to_numpy()
    Fo = F[m_out.to_numpy()]
    predL = np.linalg.norm(Fo-cl,axis=1) < np.linalg.norm(Fo-cr,axis=1)
    err = (W*(predL!=y)).sum()/W.sum()
    print(f'{name:12s} between/within {between/within:5.2f}  bench-to-nearest-team/within {bench_near/within:5.2f}  nearest-median-centre row error {err:.3f}  (nL={len(L)} nR={len(R)} nB={len(B)})')
