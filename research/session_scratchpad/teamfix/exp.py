import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from common import load, F, current
d = load()
P = pd.read_parquet('rows_M3.parquet').merge(
    d[['track','label']].rename(columns={'track':'track_id'}), on='track_id')

def assign(d, core, k=3, cut=2.5, pick='count', rest='other', rest_mask=None, deep=None):
    """core: bool mask over d (tracks that fit scaler + kmeans).
    rest: what non-core featured tracks get: 'other' or 'nearest' (nearest team centre if within cut).
    rest_mask: optional extra condition for non-core tracks to be eligible for 'nearest'."""
    has = d[F].notna().all(axis=1).to_numpy()
    core = core & has
    X = d[F].to_numpy()
    sc = StandardScaler().fit(X[core]); Z = np.full_like(X, np.nan); Z[has] = sc.transform(X[has])
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z[core])
    lab = np.full(len(d), -1); lab[core] = km.labels_
    if pick == 'count':
        s = np.bincount(km.labels_, minlength=k).astype(float)
    elif pick == 'deeprows':
        s = np.bincount(km.labels_, weights=deep[core], minlength=k)
    big = np.argsort(s)[::-1][:2]
    C = km.cluster_centers_[big]
    D = np.full((len(d), 2), np.nan); D[has] = np.linalg.norm(Z[has][:, None] - C[None], axis=2)
    near = np.nan_to_num(D).argmin(1); dmin = D.min(1)
    team = np.array([None] * len(d), dtype=object)
    inteam = core & np.isin(lab, big)
    scale = np.median(dmin[inteam])
    team[core] = 'other'
    team[inteam] = np.where(lab[inteam] == big[0], 'team_A', 'team_B')
    nc = has & ~core
    team[nc] = 'other'
    if rest == 'nearest':
        el = nc if rest_mask is None else nc & rest_mask
        team[el] = np.where(near[el] == 0, 'team_A', 'team_B')
    elif rest == 'nearest_all':  # also third-cluster core members
        el = has & ~inteam if rest_mask is None else has & ~inteam & rest_mask
        team[el] = np.where(near[el] == 0, 'team_A', 'team_B')
    rej = has & (dmin / scale > cut) & np.isin(team, ['team_A', 'team_B'])
    team[rej] = 'other'
    return pd.Series(team, index=d.index)

def evaluate(team, tag='', show=False):
    m = dict(zip(d.track, team))
    p = P.assign(team=P.track_id.map(m))
    pl = p.label != 'nonplayer'
    tab = pd.crosstab(p.label, p.team.fillna('NA'))
    sub = tab.reindex(index=['left','right'], columns=['team_A','team_B'], fill_value=0).to_numpy()
    r, c = linear_sum_assignment(-sub)
    mp = {('team_A','team_B')[j]: ('left','right')[i] for i, j in zip(r, c)}
    corr = (p.team.map(mp) == p.label)
    teamed = p.team.isin(['team_A','team_B'])
    # roster cap as player_filter: drop NA, cap 11 per (frame, team) incl 'other', longest tracks first
    q = p[p.team.notna()].copy()
    q['L'] = q.groupby('track_id').track_id.transform('size')
    q = q.sort_values('L', ascending=False).groupby(['frame','team'], sort=False).head(11)
    qc = q.team.map(mp) == q.label; qt = q.team.isin(['team_A','team_B']); qpl = q.label != 'nonplayer'
    res = dict(
        acc=round(corr[teamed & pl].mean(), 3),
        pl_correct=round(corr[pl].mean(), 3),
        pl_other=round((p.team[pl] == 'other').mean(), 3),
        np_teamed=int((teamed & ~pl).sum()),
        cap_acc=round(qc[qt & qpl].mean(), 3),
        cap_pl_correct=round(qc[qpl].sum() / pl.sum(), 3),
        cap_np_teamed=int((qt & ~qpl).sum()))
    print(f'{tag:42s}', res)
    if show: print(tab)
    return res
