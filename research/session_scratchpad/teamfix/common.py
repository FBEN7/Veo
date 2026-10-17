import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
CSV='/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/tracks_M3.csv'
F=['cos','sin','sat','val']
def load():
    return pd.read_csv(CSV)

def current(d, k=3, cut=2.5, fit_mask=None, pick='size', weights=None, assign_all=True):
    """Re-implementation of assign_teams_v2 selection on the CSV features.
    fit_mask: which featured tracks k-means/scaler are fitted on (None = all).
    Non-fit tracks are given the nearest team centre (subject to cut)."""
    has = d[F].notna().all(axis=1)
    X = d.loc[has, F].to_numpy()
    fm = np.ones(has.sum(), bool) if fit_mask is None else fit_mask[has.to_numpy()]
    sc = StandardScaler().fit(X[fm]); Z = sc.transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z[fm])
    lab_fit = km.labels_
    if pick == 'size':
        score = np.bincount(lab_fit, minlength=k).astype(float)
    else:
        w = weights[has.to_numpy()][fm]
        score = np.bincount(lab_fit, weights=w, minlength=k)
    big = np.argsort(score)[::-1][:2]
    C = km.cluster_centers_[big]
    dist_all = np.linalg.norm(Z[:, None] - C[None], axis=2)
    near = dist_all.argmin(1); dmin = dist_all.min(1)
    # cluster label for every featured track: fit tracks keep their k-means label,
    lab_all = km.predict(Z)
    lab_all[fm] = lab_fit
    inside = np.isin(lab_all, big)
    scale = np.median(dmin[fm & inside])
    team = np.where(inside, np.where(lab_all == big[0], 'team_A', 'team_B'), 'other').astype(object)
    if assign_all:
        # non-fit tracks: nearest team centre unless beyond cut
        nf = ~fm
        team[nf] = np.where(near[nf] == 0, 'team_A', 'team_B')
    team[dmin / scale > cut] = 'other'
    out = pd.Series(None, index=d.index, dtype=object)
    out[has] = team
    return out

def score(d, team, verbose=True, tag=''):
    t = team.fillna('NA')
    pl = d.label != 'nonplayer'
    tab = pd.crosstab(d.label, t, values=d.rows, aggfunc='sum').fillna(0)
    # one-to-one map team_A/team_B to left/right by rows
    sub = tab.reindex(index=['left','right'], columns=['team_A','team_B'], fill_value=0)
    r, c = linear_sum_assignment(-sub.to_numpy())
    correct = sub.to_numpy()[r, c].sum(); teamed = sub.to_numpy().sum()
    prow = d.loc[pl, 'rows'].sum()
    feat_rows = d.loc[pl & d.cos.notna(), 'rows'].sum()
    np_team = tab.loc['nonplayer', [x for x in ['team_A','team_B'] if x in tab.columns]].sum() if 'nonplayer' in tab.index else 0
    res = dict(acc_among_teamed=round(correct/teamed,3), player_rows_correct=round(correct/prow,3),
               player_rows_teamed=round(teamed/prow,3), player_rows_other=round(tab.loc[['left','right']].get('other',pd.Series([0,0])).sum()/prow,3),
               nonplayer_rows_teamed=int(np_team), nonplayer_rows=int(d.loc[~pl,'rows'].sum()))
    if verbose:
        print(tag, res)
        print(tab.astype(int))
    return res
