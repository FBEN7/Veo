import pandas as pd, numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
D = pd.read_csv('/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/tracks_M3.csv')
F = D[D.cos.notna()].reset_index(drop=True)
GK = {1, 12}  # to verify

def feat(df, kind):
    if kind == 'hsv4':
        return df[['cos','sin','sat','val']].to_numpy()
    if kind == 'chroma':
        return np.c_[df.sat*df.cos, df.sat*df.sin, df.val]
    raise ValueError(kind)

def assign(df, kind='hsv4', cut=2.5, fit_mask=None, select='count', seed=0, scale=True, k=3):
    X = feat(df, kind)
    fm = np.ones(len(df), bool) if fit_mask is None else np.asarray(fit_mask)
    sc = StandardScaler().fit(X[fm]) if scale else None
    Z = sc.transform(X) if scale else X
    km = KMeans(k, n_init=10, random_state=seed).fit(Z[fm])
    lab = km.predict(Z)
    if select == 'count':
        counts = np.bincount(lab[fm], minlength=k)
    else:
        counts = select(df, lab, fm)
    big = np.argsort(counts)[::-1][:2]
    mp = {int(big[0]): 'team_A', int(big[1]): 'team_B'}
    team = np.array([mp.get(l, 'other') for l in lab], dtype=object)
    team[~fm] = 'other'
    C = km.cluster_centers_[[big[0], big[1]]]
    dist = np.linalg.norm(Z[:, None] - C[None], axis=2).min(1)
    inside = (team != 'other') & fm
    s = np.median(dist[inside])
    team[(dist / s > cut) & (team != 'other')] = 'other'
    return pd.Series(team, index=df.index), dist / s, lab

def metrics(df, team, exclude_gk=False):
    d = df.assign(t=team.values)
    pl = d[d.label.isin(['left','right'])]
    if exclude_gk:
        pl = pl[~pl.person.isin(GK)]
    named = pl[pl.t.isin(['team_A','team_B'])]
    tab = named.pivot_table(index='t', columns='label', values='rows', aggfunc='sum', fill_value=0)
    tab = tab.reindex(index=['team_A','team_B'], columns=['left','right'], fill_value=0).to_numpy()
    right = max(tab[0,0]+tab[1,1], tab[0,1]+tab[1,0])
    acc = right / max(named.rows.sum(), 1)
    cov = named.rows.sum() / pl.rows.sum()
    npl = d[d.label=='nonplayer']
    np_in = npl[npl.t.isin(['team_A','team_B'])].rows.sum()
    return dict(acc=round(acc,3), cov=round(cov,3), recall_proxy=round(right/pl.rows.sum(),3),
                np_rows_in_team=int(np_in), np_share=round(np_in/max(npl.rows.sum(),1),3))
