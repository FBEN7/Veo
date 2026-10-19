"""Light simulation of team selection rules on tracks_M3.csv (track-level)."""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings('ignore')
CSV = '/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/tracks_M3.csv'
NF = 2250; F = ['cos', 'sin', 'sat', 'val']; GK = {1, 12}; CUT = 2.5

def load():
    d = pd.read_csv(CSV)
    return d.dropna(subset=['cos']).copy().reset_index(drop=True)

def acc(df, col):
    o = df[df.label.isin(['left', 'right']) & ~df.person.isin(GK)]
    named = o[o[col].isin(['A', 'B'])]
    t = pd.crosstab(named[col], named.label, values=named.rows, aggfunc='sum') \
        .reindex(index=['A', 'B'], columns=['left', 'right']).fillna(0).to_numpy()
    r, c = linear_sum_assignment(-t)
    return round(t[r, c].sum() / max(t.sum(), 1), 3), round(t[r, c].sum() / o.rows.sum(), 3)

def residual(X, team, centres):
    dist = np.linalg.norm(X[:, None, :] - centres[None], axis=2).min(1)
    inside = team != 'o'
    scale = np.median(dist[inside]) if inside.any() else np.median(dist)
    out = team.copy(); out[(dist / scale > CUT) & inside] = 'o'
    return out

def assign(g, rule, inner=0.0, share_cut=0.70):
    X = StandardScaler().fit_transform(g[F]); km = KMeans(3, n_init=10, random_state=0).fit(X)
    lab = km.labels_; C = km.cluster_centers_
    info = {}
    if rule == 'count':
        top = np.argsort(np.bincount(lab, minlength=3))[::-1][:2]
        team = np.array(['A' if l == top[0] else 'B' if l == top[1] else 'o' for l in lab]); cen = C[top]
    else:
        occ = np.array([g.rows[(lab == c) & (g.depth_med >= inner)].sum() / NF for c in range(3)])
        share = occ / occ.sum(); top = np.argsort(occ)[::-1][:2]
        info = dict(occ=occ.round(1).tolist(), share=share.round(2).tolist())
        if share[top[0]] > share_cut:
            m = lab == top[0]; sk = KMeans(2, n_init=10, random_state=0).fit(X[m])
            team = np.full(len(g), 'o', dtype=object); team[np.where(m)[0]] = np.where(sk.labels_ == 0, 'A', 'B')
            cen = sk.cluster_centers_; info['split'] = True
            info['half_occ'] = [round(g.rows[np.where(m)[0][sk.labels_ == h]].sum() / NF, 1) for h in (0, 1)]
        else:
            team = np.array(['A' if l == top[0] else 'B' if l == top[1] else 'o' for l in lab]); cen = C[top]
    g = g.copy(); g['t'] = residual(X, np.asarray(team, dtype=object), cen)
    b = g[g.label == 'bench']
    return dict(acc_recall=acc(g, 't'), bench_on_team=round(b[b.t.isin(['A', 'B'])].rows.sum() / NF, 2), **info), g

def bench(nb, hue_deg, sat, val, seed):
    rng = np.random.default_rng(seed)
    ang = (np.deg2rad(hue_deg) if hue_deg is not None else rng.uniform(0, 2 * np.pi, nb)) + rng.normal(0, 0.15, nb)
    return pd.DataFrame({'cos': np.cos(ang), 'sin': np.sin(ang),
                         'sat': np.clip(sat + rng.normal(0, 0.05, nb), 0, 1),
                         'val': np.clip(val + rng.normal(0, 0.04, nb), 0, 1),
                         'rows': rng.integers(100, 1500, nb), 'depth_med': rng.uniform(-0.3, 0.05, nb),
                         'label': 'bench', 'person': np.nan})
