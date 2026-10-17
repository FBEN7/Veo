"""Light re-clustering of M3 track medians under alternative kit features.

Replicates assign_teams_v2's selection (k=3, two largest clusters = teams,
RESIDUAL_CUT relative to median in-team distance) on the per-track medians in
tracks_M3.csv. Features other than the current one are approximations built
from the track medians (not from pixels).
"""
import sys
import numpy as np
import pandas as pd
import cv2
from itertools import permutations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

d = pd.read_csv(sys.argv[1])

d = d[d['cos'].notna()].copy()
GK = {1, 12}  # persons whose kit is a goalkeeper's on M3 (read off the table)
d['gk'] = d.person.isin(GK) & (d.label != 'nonplayer')

ang = np.arctan2(d['sin'], d['cos'])
s, v = d['sat'].to_numpy(), d['val'].to_numpy()
c, si = d['cos'].to_numpy(), d['sin'].to_numpy()


def hsv_to_lab(row_h, row_s, row_v):
    # hue in OpenCV units 0-179
    px = np.array([[[row_h, row_s * 255, row_v * 255]]], dtype=np.float32)
    px[..., 0] = px[..., 0] * 2  # float HSV in OpenCV uses H in degrees
    px[..., 1:] /= 255.0
    rgb = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2Lab)[0, 0]
    return lab  # L 0-100, a,b ~ -127..127


hdeg = (np.degrees(ang) % 360) / 2
lab = np.array([hsv_to_lab(h, ss, vv) for h, ss, vv in zip(hdeg, s, v)])

FEATS = {
    'current (cos,sin,s,v) std': (np.c_[c, si, s, v], True),
    'hsv chroma (s cos,s sin,v) raw': (np.c_[s * c, s * si, v], False),
    'hsv chroma std': (np.c_[s * c, s * si, v], True),
    'cone (sv cos,sv sin,v) raw': (np.c_[s * v * c, s * v * si, v], False),
    'Lab raw (L/100,a/100,b/100)': (lab / 100.0, False),
    'Lab std': (lab, True),
}


def run(X, std, cut=2.5):
    F = StandardScaler().fit_transform(X) if std else X.copy()
    km = KMeans(3, n_init=10, random_state=0).fit(F)
    lab_ = km.labels_
    cnt = np.bincount(lab_, minlength=3)
    big = np.argsort(cnt)[::-1][:2]
    mp = {int(big[0]): 'team_A', int(big[1]): 'team_B'}
    team = np.array([mp.get(int(l), 'other') for l in lab_], dtype=object)
    cent = km.cluster_centers_[sorted(mp)]
    dist = np.linalg.norm(F[:, None] - cent[None], axis=2).min(1)
    inside = team != 'other'
    sc = np.median(dist[inside])
    team[(dist / sc > cut) & inside] = 'other'
    return team, cnt, dist / sc


def score(team, name, cnt, rd):
    t = d.assign(t=team)
    w = t.rows * t.match_share
    pl = t[t.label != 'nonplayer']
    best = None
    for a, b in permutations(['left', 'right']):
        m = {'team_A': a, 'team_B': b}
        corr = (pl.t.map(m) == pl.label)
        acc_all = (corr * pl.rows).sum() / pl.rows.sum()          # unassigned count wrong
        asg = pl.t != 'other'
        acc_asg = (corr * pl.rows)[asg].sum() / pl.rows[asg].sum()
        if best is None or acc_all > best[0]:
            best = (acc_all, acc_asg, m)
    acc_all, acc_asg, m = best
    cov = pl.rows[pl.t != 'other'].sum() / pl.rows.sum()
    npl = t[t.label == 'nonplayer']
    np_in = npl.rows[npl.t != 'other'].sum() / max(1, npl.rows.sum())
    gk = t[t.gk]
    outf = pl[~pl.gk]
    acc_nogk = ((outf.t.map(m) == outf.label) * outf.rows).sum() / outf.rows.sum()
    print(f'{name:34s} k-sizes {sorted(cnt.tolist(), reverse=True)}  '
          f'acc(all rows) {acc_all:.3f}  acc(outfield) {acc_nogk:.3f}  acc(assigned) {acc_asg:.3f}  '
          f'player cov {cov:.3f}  nonplayer rows given team {np_in:.3f}  '
          f'GK rows given team {gk.rows[gk.t != "other"].sum()}/{gk.rows.sum()}')
    return t


for name, (X, std) in FEATS.items():
    team, cnt, rd = run(X, std)
    t = score(team, name, cnt, rd)
    if '-v' in sys.argv:
        print(pd.crosstab([t.label, t.gk], t.t, values=t.rows, aggfunc='sum').fillna(0).astype(int))

# sanity: does replicated 'current' match the recorded team column?
team, _, _ = run(*FEATS['current (cos,sin,s,v) std'])
rec = d.team.fillna('NA').to_numpy()
print('replication agreement with recorded team column:', (team == rec).mean(), 'n', len(d))
