from exp import *
S = pd.read_parquet('det_side_M3.parquet')
role = pd.read_csv('roles_M3.csv', index_col=0).role
S['oside'] = S.side.where(S.person.map(role) == 'player')
N_OUT = 45000
TEAMS = ('team_A','team_B')
def ts(team, side):
    df = pd.DataFrame({'team': team, 'side': side}).dropna(subset=['side'])
    named = df[df.team.isin(TEAMS)]
    cov = len(named) / max(len(df), 1)
    tab = pd.crosstab(named.team, named.side); r, c = linear_sum_assignment(-tab.to_numpy())
    right = tab.to_numpy()[r, c].sum()
    return right / len(named), cov, right / N_OUT
def ev3(team, tag):
    m = dict(zip(d.track, team))
    p = P[['frame','track_id','label']].copy(); p['team'] = p.track_id.map(m)
    p = p.merge(S[['frame','track_id','oside']], on=['frame','track_id'], how='left')
    q = p[p.team.notna()].copy()
    q['L'] = q.groupby('track_id').track_id.transform('size')
    q = q.sort_values('L', ascending=False).groupby(['frame','team'], sort=False).head(11)
    a0, c0, r0 = ts(p.team, p.oside); a, c, r = ts(q.team, q.oside)
    nonpl = int((q.oside.isna() & q.team.isin(TEAMS)).sum())
    print(f'{tag:40s} pre-cap acc {a0:.3f} cov {c0:.3f} rec {r0:.3f} | after cap acc {a:.3f} cov {c:.3f} rec {r:.3f} | unmatched-or-GK dets kept in a team {nonpl}')
if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    ev3(d.team, 'shipped')
    for D in [-0.1, 0.0, 0.1, 0.15, 0.2, 0.3, 0.5]:
        ev3(assign(d, (d.depth_med >= D).to_numpy(), rest='other'), f'core depth_med>={D}, rest=other')
    ev3(assign(d, (d.depth_med >= 0.15).to_numpy(), rest='nearest'), 'core depth_med>=0.15, rest=nearest')
    for cut in [2.0, 3.0]:
        ev3(assign(d, (d.depth_med >= 0.15).to_numpy(), rest='other', cut=cut), f'core depth_med>=0.15, cut {cut}')
