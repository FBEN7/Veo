from exp import *
import sys
sys.path.insert(0,'/home/user/Veo')
from eval_soccertrack_v2 import team_accuracy
S = pd.read_parquet('det_side_M3.parquet')
def ev2(team, tag):
    m = dict(zip(d.track, team))
    p = P[['frame','track_id','label']].copy(); p['team'] = p.track_id.map(m)
    p = p.merge(S[['frame','track_id','side']], on=['frame','track_id'], how='left')
    q = p[p.team.notna()].copy()
    q['L'] = q.groupby('track_id').track_id.transform('size')
    q = q.sort_values('L', ascending=False).groupby(['frame','team'], sort=False).head(11)
    acc_pre, n_pre = team_accuracy(p.team, p.side)
    acc, n = team_accuracy(q.team, q.side)
    # matched GT detections that survive the cap in one of the two team labels, mapped correctly
    tot = S.side.notna().sum()
    df = q.dropna(subset=['side']); top = df.team.value_counts().index[:2]
    tab = pd.crosstab(df[df.team.isin(top)].team, df[df.team.isin(top)].side)
    r, c = linear_sum_assignment(-tab.to_numpy()); corr = tab.to_numpy()[r, c].sum()
    unmatched_in_team = int((q.side.isna() & q.team.isin(['team_A','team_B'])).sum())
    print(f'{tag:34s} team_acc(pre-cap) {acc_pre:.3f}  team_acc(after cap) {acc:.3f} n={n}  correct/all-matched {corr/tot:.3f}  unmatched dets kept in a team {unmatched_in_team}')
if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    ev2(d.team, 'shipped')
    for D in [0.0, 0.1, 0.15, 0.2, 0.3]:
        ev2(assign(d, (d.depth_med >= D).to_numpy(), rest='other'), f'core depth_med>={D}, rest=other')
    ev2(assign(d, (d.depth_med >= 0.15).to_numpy(), rest='nearest'), 'core depth_med>=0.15, rest=nearest')
