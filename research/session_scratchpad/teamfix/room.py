from exp import *
import warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
D=0.15
has=d[F].notna().all(axis=1).to_numpy(); core=(d.depth_med>=D).to_numpy()&has
X=d[F].to_numpy(); sc=StandardScaler().fit(X[core]); Z=np.full_like(X,np.nan); Z[has]=sc.transform(X[has])
km=KMeans(3,n_init=10,random_state=0).fit(Z[core]); lab=np.full(len(d),-1); lab[core]=km.labels_
big=np.argsort(np.bincount(km.labels_))[::-1][:2]; C=km.cluster_centers_[big]
Dm=np.full((len(d),2),np.nan); Dm[has]=np.linalg.norm(Z[has][:,None]-C[None],axis=2)
near=np.nanargmin(np.nan_to_num(Dm,nan=1e9),1); dmin=np.nanmin(Dm,1)
base=assign(d,(d.depth_med>=D).to_numpy(),rest='other')
names={0:'team_A',1:'team_B'}
# which name corresponds to big[0]? in assign(): big[0]->team_A
m=dict(zip(d.track,base))
p=P.assign(team=P.track_id.map(m))
cnt=p[p.team.isin(['team_A','team_B'])].groupby(['frame','team']).size().unstack(fill_value=0)
lab2side=pd.crosstab(p.team,p.label)
print(lab2side)
print('placed tracks per frame: median', cnt.median().to_dict(), 'p90', cnt.quantile(.9).to_dict(), 'share frames >11', (cnt>11).mean().round(3).to_dict())
# room score for each unplaced featured track
rows=[]
for i,r in d.iterrows():
    if not has[i] or base[i] in ('team_A','team_B'): continue
    tn=names[near[i]]
    fr=p.loc[p.track_id==r.track,'frame']
    c=cnt.reindex(fr)[tn].fillna(0)
    rows.append(dict(track=r.track,rows=r.rows,label=r.label,core=core[i],cl=lab[i],near=tn,r=dmin[i]/np.median(dmin[core&np.isin(lab,big)]),
                     cnt_med=c.median(),room_share=(c<11).mean()))
R=pd.DataFrame(rows).sort_values('rows',ascending=False)
print(R.round(2).to_string())
R.to_csv('room.csv',index=False)
