from common import *
rng = np.random.default_rng(0)
bench = F[(F.label=='nonplayer') & (F.depth_med<0)]
print('bench tracks', len(bench), bench.rows.sum())
def occ_select(df, lab, fm, inner=0.25):
    rows = np.where(df.depth_med.to_numpy()>=inner, df.rows.to_numpy(), 0)
    return np.bincount(lab[fm], weights=rows[fm], minlength=3)
def rows_select(df, lab, fm):
    return np.bincount(lab[fm], weights=df.rows.to_numpy()[fm], minlength=3)
def make(copies, inside=False):
    parts=[F]
    for c in range(copies):
        b=bench.copy()
        for col in ['cos','sin','sat','val']:
            b[col]=b[col]+rng.normal(0,0.02,len(b))
        b['sat']=b.sat.clip(0,1)
        if inside: b['depth_med']=0.5
        b['track']=b.track+1000*(c+1)
        parts.append(b)
    if inside and copies==0:
        pass
    df=pd.concat(parts,ignore_index=True)
    if inside:
        df.loc[df.track.isin(bench.track),'depth_med']=0.5
    return df
for inside in [False, True]:
  for copies in [0,2,3,4,6]:
    df=make(copies, inside)
    core=(df.depth_med>=0).to_numpy()
    out=[]
    for name,kind,fm,sel in [('hsv4',  'hsv4',None,'count'),
                             ('chroma','chroma',None,'count'),
                             ('hsv4+core0','hsv4',core,'count'),
                             ('chroma+core0','chroma',core,'count'),
                             ('chroma+rows','chroma',None,rows_select),
                             ('hsv4+occ.25','hsv4',None,occ_select)]:
        t,r,l=assign(df,kind,fit_mask=fm,select=sel)
        m=metrics(df,t,True)
        out.append(f"{name}: {m['acc']}/{m['cov']}/{m['recall_proxy']}/np{m['np_rows_in_team']}")
    print('inside' if inside else 'outside', 'bench tracks', len(bench)*(copies+1), ' | '.join(out))
