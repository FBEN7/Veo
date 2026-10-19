import sys, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
exec(open('feat_m3.py').read().split("FEATS = {")[0])
exec("def run(" + open('feat_m3.py').read().split("def run(")[1].split("def score(")[0])
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
FEATS = {
 'cur': (np.c_[c, si, s, v], True),
 'chS': (np.c_[s*c, s*si, v], True),
 'chR': (np.c_[s*c, s*si, v], False),
 'coR': (np.c_[s*v*c, s*v*si, v], False),
 'coS': (np.c_[s*v*c, s*v*si, v], True),
}
out = d[['track','rows','label','person','gk','sat','val','depth_med']].copy()
out['hue'] = hdeg.round(0)
for k,(X,std) in FEATS.items():
    t,_,rd = run(X,std)
    out[k]=t; out[k+'_r']=rd.round(1)
# map teams to sides by majority rows
for k in FEATS:
    m={}
    for tm in ('team_A','team_B'):
        g=out[(out[k]==tm)&(out.label!='nonplayer')]
        m[tm]=g.groupby('label').rows.sum().idxmax()
    out[k]=out[k].map(lambda x: m.get(x,'oth')).str[:3]
bad = out[(out.label!='nonplayer') & ~out.gk & out[list(FEATS)].ne(out.label.str[:3],axis=0).any(axis=1)]
print(bad.sort_values('rows',ascending=False).round(2).to_string())
print(out[out.label=='nonplayer'].sort_values('rows',ascending=False).round(2).to_string())
