import json, sys, numpy as np
B='/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/'
res={}
for f in ['U_a.json','U_b.json','Urgb_a.json','Urgb_b.json']:
    d=json.load(open(B+f))
    for r in d['results']:
        w=r['window']
        if w not in res: res[w]=dict(r)
        else:
            for k,v in r.items():
                if k not in res[w]: res[w][k]=v
results=list(res.values())
assert all(r['window'].startswith('st2_118578') for r in results)
def cf(a,b,draws=4000,seed=0):
    da,db,keys=[],[],[]
    for r in results:
        for p,fa,fb in zip(r['frames']['passes'],r[a]['ff frames'],r[b]['ff frames']):
            if None in fa or None in fb: continue
            da.append(np.mean(fa)); db.append(np.mean(fb)); keys.append(p['block'])
    da,db=np.array(da),np.array(db); d=da-db
    _,bi=np.unique(keys,return_inverse=True)
    S,N=np.bincount(bi,weights=d),np.bincount(bi)
    rng=np.random.default_rng(seed); draw=rng.integers(0,len(S),(draws,len(S)))
    boots=S[draw].sum(1)/N[draw].sum(1)
    lo,hi=np.percentile(boots,[2.5,97.5])
    return len(d),len(S),round(da.mean(),3),round(db.mean(),3),round(d.mean(),3),round(lo,3),round(hi,3)
for a,b in [('ball_v2.4000.pt','coco'),('ball_v2.4000.pt:motion-zero','coco'),('ball_v2rgb.4000.pt','coco'),('ball_heatmap.3000.pt','coco'),('ball_v2.4000.pt','ball_v2.4000.pt:motion-zero'),('ball_v2.4000.pt','ball_heatmap.3000.pt'),('ball_v2rgb.3250.pt','coco')]:
    print(a,'-',b,cf(a,b))
print('--- extra')
for a,b in [('ball_v2rgb.4000.pt','ball_heatmap.3000.pt'),('ball_v2rgb.3250.pt','ball_heatmap.3000.pt')]:
    print(a,'-',b,cf(a,b))
results[:] = [r for r in results if '_2nd_' in r['window']]
for a,b in [('ball_v2rgb.4000.pt','ball_heatmap.3000.pt'),('ball_v2.4000.pt','ball_heatmap.3000.pt')]:
    print('2nd half',a,'-',b,cf(a,b))
