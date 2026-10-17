import json, numpy as np
B='/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/'
res={}
for f in ['U_a.json','U_b.json','Urgb_a.json','Urgb_b.json']:
    d=json.load(open(B+f))
    for r in d['results']:
        w=r['window']
        res.setdefault(w,{}).update({k:v for k,v in r.items() if k not in res.get(w,{})})
for det in ['coco','ball_heatmap.3000.pt','ball_v2.4000.pt','ball_v2.4000.pt:motion-zero','ball_v2rgb.4000.pt']:
    pp=[];ah=[]
    for w,r in res.items():
        e=r[det]; pp+=e['per pass']; ah.append((e['action hit'],r['actions']))
    a=sum(x*n for x,n in ah)/sum(n for _,n in ah)
    print('%-30s 1m-hit %.3f (n=%d) action hit %.3f'%(det,np.mean(pp),len(pp),a))
