import pickle, numpy as np, sys
import pandas as pd
base='/home/user/Veo/.cache/soccertrack_v2/'
files={'M3':base+'reid_val/st2_118575_1st_f015000.pkl','M2':base+'reid_report/st2_117093_1st_f015000.pkl','M10':base+'reid_report/st2_132877_1st_f015000.pkl'}
allh=[]
for k,f in files.items():
    b=pickle.load(open(f,'rb'))
    h=np.concatenate([np.array(p['h']) for p in b['people'].values()])
    # also actual crop heights
    ch=np.concatenate([np.array([c.shape[0] for c in p['crops']]) for p in b['people'].values()])
    roles={}
    for p in b['people'].values(): roles[p['role']]=roles.get(p['role'],0)+len(p['h'])
    allh.append(h)
    print(k,len(h),roles,'pct5/10/50/90/95',np.percentile(h,[5,10,50,90,95]).round(1),'in35-70',np.mean((h>=35)&(h<=70)).round(3),'crop rows equal h?',np.mean(np.abs(ch-h)<=1).round(3))
h=np.concatenate(allh)
print('ALL',len(h),np.percentile(h,[5,10,25,50,75,90,95]).round(1),'in35-70',np.mean((h>=35)&(h<=70)).round(3),'in35-80',np.mean((h>=35)&(h<=80)).round(3), '<35',np.mean(h<35).round(3),'>70',np.mean(h>70).round(3))
for w,mm in [('117093','M2'),('118575','M3'),('132877','M10')]:
    t=pd.read_parquet(f'/home/user/Veo/output_st2_{w}_1st_f015000/tracks.parquet')
    p=t[t.cls=='player'].crop_h
    print(mm,'tracks player det',len(p),np.percentile(p,[5,10,50,90,95]).round(1),'in35-70',np.mean((p>=35)&(p<=70)).round(3))
