import json, numpy as np, pandas as pd, cv2
O='/home/user/Veo/output_st2_118575_1st_f015000/'
def load_rows():
    t=pd.read_parquet(O+'tracks_teams.parquet')
    p=t[t.cls=='player'].copy()
    pitch=json.load(open(O+'pitch.json'))['pitch']
    poly=np.asarray(pitch,np.float32).reshape(-1,1,2)
    p['depth_px']=[cv2.pointPolygonTest(poly,(float(x),float(y)),True) for x,y in zip(p.px,p.py)]
    h=p.groupby('track_id').crop_h.transform('median')
    p['depth']=p.depth_px/h
    return p
if __name__=='__main__':
    import time; s=time.time(); p=load_rows(); print(time.time()-s, len(p))
    p.to_parquet('/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/teamfix/rows_M3.parquet')
