import sys, time, cv2, numpy as np, pandas as pd
sys.path.insert(0,'.')
from ultralytics import YOLO
from eval_soccertrack_v2 import truth, match_feet
from eval_identity_gsr import match as match_iou
from src.detect_track_hybrid import _grass_mask, _near_pitch
n='data/soccertrack_v2/st2_118575_1st_f015000'
gt=truth(n+'_gt.parquet'); cap=cv2.VideoCapture(n+'.mp4')
frames=list(np.linspace(0,4499,30).astype(int)); ims={}
for f in frames:
    cap.set(1,int(f)); ok,im=cap.read(); ims[f]=im
G=gt[gt.frame.isin(frames)]
for name,sz in (('yolov8s.pt',1280),('yolov8n.pt',4096),('yolov8s.pt',4096),('yolov8m.pt',4096)):
    m=YOLO(name); rows=[]; t=time.time()
    for f,im in ims.items():
        r=m(im,imgsz=sz,verbose=False,classes=[0],conf=0.05)[0]; mask=_grass_mask(im)
        for (x0,y0,x1,y1),c in zip(r.boxes.xyxy.numpy(), r.boxes.conf.numpy()):
            if y1-y0<16 or not _near_pitch(mask,(x0+x1)/2,y1,12): continue
            rows.append(dict(frame=f,x0=x0,y0=y0,x1=x1,y1=y1,conf=c))
    dt=(time.time()-t)/len(ims); D=pd.DataFrame(rows)
    for c in (0.25,):
        d=D[D.conf>=c].reset_index(drop=True)
        pf=match_feet(d,G).notna().sum(); pi=match_iou(d,G).notna().sum()
        print(f"{name} {sz} {dt:.1f}s/frame conf{c}: dets/frame {len(d)/len(ims):.1f}  feet R {pf/len(G):.3f} P {pf/len(d):.3f}   iou R {pi/len(G):.3f} P {pi/len(d):.3f}", flush=True)
