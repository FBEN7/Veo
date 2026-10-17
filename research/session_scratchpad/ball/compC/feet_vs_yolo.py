"""GSR feet (projected through the calibration) against YOLO person feet
(box bottom-middle) on a few frames of a training window: per region,
median offset detected - GSR in pixels and in metres."""
import sys, json
sys.path.insert(0, "/home/user/Veo")
import cv2, numpy as np
from ultralytics import YOLO
from src import soccertrack_v2 as st
from src.ball_heatmap import pitch_to_image
w = sys.argv[1]
locs = [int(x) for x in sys.argv[2:]]
info = json.loads(open(f"/home/user/Veo/data/soccertrack_v2/{w}.json").read())
cal = st.calibration(info["match"])
g = st.gsr_table(info["match"], info["half"])
model = YOLO("/home/user/Veo/yolov8s.pt")
cap = cv2.VideoCapture(f"/home/user/Veo/data/soccertrack_v2/{w}.mp4")
k, rows = 0, []
while k <= max(locs):
    ok = cap.grab()
    if k in locs:
        im = cap.retrieve()[1]
        r = model(im, imgsz=4096, conf=0.3, classes=[0], verbose=False)[0]
        det = np.array([((x0 + x1) / 2, y1) for x0, y0, x1, y1 in r.boxes.xyxy.cpu().numpy()])
        p = g[g.frame == info["start_frame"] + k]
        uv = pitch_to_image(p[["x", "y"]].to_numpy(float), cal)
        for (u, v), xy in zip(uv, p[["x", "y"]].to_numpy(float)):
            if not len(det):
                continue
            d = np.hypot(*(det - (u, v)).T)
            j = int(np.argmin(d))
            if d[j] > 80:
                continue
            dm = st.image_to_pitch([det[j]], cal)[0] - xy
            rows.append((k, u, v, det[j][0] - u, det[j][1] - v, dm[0], dm[1], xy[1]))
    k += 1
a = np.array(rows)
print(w, len(a), "matched GSR feet (within 80 px of a detected person)")
for name, m in (("all", np.ones(len(a), bool)), ("far y<-10", a[:, 7] < -10), ("mid", np.abs(a[:, 7]) <= 10), ("near y>10", a[:, 7] > 10)):
    if m.sum():
        b = a[m]
        print(f"  {name:10s} n={m.sum():3d} px dx median {np.median(b[:,3]):+.0f} dy {np.median(b[:,4]):+.0f} |d| median {np.median(np.hypot(b[:,3],b[:,4])):.0f}"
              f"; metres dx {np.median(b[:,5]):+.2f} dy {np.median(b[:,6]):+.2f} |d| median {np.median(np.hypot(b[:,5],b[:,6])):.2f}")
