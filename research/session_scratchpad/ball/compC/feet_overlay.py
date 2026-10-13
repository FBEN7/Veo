import sys
sys.path.insert(0, "/home/user/Veo")
import cv2, numpy as np, pandas as pd
from src import soccertrack_v2 as st
from src.ball_heatmap import pitch_to_image
S = "/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v2_smoke"
cal = st.calibration("118577")
g = st.gsr_table("118577", 1)
cap = cv2.VideoCapture("/home/user/Veo/data/soccertrack_v2/st2_118577_1st_f030000.mp4")
want = {653: (3380 - 200, 408 - 150), 676: (3413 - 200, 400 - 150)}
k = 0
tiles = []
while k <= max(want):
    ok, im = cap.read()
    if k in want:
        x0, y0 = want[k]
        f = 30000 + k
        p = g[g.frame == f]
        crop = im[y0:y0 + 620, x0:x0 + 720].copy()
        uv = pitch_to_image(p[["x", "y"]].to_numpy(float), cal)
        for (u, v), r in zip(uv, p.itertuples()):
            cv2.circle(crop, (int(u - x0), int(v - y0)), 5, (255, 0, 255), 2)       # projected feet
            cv2.rectangle(crop, (int(r.bx - x0), int(r.by - y0)), (int(r.bx + r.bw - x0), int(r.by + r.bh - y0)), (255, 255, 0), 1)
            print(f, r.player_id, "proj", np.round([u, v]), "box bottom-mid", (r.bx + r.bw / 2, r.by + r.bh))
        tiles.append(crop)
    k += 1
cv2.imwrite(f"{S}/feet_overlay.png", np.vstack(tiles))
