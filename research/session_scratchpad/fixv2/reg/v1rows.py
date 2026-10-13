"""v1 detect() rows on 3 frames of a training window, from the cwd's src."""
import json, sys
sys.path.insert(0, ".")
import cv2, numpy as np, torch
from src import ball_heatmap as bh, soccertrack_v2 as st
from src.paths import DATA_DIR
torch.set_num_threads(2)
w = "st2_118575_1st_f015000"
info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
m = bh.build(); m.load_state_dict(torch.load("/home/user/Veo/.cache/ball_heatmap.3000.pt")); m.eval()
cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{w}.mp4"))
out = []
for i in range(3):
    ok, im = cap.read()
    out.append(bh.detect(m, im, info["pitch"], tau=0.01, top_k=8, cal=st.calibration(info["match"])).tolist())
np.save(sys.argv[1], np.array(json.dumps(out)))
