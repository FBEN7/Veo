# Kaggle GPU kernel: why the player detector misses M9's players. On every
# 25th frame of a few M9 clips (and M7 clips as a control), every YOLO person
# detection at confidence >= 0.05 at the pipeline's imgsz (native width), for
# yolov8s (the pipeline's), yolov8m and yolov8l, with the pipeline's
# near-grass flag (src/grass._near_pitch on _grass_mask). Recall against the
# truth is computed back home (the labels never come here). Written by
# push_m9_players.py; output /kaggle/working/m9_players.parquet.
import glob, os, shutil, subprocess, sys, time

CLIPS = __CLIPS__
MODELS = ["yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
STEP = 25
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise SystemExit(f"{name} not in the inputs")
    return hits[0]


root = "/home/user/Veo"
shutil.copytree(os.path.dirname(find("eval_soccertrack_v2.py")), root, dirs_exist_ok=True)
assets = os.path.dirname(find("ball_v3b.1500.pt"))
wheels = sorted(glob.glob(os.path.join(assets, "*.whl")))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", *wheels], check=True)
shutil.copy(os.path.join(assets, "yolov8s.pt"), "/kaggle/working/yolov8s.pt")
log("setup done")
sys.path.insert(0, root)
os.chdir("/kaggle/working")

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from src.grass import _grass_mask, _near_pitch  # noqa: E402

rows = []
for name in MODELS:
    model = YOLO(name)
    for w in CLIPS:
        cap = cv2.VideoCapture(find(f"{w}.mp4"))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        imgsz = -(-width // 32) * 32
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for f in range(0, n, STEP):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                break
            res = model(frame, verbose=False, conf=0.05, imgsz=imgsz, classes=[0], device=0)[0]
            mask = _grass_mask(frame)
            for (x0, y0, x1, y1), c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
                rows.append({"model": name, "clip": w, "frame": f, "conf": c, "x0": x0, "y0": y0,
                             "x1": x1, "y1": y1, "near": _near_pitch(mask, (x0 + x1) / 2, y1)})
        cap.release()
        log(name, w, len(rows))
pd.DataFrame(rows).to_parquet("/kaggle/working/m9_players.parquet")
log("done")
