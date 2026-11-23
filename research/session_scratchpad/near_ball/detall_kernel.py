# Kaggle GPU kernel (pushed by push_detall.py, plan_v4.json step 1): yolov8s
# on the even frames 300..1125 of each unit (UNITS, filled in by the push
# script), imgsz 4096, conf 0.10, NMS IoU 0.70, person class; every box kept.
# Units split over the two GPUs. Output: /kaggle/working/det/<unit>.parquet
# (frame, x0, y0, x1, y1, conf) and logs.
import glob, os, subprocess, sys, time

UNITS = __UNITS__
T0 = time.time()


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None


import hashlib, json
stage = os.path.dirname(find("weights_sha256.json"))       # veo-nearball-stage
want = json.load(open(f"{stage}/weights_sha256.json"))["yolov8s.pt"]
got = hashlib.sha256(open(f"{stage}/yolov8s.pt", "rb").read()).hexdigest()
assert got == want, (got, want)
print("yolov8s.pt sha256", got, flush=True)
code = os.path.dirname(find("ultralytics-8.4.183-py3-none-any.whl"))
wheels = sorted(glob.glob(os.path.join(code, "*.whl")))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", *wheels], check=True)
os.environ["YOLO_OFFLINE"] = "1"
os.makedirs("/kaggle/working/det", exist_ok=True)
subprocess.run(["nvidia-smi", "-L"], check=True)
with open("/kaggle/working/units.txt", "w") as f:
    f.write("\n".join(UNITS))
print(f"[{time.time() - T0:5.0f}s] setup done, {len(UNITS)} units", flush=True)

worker = r'''
import glob, sys, time
import cv2
import pandas as pd
from ultralytics import YOLO
stage, part = sys.argv[1], int(sys.argv[2])
dev = f"cuda:{part}"
model = YOLO(f"{stage}/yolov8s.pt")
units = open("/kaggle/working/units.txt").read().split()[part::2]
LO, HI = 300, 1125
def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None
t0 = time.time()
for i, w in enumerate(units):
    video = find(f"{w}.mp4")
    if video is None:
        print(f"{i + 1}/{len(units)} {w}: MISSING", flush=True)
        continue
    cap = cv2.VideoCapture(video)
    rows, idx, n = [], 0, 0
    while idx <= HI:
        if not cap.grab():
            break
        if idx >= LO and idx % 2 == 0:
            ok, img = cap.retrieve()
            r = model(img, imgsz=4096, conf=0.10, iou=0.70, classes=[0], device=dev, verbose=False)[0]
            b = r.boxes
            for (x0, y0, x1, y1), c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
                rows.append((idx, float(x0), float(y0), float(x1), float(y1), float(c)))
            n += 1
        idx += 1
    cap.release()
    pd.DataFrame(rows, columns=["frame", "x0", "y0", "x1", "y1", "conf"]).to_parquet(f"/kaggle/working/det/{w}.parquet")
    print(f"[{time.time() - t0:6.0f}s] {i + 1}/{len(units)} {w}: {n} frames, {len(rows)} boxes", flush=True)
'''
with open("/kaggle/working/worker.py", "w") as f:
    f.write(worker)
procs = [subprocess.Popen([sys.executable, "/kaggle/working/worker.py", stage, str(k)],
                          stdout=open(f"/kaggle/working/log_{k}.txt", "w"), stderr=subprocess.STDOUT) for k in (0, 1)]
for p in procs:
    p.wait()
for k in (0, 1):
    print(open(f"/kaggle/working/log_{k}.txt").read()[-3000:], flush=True)
print(f"[{time.time() - T0:5.0f}s] done", flush=True)
