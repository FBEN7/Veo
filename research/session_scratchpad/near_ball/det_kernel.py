# Kaggle GPU kernel (pushed by push_det.py, plan_v2.json): the player
# detector's variants on the action frames of veo-nearball-stage/frames.parquet
# (unit, frame). Per clip, one sequential pass through its video (the
# shot-clip datasets); at each listed frame, every pass below; person boxes
# kept. Clips split over the two GPUs. Output: /kaggle/working/det/<unit>.parquet
# (frame, pass, x0, y0, x1, y1, conf) and logs.
import glob, os, subprocess, sys, time

T0 = time.time()


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None


stage = os.path.dirname(find("frames.parquet"))
code = os.path.dirname(find("ultralytics-8.4.183-py3-none-any.whl"))
wheels = sorted(glob.glob(os.path.join(code, "*.whl")))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", *wheels], check=True)
os.environ["YOLO_OFFLINE"] = "1"
os.makedirs("/kaggle/working/det", exist_ok=True)
subprocess.run(["nvidia-smi", "-L"], check=True)
print(f"[{time.time() - T0:5.0f}s] setup done", flush=True)

worker = r'''
import glob, sys, time
import cv2
import pandas as pd
from ultralytics import YOLO
stage, part = sys.argv[1], int(sys.argv[2])
dev = f"cuda:{part}"
PASSES = {  # name: (weights, conf, iou, augment)
    "s_c10": ("yolov8s.pt", 0.10, 0.70, False),
    "s_iou85": ("yolov8s.pt", 0.20, 0.85, False),
    "m": ("yolov8m.pt", 0.20, 0.70, False),
    "x": ("yolov8x.pt", 0.20, 0.70, False),
    "s_tta": ("yolov8s.pt", 0.20, 0.70, True),
}
models = {w: YOLO(f"{stage}/{w}") for w in {p[0] for p in PASSES.values()}}
fr = pd.read_parquet(f"{stage}/frames.parquet")
units = sorted(set(fr.unit))[part::2]
def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None
t0 = time.time()
for i, w in enumerate(units):
    video = find(f"{w}.mp4")
    if video is None:
        print(f"{i + 1}/{len(units)} {w}: MISSING", flush=True)
        continue
    want = set(fr[fr.unit == w].frame.astype(int))
    cap = cv2.VideoCapture(video)
    rows, idx, last = [], 0, max(want)
    while idx <= last:
        ok = cap.grab()
        if not ok:
            break
        if idx in want:
            ok, img = cap.retrieve()
            for name, (wt, conf, iou, aug) in PASSES.items():
                r = models[wt](img, imgsz=4096, conf=conf, iou=iou, classes=[0], augment=aug, device=dev, verbose=False)[0]
                b = r.boxes
                for (x0, y0, x1, y1), c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
                    rows.append((idx, name, float(x0), float(y0), float(x1), float(y1), float(c)))
        idx += 1
    cap.release()
    pd.DataFrame(rows, columns=["frame", "pass", "x0", "y0", "x1", "y1", "conf"]).to_parquet(f"/kaggle/working/det/{w}.parquet")
    print(f"[{time.time() - t0:6.0f}s] {i + 1}/{len(units)} {w}: {len(want)} frames, {len(rows)} boxes", flush=True)
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
