# Kaggle GPU kernel for the hybrid ball (written by push_hybrid.py). Inputs:
# veo-code (code tree, wheels), veo-hybrid-stage (frames.parquet: unit,
# frame, model; the v4r checkpoints) and the shot-clip datasets (M7 / M9
# clips only those in opened.json, cut on the repo machine through
# src/unseal.py). Per unit, detect_ball_heatmap.dense_v2 with the unit's v4r
# checkpoint at its listed frames: the rows the v3 detector gave
# (.cache/v3_balls_<w>.parquet's format). Units split over the two GPUs.
# Output: /kaggle/working/rows/<unit>.parquet and logs.
import glob, os, shutil, subprocess, sys, time

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
os.makedirs(os.path.join(root, ".cache/soccertrack_v2/calib"), exist_ok=True)
for f in glob.glob("/kaggle/input/**/*", recursive=True):
    if os.path.isfile(f) and f.endswith(("_camera_intrinsics.npz", "_homography.npy")):
        shutil.copy(f, os.path.join(root, ".cache/soccertrack_v2/calib", os.path.basename(f)))
wheels = sorted(glob.glob(os.path.join(assets, "*.whl")))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", *wheels], check=True)
hub = os.path.expanduser("~/.cache/torch/hub/checkpoints")
os.makedirs(hub, exist_ok=True)
shutil.copy(os.path.join(assets, "resnet18-f37072fd.pth"), hub)
stage = os.path.dirname(find("frames.parquet"))
os.makedirs("/kaggle/working/rows", exist_ok=True)
os.makedirs("/kaggle/working/logs", exist_ok=True)
subprocess.run(["nvidia-smi", "-L"], check=True)
log("setup done")

worker = r'''
import glob, json, sys, time
from pathlib import Path
import pandas as pd
import torch
sys.path.insert(0, "/home/user/Veo")
import detect_ball_heatmap as dbh
from src import ball_heatmap as bh
from src import soccertrack_v2 as st
torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
stage, part = sys.argv[1], int(sys.argv[2])
fr = pd.read_parquet(f"{stage}/frames.parquet")
units = sorted(set(fr.unit))[part::2]
models = {}
def load(name):
    if name not in models:
        ck = f"{stage}/{name}.pt"
        m, info = bh.load_v2(ck)
        m.to("cuda")
        fwd = m.forward
        m.forward = lambda x, fwd=fwd: fwd(x.to("cuda", non_blocking=True)).float().cpu()
        models[name] = (m, info, bh.load_calibration(ck))
    return models[name]
def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None
t0 = time.time()
for i, w in enumerate(units):
    g = fr[fr.unit == w]
    if find(f"{w}.json") is None or find(f"{w}.mp4") is None:
        print(f"{i + 1}/{len(units)} {w}: MISSING clip, skipped", flush=True)
        continue
    info = json.loads(Path(find(f"{w}.json")).read_text())
    m, mi, calib = load(g.model.iloc[0])
    df, done = dbh.dense_v2(find(f"{w}.mp4"), info, m, sorted(g.frame.astype(int)), calib=calib,
                            cal=st.calibration(str(info["match"])), tau=0.05, top_k=8, log=lambda *a, **k: None,
                            motion_zero=mi["rgb_only"], sigma_d=mi["sigma_d"], sigma_f=mi["sigma_f"])
    df.to_parquet(f"/kaggle/working/rows/{w}.parquet")
    print(f"{i + 1}/{len(units)} {w}: {len(done)} of {len(g)} frames, {len(df)} rows [{time.time() - t0:.0f} s]", flush=True)
'''
open("/kaggle/working/worker.py", "w").write(worker)
procs = []
for g in (0, 1):
    f = open(f"/kaggle/working/logs/detect_{g}.log", "w")
    procs.append((subprocess.Popen([sys.executable, "/kaggle/working/worker.py", stage, str(g)], stdout=f,
                                   stderr=subprocess.STDOUT, cwd=root,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))), f))
codes = [p.wait() for p, _ in procs]
for g, (_, f) in enumerate(procs):
    f.close()
    print(f"--- detect_{g} (exit {codes[g]})", open(f.name).read()[-2000:], sep="\n", flush=True)
log("done", codes)
if any(codes):
    raise SystemExit(f"detection failed: {codes}")
