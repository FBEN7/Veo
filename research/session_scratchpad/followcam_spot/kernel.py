"""follow-cam spotting (plan.json, plan2.json): the Kaggle side. Extracts the
crops' frames, trains T-DEED (store pass then training), saves the last epoch,
scores videos with infer.py. Parameters are filled in by push.py (P below):
mode "both" (the pilot), "train" (plan 2, kernel 1: keeps the checkpoint) or
"infer" (plan 2, kernel 2: scores with a checkpoint from an input dataset)."""
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from multiprocessing import Pool

P = __PARAMS__
MODE = P.get("mode", "both")
JPEG_Q = P.get("jpeg_q", 92)
MIN_FREE = 5e9   # plan2: stop if the disk runs under 5 GB during extraction
T0 = time.time()
stage = os.path.dirname(glob.glob("/kaggle/input/**/stage_marker.json", recursive=True)[0])
W = "/kaggle/working"
REPO, FR, LB, STORE, CK = f"{W}/T-DEED", "/tmp/frames", "/tmp/labels", "/tmp/store", f"{W}/ckpt"
log = open(f"{W}/kernel_log.txt", "a")


def say(*a):
    print(*a, flush=True)
    print(f"[{time.time() - T0:7.0f} s]", *a, file=log, flush=True)


def extract(name):
    import cv2
    d = os.path.join(FR, name)
    os.makedirs(d, exist_ok=True)
    cap, k = cv2.VideoCapture(os.path.join(stage, f"{name}.mp4")), 0
    while True:
        if k % 2000 == 0 and shutil.disk_usage("/tmp").free < MIN_FREE:
            return name, -1
        ok, img = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(d, f"frame{2 * k}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        k += 1
    return name, k


def finish(keep, counts):
    with tarfile.open(f"{W}/result.tar.gz", "w:gz") as t:
        for f in keep:
            t.add(f, arcname=os.path.relpath(f, W))
    json.dump({"params": P, "frames": counts, "seconds": round(time.time() - T0)}, open(f"{W}/record.json", "w"))
    shutil.rmtree(REPO, ignore_errors=True)
    shutil.rmtree(CK, ignore_errors=True)
    shutil.rmtree(f"{W}/scores", ignore_errors=True)


subprocess.run(["nvidia-smi", "-L"])
if os.path.exists(os.path.join(stage, "tdeed.tar.gz")):
    with tarfile.open(os.path.join(stage, "tdeed.tar.gz")) as t:
        t.extractall(W)
else:   # Kaggle unpacks archives on upload: tdeed/T-DEED
    shutil.copytree(os.path.join(stage, "tdeed", "T-DEED"), REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "SoccerNet", "tabulate"], check=False)
cfg = json.load(open(os.path.join(REPO, "config", "SoccerNetBall", "SoccerNetBall_challenge1.json")))
cfg.pop("pretrain", None)
cfg.update(frame_dir=FR, save_dir=CK, store_dir=STORE, store_mode="store", num_epochs=P["epochs"],
           epoch_num_frames=P["epoch_frames"], warm_up_epochs=P["warmup"], start_val_epoch=10 ** 6,
           criterion="loss", only_test=False, num_workers=4)
cpath = os.path.join(REPO, "config", "SoccerNetBall", "SoccerNetBall_followcam.json")
json.dump(cfg, open(cpath, "w"), indent=1)
counts = {}
ck = os.path.join(CK, "SoccerNetBall_followcam", "checkpoint_last.pt")

if MODE in ("both", "train"):
    videos = P["train"] + P["val"]
    say("free disk GB", round(shutil.disk_usage("/tmp").free / 1e9, 1), "jpeg quality", JPEG_Q)
    with Pool(4) as pool:
        counts = dict(pool.map(extract, videos))
    say("frames", counts, "free disk GB", round(shutil.disk_usage("/tmp").free / 1e9, 1))
    if any(c < 0 for c in counts.values()):
        say("STOP: disk under 5 GB during extraction (plan2: re-extract all halves at JPEG quality 80)")
        finish(glob.glob(f"{W}/*.txt"), counts)
        sys.exit(0)
    for v in videos:
        os.makedirs(os.path.join(LB, v), exist_ok=True)
        shutil.copy(os.path.join(stage, f"{v}.labels.json"), os.path.join(LB, v, "Labels-ball.json"))
    dd = os.path.join(REPO, "data", "soccernetball")
    for split, vs in (("train", P["train"]), ("val", P["val"]), ("test", P["val"])):
        json.dump([{"video": v, "num_frames": 2 * counts[v]} for v in vs], open(os.path.join(dd, f"{split}.json"), "w"))
    open(os.path.join(dd, "labels_path.txt"), "w").write(LB)
    # store pass (exits 1 by design), then training; the batch of 4 is accumulated over
    # 2 or 4 steps to fit a 16 GB T4 (same effective batch)
    for mode, ag in [("store", 1)] + [("load", a) for a in P.get("ag", [2, 4])]:
        cfg["store_mode"] = mode
        json.dump(cfg, open(cpath, "w"), indent=1)
        t = time.time()
        r = subprocess.run([sys.executable, "train_tdeed.py", "--model", "SoccerNetBall_followcam", "-ag", str(ag)], cwd=REPO,
                           stdout=open(f"{W}/train_{mode}_ag{ag}.txt", "w"), stderr=subprocess.STDOUT,
                           env={**os.environ, "WANDB_MODE": "disabled", "CUDA_VISIBLE_DEVICES": "0",
                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        say(mode, "ag", ag, "exit", r.returncode, f"{time.time() - t:.0f} s")
        if mode == "load" and r.returncode == 0:
            break
    shutil.rmtree(FR, ignore_errors=True)
else:   # infer: the checkpoint comes from an input dataset (plan2, kernel 1's last epoch)
    ck = glob.glob("/kaggle/input/**/checkpoint_last.pt", recursive=True)[0]
    say("checkpoint", ck)

if MODE in ("both", "infer") and os.path.exists(ck):
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(stage, "infer.py"), "--repo", REPO, "--config", cpath, "--ckpt", ck,
                        "--videos", stage, "--names", *P["infer"], "--out", f"{W}/scores"],
                       stdout=open(f"{W}/infer.txt", "w"), stderr=subprocess.STDOUT)
    say("infer exit", r.returncode, f"{time.time() - t:.0f} s")
keep = glob.glob(f"{W}/*.txt") + glob.glob(f"{W}/scores/*.npz") + glob.glob(f"{CK}/*/loss.json")
if P.get("keep_ckpt") and os.path.exists(ck) and MODE != "infer":
    keep.append(ck)
finish(keep, counts)
