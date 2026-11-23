"""follow-cam spotting (plan.json, plan2.json): stage the Kaggle datasets
(crops, labels, patched T-DEED, infer.py; plan 2's checkpoint) and push /
collect the kernel.

    python push.py stage | stage5              # builds + uploads the pilot's / plan 2's dataset
    python push.py stage-ckpt                  # plan 2: kernel 1's last epoch as a dataset (after collect train5)
    python push.py push timing|pilot [--epochs N --epoch-frames F --warmup W]
    python push.py push train5 | infer5 [--jpeg-q 80]
    python push.py status
    python push.py collect <name>              # downloads the kernel's output into runs/<name>/
"""
import argparse
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCR = Path("/tmp/claude-0/-home-user-veo/b69e5b60-e0fe-5920-8f15-285768d7ea57/scratchpad")
CROPS = SCR / "followcam_crops"
OWNER, SLUG = "francoisbernard", "veo-gpu-run"
DS, DS5, DSCK = "veo-followcam-pilot", "veo-followcam-5m", "veo-followcam-ckpt5"
TRAIN = ["118575_1", "118575_2", "118576_1", "118576_2"]
VAL = ["117093_1", "117093_2"]
TRAIN5 = [f"{m}_{h}" for m in ("118575", "118576", "118577", "118578", "128058") for h in (1, 2)]
EVAL5 = VAL + ["132877_1", "132877_2"]


def kaggle(*args, check=True):
    env = {**os.environ, "KAGGLE_API_TOKEN": os.environ.get("KAGGLE_TOKEN", "")}
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, env=env)
    if check and r.returncode:
        raise SystemExit(f"kaggle {' '.join(args[:2])}: {r.stderr[-800:]}")
    return r.stdout


def upload(d, name):
    (d / "dataset-metadata.json").write_text(json.dumps({"title": name, "id": f"{OWNER}/{name}", "licenses": [{"name": "CC-BY-4.0"}]}))
    out = kaggle("datasets", "create", "-p", str(d), check=False)
    if "already" in out.lower() or "exists" in out.lower():
        kaggle("datasets", "version", "-p", str(d), "-m", "restage")
    for _ in range(240):
        st = (kaggle("datasets", "status", f"{OWNER}/{name}", check=False).strip().splitlines() or [""])[-1]
        if st == "ready":
            break
        time.sleep(30)
    else:
        raise SystemExit("dataset not ready")
    shutil.rmtree(d)
    print("staged", name)


def stage(name, videos, plan):
    d = SCR / f"stage_{name}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for v in videos:
        src = CROPS / v
        if not (src / "done.json").exists():
            raise SystemExit(f"{v} not generated")
        os.link(src / "crops.mp4", d / f"{v}.mp4")
        shutil.copy(src / "Labels-ball.json", d / f"{v}.labels.json")
        shutil.copy(src / "done.json", d / f"{v}.done.json")
    shutil.copy(SCR / "tdeed.tar.gz", d / "tdeed.tar.gz")
    shutil.copy(HERE / "infer.py", d / "infer.py")
    (d / "stage_marker.json").write_text(json.dumps({"plan sha256": (HERE / f"{plan}.sha256").read_text().split()[0]}))
    upload(d, name)


def stage_ckpt():
    src = SCR / "result_train5" / "ckpt" / "SoccerNetBall_followcam"
    d = SCR / "stage_ckpt5"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for f in ("checkpoint_last.pt", "loss.json"):
        shutil.copy(src / f, d / f)
    upload(d, DSCK)


RUNS = {   # name -> (datasets, params); plan 2 keeps the pilot's schedule
    "timing": ([DS], {"train": TRAIN, "val": VAL, "infer": VAL}),
    "pilot": ([DS], {"train": TRAIN, "val": VAL, "infer": VAL, "keep_ckpt": True}),
    "train5": ([DS5], {"mode": "train", "train": TRAIN5, "val": VAL, "infer": [], "keep_ckpt": True, "ag": [4],
                       "epochs": 20, "epoch_frames": 100000, "warmup": 2}),
    "infer5": ([DS5, DSCK], {"mode": "infer", "train": [], "val": [], "infer": EVAL5,
                             "epochs": 20, "epoch_frames": 100000, "warmup": 2}),
}


def push(name, epochs, epoch_frames, warmup, jpeg_q):
    datasets, base = RUNS[name]
    p = {"name": name, "epochs": epochs, "epoch_frames": epoch_frames, "warmup": warmup, "keep_ckpt": False,
         "jpeg_q": jpeg_q, **base}
    d = SCR / f"kernel_{name}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "kernel.py").write_text((HERE / "kernel.py").read_text().replace("__PARAMS__", repr(p)))
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": "kernel.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [f"{OWNER}/{x}" for x in datasets],
        "competition_sources": [], "kernel_sources": []}, indent=1))
    out = kaggle("kernels", "push", "-p", str(d), "--accelerator", "NvidiaTeslaT4")
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / f"{name}.pushed.json").write_text(json.dumps({**p, "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    print(out.strip()[-300:])


def collect(name):
    out = HERE / "runs" / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    tmp = SCR / f"out_{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    kaggle("kernels", "output", f"{OWNER}/{SLUG}", "-p", str(tmp), check=False)
    for f in ("record.json", "kernel_log.txt", "infer.txt", *[x.name for x in tmp.glob("train_*.txt")]):
        if (tmp / f).exists():
            shutil.copy(tmp / f, out / f)
    if (tmp / "result.tar.gz").exists():
        with tarfile.open(tmp / "result.tar.gz") as t:
            t.extractall(SCR / f"result_{name}", filter="data")
    print(sorted(p.name for p in out.iterdir()), "| result in", SCR / f"result_{name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--epoch-frames", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--jpeg-q", type=int, default=92)
    a = ap.parse_args()
    if a.cmd == "stage":
        stage(DS, TRAIN + VAL, "plan.json")
    elif a.cmd == "stage5":
        stage(DS5, TRAIN5 + EVAL5, "plan2.json")
    elif a.cmd == "stage-ckpt":
        stage_ckpt()
    elif a.cmd == "push":
        push(a.name, a.epochs, a.epoch_frames, a.warmup, a.jpeg_q)
    elif a.cmd == "status":
        print(kaggle("kernels", "status", f"{OWNER}/{SLUG}", check=False))
    elif a.cmd == "collect":
        collect(a.name)
