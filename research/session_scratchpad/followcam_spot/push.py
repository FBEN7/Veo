"""follow-cam pilot (plan.json): stage the Kaggle dataset (crops, labels,
patched T-DEED, infer.py) and push / collect the kernel.

    python push.py stage                       # builds + uploads francoisbernard/veo-followcam-pilot
    python push.py push timing|pilot [--epochs N --epoch-frames F --warmup W]
    python push.py status
    python push.py collect timing|pilot        # downloads the kernel's output into runs/<name>/
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
OWNER, DS, SLUG = "francoisbernard", "veo-followcam-pilot", "veo-gpu-run"
TRAIN = ["118575_1", "118575_2", "118576_1", "118576_2"]
VAL = ["117093_1", "117093_2"]


def kaggle(*args, check=True):
    env = {**os.environ, "KAGGLE_API_TOKEN": os.environ.get("KAGGLE_TOKEN", "")}
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, env=env)
    if check and r.returncode:
        raise SystemExit(f"kaggle {' '.join(args[:2])}: {r.stderr[-800:]}")
    return r.stdout


def stage():
    d = SCR / "stage_followcam"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for v in TRAIN + VAL:
        src = CROPS / v
        if not (src / "done.json").exists():
            raise SystemExit(f"{v} not generated")
        os.link(src / "crops.mp4", d / f"{v}.mp4")
        shutil.copy(src / "Labels-ball.json", d / f"{v}.labels.json")
        shutil.copy(src / "done.json", d / f"{v}.done.json")
    shutil.copy(SCR / "tdeed.tar.gz", d / "tdeed.tar.gz")
    shutil.copy(HERE / "infer.py", d / "infer.py")
    (d / "stage_marker.json").write_text(json.dumps({"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0]}))
    (d / "dataset-metadata.json").write_text(json.dumps({"title": DS, "id": f"{OWNER}/{DS}", "licenses": [{"name": "CC-BY-4.0"}]}))
    out = kaggle("datasets", "create", "-p", str(d), check=False)
    if "already" in out.lower() or "exists" in out.lower():
        kaggle("datasets", "version", "-p", str(d), "-m", "restage")
    for _ in range(240):
        st = (kaggle("datasets", "status", f"{OWNER}/{DS}", check=False).strip().splitlines() or [""])[-1]
        if st == "ready":
            break
        time.sleep(30)
    else:
        raise SystemExit("dataset not ready")
    shutil.rmtree(d)
    print("staged", DS)


def push(name, epochs, epoch_frames, warmup):
    p = {"name": name, "train": TRAIN, "val": VAL, "infer": VAL, "epochs": epochs, "epoch_frames": epoch_frames,
         "warmup": warmup, "keep_ckpt": name == "pilot"}
    d = SCR / f"kernel_{name}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "kernel.py").write_text((HERE / "kernel.py").read_text().replace("__PARAMS__", repr(p)))
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": "kernel.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [f"{OWNER}/{DS}"],
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
    for f in ("record.json", "kernel_log.txt", "train_store.txt", "train_load.txt", "infer.txt"):
        if (tmp / f).exists():
            shutil.copy(tmp / f, out / f)
    if (tmp / "result.tar.gz").exists():
        with tarfile.open(tmp / "result.tar.gz") as t:
            t.extractall(SCR / f"result_{name}")
    print(sorted(p.name for p in out.iterdir()), "| result in", SCR / f"result_{name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--epoch-frames", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "stage":
        stage()
    elif a.cmd == "push":
        push(a.name, a.epochs, a.epoch_frames, a.warmup)
    elif a.cmd == "status":
        print(kaggle("kernels", "status", f"{OWNER}/{SLUG}", check=False))
    elif a.cmd == "collect":
        collect(a.name)
