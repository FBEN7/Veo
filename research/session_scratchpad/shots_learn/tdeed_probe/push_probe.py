"""tdeed_probe (plan.json): upload the stage (T-DEED code, checkpoint, crop
centres, run_tdeed.py) and push two kernels: the clips, the windows.

    KAGGLE_TOKEN=... python push_probe.py stage STAGE_DIR    # builds + uploads veo-tdeed-stage
    KAGGLE_TOKEN=... python push_probe.py push                # both kernels
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = "francoisbernard"
STAGE_DS = "veo-tdeed-stage"
CLIP_DS = ["veo-st2-dev2", *[f"veo-st2-m5-b{i}" for i in range(4)], *[f"veo-st2-train-b{i}" for i in range(10)]]
WIN_DS = [f"veo-st2-neg-b{i}" for i in range(4)] + ["veo-st2-dev1"]
SCRATCH = Path("/tmp/claude-0/-home-user-veo/b69e5b60-e0fe-5920-8f15-285768d7ea57/scratchpad")
KERNEL = '''import glob, json, os, subprocess, sys, tarfile
UNITS = __UNITS__
stage = os.path.dirname(glob.glob("/kaggle/input/**/units.json", recursive=True)[0])
repo = os.path.dirname(glob.glob("/kaggle/input/**/train_tdeed.py", recursive=True)[0])
subprocess.run(["nvidia-smi", "-L"])
half = [UNITS[0::2], UNITS[1::2]]
ps = [subprocess.Popen([sys.executable, os.path.join(stage, "run_tdeed.py"), "--repo", repo,
                        "--ckpt", os.path.join(stage, "SoccerNetBall_challenge1.pt"), "--stage", stage,
                        "--videos", "/kaggle/input", "--out", "/kaggle/working/out", "--units", *h],
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)},
                       stdout=open(f"/kaggle/working/log{i}.txt", "w"), stderr=subprocess.STDOUT)
      for i, h in enumerate(half) if h]
codes = [p.wait() for p in ps]
done = sorted(glob.glob("/kaggle/working/out/*.npz"))
with tarfile.open("/kaggle/working/probs.tar.gz", "w:gz") as t:
    for f in done:
        t.add(f, arcname=os.path.basename(f))
json.dump({"codes": codes, "done": len(done), "asked": len(UNITS)}, open("/kaggle/working/record.json", "w"))
'''


def kaggle(*args):
    env = {**os.environ, "KAGGLE_API_TOKEN": os.environ.get("KAGGLE_TOKEN", "")}
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(f"kaggle {' '.join(args[:2])}: {r.stderr[-800:]}")
    return r.stdout


def stage(src):
    d = SCRATCH / "tdeed_stage_ds"
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SCRATCH / "ext/tdeed", d / "tdeed", ignore=shutil.ignore_patterns(".git", "__pycache__"))
    shutil.copy(SCRATCH / "ext/ckpt/SoccerNetBall_challenge1.pt", d)
    for f in ("units.json", "centres.parquet"):
        shutil.copy(Path(src) / f, d)
    shutil.copy(HERE / "run_tdeed.py", d)
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": STAGE_DS, "id": f"{OWNER}/{STAGE_DS}", "licenses": [{"name": "other"}]}))
    print(kaggle("datasets", "create", "-p", str(d), "-r", "zip")[-300:])
    for _ in range(60):
        if kaggle("datasets", "status", f"{OWNER}/{STAGE_DS}").strip().splitlines()[-1] == "ready":
            return print("stage ready")
        time.sleep(20)
    raise SystemExit("stage not ready")


def push():
    units = json.loads((SCRATCH / "tdeed_stage/units.json").read_text())
    for slug, names, ds in (("veo-gpu-run", [u["name"] for u in units if u["span"]], CLIP_DS),
                            ("veo-gpu-run2", [u["name"] for u in units if not u["span"]], WIN_DS)):
        d = SCRATCH / f"tdeed_kernel_{slug}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        (d / "kernel.py").write_text(KERNEL.replace("__UNITS__", json.dumps(names)))
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{OWNER}/{slug}", "title": slug, "code_file": "kernel.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False,
            "dataset_sources": [f"{OWNER}/{STAGE_DS}", *[f"{OWNER}/{x}" for x in ds]],
            "competition_sources": [], "kernel_sources": []}, indent=1))
        print(slug, len(names), kaggle("kernels", "push", "-p", str(d), "--accelerator", "NvidiaTeslaT4")[-200:])


if __name__ == "__main__":
    {"stage": lambda: stage(sys.argv[2]), "push": push}[sys.argv[1]]()
