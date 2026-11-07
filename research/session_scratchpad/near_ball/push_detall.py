"""Push plan_v4's step 1, the GPU detection pass (detall_kernel.py): the 223
units of actions.parquet; yolov8s.pt from veo-nearball-stage (sha256 checked
in weights_sha256.json), the ultralytics wheels from veo-code, the videos from
the shot-clip datasets (M7 / M9: veo-st2-dev2, opened clips only, through
src/unseal.py).

    VEO_UNSEAL=... KAGGLE_API_TOKEN=... python push_detall.py [--dry]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
OWNER = "francoisbernard"
SEALED = {"128057", "132831"}
CLIP_DS = ["veo-st2-dev2", *[f"veo-st2-m5-b{i}" for i in range(4)], *[f"veo-st2-train-b{i}" for i in range(10)]]


def main():
    from src.unseal import unsealed

    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(HERE / "push_detall"))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    for m in sorted(SEALED):
        if not unsealed(m, "near_ball plan_v4: GPU detection pass on the opened M7 / M9 shot clips (Kaggle)"):
            raise SystemExit("VEO_UNSEAL needed")
    units = sorted(set(pd.read_parquet(HERE / "stages.parquet").unit))
    opened = set(json.loads((ROOT / "research/session_scratchpad/kaggle/opened.json").read_text())["clips"])
    shut = [u for u in units if u.split("_")[1] in SEALED and u not in opened]
    if shut:
        raise SystemExit(f"sealed clips not opened: {shut}")
    kd = Path(args.work)
    shutil.rmtree(kd, ignore_errors=True)
    kd.mkdir(parents=True)
    src = (HERE / "detall_kernel.py").read_text().replace("__UNITS__", json.dumps(units))
    (kd / "kernel.py").write_text(src)
    (kd / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/veo-nearball-detall", "title": "veo-nearball-detall", "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [f"{OWNER}/veo-code", f"{OWNER}/veo-nearball-stage", *[f"{OWNER}/{d}" for d in CLIP_DS]],
        "competition_sources": [], "kernel_sources": []}, indent=1))
    print(len(units), "units")
    if args.dry:
        return
    subprocess.run(["kaggle", "kernels", "push", "-p", str(kd)], check=True)


if __name__ == "__main__":
    main()
