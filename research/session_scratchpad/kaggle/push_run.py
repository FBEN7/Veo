"""Push the window-pipeline kernel (run/kernel_template.py) for some windows,
with the code dataset and one video dataset as inputs.

    KAGGLE_API_TOKEN=... python push_run.py --data francoisbernard/veo-st2-w-check W1 [W2 ...]
"""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="the batch's video dataset")
ap.add_argument("--slug", default="veo-gpu-run")
ap.add_argument("--allow-seen", action="store_true")
ap.add_argument("windows", nargs="+")
args = ap.parse_args()
d = HERE / "run"
(d / "kernel.py").write_text((d / "kernel_template.py").read_text()
                             .replace("__WINDOWS__", json.dumps(args.windows))
                             .replace("__EXTRA__", json.dumps(["--allow-seen"] if args.allow_seen else [])))
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": f"francoisbernard/{args.slug}", "title": args.slug, "code_file": "kernel.py",
    "language": "python", "kernel_type": "script", "is_private": True,
    "enable_gpu": True, "enable_internet": False,
    "dataset_sources": ["francoisbernard/veo-code", args.data],
    "competition_sources": [], "kernel_sources": []}, indent=1))
subprocess.run(["kaggle", "kernels", "push", "-p", str(d), "--accelerator", "NvidiaTeslaT4"],
               check=True)
