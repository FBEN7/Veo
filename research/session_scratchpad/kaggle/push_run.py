"""Push the window-pipeline kernel (run/kernel_template.py) for some windows,
with the code dataset and one video dataset as inputs.

    KAGGLE_API_TOKEN=... python push_run.py W1 [W2:seen ...] --data francoisbernard/veo-st2-w-check
"""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, nargs="+", help="the batch's video dataset(s)")
ap.add_argument("--slug", default="veo-gpu-run")
ap.add_argument("--cut", action="store_true", help="cut the clips on Kaggle (internet, HF_TOKEN secret)")
ap.add_argument("--plan", help="a clips_*.json plan: its clips become the units (W[:seen]@span +-50)")
ap.add_argument("--seen", nargs="*", default=[], help="with --plan: matches whose ball needs --allow-seen")
ap.add_argument("--only", nargs=2, type=int, metavar=("I", "J"), help="with --plan: clips [I, J) of it")
ap.add_argument("--extra", default="", help='more run_windows.py options in one string, e.g. --extra="--players-only --pitch-margin 0"')
ap.add_argument("windows", nargs="*", help="W, or W:seen (ball with --allow-seen)")
args = ap.parse_args()
if args.plan:
    clips = json.loads(Path(args.plan).read_text())["clips"]
    if args.only:
        clips = clips[args.only[0]:args.only[1]]
    for c in clips:
        a, b = c["span_local"]
        args.windows.append(f"{c['name']}{':seen' if c['match'] in args.seen else ''}"
                            f"@{max(a - 50, 0)}-{b + 51}")
d = HERE / "run"
(d / "kernel.py").write_text((d / "kernel_template.py").read_text()
                             .replace("__WINDOWS__", json.dumps(args.windows))
                             .replace("__CUT__", "True" if args.cut else "False")
                             .replace("__EXTRA__", json.dumps(args.extra.split())))
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": f"francoisbernard/{args.slug}", "title": args.slug, "code_file": "kernel.py",
    "language": "python", "kernel_type": "script", "is_private": True,
    "enable_gpu": True, "enable_internet": bool(args.cut),
    "dataset_sources": ["francoisbernard/veo-code", *args.data],
    "competition_sources": [], "kernel_sources": []}, indent=1))
subprocess.run(["kaggle", "kernels", "push", "-p", str(d), "--accelerator", "NvidiaTeslaT4"],
               check=True)
