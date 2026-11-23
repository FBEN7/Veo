"""Push the M9 player-detection diagnostic kernel (run/m9_players_template.py).

    KAGGLE_API_TOKEN=... python push_m9_players.py --stage DIR CLIP ...
"""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True)
ap.add_argument("--slug", default="veo-m9-players")
ap.add_argument("--models", nargs="+", default=["yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
ap.add_argument("clips", nargs="+")
args = ap.parse_args()
k = Path(args.stage)
k.mkdir(parents=True, exist_ok=True)
(k / "kernel.py").write_text((HERE / "run/m9_players_template.py").read_text().replace("__CLIPS__", json.dumps(args.clips)).replace("__MODELS__", json.dumps(args.models)))
(k / "kernel-metadata.json").write_text(json.dumps({
    "id": f"francoisbernard/{args.slug}", "title": args.slug, "code_file": "kernel.py",
    "language": "python", "kernel_type": "script", "is_private": True,
    "enable_gpu": True, "enable_internet": True,
    "dataset_sources": ["francoisbernard/veo-code", "francoisbernard/veo-st2-dev2"],
    "competition_sources": [], "kernel_sources": []}, indent=1))
subprocess.run(["kaggle", "kernels", "push", "-p", str(k)], check=True)
