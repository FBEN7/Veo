"""Stage and push the v4 GPU kernel (kernel_template.py).

1. The stage dataset veo-ball-<recipe>-stage, flat: the ball_v4 scripts,
   prepare.py's stage/ (targets, codes, test frames) and the click page's
   manifest; created, or versioned if it exists.
2. The kernel veo-ball-<recipe> over veo-code, veo-ball-v4-stream, the stage
   and the clip datasets of the clicked units (and the M6 guard windows,
   veo-st2-w-check and veo-st2-guard2).

    KAGGLE_API_TOKEN=... python push_v4.py --clip-datasets clip_dataset.json [--recipe v4r] [--steps 600] [--dry]
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OWNER = "francoisbernard"
SCRIPTS = ("prepare.py", "stream_subset.py", "cut_v4.py", "train_v4.py", "eval_v4.py")
GUARD_DS = (f"{OWNER}/veo-st2-w-check", f"{OWNER}/veo-st2-guard2")   # the M6 guard windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-datasets", required=True, help="json {unit: owner/dataset}")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--recipe", default="v4", help="train_v4.py --recipe")
    ap.add_argument("--work", default=str(HERE / "push"))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    work = Path(args.work)
    ds, kd = work / "stage_ds", work / "kernel"
    shutil.rmtree(work, ignore_errors=True)
    ds.mkdir(parents=True)
    kd.mkdir(parents=True)
    for n in SCRIPTS:
        shutil.copy(HERE / n, ds / n)
    for n in ("targets.parquet", "codes.npy", "test_frames.parquet"):
        shutil.copy(HERE / "stage" / n, ds / n)
    shutil.copy(ROOT / "research/session_scratchpad/ball_clicks/manifest.json", ds / "manifest.json")
    slug = f"{OWNER}/veo-ball-{args.recipe}-stage"          # one stage dataset per recipe
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug.split("/")[1], "id": slug, "licenses": [{"name": "CC-BY-4.0"}]}))
    units = set(pd.read_parquet(HERE / "stage/test_frames.parquet").unit)
    of = json.loads(Path(args.clip_datasets).read_text())
    guards = {"st2_118578_2nd_f015000", "st2_118578_2nd_f037500"}
    missing = sorted(u for u in units if u not in of and u not in guards)
    if missing:
        raise SystemExit(f"no dataset for {missing}")
    clips = sorted({of[u] for u in units if u in of} | set(GUARD_DS))
    probe = (HERE / "stage_stream/files.txt").read_text().split()[0]
    (kd / "kernel.py").write_text((HERE / "kernel_template.py").read_text()
                                  .replace("__STEPS__", str(args.steps))
                                  .replace("__RECIPE__", json.dumps(args.recipe))
                                  .replace("__PROBE__", json.dumps(probe)))
    (kd / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/veo-ball-{args.recipe}", "title": f"veo-ball-{args.recipe}", "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [f"{OWNER}/veo-code", f"{OWNER}/veo-ball-v4-stream", slug, *clips],
        "competition_sources": [], "kernel_sources": []}, indent=1))
    print("datasets:", clips)
    if args.dry:
        return
    exists = subprocess.run(["kaggle", "datasets", "status", slug], capture_output=True, text=True)
    if exists.returncode == 0 and "ready" in exists.stdout:
        subprocess.run(["kaggle", "datasets", "version", "-p", str(ds), "-m", "v4 stage"], check=True)
    else:
        subprocess.run(["kaggle", "datasets", "create", "-p", str(ds)], check=True)
    print(f"stage dataset sent; once `kaggle datasets status {slug}` reads ready:\n"
          f"  kaggle kernels push -p {kd}")


if __name__ == "__main__":
    main()
