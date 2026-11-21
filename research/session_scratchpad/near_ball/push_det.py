"""Stage and push the detector-variants kernel (det_kernel.py, plan_v2.json):
the action frames (actions.parquet, each at its even frame) and the three
YOLO weights go up as veo-nearball-stage; the GPU kernel veo-nearball-det
reads them with veo-code (ultralytics wheels) and the shot-clip datasets
(M7 / M9: veo-st2-dev2, opened clips only, through src/unseal.py).

    VEO_UNSEAL=... KAGGLE_API_TOKEN=... python push_det.py --weights DIR [--dry]
"""
import argparse
import hashlib
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
    ap.add_argument("--weights", required=True, help="folder with yolov8m.pt and yolov8x.pt")
    ap.add_argument("--work", default=str(HERE / "push"))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    for m in sorted(SEALED):
        if not unsealed(m, "near_ball plan_v2: detector variants on the opened M7 / M9 shot clips (Kaggle)"):
            raise SystemExit("VEO_UNSEAL needed")
    a = pd.read_parquet(HERE / "actions.parquet")
    fr = a.assign(frame=a.frame - a.frame % 2)[["unit", "frame"]].drop_duplicates().sort_values(["unit", "frame"])
    opened = set(json.loads((ROOT / "research/session_scratchpad/kaggle/opened.json").read_text())["clips"])
    shut = sorted(u for u in set(fr.unit) if u.split("_")[1] in SEALED and u not in opened)
    if shut:
        raise SystemExit(f"sealed clips not opened: {shut}")
    work = Path(args.work)
    ds, kd = work / "stage_ds", work / "kernel"
    shutil.rmtree(work, ignore_errors=True)
    ds.mkdir(parents=True)
    kd.mkdir(parents=True)
    fr.to_parquet(ds / "frames.parquet")
    sha = {}
    for src in (ROOT / "yolov8s.pt", Path(args.weights) / "yolov8m.pt", Path(args.weights) / "yolov8x.pt"):
        shutil.copy(src, ds / src.name)
        sha[src.name] = hashlib.sha256(src.read_bytes()).hexdigest()
    (ds / "weights_sha256.json").write_text(json.dumps(sha, indent=1))
    slug = f"{OWNER}/veo-nearball-stage"
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": "veo-nearball-stage", "id": slug, "licenses": [{"name": "other"}]}))
    shutil.copy(HERE / "det_kernel.py", kd / "kernel.py")
    (kd / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/veo-nearball-det", "title": "veo-nearball-det", "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [f"{OWNER}/veo-code", slug, *[f"{OWNER}/{d}" for d in CLIP_DS]],
        "competition_sources": [], "kernel_sources": []}, indent=1))
    (HERE / "weights_sha256.json").write_text(json.dumps(sha, indent=1))
    print(fr.unit.nunique(), "units,", len(fr), "frames;", sha)
    if args.dry:
        return
    subprocess.run(["kaggle", "datasets", "create", "-p", str(ds)], check=True)
    print(f"once {slug} lists its files: kaggle kernels push -p {kd}")


if __name__ == "__main__":
    main()
