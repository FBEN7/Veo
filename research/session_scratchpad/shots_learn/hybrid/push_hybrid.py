"""Stage and push the hybrid ball's GPU kernel (kernel_template.py).

frames.parquet (from windows.py: unit, frame, model) and the two v4r
checkpoints (.cache/ball_v4/ball_v4r{A,B}.600.pt, with calibration and
provenance) go up as veo-hybrid-stage; the kernel veo-hybrid-ball reads
them with veo-code and the units' clip datasets (M7 / M9 clips: veo-st2-dev2,
opened through src/unseal.py, logged here).

    VEO_UNSEAL=... KAGGLE_API_TOKEN=... python push_hybrid.py --clip-datasets clip_dataset.json
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
SEALED_DS = f"{OWNER}/veo-st2-dev2"
CKPTS = ("v4rA.600", "v4rB.600")


def main():
    from src.unseal import unsealed

    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-datasets", required=True)
    ap.add_argument("--work", default=str(HERE / "push"))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    fr = pd.read_parquet(HERE / "frames.parquet")
    for m in sorted({u.split("_")[1] for u in fr.unit} & {"128057", "132831"}):
        if not unsealed(m, "shots hybrid: v4r ball rows on the opened M7 / M9 shot clips (Kaggle)"):
            raise SystemExit("VEO_UNSEAL needed")
    work = Path(args.work)
    ds, kd = work / "stage_ds", work / "kernel"
    shutil.rmtree(work, ignore_errors=True)
    ds.mkdir(parents=True)
    kd.mkdir(parents=True)
    shutil.copy(HERE / "frames.parquet", ds / "frames.parquet")
    for c in CKPTS:
        src = ROOT / f".cache/ball_v4/ball_{c}.pt"
        for suf in ("", ".calib.json", ".provenance.json"):
            shutil.copy(f"{src}{suf}", ds / f"{c}.pt{suf}")
    slug = f"{OWNER}/veo-hybrid-stage"
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": "veo-hybrid-stage", "id": slug, "licenses": [{"name": "CC-BY-4.0"}]}))
    of = json.loads(Path(args.clip_datasets).read_text())
    units = sorted(set(fr.unit))
    sealed = [u for u in units if u.split("_")[1] in {"128057", "132831"}]
    opened = set(json.loads((ROOT / "research/session_scratchpad/kaggle/opened.json").read_text())["clips"])
    if set(sealed) - opened:
        raise SystemExit(f"sealed clips not opened: {sorted(set(sealed) - opened)}")
    missing = [u for u in units if u not in of and u not in sealed]
    if missing:
        raise SystemExit(f"no dataset for {missing}")
    clips = sorted({of[u] for u in units if u in of} | ({SEALED_DS} if sealed else set()))
    (kd / "kernel.py").write_text((HERE / "kernel_template.py").read_text())
    (kd / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/veo-hybrid-ball", "title": "veo-hybrid-ball", "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [f"{OWNER}/veo-code", slug, *clips],
        "competition_sources": [], "kernel_sources": []}, indent=1))
    print(len(units), "units,", len(fr), "frames; datasets:", clips)
    if args.dry:
        return
    subprocess.run(["kaggle", "datasets", "create", "-p", str(ds)], check=True)
    print(f"once {slug} lists its files: kaggle kernels push -p {kd}")


if __name__ == "__main__":
    main()
