"""Stage and push the kit-colour kernel (kit_kernel.py, plan_v3.json): the
clips' player rows (even frames 300-1125 of .cache/gpu_shots3's metric) and
team_assignment_v2.py go up as veo-kit-stage; the CPU kernel veo-kit-teams
reads them with the shot-clip datasets (M7 / M9: veo-st2-dev2, the clips in
kaggle/opened.json, opened through src/unseal.py).

    VEO_UNSEAL=... KAGGLE_API_TOKEN=... python push_kit.py [--dry]
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
TREE = ROOT / ".cache/gpu_shots3"
SEALED = {"128057", "132831"}
CLIP_DS = ["veo-st2-dev2", *[f"veo-st2-m5-b{i}" for i in range(4)], *[f"veo-st2-train-b{i}" for i in range(10)]]
LO, HI = 300, 1125


def main():
    from src.unseal import unsealed

    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(HERE / "push"))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    for m in sorted(SEALED):
        if not unsealed(m, "team passes plan_v3: kit colours of the opened M7 / M9 shot clips (Kaggle)"):
            raise SystemExit("VEO_UNSEAL needed")
    units = sorted(p.name for p in TREE.iterdir() if p.name.startswith("st2_"))
    opened = set(json.loads((ROOT / "research/session_scratchpad/kaggle/opened.json").read_text())["clips"])
    shut = [u for u in units if u.split("_")[1] in SEALED and u not in opened]
    if shut:
        raise SystemExit(f"sealed clips not opened: {shut}")
    rows = []
    for w in units:
        m = pd.read_parquet(TREE / w / f"output_{w}" / "metric_v3s_link.parquet",
                            columns=["frame", "track_id", "cls", "px_raw", "py_raw", "crop_h"])
        m = m[(m.cls == "player") & (m.frame >= LO) & (m.frame <= HI) & (m.frame % 2 == 0)]
        rows.append(m.drop(columns="cls").assign(unit=w))
    players = pd.concat(rows, ignore_index=True)
    work = Path(args.work)
    ds, kd = work / "stage_ds", work / "kernel"
    shutil.rmtree(work, ignore_errors=True)
    ds.mkdir(parents=True)
    kd.mkdir(parents=True)
    players.to_parquet(ds / "players.parquet")
    shutil.copy(ROOT / "src/team_assignment_v2.py", ds / "team_assignment_v2.py")
    slug = f"{OWNER}/veo-kit-stage"
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": "veo-kit-stage", "id": slug, "licenses": [{"name": "CC-BY-4.0"}]}))
    shutil.copy(HERE / "kit_kernel.py", kd / "kernel.py")
    (kd / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/veo-kit-teams", "title": "veo-kit-teams", "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [slug, *[f"{OWNER}/{d}" for d in CLIP_DS]],
        "competition_sources": [], "kernel_sources": []}, indent=1))
    print(len(units), "units,", len(players), "player rows")
    if args.dry:
        return
    subprocess.run(["kaggle", "datasets", "create", "-p", str(ds)], check=True)
    print(f"once {slug} lists its files: kaggle kernels push -p {kd}")


if __name__ == "__main__":
    main()
