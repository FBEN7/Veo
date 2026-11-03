"""Push plan_v4's step 2 (addendum 2), the CPU kernel track_kernel.py: the
code under test (src/ constants.py, player_filter.py,
team_assignment_v2.py, with their sha256) goes up as veo-nearball-track-stage;
the kernel reads the GPU pass's output (kernel veo-nearball-detall), the
supervision wheel from veo-code and the videos and clip files from the
shot-clip datasets (M7 / M9: veo-st2-dev2, opened clips only).

    VEO_UNSEAL=... KAGGLE_API_TOKEN=... python push_track.py stage|kernel [--dry]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
OWNER = "francoisbernard"
SEALED = {"128057", "132831"}
CLIP_DS = ["veo-st2-dev2", *[f"veo-st2-m5-b{i}" for i in range(4)], *[f"veo-st2-train-b{i}" for i in range(10)]]
CODE = ["src/__init__.py", "src/constants.py", "src/player_filter.py", "src/team_assignment_v2.py"]


def main():
    from src.unseal import unsealed

    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stage", "kernel"])
    ap.add_argument("--work", default=str(HERE / "push_track"))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    work = Path(args.work)
    slug = f"{OWNER}/veo-nearball-track-stage"
    if args.what == "stage":
        ds = work / "stage_ds"
        shutil.rmtree(ds, ignore_errors=True)
        ds.mkdir(parents=True)
        sha = {}
        for f in CODE[1:]:                      # flat; the kernel rebuilds src/
            shutil.copy(ROOT / f, ds / Path(f).name)
            sha[f] = hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
        (ds / "track_stage.json").write_text(json.dumps({"plan": "near_ball plan_v4 + addenda 1, 2", "sha256": sha}, indent=1))
        (ds / "dataset-metadata.json").write_text(json.dumps(
            {"title": "veo-nearball-track-stage", "id": slug, "licenses": [{"name": "other"}]}))
        (HERE / "track_stage_sha256.json").write_text(json.dumps(sha, indent=1))
        print(sha)
        if not args.dry:
            subprocess.run(["kaggle", "datasets", "create", "-p", str(ds), ], check=True)
        return
    for m in sorted(SEALED):
        if not unsealed(m, "near_ball plan_v4 step 2: tracker variants on the opened M7 / M9 shot clips (Kaggle CPU)"):
            raise SystemExit("VEO_UNSEAL needed")
    kd = work / "kernel"
    shutil.rmtree(kd, ignore_errors=True)
    kd.mkdir(parents=True)
    shutil.copy(HERE / "track_kernel.py", kd / "kernel.py")
    (kd / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/veo-nearball-track", "title": "veo-nearball-track", "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [f"{OWNER}/veo-code", slug, *[f"{OWNER}/{d}" for d in CLIP_DS]],
        "competition_sources": [], "kernel_sources": [f"{OWNER}/veo-nearball-detall"]}, indent=1))
    if not args.dry:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(kd)], check=True)


if __name__ == "__main__":
    main()
