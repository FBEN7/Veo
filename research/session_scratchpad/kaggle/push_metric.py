"""Push the CPU-only metric kernel (run/metric_kernel_template.py) for units
detected on the GPU before metric_step.py ran there: stage their returned
events_v1 runs and v3 rows as a private dataset (directories kept), then push.

    KAGGLE_API_TOKEN=... python push_metric.py --videos DS1 DS2 ... --units W1 W2 ...
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--videos", nargs="+", required=True)
ap.add_argument("--units", nargs="+", required=True)
ap.add_argument("--stage", required=True)
ap.add_argument("--slug", default="veo-cpu-metric")
ap.add_argument("--inputs", default="veo-metric-inputs", help="the staged inputs' dataset name (a new one per batch)")
args = ap.parse_args()
st = Path(args.stage)
shutil.rmtree(st, ignore_errors=True)
(st / "data").mkdir(parents=True)
for w in args.units:
    g = ROOT / ".cache/gpu" / w
    shutil.copytree(g / f"output_{w}" / "events_v1", st / "data" / f"output_{w}" / "events_v1")
    for n in (f"v3_balls_{w}.parquet", f"v3_balls_{w}.parquet.meta.json"):
        shutil.copy(g / ".cache" / n, st / "data" / n)
(st / "data/dataset-metadata.json").write_text(json.dumps(
    {"title": args.inputs, "id": f"francoisbernard/{args.inputs}",
     "licenses": [{"name": "CC-BY-4.0"}]}))
subprocess.run(["kaggle", "datasets", "create", "-p", str(st / "data"), "--dir-mode", "tar"])
k = st / "kernel"
k.mkdir()
(k / "kernel.py").write_text((HERE / "run/metric_kernel_template.py").read_text()
                             .replace("__UNITS__", json.dumps(args.units)))
(k / "kernel-metadata.json").write_text(json.dumps({
    "id": f"francoisbernard/{args.slug}", "title": args.slug, "code_file": "kernel.py",
    "language": "python", "kernel_type": "script", "is_private": True,
    "enable_gpu": False, "enable_internet": False,
    "dataset_sources": ["francoisbernard/veo-code", f"francoisbernard/{args.inputs}", *args.videos],
    "competition_sources": [], "kernel_sources": []}, indent=1))
print("staged; push after the dataset is ready:", k)
