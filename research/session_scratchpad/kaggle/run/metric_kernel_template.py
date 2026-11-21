# Kaggle CPU kernel (no GPU): metric_step.py for units already detected on
# the GPU whose pitch tracks were not made there. Inputs: veo-code, the
# units' video datasets, and veo-metric-inputs (each unit's returned
# output_<w>/events_v1/ as a tree, and its v3_balls_<w>.parquet + sidecar). The repo goes to
# /home/user/Veo; each unit's metric_v3s_link.parquet and v3s_rows.parquet
# go to /kaggle/working/<w>.metric.tar.gz. Written by push_metric.py.
import glob, json, os, shutil, subprocess, sys, tarfile

UNITS = __UNITS__


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise SystemExit(f"{name} not in the inputs")
    return hits[0]


root = "/home/user/Veo"
shutil.copytree(os.path.dirname(find("eval_soccertrack_v2.py")), root, dirs_exist_ok=True)
for d in ("data/soccertrack_v2", ".cache/soccertrack_v2/calib"):
    os.makedirs(os.path.join(root, d), exist_ok=True)
for f in glob.glob("/kaggle/input/**/*", recursive=True):
    n = os.path.basename(f)
    if not os.path.isfile(f):
        continue
    if n.startswith("st2_") and n.endswith((".mp4", ".json", "_gt.parquet", "_events.csv")):
        dst = os.path.join(root, "data/soccertrack_v2", n)
        os.path.exists(dst) or os.symlink(f, dst)
    elif n.endswith(("_camera_intrinsics.npz", "_homography.npy")):
        shutil.copy(f, os.path.join(root, ".cache/soccertrack_v2/calib", n))
assets = os.path.dirname(find("ball_v3b.1500.pt"))
wheels = sorted(glob.glob(os.path.join(assets, "*.whl")))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", *wheels], check=True)
os.environ["YOLO_OFFLINE"] = "1"
for w in UNITS:
    run = [p for p in glob.glob(f"/kaggle/input/**/output_{w}", recursive=True) if os.path.isdir(p)][0]
    shutil.copytree(run, os.path.join(root, f"output_{w}"), dirs_exist_ok=True)
    cj = os.path.join(root, f"output_{w}", "events_v1", "clip.json")
    clip = json.load(open(cj))
    clip["path"] = os.path.join(root, "data/soccertrack_v2", f"{w}.mp4")   # where the video is here
    json.dump(clip, open(cj, "w"), indent=2)
    os.makedirs(os.path.join(root, ".cache"), exist_ok=True)
    for n in (f"v3_balls_{w}.parquet", f"v3_balls_{w}.parquet.meta.json"):
        shutil.copy(find(n), os.path.join(root, ".cache", n))
    r = subprocess.run([sys.executable, os.path.join(root, "research/session_scratchpad/kaggle/metric_step.py"), w],
                       cwd=root, env={**os.environ, "PYTHONPATH": root})
    out = os.path.join(root, f"output_{w}")
    with tarfile.open(f"/kaggle/working/{w}.metric.tar.gz", "w:gz") as t:
        for n in ("metric_v3s_link.parquet", "v3s_rows.parquet"):
            if os.path.exists(os.path.join(out, n)):
                t.add(os.path.join(out, n), arcname=f"output_{w}/{n}")
    print(w, "ok" if r.returncode == 0 else "FAILED", flush=True)
