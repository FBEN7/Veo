# Kaggle kernel entry point (`kaggle kernels push -p .` from this folder): unpack the private dataset
# (code tarball, one SoccerTrack v2 window, weights), install ultralytics,
# run bench.py. Outputs land in /kaggle/working.
import glob, os, shutil, subprocess, sys, tarfile

def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        subprocess.run(["find", "/kaggle/input", "-maxdepth", "4"])
        raise SystemExit(f"{name} not in the inputs")
    return hits[0]


# Kaggle extracts an uploaded archive: the code arrives as a tree.
src = os.path.dirname(find("ball_v3b.1500.pt"))
root = "/kaggle/working/veo"
code = find("detect_ball_heatmap.py")
if code.endswith(".tar.gz"):
    with tarfile.open(code) as t:
        t.extractall(root)
else:
    shutil.copytree(os.path.dirname(code), root, dirs_exist_ok=True)
for d in ("data/soccertrack_v2", ".cache/soccertrack_v2/calib"):
    os.makedirs(os.path.join(root, d), exist_ok=True)
for f in glob.glob(os.path.join(src, "*")):
    n = os.path.basename(f)
    if n.endswith((".mp4", "_gt.parquet")) or (n.startswith("st2_") and n.endswith(".json")):
        os.symlink(f, os.path.join(root, "data/soccertrack_v2", n))
    elif n.endswith((".npz", ".npy")):
        shutil.copy(f, os.path.join(root, ".cache/soccertrack_v2/calib", n))
    elif n.startswith("ball_v3b"):
        shutil.copy(f, os.path.join(root, ".cache", n))
    elif n == "yolov8s.pt":
        shutil.copy(f, root)
        shutil.copy(f, "/kaggle/working")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "supervision"], check=True)
subprocess.run(["nvidia-smi"])
subprocess.run([sys.executable, os.path.join(root, "research/session_scratchpad/kaggle/bench.py"),
                "--root", root], check=True, cwd="/kaggle/working")
