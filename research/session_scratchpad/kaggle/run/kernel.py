# Kaggle kernel for the window pipeline on GPU (written by push_run.py from
# kernel_template.py; WINDOWS filled in there). Inputs: the private datasets
# veo-code (code tree, wheels, weights) and the batch's video dataset. The
# kernel has no internet. The repo goes to /home/user/Veo, the path its
# scripts expect; results (<w>.tar.gz, <w>.json) to /kaggle/working.
import glob, os, shutil, subprocess, sys

WINDOWS = ["st2_128057_1st_f004663", "st2_128057_1st_f026111", "st2_128057_1st_f030595", "st2_128057_1st_f033054", "st2_128057_1st_f046616", "st2_128057_1st_f047719", "st2_128057_1st_f050874", "st2_128057_1st_f057953", "st2_128057_1st_f059492", "st2_128057_1st_f061990", "st2_132831_2nd_f005307", "st2_132831_2nd_f007583", "st2_132831_2nd_f010419", "st2_132831_2nd_f012094", "st2_132831_2nd_f021348", "st2_132831_2nd_f028849", "st2_132831_2nd_f031521", "st2_132831_2nd_f036505", "st2_132831_2nd_f041118", "st2_132831_2nd_f045484", "st2_132831_2nd_f046495", "st2_132831_2nd_f050296", "st2_132831_2nd_f051396", "st2_132831_2nd_f058268", "st2_132831_2nd_f060296", "st2_132831_2nd_f063240", "st2_128057_1st_f030000"]


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        subprocess.run(["find", "/kaggle/input", "-maxdepth", "4"])
        raise SystemExit(f"{name} not in the inputs")
    return hits[0]


root = "/home/user/Veo"
code = os.path.dirname(find("eval_soccertrack_v2.py"))
shutil.copytree(code, root, dirs_exist_ok=True)
assets = os.path.dirname(find("ball_v3b.1500.pt"))
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
for n in ("ball_v3b.1500.pt", "ball_v3b.1500.pt.calib.json", "ball_v3b.1500.pt.provenance.json"):
    shutil.copy(os.path.join(assets, n), os.path.join(root, ".cache", n))
shutil.copy(os.path.join(assets, "yolov8s.pt"), root)
subprocess.run(["nvidia-smi", "-L"], check=True)
os.makedirs(os.path.join(root, ".cache"), exist_ok=True)
wheels = sorted(glob.glob(os.path.join(assets, "*.whl")))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", *wheels],
               check=True)
hub = os.path.expanduser("~/.cache/torch/hub/checkpoints")
os.makedirs(hub, exist_ok=True)
shutil.copy(os.path.join(assets, "resnet18-f37072fd.pth"), hub)
os.environ["YOLO_OFFLINE"] = "1"
subprocess.run([sys.executable, os.path.join(root, "research/session_scratchpad/kaggle/run_windows.py"),
                "--root", root, *WINDOWS], check=True, cwd="/kaggle/working")
