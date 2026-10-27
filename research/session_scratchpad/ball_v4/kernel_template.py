# Kaggle GPU kernel for ball detector v4 (written by push_v4.py). Inputs:
# veo-code (code tree, wheels, v3b), veo-ball-v4-stream (v2's stream strips),
# veo-ball-v4-stage (ball_v4 scripts, prepare.py's stage, the click page's
# manifest) and the shot-clip datasets. No internet. Steps: cut the clicked
# strips (CPU, PARTS processes), train fold A on GPU 0 and fold B on GPU 1,
# then v3b and v4 on the test frames (one process per GPU). Results to
# /kaggle/working (checkpoints, provenance, detections, summary, logs).
import glob, os, shutil, subprocess, sys, time

STEPS = __STEPS__
STREAM_PROBE = __PROBE__          # one of the stream strips: its directory is the stream dataset
PARTS = 3
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        subprocess.run(["find", "/kaggle/input", "-maxdepth", "3"])
        raise SystemExit(f"{name} not in the inputs")
    return hits[0]


def run_all(cmds, name):
    """Run the commands side by side ((argv, env) pairs), logs to logs/; raise if one fails."""
    procs = []
    for i, (argv, env) in enumerate(cmds):
        f = open(f"/kaggle/working/logs/{name}_{i}.log", "w")
        procs.append((subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, cwd=root,
                                       env=dict(os.environ, **env)), f))
    codes = [p.wait() for p, _ in procs]
    for i, (_, f) in enumerate(procs):
        f.close()
        print(f"--- {name}_{i} (exit {codes[i]})", open(f.name).read()[-3000:], sep="\n", flush=True)
    if any(codes):
        raise SystemExit(f"{name} failed: {codes}")
    log(name, "done")


root = "/home/user/Veo"
shutil.copytree(os.path.dirname(find("eval_soccertrack_v2.py")), root, dirs_exist_ok=True)
assets = os.path.dirname(find("ball_v3b.1500.pt"))
os.makedirs(os.path.join(root, ".cache/soccertrack_v2/calib"), exist_ok=True)
for f in glob.glob("/kaggle/input/**/*", recursive=True):
    if os.path.isfile(f) and f.endswith(("_camera_intrinsics.npz", "_homography.npy")):
        shutil.copy(f, os.path.join(root, ".cache/soccertrack_v2/calib", os.path.basename(f)))
for n in ("ball_v3b.1500.pt", "ball_v3b.1500.pt.calib.json", "ball_v3b.1500.pt.provenance.json"):
    shutil.copy(os.path.join(assets, n), os.path.join(root, ".cache", n))
wheels = sorted(glob.glob(os.path.join(assets, "*.whl")))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", *wheels], check=True)
hub = os.path.expanduser("~/.cache/torch/hub/checkpoints")
os.makedirs(hub, exist_ok=True)
shutil.copy(os.path.join(assets, "resnet18-f37072fd.pth"), hub)
stage_in = os.path.dirname(find("test_frames.parquet"))          # the stage dataset (flat)
v4 = os.path.join(root, "research/session_scratchpad/ball_v4")
os.makedirs(os.path.join(v4, "stage"), exist_ok=True)
for f in glob.glob(os.path.join(stage_in, "*.py")):
    shutil.copy(f, v4)
for n in ("targets.parquet", "codes.npy", "test_frames.parquet"):
    shutil.copy(os.path.join(stage_in, n), os.path.join(v4, "stage", n))
os.makedirs(os.path.join(root, "research/session_scratchpad/ball_clicks"), exist_ok=True)
shutil.copy(os.path.join(stage_in, "manifest.json"),
            os.path.join(root, "research/session_scratchpad/ball_clicks/manifest.json"))
stream = os.path.dirname(find(STREAM_PROBE))
clips = sorted(set(os.path.dirname(f) for f in glob.glob("/kaggle/input/**/st2_*.mp4", recursive=True)))
os.makedirs("/kaggle/working/logs", exist_ok=True)
subprocess.run(["nvidia-smi", "-L"], check=True)
log("setup done;", len(clips), "clip dirs")

py, W = sys.executable, "/kaggle/working"
strips = "/kaggle/strips"
stage = os.path.join(v4, "stage")
run_all([([py, f"{v4}/cut_v4.py", "--stage", stage, "--clips", *clips, "--out", strips,
           "--part", str(i), "--parts", str(PARTS)], {}) for i in range(PARTS)], "cut")
subprocess.run([py, f"{v4}/cut_v4.py", "--stage", stage, "--out", strips, "--merge", "--stream", stream],
               check=True, cwd=root)
run_all([([py, f"{v4}/train_v4.py", "--fold", k, "--strips", strips, "--out", f"{W}/ball_v4{k}",
           "--steps", str(STEPS)], {"CUDA_VISIBLE_DEVICES": str(g), "OMP_NUM_THREADS": "2"})
         for g, k in enumerate("AB")], "train")
ckpts = [f"v3b={root}/.cache/ball_v3b.1500.pt"] + [
    f"v4{k}.{s}={W}/ball_v4{k}.{s}.pt" for k in "AB" for s in (STEPS // 2, STEPS)]
run_all([([py, f"{v4}/eval_v4.py", "--stage", stage, "--clips", *clips, "--ckpt", *ckpts,
           "--out", f"{W}/eval_part{g}.parquet", "--part", str(g), "--parts", "2"],
          {"CUDA_VISIBLE_DEVICES": str(g)}) for g in (0, 1)], "eval")
import pandas as pd  # noqa: E402
pd.concat([pd.read_parquet(f"{W}/eval_part{g}.parquet") for g in (0, 1)],
          ignore_index=True).to_parquet(f"{W}/eval.parquet")
subprocess.run([py, f"{v4}/eval_v4.py", "--stage", stage, "--score", f"{W}/eval.parquet"], check=True, cwd=root)
shutil.copy(f"{strips}/labels.parquet", f"{W}/strips_labels.parquet")
log("all done")
