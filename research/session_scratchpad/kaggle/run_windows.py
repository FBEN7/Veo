"""The window pipeline on the Kaggle GPU (backend_note.json): for each window,
the same steps as research/session_scratchpad/ball/events_m1/driver.sh, minus
the CPU-only ones that run back here from the returned files (marks.py, v2):

1. players: eval_soccertrack_v2.py --run --model yolov8s.pt --stride 2
   --pitch-margin 50 -> output_<w>/ (ultralytics picks the GPU; fp32);
2. teams and events_v1: research/session_scratchpad/events_run.py <w> events_v1;
3. ball: detect_ball_heatmap.py --v2 with ball_v3b.1500.pt at stride 1 over
   the whole window -> .cache/v3_balls_<w>.parquet (+ sidecar), the model on
   cuda through a wrapped `ball_heatmap.load_v2` (repo code unchanged; one
   run instead of dense_a2.py's chunks, which give the same rows).

Each window's files go to /kaggle/working/<w>.tar.gz with timings in
<w>.json. A window that fails is logged, what it finished is packed, and the
next one runs. `--allow-seen` passes the flag to the ball step, for windows
whose CPU v3 run had it.

    python run_windows.py --root /home/user/Veo [--allow-seen] W1 W2 ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path


def ball_gpu(root: Path, w: str, allow_seen: bool) -> dict:
    """detect_ball_heatmap.py --v2 on the whole window, the model on cuda."""
    import torch

    sys.path.insert(0, str(root))
    import detect_ball_heatmap as dbh
    from src import ball_heatmap as bh

    orig = bh.load_v2

    def load_on(ckpt):
        model, info = orig(ckpt)
        model.to("cuda")
        fwd = model.forward
        model.forward = lambda x: fwd(x.to("cuda", non_blocking=True)).float().cpu()
        return model, info

    bh.load_v2 = load_on
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    sys.argv = ["detect_ball_heatmap.py", "--v2", "--window", w,
                "--weights", ".cache/ball_v3b.1500.pt", "--stride", "1",
                "--threads", "4", "--out", f".cache/v3_balls_{w}.parquet"]
    if allow_seen:
        sys.argv.append("--allow-seen")
    try:
        dbh.main()
    finally:
        bh.load_v2 = orig
    return {}


def pack(root: Path, w: str, out: Path):
    with tarfile.open(out / f"{w}.tar.gz", "w:gz") as t:
        if (root / f"output_{w}").exists():
            t.add(root / f"output_{w}", arcname=f"output_{w}")
        for f in (root / ".cache").glob(f"v3_balls_{w}.parquet*"):
            t.add(f, arcname=f".cache/{f.name}")


def run(root: Path, w: str, rec: dict, allow_seen: bool):
    env = {**os.environ, "PYTHONPATH": str(root)}
    rec["steps"] = {}

    def step(name, fn):
        t = time.perf_counter()
        fn()
        rec["steps"][name] = round(time.perf_counter() - t, 1)
        print(f"{w}: {name} {rec['steps'][name]} s", flush=True)

    def sh(*cmd):
        subprocess.run([sys.executable, *cmd], cwd=root, env=env, check=True)

    step("players", lambda: sh("eval_soccertrack_v2.py", f"data/soccertrack_v2/{w}.mp4",
                               "--out", f"output_{w}", "--run", "--model", "yolov8s.pt",
                               "--stride", "2", "--pitch-margin", "50"))
    step("events_v1", lambda: sh("research/session_scratchpad/events_run.py", w, "events_v1"))
    os.chdir(root)
    step("ball_v3", lambda: ball_gpu(root, w, allow_seen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--allow-seen", action="store_true",
                    help="v3 on windows of a match it was trained on (as their CPU runs)")
    ap.add_argument("windows", nargs="+")
    args = ap.parse_args()
    import torch

    root, out = Path(args.root), Path.cwd()
    for w in args.windows:
        rec = {"window": w, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "allow_seen": args.allow_seen}
        try:
            run(root, w, rec, args.allow_seen)
            rec["ok"] = True
        except BaseException:          # detect_ball_heatmap refuses with SystemExit
            rec["ok"] = False
            rec["error"] = traceback.format_exc()[-3000:]
            print(rec["error"], flush=True)
        os.chdir(out)
        pack(root, w, out)             # what finished, even after a failure
        (out / f"{w}.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
