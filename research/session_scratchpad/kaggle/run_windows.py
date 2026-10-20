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
<w>.json. A window that fails is logged and the next one runs.

    python run_windows.py --root /kaggle/working/veo W1 W2 ...
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


def ball_gpu(root: Path, w: str) -> dict:
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
    try:
        dbh.main()
    finally:
        bh.load_v2 = orig
    return {}


def run(root: Path, w: str, out: Path) -> dict:
    env = {**os.environ, "PYTHONPATH": str(root)}
    rec = {"window": w, "steps": {}}

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
    step("ball_v3", lambda: ball_gpu(root, w))
    with tarfile.open(out / f"{w}.tar.gz", "w:gz") as t:
        t.add(root / f"output_{w}", arcname=f"output_{w}")
        for f in (root / ".cache").glob(f"v3_balls_{w}.parquet*"):
            t.add(f, arcname=f".cache/{f.name}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("windows", nargs="+")
    args = ap.parse_args()
    import torch

    root, out = Path(args.root), Path.cwd()
    for w in args.windows:
        rec = {"window": w, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
        try:
            rec.update(run(root, w, out))
            rec["ok"] = True
        except Exception:
            rec["ok"] = False
            rec["error"] = traceback.format_exc()[-3000:]
            print(rec["error"], flush=True)
        (out / f"{w}.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
