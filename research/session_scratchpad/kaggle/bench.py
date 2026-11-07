"""GPU benchmark on Kaggle: yolov8s players at full width (imgsz 4096) and
the v3 ball detector (`detect_ball_heatmap.py --v2`, ball_v3b.1500.pt) on
one SoccerTrack v2 window (M2, st2_117093_1st_f015000).

Runs inside the kernel (`kernel.py` unpacks the code and data first). The
ball model is moved to the GPU by wrapping `ball_heatmap.load_v2`, the repo
code is unchanged (`--allow-seen`: ball_v3b was trained on M2, and this
is a timing, not a score); a short CPU run of the same frames checks that both give
the same candidates. Writes bench.json and the parquets to the cwd.

    python bench.py --root /kaggle/working/veo --n-ball 500 --n-player 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def ball_run(root: Path, window: str, n: int, device: str, out: Path, threads: int):
    import torch

    sys.path.insert(0, str(root))
    os.chdir(root)
    import detect_ball_heatmap as dbh
    from src import ball_heatmap as bh

    forward = {"s": 0.0, "n": 0}
    orig = bh.load_v2

    def load_on(ckpt):
        model, info = orig(ckpt)
        model.to(device)
        fwd = model.forward

        def timed(x):
            t = time.perf_counter()
            y = fwd(x.to(device, non_blocking=True)).float().cpu()
            forward["s"] += time.perf_counter() - t
            forward["n"] += 1
            return y

        model.forward = timed
        return model, info

    bh.load_v2 = load_on
    sys.argv = ["detect_ball_heatmap.py", "--v2", "--window", window,
                "--weights", ".cache/ball_v3b.1500.pt", "--stride", "1",
                "--range", "0", str(n), "--allow-seen", "--threads", str(threads), "--out", str(out)]
    t0 = time.perf_counter()
    dbh.main()
    wall = time.perf_counter() - t0
    bh.load_v2 = orig
    if device == "cuda":
        torch.cuda.synchronize()
    return {"device": device, "frames": n, "wall_s": wall, "wall_s_per_frame": wall / n,
            "forward_s_per_frame": forward["s"] / max(forward["n"], 1),
            "forwards": forward["n"]}


def player_run(video: Path, n: int, half: bool):
    import cv2
    import torch
    from ultralytics import YOLO

    model = YOLO("yolov8s.pt")
    cap = cv2.VideoCapture(str(video))
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    for f in frames[:5]:                       # warm-up
        model(f, verbose=False, conf=0.1, imgsz=4096, classes=[0, 32], half=half)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    counts = []
    for f in frames:
        r = model(f, verbose=False, conf=0.1, imgsz=4096, classes=[0, 32], half=half)[0]
        counts.append(int((r.boxes.cls == 0).sum()))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    # Decode alone, for the share of a real run's time it takes.
    cap = cv2.VideoCapture(str(video))
    t1 = time.perf_counter()
    for _ in range(len(frames)):
        cap.read()
    dec = (time.perf_counter() - t1) / len(frames)
    return {"half": half, "frames": len(frames), "s_per_frame": dt / len(frames),
            "decode_s_per_frame": dec,
            "persons_mean": sum(counts) / len(counts)}


def compare(gpu: Path, cpu: Path) -> dict:
    import numpy as np
    import pandas as pd

    g, c = pd.read_parquet(gpu), pd.read_parquet(cpu)
    g = g[g.frame <= c.frame.max()] if len(c) else g.iloc[:0]
    out = {"rows_gpu": len(g), "rows_cpu": len(c)}
    if len(g) == len(c) and len(c):
        a = g.sort_values(["frame", "logit"]).reset_index(drop=True)
        b = c.sort_values(["frame", "logit"]).reset_index(drop=True)
        out["max_abs_logit_diff"] = float(np.abs(a.logit - b.logit).max())
        out["max_px_diff"] = float(np.hypot(a.px - b.px, a.py - b.py).max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--window", default="st2_117093_1st_f015000")
    ap.add_argument("--n-ball", type=int, default=500)
    ap.add_argument("--n-cpu", type=int, default=40)
    ap.add_argument("--n-player", type=int, default=300)
    args = ap.parse_args()
    import torch

    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    root, here = Path(args.root), Path.cwd()
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "cpus": os.cpu_count(), "window": args.window}
    video = root / "data" / "soccertrack_v2" / f"{args.window}.mp4"
    res["players"] = [player_run(video, args.n_player, half) for half in (False, True)]
    print(json.dumps(res["players"]), flush=True)
    res["ball_gpu"] = ball_run(root, args.window, args.n_ball, "cuda",
                               here / "balls_gpu.parquet", threads=4)
    print(json.dumps(res["ball_gpu"]), flush=True)
    res["ball_cpu"] = ball_run(root, args.window, args.n_cpu, "cpu",
                               here / "balls_cpu.parquet", threads=4)
    res["gpu_vs_cpu"] = compare(here / "balls_gpu.parquet", here / "balls_cpu.parquet")
    (here / "bench.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
