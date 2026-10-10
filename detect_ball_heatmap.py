"""Ball candidates from the panorama heatmap (`src/ball_heatmap.py`) on a
SoccerTrack v2 window, as the ball rows of the pipeline's
`tracks.parquet` (`detection_method` "heatmap", `confidence` its
probability, `crop_h` the ball's size there through the calibration).

    python detect_ball_heatmap.py --window W --weights CKPT --tau 0.2 \\
        --stride 2 --out balls.parquet
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
import pandas as pd

from src import soccertrack_v2 as st
from src.ball_heatmap import build, detect
from src.paths import DATA_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    info = json.loads((DATA_DIR / "soccertrack_v2" / f"{args.window}.json").read_text())
    cal = st.calibration(info["match"])
    model = build()
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{args.window}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    rows, idx, t0 = [], 0, time.time()
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if idx % args.stride == 0:
            for u, v, p, size in detect(model, image, info["pitch"], tau=args.tau,
                                        top_k=args.top_k, cal=cal):
                rows.append({"frame": idx, "time_s": idx / fps, "track_id": -1,
                             "cls": "ball", "px": u, "py": v, "crop_h": size,
                             "crop_w": float("nan"),
                             "detection_method": "heatmap", "confidence": p})
        if idx % 500 == 0:
            print(f"  frame {idx}: {len(rows)} candidates "
                  f"[{time.time() - t0:.0f} s]", flush=True)
        idx += 1
    cap.release()
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"  {len(rows)} candidates on {idx} frames -> {args.out}")


if __name__ == "__main__":
    main()
