"""Ball candidates from the detector fine-tuned on this footage.

`train_ball_detector.py` trains a one-class ball detector on hand-clicked
frames and tests it by match. This runs a trained one over a whole clip,
on every frame at the footage's own width -- the scale it was trained at
-- and stores its candidates next to the clip's tracks. `with_candidates`
then puts them in place of the COCO detector's ball rows, leaving the
players as they were.

The candidates are derived from footage under the data agreement and, like
every run output, are not committed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CANDIDATES_FILE = "ball_candidates.parquet"
# The lowest confidence kept; the ball path weighs confidence itself.
CONF = 0.05


def run_clip(out_dir: Path, info: dict, weights: Path, conf: float = CONF,
             overwrite: bool = False, verbose: bool = True) -> Path:
    """Detect the ball on every frame of a clip; returns the cache path."""
    import cv2
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    cache = out_dir / CANDIDATES_FILE
    if cache.exists() and not overwrite:
        return cache
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(info["path"])
    width, fps = int(info["width"]), float(info["fps"])
    rows, index = [], -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        result = model(frame, verbose=False, conf=conf, imgsz=width)[0]
        for (x0, y0, x1, y1), score in zip(result.boxes.xyxy.cpu().numpy(),
                                           result.boxes.conf.cpu().numpy()):
            rows.append({"frame": index, "time_s": index / fps,
                         "track_id": -1, "cls": "ball",
                         "px": float(x0 + x1) / 2.0,
                         "py": float(y0 + y1) / 2.0,
                         "crop_h": float(y1 - y0),
                         "detection_method": "ball_detector",
                         "confidence": float(score)})
    cap.release()
    table = pd.DataFrame(rows)
    table.attrs["weights"] = str(weights)
    table.to_parquet(cache)
    if verbose:
        print(f"  [ball detector] {len(rows)} candidates on {index + 1} "
              f"frames from {Path(weights).parent.parent.name}")
    return cache


def with_candidates(tracks: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """The tracks with the fine-tuned detector's ball rows, if cached."""
    cache = Path(out_dir) / CANDIDATES_FILE
    if not cache.exists():
        return tracks
    balls = pd.read_parquet(cache)
    players = tracks[tracks.cls != "ball"]
    keep = [c for c in tracks.columns if c in balls.columns]
    return pd.concat([players, balls[keep]], ignore_index=True)
