"""Ball candidates at full resolution, where the goal is in view.

The clips were run through the detector at 640 or 960 pixels wide while the
footage is 1280, so a ball a few pixels across arrived at half that or less.
Traced at the labelled shot moments, re-detecting at the footage's own
width found a ball candidate on 47 of 50 frames around stoke_4207's shot at
66 s where the stored detections had 21, and 30 where they had 12 around
stoke_1302's at 20 s (at the lowest confidence the clips use).

Running the whole clip at full width costs about a second a frame on this
machine, so it is spent where it pays: frames where the goal is in view,
which is where shots, goals and most balls going out happen. The extra
candidates join the stored ones and `ball_path` chooses among them all.

Results are cached per clip in `tracks_fullres.parquet` next to the stored
tracks, with the frames already scanned, so a second run only scans what is
new. Like every run output, it is derived from footage under the data
agreement and is not committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import BALL_CLASS_ID

FULLRES_FILE = "tracks_fullres.parquet"
SCANNED_FILE = "fullres_frames.json"

# A full-resolution candidate this close to a stored one, in pixels, is the
# same object seen twice and is not added again.
DUPLICATE_PX = 6.0


def goal_view_frames(goal_poses, grid: int, n_frames: int) -> list[int]:
    """Every frame within half a grid step of a solved goal pose."""
    half = grid // 2
    frames = set()
    for g in goal_poses:
        frames.update(range(max(0, int(g) - half),
                            min(n_frames, int(g) + half + 1)))
    return sorted(frames)


def densify(out_dir: Path, info: dict, frames, conf: float,
            model_name: str = "yolov8m.pt", verbose: bool = True) -> int:
    """Detect the ball at full width on `frames`; returns candidates added."""
    import cv2
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    cache, scanned_path = out_dir / FULLRES_FILE, out_dir / SCANNED_FILE
    have = pd.read_parquet(cache) if cache.exists() else None
    scanned = (set(json.loads(scanned_path.read_text()))
               if scanned_path.exists() else set())
    todo = [int(f) for f in frames if int(f) not in scanned]
    if not todo:
        return 0

    stored = pd.read_parquet(out_dir / "tracks.parquet")
    stored = stored[stored.cls == "ball"]
    by_frame = {f: g[["px", "py"]].to_numpy() for f, g in
                stored.groupby("frame")}

    model = YOLO(model_name)
    cap = cv2.VideoCapture(info["path"])
    width, fps = int(info["width"]), float(info["fps"])
    rows, last = [], -2
    for index in todo:
        if index != last + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        last = index
        if not ok:
            continue
        result = model(frame, verbose=False, conf=conf, imgsz=width,
                       classes=[BALL_CLASS_ID])[0]
        known = by_frame.get(index, np.empty((0, 2)))
        for (x0, y0, x1, y1), score in zip(result.boxes.xyxy.cpu().numpy(),
                                           result.boxes.conf.cpu().numpy()):
            px, py = float(x0 + x1) / 2.0, float(y0 + y1) / 2.0
            if len(known) and np.min(np.hypot(known[:, 0] - px,
                                              known[:, 1] - py)) < DUPLICATE_PX:
                continue
            rows.append({"frame": index, "time_s": index / fps,
                         "track_id": -1, "cls": "ball", "px": px, "py": py,
                         "crop_h": float(y1 - y0),
                         "detection_method": "yolo_fullres",
                         "confidence": float(score)})
    cap.release()

    added = pd.DataFrame(rows)
    merged = (added if have is None else
              pd.concat([have, added], ignore_index=True))
    if not merged.empty:
        merged.to_parquet(cache)
    scanned_path.write_text(json.dumps(sorted(scanned | set(todo))))
    if verbose:
        print(f"  [full res] scanned {len(todo)} goal-view frames at "
              f"{width} px, {len(rows)} new ball candidates")
    return len(rows)


def with_fullres(tracks: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """The stored tracks plus any cached full-resolution ball candidates."""
    cache = Path(out_dir) / FULLRES_FILE
    if not cache.exists():
        return tracks
    extra = pd.read_parquet(cache)
    return pd.concat([tracks, extra[[c for c in extra.columns
                                     if c in tracks.columns]]],
                     ignore_index=True)
