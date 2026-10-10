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

    out_dir = Path(out_dir)
    cache = out_dir / CANDIDATES_FILE
    if cache.exists() and not overwrite:
        return cache
    width, fps = int(info["width"]), float(info["fps"])
    model, imgsz = _model(Path(weights), width, int(info["height"]))
    cap = cv2.VideoCapture(info["path"])
    rows, index = [], -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        result = model(frame, verbose=False, conf=conf, imgsz=imgsz)[0]
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


def _model(weights: Path, width: int, height: int):
    """The detector and the input size to run it at.

    An OpenVINO export beside the weights (`yolo export format=openvino
    imgsz=736,1280`) runs 1.9 times faster on CPU and, measured on 60
    frames, returns the same boxes and confidences to the last digit. It is
    exported for one input shape, so it is used only for footage of that
    shape; anything else runs the PyTorch weights at the footage's width.
    """
    from ultralytics import YOLO

    shape = (-(-height // 32) * 32, width)
    export = weights.parent / f"{weights.stem}_openvino_model"
    meta = export / "metadata.yaml"
    if meta.exists():
        import yaml

        if tuple(yaml.safe_load(meta.read_text()).get("imgsz", ())) == shape:
            return YOLO(str(export), task="detect"), shape
    return YOLO(str(weights)), width


# In `union`, a fine-tuned candidate this close to a COCO one on the same
# frame is the same object and is not added twice.
SAME_PX = 6.0


# In `fill`, COCO candidates are used only on frames where the fine-tuned
# detector has none at this confidence or above.
FILL_BELOW = 0.25


def with_candidates(tracks: pd.DataFrame, out_dir: Path,
                    union: bool = False, fill: bool = False) -> pd.DataFrame:
    """The tracks with the fine-tuned detector's ball rows, if cached.

    By default they replace the COCO ball rows. With `union` both are kept
    and the ball path chooses among them: measured on six windows, each
    detector found labelled shots the other missed -- five between them,
    none by both.
    """
    import numpy as np

    cache = Path(out_dir) / CANDIDATES_FILE
    if not cache.exists():
        return tracks
    balls = pd.read_parquet(cache)
    keep = [c for c in tracks.columns if c in balls.columns]
    if fill:
        # Taken as equals, COCO's confident look-alikes pulled the path off
        # the ball: on stoke_7001 through its labelled shots it sat on the
        # clicked ball 29 times with the fine-tuned candidates alone and 0
        # times with both. So COCO only fills frames the fine-tuned
        # detector leaves empty.
        sure = set(balls.frame[balls.confidence >= FILL_BELOW].astype(int))
        coco = tracks[(tracks.cls == "ball") & ~tracks.frame.isin(sure)]
        players = tracks[tracks.cls != "ball"]
        return pd.concat([players, coco, balls[keep]], ignore_index=True)
    if not union:
        players = tracks[tracks.cls != "ball"]
        return pd.concat([players, balls[keep]], ignore_index=True)
    coco = tracks[tracks.cls == "ball"]
    near = {f: g[["px", "py"]].to_numpy() for f, g in coco.groupby("frame")}
    new = []
    for row in balls.itertuples(index=False):
        have = near.get(row.frame)
        if have is not None and len(have) and np.min(np.hypot(
                have[:, 0] - row.px, have[:, 1] - row.py)) < SAME_PX:
            continue
        new.append(row)
    extra = pd.DataFrame(new, columns=balls.columns)
    return pd.concat([tracks, extra[keep]], ignore_index=True)
