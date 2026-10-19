"""Goal corners for clips nobody clicked, from the corner keypoint model.

The goal placer is calibrated once per clip from clicked corners
(`check_goal_corners.py`): with the centre circle they fix where the camera
stands. The windows cut around every SoccerNet-labelled shot
(`cut_shot_clips_more.py`) have no clicks, and asking for them would cap the
shot classifier's training data at what can be clicked. This finds the
corners with the keypoint model (`train_goal_keypoints.py`) on frames spread
over the clip and writes them in the clicked file's format, so the rest of
the calibration runs unchanged.

Only confident frames with all four corners inside the picture are kept,
and only of one goal: a window can show both ends, and the bundle that
fixes the camera assumes one. The two ends are told apart by which post
looks taller -- the one nearer the camera, which is the left post at one
end and the right post at the other -- and the end seen on more frames is
kept.

    python auto_goal_corners.py output_stoke_0102_1shots ... \\
        --weights .cache/goal_keypoints/without_clips/run/weights/best.pt \\
        --out goal_corners_auto.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.goal_placer import corners_from_keypoints

# Frames tried per clip, spread evenly, and kept at most.
TRIED = 60
KEPT = 20
MIN_GOAL_CONFIDENCE = 0.5


def near_post(corners) -> int:
    """0 if the left post looks taller, 1 if the right one does."""
    left = abs(corners[1][1] - corners[0][1])
    right = abs(corners[2][1] - corners[3][1])
    return 0 if left > right else 1


def clip_rows(out_dir: Path, model, verbose: bool = True):
    import cv2

    from check_goal_corners import looks_mirrored
    from src.replays import for_clip, in_replay

    info = json.loads((out_dir / "clip.json").read_text())
    width, height = int(info["width"]), int(info["height"])
    spans = for_clip(out_dir, info)
    cap = cv2.VideoCapture(info["path"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip = out_dir.name.replace("output_", "")
    found = []
    for frame in np.linspace(0, total - 1, TRIED).astype(int):
        if in_replay([int(frame)], spans)[0]:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = cap.read()
        if not ok:
            continue
        result = model.predict(image, verbose=False, conf=MIN_GOAL_CONFIDENCE,
                               imgsz=640)[0]
        corners = corners_from_keypoints(result, width, height)
        if corners is None or any(c is None for c in corners):
            continue
        norm = [[round(c[0] / width, 5), round(c[1] / height, 5)]
                for c in corners]
        if looks_mirrored(norm):
            continue
        found.append((int(frame), norm, near_post(corners)))
    cap.release()
    if not found:
        if verbose:
            print(f"  {clip}: no goal with four corners on {TRIED} frames")
        return []
    ends = [f[2] for f in found]
    end = max(set(ends), key=ends.count)
    kept = [f for f in found if f[2] == end]
    if len(kept) > KEPT:
        kept = [kept[i] for i in np.linspace(0, len(kept) - 1,
                                             KEPT).astype(int)]
    if verbose:
        print(f"  {clip}: four corners on {len(found)} frames, "
              f"{len(kept)} kept of the end seen on {ends.count(end)}")
    return [{"id": f"{clip}:{frame}", "clip": clip, "frame": frame,
             "corners": norm, "out_dir": str(out_dir), "source": "keypoints"}
            for frame, norm, _ in kept]


def add_clip(out_dir: Path, weights: Path, path: Path, model=None):
    """Find a clip's corners and add them to the corner file at `path`,
    unless it has some already."""
    blob = (json.loads(path.read_text()) if path.exists() else
            {"order": "left base, left top, right top, right base",
             "frames": []})
    clip = out_dir.name.replace("output_", "")
    if any(r["clip"] == clip for r in blob["frames"]):
        return
    if model is None:
        from ultralytics import YOLO

        model = YOLO(str(weights))
    rows = clip_rows(out_dir, model)
    if rows:
        info = json.loads((out_dir / "clip.json").read_text())
        blob["width"], blob["height"] = info["width"], info["height"]
    blob["frames"] += rows
    path.write_text(json.dumps(blob, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", nargs="+")
    ap.add_argument("--weights", required=True,
                    help="a corner model from train_goal_keypoints.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    for out_dir in map(Path, args.out_dirs):
        add_clip(out_dir, Path(args.weights), Path(args.out), model)


if __name__ == "__main__":
    main()
