"""Find the goal's four corners, not a box, and test it on a match it never saw.

The goal detector learned a box drawn from the top of the left post to the
base of the right post (`goal_pose.BOX_FROM`, `BOX_TO`), which is what
`pose_from_box` reads. Its shape depends on where the camera stands: seen
from the side, as at Reading, that box is a thin strip; at Stoke it is
tall. Retrained with 92 more goals and tested by match
(`train_goal_detector2.py`), the detector trained on one ground found the
other's goals -- at Stoke with confidence 0.4-0.7 -- but drew the other
ground's box shape over them, overlapping the labels by about 27%: 18 of
102 Stoke goals and 2 of 78 Reading ones at IoU 0.5. A goal in perspective
is a slanted quadrilateral, and a box is the wrong thing to learn.

This learns the quadrilateral: a keypoint model (yolov8n-pose) with the
four corners as keypoints, in the clicked order -- left post base, left
post top, right post top, right post base -- and the box enclosing them,
which is the same shape from any side. A pose then comes from the corners
themselves (`goal_pose.pose_at`).

## Data

Every frame with clicked corners (25 from the first round, 92 from the
second) and every frame labelled as having no goal. A corner clicked at
the edge of the picture is where the goal is cut off, not the corner, and
is marked invisible so it is not learned. Frames with a goal but no
corners are left out: they cannot be a positive without corners, and must
not be a negative.

## Tested by match, and by clip

Stoke v Huddersfield and Reading v Fulham are each held out whole, or
(`--fold clips`) the three clips the current box detector was trained
without, which is the fair comparison for a camera fixed at one ground. On the
held-out match's corner-labelled frames: how often the goal is found, and
how far, in pixels at 1280 x 720, the corners land from the clicks. The
current box detector is scored on the two corners its box stands for --
left post top and right post base -- on the same frames. It saw frames of
every one of these clips in training, so the comparison flatters it.

    python train_goal_keypoints.py --round1-corners goal_corners_r1.json \\
        --round2 goal_labels.json --corners goal_corners.json \\
        --round1 goal_dataset.json --baseline <current best.pt>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from propagate_goal_labels import DIRS
from src.goal_pose import BOX_FROM, BOX_TO
from src.paths import CACHE_DIR
from train_goal_detector2 import MATCH

# A corner within this share of the frame of its edge was clicked where the
# goal is cut off.
EDGE = 0.004
# Padding round the corners' enclosing box, as a share of the frame.
PAD = 0.01
CONFS = (0.05, 0.15, 0.25)
# A detection is on the goal when its corners are this close, on average,
# to the clicked ones (pixels at 1280 x 720).
ON_GOAL_PX = 40.0


def visible(corner) -> bool:
    return (corner is not None and EDGE < corner[0] < 1 - EDGE
            and EDGE < corner[1] < 1 - EDGE)


def label_line(corners) -> str:
    """One YOLO-pose line: class, box, then x y visibility per corner."""
    pts = np.array([c for c in corners if c is not None])
    x0, y0 = np.clip(pts.min(axis=0) - PAD, 0, 1)
    x1, y1 = np.clip(pts.max(axis=0) + PAD, 0, 1)
    parts = [0, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    for c in corners:
        if visible(c):
            parts += [c[0], c[1], 2]
        else:
            parts += [0.0, 0.0, 0]
    return " ".join(f"{p:.6f}" if isinstance(p, float) else str(p)
                    for p in parts)


def rows_from(round1: Path, round1_corners: Path, round2: Path,
              corners: Path):
    """Frames with corners (positives) or no goal (negatives)."""
    rows = {}
    for f in json.loads(round1_corners.read_text())["frames"]:
        if f.get("corners") and sum(c is not None for c in f["corners"]) >= 2:
            rows[(f["clip"], int(f["frame"]))] = f["corners"]
    for f in json.loads(corners.read_text())["frames"]:
        if f.get("corners") and sum(c is not None for c in f["corners"]) >= 2:
            rows[(f["clip"], int(f["frame"]))] = f["corners"]
    for r in json.loads(round1.read_text())["rows"]:
        if r["box"] is None:
            rows.setdefault((r["clip"], int(r["frame"])), None)
    for r in json.loads(round2.read_text())["frames"]:
        if r.get("label") and r["label"].get("goal") is None:
            rows.setdefault((r["clip"], int(r["frame"])), None)
    return [{"clip": c, "frame": f, "corners": v}
            for (c, f), v in sorted(rows.items())]


def write_split(rows, root: Path, split: str) -> int:
    images, labels = root / "images" / split, root / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    written = 0
    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append(r)
    for clip, group in sorted(by_clip.items()):
        info = json.loads((Path(DIRS[clip]) / "clip.json").read_text())
        cap = cv2.VideoCapture(info["path"])
        for r in group:
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
            ok, frame = cap.read()
            if not ok:
                continue
            stem = f"{clip}_{r['frame']:06d}"
            cv2.imwrite(str(images / f"{stem}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            (labels / f"{stem}.txt").write_text(
                label_line(r["corners"]) if r["corners"] else "")
            written += 1
        cap.release()
    return written


def corner_error(pred, truth, w, h, which=(0, 1, 2, 3)):
    """Mean pixel distance over the clicked, visible corners in `which`."""
    d = [np.hypot((p[0] - t[0]) * w, (p[1] - t[1]) * h)
         for i, (p, t) in enumerate(zip(pred, truth))
         if i in which and p is not None and visible(t)]
    return float(np.mean(d)) if d else None


def evaluate(model, rows, kind: str, label: str):
    """Goals found, corner error, and boxes on goal-less frames."""
    results = []
    for r in rows:
        info = json.loads((Path(DIRS[r["clip"]]) / "clip.json").read_text())
        cap = cv2.VideoCapture(info["path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        h, w = frame.shape[:2]
        res = model.predict(frame, conf=min(CONFS), verbose=False)[0]
        dets = []
        for k in range(len(res.boxes)):
            score = float(res.boxes.conf[k])
            if kind == "keypoints":
                xy = res.keypoints.xy[k].cpu().numpy()
                pred = [[x / w, y / h] for x, y in xy]
                which = (0, 1, 2, 3)
            else:
                b = res.boxes.xyxy[k].cpu().numpy()
                pred = [None] * 4
                pred[BOX_FROM] = [b[0] / w, b[1] / h]
                pred[BOX_TO] = [b[2] / w, b[3] / h]
                which = (BOX_FROM, BOX_TO)
            err = (corner_error(pred, r["corners"], w, h, which)
                   if r["corners"] else None)
            dets.append((score, err))
        results.append((r["corners"] is not None, dets))
    goals = sum(1 for g, _ in results if g)
    empty = len(results) - goals
    for conf in CONFS:
        errs, false = [], 0
        for is_goal, dets in results:
            kept = [d for d in dets if d[0] >= conf]
            if not is_goal:
                false += bool(kept)
                continue
            on = [e for _, e in kept if e is not None and e <= ON_GOAL_PX]
            if on:
                errs.append(min(on))
        print(f"  {label:>26s}  conf {conf:.2f}: goal found {len(errs):3d}/"
              f"{goals}, corner error median "
              f"{np.median(errs) if errs else float('nan'):5.1f} px, a "
              f"goal on {false:3d}/{empty} goal-less frames", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round1", required=True, help="goal_dataset.json")
    ap.add_argument("--round1-corners", required=True,
                    help="goal_corners.json of the first corner round")
    ap.add_argument("--round2", required=True,
                    help="goal_labels.json from make_goal_labeller2.py")
    ap.add_argument("--corners", required=True,
                    help="goal_corners.json of the second corner round")
    ap.add_argument("--fold", choices=["stoke", "reading", "all", "clips",
                                       "clips2"],
                    default="all",
                    help="a match held out whole, both, or 'clips': the "
                         "three clips the current box detector was trained "
                         "without (same grounds, unseen moments)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--baseline", help="the current box detector")
    args = ap.parse_args()
    from ultralytics import YOLO

    rows = rows_from(Path(args.round1), Path(args.round1_corners),
                     Path(args.round2), Path(args.corners))
    print(f"  {len(rows)} frames: {sum(1 for r in rows if r['corners'])} "
          f"goals with corners, {sum(1 for r in rows if not r['corners'])} "
          f"without a goal", flush=True)
    from train_goal_detector import VAL_CLIPS

    for held_out in (["stoke", "reading"] if args.fold == "all"
                     else [args.fold]):
        if held_out == "clips2":
            # The other three clips, so that every clip has a model that
            # has not seen it.
            unseen = lambda r: (r["clip"] in MATCH and MATCH[r["clip"]] != "veo"
                                and r["clip"] not in VAL_CLIPS
                                and not r["clip"].startswith("soccernet"))
        elif held_out == "clips":
            # A Veo camera is fixed at one ground, so what matters is a
            # detector that has seen the ground but not the moment. The
            # current box detector held these three clips out of training,
            # which makes this comparison a fair one.
            unseen = lambda r: r["clip"] in VAL_CLIPS
        else:
            unseen = lambda r, m=held_out: MATCH[r["clip"]] == m
        train = [r for r in rows if not unseen(r)]
        test = [r for r in rows if unseen(r)]
        root = CACHE_DIR / "goal_keypoints" / f"without_{held_out}"
        data = root / "data"
        if data.exists():
            shutil.rmtree(data)
        fit = [r for i, r in enumerate(train) if i % 8]
        val = [r for i, r in enumerate(train) if not i % 8]
        write_split(fit, data, "train")
        write_split(val, data, "val")
        (data / "data.yaml").write_text(
            f"path: {data.resolve()}\ntrain: images/train\nval: images/val\n"
            f"kpt_shape: [4, 3]\nflip_idx: [3, 2, 1, 0]\n"
            f"names:\n  0: goal\n")
        print(f"\n  fold: test on {held_out} ({len(test)} frames, "
              f"{sum(1 for r in test if r['corners'])} goals); train on "
              f"{len(fit)} + {len(val)} validation", flush=True)
        if args.baseline:
            evaluate(YOLO(args.baseline), test, "box", "current box detector")
        model = YOLO("yolov8n-pose.pt")
        model.train(data=str(data / "data.yaml"), epochs=args.epochs,
                    imgsz=640, batch=8, device="cpu", workers=2,
                    project=str(root), name="run", exist_ok=True,
                    verbose=False, plots=False, seed=0)
        evaluate(YOLO(str(root / "run" / "weights" / "best.pt")), test,
                 "keypoints", "corners, held out")


if __name__ == "__main__":
    main()
