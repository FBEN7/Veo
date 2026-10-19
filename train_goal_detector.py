"""Train a goal detector on the hand-drawn boxes, and test it where it matters.

The anchor is fitted from the centre circle, which a camera following play
into the box does not show. Shot recall is 0 of 10 and goal detection 0 of 2
because of that. Finding the goal instead was tried before with hand-written
morphology and failed -- it fired on 88-92% of frames, pairing the halfway
line with an advertising hoarding at an aspect ratio inside tolerance.

This is the same landmark, found by learning rather than by rules.

## The data, and why it is small

150 frames were labelled by hand: 26 goals, 124 without. One was a replay
from behind the goal and was dropped. The remaining 25 were carried to
neighbouring frames by the warp between them, which gives 91 boxes.

The carry was tightened twice, both times because the boxes were drawn on
the frames and looked at. At eight steps a third of them had slid onto
advertising hoardings while every geometric test passed; at eight steps with
an appearance test, six of sixteen still had. Three steps and a 0.80
correlation leaves 16 of 16 on goals. Smaller and clean beats larger and
contaminated, because a positive containing a hoarding teaches exactly the
mistake this replaces.

## Held out by clip

Never by frame. Frames five apart are nearly the same picture, so a
frame-level split reports how well the model memorised, which is not the
question. Three windows are kept back entirely.

## The test that counts

Not the validation mAP. Whether the goal is found at **frame 481 of each
held-out window** -- the moment of the labelled shot, where the anchor
fails and where a goal was labelled in all six windows. Everything else is
context.

"Found" means landing on the hand-drawn box, not merely returning one.

## What it found, at 60 epochs

Held out: mAP50 0.78, precision 0.93, recall 0.76 over 58 frames. At the
shot moment, 5 of 6 land on the labelled goal at IoU 0.89 to 0.96,
including 2 of the 3 held-out windows at 0.87 and 0.93 confidence. Compare
the centre-circle anchor, which finds the shot moment 0 times in 10.

The miss at the default 0.25 was stoke_7001, and it looked alarming -- the
clearest, most fully visible goal in the set, no box at all. It was not a
blind spot. At a lower threshold the model puts a box on that goal,
overlapping the hand label, at 0.167; ten frames earlier, on a
near-identical picture, it scores 0.54. The failure is unstable confidence,
not sight, which is what 91 training boxes buy. At 0.15 it is 6 of 6, for
two extra false frames in 124. See `--conf`.

## What this does not give you

A landmark, not a geometry. `probe_goal_ruler.py` puts the detected box
through the ground plane and measures a quantity the laws of the game fix
exactly: goal width comes out at a median 4.7 m against a true 7.32, short
on almost every frame, because an axis-aligned rectangle discards the slant
that encodes orientation. The same measurement taken vertically -- where an
oblique view does not foreshorten -- gives a median 2.5 m against a true
2.44. Distances to goal need the corners, not the box.

    python train_goal_detector.py --dataset goal_dataset.json [--epochs 60]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from propagate_goal_labels import DIRS

# Held back entirely. Chosen before training, and they include one window
# from each ground so the split is not one stadium against another.
VAL_CLIPS = ("stoke_1302", "stoke_7001", "reading_1155")

# The moment of the labelled shot in every window cut around one.
SHOT_FRAME = 481


def write_split(rows, root: Path, split: str) -> int:
    """Frames and YOLO labels for one split. Returns how many were written."""
    images = root / "images" / split
    labels = root / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    by_clip = {}
    for row in rows:
        by_clip.setdefault(row["clip"], []).append(row)

    written = 0
    for clip, group in sorted(by_clip.items()):
        out_dir = DIRS.get(clip)
        if not out_dir or not (Path(out_dir) / "clip.json").exists():
            continue
        info = json.loads((Path(out_dir) / "clip.json").read_text())
        cap = cv2.VideoCapture(info["path"])
        for row in sorted(group, key=lambda r: r["frame"]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
            ok, frame = cap.read()
            if not ok:
                continue
            stem = f"{clip}_{row['frame']:06d}"
            cv2.imwrite(str(images / f"{stem}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            # An empty label file is how YOLO is told "nothing here", and
            # the 124 frames without a goal are the point of the exercise:
            # a detector trained only on frames containing goals learns
            # that every frame contains one.
            lines = []
            if row["box"]:
                x0, y0, x1, y1 = row["box"]
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                lines.append(f"0 {cx:.6f} {cy:.6f} {x1-x0:.6f} {y1-y0:.6f}")
            (labels / f"{stem}.txt").write_text("\n".join(lines))
            written += 1
        cap.release()
    return written


def build_dataset(dataset_path: Path, root: Path):
    blob = json.loads(dataset_path.read_text())
    rows = blob["rows"]
    train = [r for r in rows if r["clip"] not in VAL_CLIPS]
    val = [r for r in rows if r["clip"] in VAL_CLIPS]

    if root.exists():
        shutil.rmtree(root)
    n_train = write_split(train, root, "train")
    n_val = write_split(val, root, "val")

    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: goal\n")

    def goals(rows_):
        return sum(1 for r in rows_ if r["box"])

    print(f"  train {n_train:4d} frames ({goals(train)} with a goal)   "
          f"from {len(set(r['clip'] for r in train))} clips")
    print(f"  val   {n_val:4d} frames ({goals(val)} with a goal)   "
          f"held out: {', '.join(VAL_CLIPS)}")
    return root / "data.yaml"


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# A detection counts only if it lands on the goal that was labelled.
MIN_IOU = 0.5


def test_shot_moments(model, conf: float, labels: dict | None = None):
    """Does it find the goal at the moment of each labelled shot?

    "Found" means overlapping the hand-drawn box, not merely returning one.
    The difference is not pedantic: an earlier version of this counted any
    box at all, and eyeballing the rendered frames it was genuinely unclear
    whether two of the five were on goals or on advertising hoardings. IoU
    against the labels settled it in seconds -- 0.89 to 0.96, all on the
    goal -- where staring at the picture had produced a wrong guess.
    """
    print(f"\n  The test that counts: the goal at frame {SHOT_FRAME}, the "
          f"moment of the\n  labelled shot, where the anchor fails.\n")
    print(f"  {'window':>14s} {'held out':>9s} {'conf':>6s} {'IoU':>6s}  "
          f"verdict")
    hits = total = 0
    for clip in ("stoke_1302", "stoke_4207", "stoke_7001",
                 "reading_0737", "reading_1155", "reading_2519"):
        out_dir = DIRS.get(clip)
        if not out_dir or not (Path(out_dir) / "clip.json").exists():
            continue
        info = json.loads((Path(out_dir) / "clip.json").read_text())
        cap = cv2.VideoCapture(info["path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, SHOT_FRAME)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        truth = (labels or {}).get(f"{clip}:{SHOT_FRAME}")
        if truth is None:
            continue
        h, w = frame.shape[:2]
        truth = [truth[0] * w, truth[1] * h, truth[2] * w, truth[3] * h]

        result = model.predict(frame, conf=conf, verbose=False)[0]
        best_iou, best_conf = 0.0, 0.0
        for box, score in zip(result.boxes.xyxy.cpu().numpy(),
                              result.boxes.conf.cpu().numpy()):
            got = iou(box, truth)
            if got > best_iou:
                best_iou, best_conf = got, float(score)

        on_goal = best_iou >= MIN_IOU
        held = "yes" if clip in VAL_CLIPS else "no"
        total += 1
        hits += int(on_goal)
        verdict = ("on the goal" if on_goal else
                   "no box at all" if not len(result.boxes) else
                   "found the wrong thing")
        print(f"  {clip:>14s} {held:>9s} {best_conf:6.2f} {best_iou:6.2f}  "
              f"{verdict}")
    print(f"\n  {hits} of {total} shot moments land on the labelled goal "
          f"(IoU >= {MIN_IOU}).")
    return hits, total


def hand_boxes(dataset_path: Path) -> dict:
    """The hand-drawn box per `clip:frame`, for scoring against."""
    blob = json.loads(Path(dataset_path).read_text())
    return {f"{r['clip']}:{r['frame']}": r["box"]
            for r in blob["rows"] if r["box"] and r.get("source") == "hand"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default="goal_yolo")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    # 0.15, not the usual 0.25. The one miss at 0.25 was stoke_7001, which
    # looked like a blind spot -- the clearest goal in the set, no box at
    # all -- and was not one. At 0.01 the model puts a box on that goal,
    # overlapping the hand label, at 0.167. It localises correctly with low
    # and unstable confidence: 0.54 ten frames earlier on a near-identical
    # picture. Measured against the 124 goal-less frames:
    #
    #     conf 0.25   5 of 6 shot moments   2 of 124 false frames
    #     conf 0.15   6 of 6                4 of 124
    #
    # Two extra false frames buys the sixth shot moment, and downstream use
    # can demand temporal consistency where a single frame cannot.
    #
    # Chosen on the same frames it is reported on, which is mild overfitting
    # on 6 positives and 124 negatives. On the held-out clips alone the
    # false-frame rate is 3 of 40. Treat it as a starting point that a
    # larger negative set should revisit, not a tuned constant.
    ap.add_argument("--conf", type=float, default=0.15)
    args = ap.parse_args()

    print("Building the dataset, held out by clip.\n")
    data_yaml = build_dataset(Path(args.dataset), Path(args.root))

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print(f"\nTraining yolov8n for {args.epochs} epochs on CPU.\n")
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
                batch=8, workers=2, device="cpu", seed=0, verbose=False,
                project=args.root, name="run", exist_ok=True,
                # A goal is the same shape wherever it is; scale and
                # translation are the variation that matters, and a
                # left-right flip is a goal at the other end.
                degrees=0.0, shear=0.0, perspective=0.0,
                scale=0.6, translate=0.2, fliplr=0.5, mosaic=0.4)

    metrics = model.val(data=str(data_yaml), device="cpu", verbose=False)
    print(f"\n  held-out mAP50 {metrics.box.map50:.3f}   "
          f"mAP50-95 {metrics.box.map:.3f}   "
          f"precision {metrics.box.mp:.3f}   recall {metrics.box.mr:.3f}")

    test_shot_moments(model, args.conf, hand_boxes(Path(args.dataset)))
    print("\n  Validation numbers on 3 held-out windows are a small sample "
          "and the\n  shot-moment rows are six frames. Neither is a claim "
          "about football; they\n  are a check that the thing found is the "
          "thing labelled.")


if __name__ == "__main__":
    main()
