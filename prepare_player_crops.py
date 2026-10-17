#!/usr/bin/env python3
"""Turn a Roboflow football export into a small crop archive. Run locally.

This machine cannot reach roboflow.com -- the egress proxy refuses it by
organisation policy -- and a 663-image 1080p export is far larger than the
30 MB upload limit anyway. Neither matters, because the thing being trained
is a filter over crops, and crops are tiny.

Run this where the dataset is, and upload the `.npz` it writes. It is a few
megabytes.

    pip install roboflow ultralytics
    python prepare_player_crops.py --out player_crops.npz

By default it downloads the dataset itself; point `--dataset` at an existing
export to skip that.

## Why the negatives are mined rather than sampled

The obvious negative is a random box that overlaps no annotation. Those are
mostly grass, and a classifier that separates footballers from grass has
learned nothing useful -- the pipeline never asks it about grass, it asks it
about a person-shaped detection that might be a steward.

So the negatives are mined with the same detector the pipeline deploys: run
COCO yolov8m over the dataset's images, keep every `person` detection that
matches no annotated player, goalkeeper or referee, and call those negatives.
Roboflow's annotators labelled the participants and ignored everyone else, so
what is left is exactly the population that contaminates our roster --
touchline staff, substitutes, stewards, crowd, ballboys.

That also matches the deployment condition. The filter will see COCO person
detections, so it should be trained on COCO person detections.

## Licence

Roboflow football-players-detection is CC BY 4.0: usable commercially, unlike
SoccerNet. Attribution belongs wherever this ends up shipping.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Crop size the classifier sees. A footballer is roughly 1:2, and at this size
# a 15 px-tall detection still carries a recognisable shape.
CROP_W, CROP_H = 32, 64

# Roboflow class ids in football-players-detection: 0 ball, 1 goalkeeper,
# 2 player, 3 referee.
PARTICIPANT_CLASSES = {1, 2, 3}

# A COCO person detection counts as "already annotated" above this IOU, and is
# therefore a positive rather than a negative.
IOU_MATCH = 0.40

# Confidence for mining. Low, because the pipeline itself runs at 0.05-0.20
# and the negatives should include the marginal detections it will see.
MINE_CONF = 0.10

# Detections shorter than this are too small to classify and are skipped.
MIN_CROP_PX = 12


def load_labels(path: Path, w: int, h: int):
    """YOLO-format labels -> (class, x1, y1, x2, y2) in pixels."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().strip().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(p[0])
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append((cls, (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def crop_of(img, box):
    """A fixed-size BGR crop, or None if the box is too small."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < MIN_CROP_PX:
        return None
    return cv2.resize(img[y1:y2, x1:x2], (CROP_W, CROP_H),
                      interpolation=cv2.INTER_AREA)


def download(version: int) -> Path:
    from roboflow import Roboflow
    import os
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit(
            "Set ROBOFLOW_API_KEY. Get one free at roboflow.com; the dataset "
            "itself is CC BY 4.0.")
    rf = Roboflow(api_key=key)
    ds = (rf.workspace("roboflow-jvuqo")
            .project("football-players-detection-3zvbc")
            .version(version).download("yolov8"))
    return Path(ds.location)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="existing Roboflow yolov8 export")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--out", default="player_crops.npz")
    ap.add_argument("--model", default="yolov8m.pt")
    args = ap.parse_args()

    root = Path(args.dataset) if args.dataset else download(args.version)
    splits = [d for d in ("train", "valid", "test") if (root / d).is_dir()]
    if not splits:
        raise SystemExit(f"no train/valid/test directories under {root}")

    from ultralytics import YOLO
    model = YOLO(args.model)

    crops, labels, sources = [], [], []
    for split in splits:
        images = sorted((root / split / "images").glob("*.jpg"))
        print(f"{split}: {len(images)} images")
        for n, img_path in enumerate(images, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            gt = [b for b in load_labels(
                root / split / "labels" / (img_path.stem + ".txt"), w, h)
                if b[0] in PARTICIPANT_CLASSES]
            gt_boxes = [b[1:] for b in gt]

            for box in gt_boxes:
                c = crop_of(img, box)
                if c is not None:
                    crops.append(c)
                    labels.append(1)
                    sources.append(split)

            # Hard negatives: people the annotators deliberately left out.
            res = model.predict(str(img_path), conf=MINE_CONF, classes=[0],
                                verbose=False)[0]
            for xyxy in res.boxes.xyxy.cpu().numpy():
                if any(iou(xyxy, g) >= IOU_MATCH for g in gt_boxes):
                    continue
                c = crop_of(img, xyxy)
                if c is not None:
                    crops.append(c)
                    labels.append(0)
                    sources.append(split)

            if n % 50 == 0:
                print(f"  {n}/{len(images)}  {len(crops)} crops "
                      f"({sum(labels)} participants)")

    if not crops:
        raise SystemExit("no crops extracted -- check --dataset points at a "
                         "yolov8 export with images/ and labels/")

    arr = np.stack(crops).astype(np.uint8)
    lab = np.array(labels, dtype=np.uint8)
    np.savez_compressed(args.out, crops=arr, labels=lab,
                        sources=np.array(sources))
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"\nwrote {args.out}: {len(arr)} crops "
          f"({int(lab.sum())} participants, {int((1 - lab).sum())} others), "
          f"{size_mb:.1f} MB")
    if size_mb > 28:
        print("Larger than the upload limit. Re-run with fewer images, or "
              "split the archive.")


if __name__ == "__main__":
    main()
