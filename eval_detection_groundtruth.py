#!/usr/bin/env python3
"""Measure detection accuracy against human-labelled boxes.

Detection in this project was checked for plausibility long before it was
checked for accuracy -- counts per frame, positions on the pitch, speeds that
look physical. None of that is accuracy, and the gap mattered: "100% ball
detection" was repeated for weeks on the strength of coverage counts, and
turned out to be an artefact of a cap on candidates per frame.

Source: Roboflow football-players-detection (CC BY 4.0), 663 human-labelled
1920x1080 images, 114 of them from 08fd33 -- the same match as the broadcast
validation clip. Our detector is COCO yolov8m, which has a single `person`
class, so goalkeeper, player and referee are merged for the comparison. A
detection counts as correct when it overlaps a ground-truth box of the same
class by at least IOU_THRESHOLD, matched one-to-one.

Inference runs once per image at the lowest confidence of interest; every
threshold above it is then a filter over the cached boxes. Re-running the model
per threshold costs ten times as much and measures exactly the same thing.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.constants import PERSON_CLASS_ID, BALL_CLASS_ID

DATA = Path("/tmp/fa-cv/training/football-players-detection-1/"
            "football-players-detection-1")
IOU_THRESHOLD = 0.30
MODEL = "yolov8m.pt"

# Roboflow class ids -> our two categories.
GT_BALL = {0}
GT_PERSON = {1, 2, 3}          # goalkeeper, player, referee

CONF_GRID = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def load_labels(path: Path, width: int, height: int):
    """YOLO-format labels -> (class, x1, y1, x2, y2) in pixels."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:5])
        out.append((cls,
                    (cx - w / 2) * width, (cy - h / 2) * height,
                    (cx + w / 2) * width, (cy + h / 2) * height))
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def match(preds, truths):
    """Greedy one-to-one matching by descending IoU."""
    pairs = sorted(
        ((iou(p, t), pi, ti)
         for pi, p in enumerate(preds) for ti, t in enumerate(truths)),
        key=lambda x: x[0], reverse=True,
    )
    used_p, used_t, tp = set(), set(), 0
    for score, pi, ti in pairs:
        if score < IOU_THRESHOLD:
            break
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        tp += 1
    return tp


def centre_errors(preds, truths):
    """Distance from each truth box centre to the nearest prediction centre.

    IoU on a 12-pixel box is unforgiving, so a low ball recall could in
    principle be a scoring artefact rather than a miss. Centre distance
    separates the two: if the misses were near-hits, loosening the criterion
    would recover them.
    """
    if not truths:
        return []
    out = []
    pc = [((p[0] + p[2]) / 2, (p[1] + p[3]) / 2) for p in preds]
    for t in truths:
        tc = ((t[0] + t[2]) / 2, (t[1] + t[3]) / 2)
        if not pc:
            out.append(float("inf"))
            continue
        out.append(min(float(np.hypot(c[0] - tc[0], c[1] - tc[1])) for c in pc))
    return out


def cache_detections(model, images, imgsz, lowest_conf):
    """Run the model once per image; keep every box with its score."""
    cached = []
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        truths = load_labels(
            path.parent.parent / "labels" / (path.stem + ".txt"), w, h)

        res = model(frame, verbose=False, imgsz=imgsz, conf=lowest_conf,
                    classes=[PERSON_CLASS_ID, BALL_CLASS_ID])[0]
        cls = res.boxes.cls.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        xyxy = res.boxes.xyxy.cpu().numpy()

        dets = {"person": [], "ball": []}
        for box, c, cf in zip(xyxy, cls, conf):
            key = "person" if c == PERSON_CLASS_ID else "ball"
            dets[key].append((float(cf), *(float(v) for v in box)))

        cached.append({
            "dets": dets,
            "person": [t[1:] for t in truths if t[0] in GT_PERSON],
            "ball": [t[1:] for t in truths if t[0] in GT_BALL],
        })
    return cached


def score(cached, conf_person, conf_ball):
    stats = {k: dict(tp=0, pred=0, gt=0) for k in ("person", "ball")}
    for item in cached:
        for name, thr in (("person", conf_person), ("ball", conf_ball)):
            preds = [d[1:] for d in item["dets"][name] if d[0] >= thr]
            gts = item[name]
            s = stats[name]
            s["pred"] += len(preds)
            s["gt"] += len(gts)
            s["tp"] += match(preds, gts)
    return stats


def prf(s):
    p = s["tp"] / s["pred"] if s["pred"] else 0.0
    r = s["tp"] / s["gt"] if s["gt"] else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def report(title, stats, n_images):
    print(f"\n{title}  ({n_images} images)")
    print(f"  {'class':<8} {'truth':>6} {'pred':>6} {'TP':>5} "
          f"{'precision':>10} {'recall':>8} {'F1':>7}")
    for name in ("person", "ball"):
        s = stats[name]
        p, r, f = prf(s)
        print(f"  {name:<8} {s['gt']:>6} {s['pred']:>6} {s['tp']:>5} "
              f"{p:>10.2f} {r:>8.2f} {f:>7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf-person", type=float, default=0.25)
    ap.add_argument("--conf-ball", type=float, default=0.10)
    ap.add_argument("--out", default="detection_accuracy.json")
    args = ap.parse_args()

    print("=" * 74)
    print(f"DETECTION vs HUMAN-LABELLED BOXES  (IoU >= {IOU_THRESHOLD})")
    print("=" * 74)
    print("Roboflow football-players-detection, CC BY 4.0")
    print("goalkeeper/player/referee merged: COCO yolov8m has one person class")

    model = YOLO(MODEL)
    results = {}

    splits = {
        "valid": sorted((DATA / "valid" / "images").glob("*.jpg")),
        "test": sorted((DATA / "test" / "images").glob("*.jpg")),
        "our match (08fd33, train)": [
            p for p in sorted((DATA / "train" / "images").glob("*.jpg"))
            if p.name.startswith("08fd33")
        ],
    }

    cached_by_split = {}
    for name, images in splits.items():
        if not images:
            continue
        cached = cache_detections(model, images, args.imgsz, min(CONF_GRID))
        cached_by_split[name] = cached
        stats = score(cached, args.conf_person, args.conf_ball)
        report(f"{name}  conf_person={args.conf_person} "
               f"conf_ball={args.conf_ball} imgsz={args.imgsz}",
               stats, len(cached))
        results[name] = {k: dict(v, **dict(zip(
            ("precision", "recall", "f1"), prf(v)))) for k, v in stats.items()}

    # Are the ball misses near-hits that IoU is being harsh about, or genuine
    # absences? If loosening the criterion recovers nothing, they are absences.
    print("\n" + "-" * 74)
    print("ball misses: IoU artefact, or genuine absence?")
    print("-" * 74)
    for name, cached in cached_by_split.items():
        errs = []
        for item in cached:
            preds = [d[1:] for d in item["dets"]["ball"] if d[0] >= args.conf_ball]
            errs += centre_errors(preds, item["ball"])
        if not errs:
            continue
        errs = np.array(errs)
        finite = errs[np.isfinite(errs)]
        line = f"  {name:<28}"
        for tol in (10, 20, 40):
            line += f"  within {tol:>3}px: {float((errs <= tol).mean()):.2f}"
        if len(finite):
            near = finite[finite <= 40]
            line += (f"   median when found: "
                     f"{float(np.median(near)) if len(near) else float('nan'):.1f}px")
        print(line)

    print("\n" + "-" * 74)
    print("confidence sensitivity (valid split)")
    print("-" * 74)
    if "valid" in cached_by_split:
        print(f"  {'conf':>6} {'person P':>9} {'person R':>9} "
              f"{'ball P':>8} {'ball R':>8}")
        for c in CONF_GRID:
            s = score(cached_by_split["valid"], c, c)
            pp, pr, _ = prf(s["person"])
            bp, br, _ = prf(s["ball"])
            print(f"  {c:>6.2f} {pp:>9.2f} {pr:>9.2f} {bp:>8.2f} {br:>8.2f}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
