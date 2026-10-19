"""Retrain the goal detector with the clicked goal corners, tested by match.

Shots are missed for want of a camera pose, and poses come from the goal
detector (`train_goal_detector.py`), trained on 150 frames. On stoke_7001 it
scores the goal through both labelled shots at 0.09-0.25 -- under its 0.15
cut-off -- though a pose solved from each of those boxes fits it to under a
pixel. A second labelling round (`make_goal_labeller2.py`, then
`make_corner_labeller.py`) gave 162 more frames: 92 goals with their four
corners clicked, and 70 without a goal.

## The box, from the corners

The detector does not learn the goal's outline. It learns the box the
first round's labels drew, which turned out to run from the top of the
left post to the base of the right post (`goal_pose.BOX_FROM`, `BOX_TO`),
and `pose_from_box` reads its boxes that way. So each new goal's box is
cut from its corners the same way: left post top to right post base.

## Tested by match

Never by frame or clip. The first round's split held out three clips but
trained on the other three clips of the same two matches. Here each match
is held out whole -- Stoke City v Huddersfield (stoke_*, soccernet_w1-w3)
and Reading v Fulham (reading_*, soccernet_reading) -- and the detector
trained on the other, plus the Veo window, is scored on its labelled
frames: a goal is found if a box at the confidence lands on the labelled
box at IoU 0.5 or more; any box on a frame without a goal is false. The
current detector is scored on the same frames. It saw frames of every one
of these clips in training, so the comparison flatters it.

    python train_goal_detector2.py --round1 goal_dataset.json \\
        --round2 goal_labels.json --corners goal_corners.json --fold all

Weights are derived from footage under the data agreement and are not
committed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from src.goal_pose import BOX_FROM, BOX_TO
from src.paths import CACHE_DIR
from train_goal_detector import iou, write_split

MATCH = {"stoke_1302": "stoke", "stoke_4207": "stoke", "stoke_7001": "stoke",
         "soccernet_w1": "stoke", "soccernet_w2": "stoke",
         "soccernet_w3": "stoke",
         "reading_0737": "reading", "reading_1155": "reading",
         "reading_2519": "reading", "soccernet_reading": "reading",
         "veo": "veo"}
CONFS = (0.05, 0.15, 0.25)
MIN_IOU = 0.5


def box_from_corners(corners):
    """The training box: left post top to right post base, normalised."""
    a, b = corners[BOX_FROM], corners[BOX_TO]
    if a is None or b is None:
        return None
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]),
            max(a[1], b[1])]


def rows_from(round1: Path, round2: Path, corners: Path):
    """Every labelled frame: clip, frame, box or None, and its round."""
    rows = [{"clip": r["clip"], "frame": int(r["frame"]), "box": r["box"],
             "round": 1} for r in json.loads(round1.read_text())["rows"]]
    have = {(r["clip"], r["frame"]) for r in rows}
    cornered = {f["id"]: f for f in json.loads(corners.read_text())["frames"]
                if f.get("corners")}
    for r in json.loads(round2.read_text())["frames"]:
        key = (r["clip"], int(r["frame"]))
        if key in have or not r.get("label"):
            continue
        if r["label"].get("goal") is None:
            box = None
        elif r["id"] in cornered:
            box = box_from_corners(cornered[r["id"]]["corners"])
            if box is None:
                continue          # a corner it needs was not visible
        else:
            continue              # a goal without corners: no usable box
        rows.append({"clip": r["clip"], "frame": key[1], "box": box,
                     "round": 2})
    return rows


def evaluate(model, rows, label: str):
    """Goals found at IoU >= MIN_IOU, and boxes on goal-less frames."""
    from propagate_goal_labels import DIRS

    found = []
    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append(r)
    for clip, group in sorted(by_clip.items()):
        info = json.loads((Path(DIRS[clip]) / "clip.json").read_text())
        cap = cv2.VideoCapture(info["path"])
        for r in sorted(group, key=lambda r: r["frame"]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            res = model.predict(frame, conf=min(CONFS), verbose=False)[0]
            boxes = [(float(s), [b[0] / w, b[1] / h, b[2] / w, b[3] / h])
                     for b, s in zip(res.boxes.xyxy.cpu().numpy(),
                                     res.boxes.conf.cpu().numpy())]
            found.append((r["box"], boxes))
        cap.release()
    goals = sum(1 for truth, _ in found if truth)
    empty = len(found) - goals
    out = {}
    for conf in CONFS:
        hit = sum(1 for truth, boxes in found if truth and any(
            s >= conf and iou(b, truth) >= MIN_IOU for s, b in boxes))
        false = sum(1 for truth, boxes in found if not truth and any(
            s >= conf for s, _ in boxes))
        out[conf] = (hit, false)
        print(f"  {label:>24s}  conf {conf:.2f}: goal found {hit:3d}/{goals}"
              f", a box on {false:3d}/{empty} goal-less frames", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round1", required=True, help="goal_dataset.json")
    ap.add_argument("--round2", required=True,
                    help="goal_labels.json from make_goal_labeller2.py")
    ap.add_argument("--corners", required=True,
                    help="goal_corners.json for the round-2 goals")
    ap.add_argument("--fold", choices=["stoke", "reading", "all"],
                    default="all")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--baseline",
                    help="the current detector's weights, scored on the same "
                         "held-out frames")
    args = ap.parse_args()
    from ultralytics import YOLO

    rows = rows_from(Path(args.round1), Path(args.round2), Path(args.corners))
    for r in rows:
        if r["clip"] not in MATCH:
            raise SystemExit(f"no match known for {r['clip']}")
    print(f"  {len(rows)} labelled frames, {sum(1 for r in rows if r['box'])}"
          f" with a goal ({sum(1 for r in rows if r['round'] == 2)} from "
          f"round 2)", flush=True)
    folds = ["stoke", "reading"] if args.fold == "all" else [args.fold]
    for held_out in folds:
        train = [r for r in rows if MATCH[r["clip"]] != held_out]
        test = [r for r in rows if MATCH[r["clip"]] == held_out]
        root = CACHE_DIR / "goal_detector" / f"without_{held_out}"
        data = root / "data"
        if data.exists():
            shutil.rmtree(data)
        # Validation for early stopping is a slice of the training matches,
        # never the held-out one.
        val = [r for i, r in enumerate(train) if i % 8 == 0]
        fit = [r for i, r in enumerate(train) if i % 8 != 0]
        write_split(fit, data, "train")
        write_split(val, data, "val")
        (data / "data.yaml").write_text(
            f"path: {data.resolve()}\ntrain: images/train\nval: images/val\n"
            f"names:\n  0: goal\n")
        print(f"\n  fold: test on {held_out} ({len(test)} frames, "
              f"{sum(1 for r in test if r['box'])} goals); train on "
              f"{len(fit)} + {len(val)} validation frames", flush=True)
        if args.baseline:
            evaluate(YOLO(args.baseline), test, "current detector")
        model = YOLO("yolov8n.pt")
        model.train(data=str(data / "data.yaml"), epochs=args.epochs,
                    imgsz=640, batch=8, device="cpu", workers=2,
                    project=str(root), name="run", exist_ok=True,
                    verbose=False, plots=False, seed=0)
        evaluate(YOLO(str(root / "run" / "weights" / "best.pt")), test,
                 "retrained, held out")


if __name__ == "__main__":
    main()
