"""Carry hand-drawn goal boxes to neighbouring frames.

Twenty-six goals were drawn by hand. That is too few to train a detector and
it is not the labeller's fault: a person's time is the scarce thing here, and
150 frames was the agreed budget.

The camera does not jump. A goal in frame 481 is in frame 482 a few pixels
away, and the warp between the two is something this repository already
computes -- `propagate_anchor.frame_to_frame` matches ORB features and
returns the homography. So each hand-drawn box can be carried outward until
the warp stops being trustworthy, turning 26 human labels into several
hundred machine-derived ones.

Everything here still traces back to a human click. Nothing is labelled by a
model, which is what would make this circular.

## Where it stops

Carrying a box is only safe while the warp is. Propagation halts on any of:

  * **a failed match** -- ORB found too few inliers to fit a homography;
  * **a box leaving the frame**, because a partly visible goal is exactly
    what the labeller was told to call "no goal", and inventing one here
    would contradict the labels;
  * **implausible drift** -- the box changing area by more than a factor,
    which means the warp is wrong rather than the camera having zoomed;
  * **a step limit**, because error compounds and this project has measured
    what chained warps do: drift counts multiplications, not seconds.

## Replay cameras are dropped

Two labelled frames are shot from behind the goal, after it was scored.
Those are a different camera with a different geometry, the pipeline never
needs an anchor for them, and a detector taught that view learns something
it will never be asked. Frames whose box covers more than a third of the
picture are treated as replays and left out, with a note of how many.

    python propagate_goal_labels.py --labels goal_labels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from propagate_anchor import frame_to_frame

DIRS = {
    "stoke_1302": "output_stoke_1302", "stoke_4207": "output_stoke_4207",
    "stoke_7001": "output_stoke_7001", "reading_0737": "output_reading_0737",
    "reading_1155": "output_reading_1155",
    "reading_2519": "output_reading_2519",
    "soccernet_w1": "output_soccernet", "soccernet_w2": "output_soccernet_w2",
    "soccernet_w3": "output_soccernet_w3",
    "soccernet_reading": "output_soccernet_reading", "veo": "output_veo",
}

# How far to carry a box, and in what steps. Five frames is a fifth of a
# second: far enough that the walk is cheap, near enough that ORB has plenty
# of overlap to match on.
#
# Three steps, not eight. At forty frames from the seed, roughly a third of
# carried boxes had slid onto advertising hoardings, and tightening the
# appearance test alone did not clear them: a white goal frame and a
# white-lettered board, both against a dark crowd and reduced to a small
# patch, look more alike than is comfortable. Fifteen frames multiplies the
# hand labels by six rather than sixteen, and a smaller clean set beats a
# larger contaminated one -- contaminated positives would teach the detector
# that hoardings are goals, which is the exact failure this replaces.
STEP = 5
MAX_STEPS = 3

# A box whose area changes by more than this between steps is being dragged
# by a bad warp, not by the camera.
MAX_AREA_RATIO = 1.6

# Above this share of the frame, the shot is a replay from behind the goal.
REPLAY_AREA = 0.33

# A propagated box must keep this much of itself inside the picture.
MIN_INSIDE = 0.98

# How much the crop may stop looking like the one that was drawn.
#
# The geometric stops below check that the *warp* is plausible; none of them
# checks that the box still contains a goal. Inspected at maximum drift,
# about a third of carried boxes had slid onto hoardings or crowd while
# every geometric test still passed -- a warp can be perfectly well
# conditioned and still be tracking the wrong thing.
#
# So the crop is compared with the one the labeller drew, as a small
# normalised greyscale patch. A goal that stays a goal correlates highly
# with itself; a box that has wandered onto an advertising board does not.
APPEARANCE_MIN = 0.80
PATCH = 40


def patch_of(frame, box):
    """A small normalised greyscale crop, for comparing appearance."""
    h, w = frame.shape[:2]
    x0 = int(max(0, min(box[0], 1.0) * w))
    x1 = int(max(0, min(box[2], 1.0) * w))
    y0 = int(max(0, min(box[1], 1.0) * h))
    y1 = int(max(0, min(box[3], 1.0) * h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (PATCH, PATCH)).astype(np.float32)
    crop -= crop.mean()
    norm = float(np.linalg.norm(crop))
    return None if norm < 1e-6 else crop / norm


def looks_alike(a, b) -> float:
    """Correlation between two normalised patches, -1 to 1."""
    if a is None or b is None:
        return -1.0
    return float(np.sum(a * b))


def warp_box(box, warp, width, height):
    """Move a normalised box through an image-to-image homography."""
    x0, y0, x1, y1 = box
    corners = np.array([[x0 * width, x1 * width, x1 * width, x0 * width],
                        [y0 * height, y0 * height, y1 * height, y1 * height],
                        [1.0, 1.0, 1.0, 1.0]])
    moved = warp @ corners
    if np.any(np.abs(moved[2]) < 1e-9):
        return None
    moved = moved[:2] / moved[2]
    nx0, nx1 = float(moved[0].min()), float(moved[0].max())
    ny0, ny1 = float(moved[1].min()), float(moved[1].max())
    return [nx0 / width, ny0 / height, nx1 / width, ny1 / height]


def inside_fraction(box):
    x0, y0, x1, y1 = box
    whole = max((x1 - x0) * (y1 - y0), 1e-9)
    cx0, cy0 = max(x0, 0.0), max(y0, 0.0)
    cx1, cy1 = min(x1, 1.0), min(y1, 1.0)
    if cx1 <= cx0 or cy1 <= cy0:
        return 0.0
    return (cx1 - cx0) * (cy1 - cy0) / whole


def area(box):
    return max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)


def walk(cap, info, start_frame, box, direction: int):
    """Carry one box away from its labelled frame, until it is unsafe."""
    width, height = info["width"], info["height"]
    total = int(info["n_frames"])
    out = []
    current, here = box, start_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, here)
    ok, previous = cap.read()
    if not ok:
        return out
    seed_patch = patch_of(previous, box)

    for _ in range(MAX_STEPS):
        nxt = here + direction * STEP
        if nxt < 0 or nxt >= total:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, nxt)
        ok, frame = cap.read()
        if not ok:
            break
        warp, inliers = frame_to_frame(previous, frame)
        if warp is None:
            break
        moved = warp_box(current, warp, width, height)
        if moved is None:
            break
        ratio = area(moved) / area(current)
        if not (1 / MAX_AREA_RATIO <= ratio <= MAX_AREA_RATIO):
            break
        if inside_fraction(moved) < MIN_INSIDE:
            break
        alike = looks_alike(seed_patch, patch_of(frame, moved))
        if alike < APPEARANCE_MIN:
            break
        out.append({"frame": nxt, "box": [round(v, 4) for v in moved],
                    "inliers": int(inliers), "alike": round(alike, 3)})
        current, previous, here = moved, frame, nxt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="goal_dataset.json")
    args = ap.parse_args()

    blob = json.loads(Path(args.labels).read_text())
    rows = blob["frames"]
    positives = [r for r in rows if r["label"] and r["label"]["goal"]]
    negatives = [r for r in rows if r["label"] and r["label"]["goal"] is None]

    replays = [r for r in positives
               if area(r["label"]["goal"]) > REPLAY_AREA]
    usable = [r for r in positives if r not in replays]
    print(f"hand labels: {len(positives)} goals, {len(negatives)} without\n"
          f"  dropped as replay views (box over "
          f"{REPLAY_AREA:.0%} of frame): {len(replays)}")
    for r in replays:
        print(f"    {r['id']}  area {area(r['label']['goal']):.0%}")

    dataset = []
    for row in usable:
        dataset.append({"clip": row["clip"], "frame": row["frame"],
                        "box": row["label"]["goal"], "source": "hand"})
    for row in negatives:
        dataset.append({"clip": row["clip"], "frame": row["frame"],
                        "box": None, "source": "hand"})

    print(f"\n  {'clip':>18s} {'seed':>5s} {'carried':>8s}")
    by_clip = {}
    for row in usable:
        by_clip.setdefault(row["clip"], []).append(row)

    for clip, seeds in sorted(by_clip.items()):
        out_dir = DIRS.get(clip)
        if not out_dir or not (Path(out_dir) / "clip.json").exists():
            continue
        info = json.loads((Path(out_dir) / "clip.json").read_text())
        cap = cv2.VideoCapture(info["path"])
        carried = 0
        for seed in seeds:
            for direction in (1, -1):
                for got in walk(cap, info, seed["frame"],
                                seed["label"]["goal"], direction):
                    dataset.append({"clip": clip, "frame": got["frame"],
                                    "box": got["box"], "source": "carried",
                                    "from": seed["frame"],
                                    "alike": got.get("alike")})
                    carried += 1
        cap.release()
        print(f"  {clip:>18s} {len(seeds):5d} {carried:8d}", flush=True)

    goals = [d for d in dataset if d["box"]]
    Path(args.out).write_text(json.dumps(
        {"positives": len(goals), "negatives": len(dataset) - len(goals),
         "rows": dataset}, indent=1))
    print(f"\n  {len(goals)} goal boxes ({len(usable)} drawn by hand, "
          f"{len(goals) - len(usable)} carried),\n"
          f"  {len(dataset) - len(goals)} frames without a goal "
          f"-> {args.out}")
    print("\n  Every box here descends from a human click. Nothing was "
          "labelled by a\n  model, which is what would make this circular.")


if __name__ == "__main__":
    main()
