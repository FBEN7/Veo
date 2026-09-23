"""Carry an anchor to neighbouring frames, and measure how far it survives.

An anchor is only available where the markings allow it: 3 to 11 frames of
30, because the horizon needs a penalty area and the circle needs midfield.
Everything downstream -- a shot's distance from goal, its angle, which half
of the pitch it happened in -- needs one on the frame the shot is actually
on.

A frame without its own anchor can borrow the nearest one. If W maps image
points from the anchored frame a to the target frame b, then the pitch map on
b is simply H_a W^-1: the pitch has not moved, only the camera. W is measured
by matching features between the two frames rather than modelled from the
camera's motion, so it needs no focal length and inherits no assumption about
how the camera turned.

The question is how far this reaches, and there is reason to expect not very.
This project already found that chaining homographies between frames
"degenerates into a radial smear" over hundreds of pairs, because pairwise
error compounds. Here each borrow is a single hop from an anchored frame
rather than a chain, which should do better -- but a single hop over a large
gap has its own problem: the two views stop overlapping, and a homography
fitted on whatever few features remain is a homography fitted on nothing.

So the output is a curve, not a number: error against how far the anchor was
carried, scored the same way anchors are scored -- markings the fit never saw,
against a measured chance floor.

    python propagate_anchor.py [--frames 60]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from fit_pitch_anchor import (CHANCE_OFFSET_M, CHANCE_TRIALS, anchored_frames,
                              marking_error, plausible_anchor)
from probe_pitch_lines import CLIPS, line_segments
from src import pitch_model as pm

# Gaps to test, in frames. At 25 fps these run from a third of a second to
# eight seconds.
GAPS = (8, 25, 50, 100, 200)

ORB_FEATURES = 4000
MIN_MATCHES = 40
MIN_INLIERS = 25
RANSAC_PX = 3.0


def frame_to_frame(anchor_frame, target_frame):
    """The homography taking points in one frame to the other.

    Features are matched rather than the camera's motion being modelled: for
    a camera that turns without travelling, every point in the image moves by
    the same homography whatever its depth, so the stands are as useful as
    the pitch and there is no focal length to get wrong. Players move on
    their own and are left for RANSAC to reject, being a minority of what ORB
    finds.
    """
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    grey_a = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY)
    grey_b = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    kp_a, desc_a = orb.detectAndCompute(grey_a, None)
    kp_b, desc_b = orb.detectAndCompute(grey_b, None)
    if desc_a is None or desc_b is None:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    good = [p[0] for p in pairs
            if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_MATCHES:
        return None, len(good)
    src = np.float32([kp_a[m.queryIdx].pt for m in good])
    dst = np.float32([kp_b[m.trainIdx].pt for m in good])
    warp, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if warp is None or inliers is None or int(inliers.sum()) < MIN_INLIERS:
        return None, int(inliers.sum()) if inliers is not None else 0
    return warp, int(inliers.sum())


def chance_floor(homography, segments, info, rng):
    """What the same markings score under a displaced anchor."""
    scores = []
    for _ in range(CHANCE_TRIALS):
        offset = rng.uniform(-CHANCE_OFFSET_M, CHANCE_OFFSET_M, 2)
        shifted = np.eye(3)
        shifted[:2, 2] = offset
        value = marking_error(shifted @ homography, segments, info)
        if np.isfinite(value):
            scores.append(value)
    return float(np.median(scores)) if scores else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("An anchor borrowed by a neighbouring frame, scored on markings "
          "the anchor\nnever saw. The question is how far it carries.\n")

    totals = {gap: [] for gap in GAPS}
    floors = {gap: [] for gap in GAPS}
    attempts = {gap: 0 for gap in GAPS}
    coverage = []
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info, anchors = anchored_frames(path, args.frames, rng)
        if len(anchors) < 1:
            print(f"{name}: no anchored frames\n")
            continue

        cap = cv2.VideoCapture(info["path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        per_gap = {gap: [] for gap in GAPS}

        for anchor_idx, homography in anchors:
            cap.set(cv2.CAP_PROP_POS_FRAMES, anchor_idx)
            ok, anchor_frame = cap.read()
            if not ok:
                continue
            for gap in GAPS:
                for direction in (-1, 1):
                    target = anchor_idx + direction * gap
                    if not (0 <= target < total):
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    ok, target_frame = cap.read()
                    if not ok:
                        continue
                    attempts[gap] += 1
                    warp, _ = frame_to_frame(anchor_frame, target_frame)
                    if warp is None:
                        continue
                    try:
                        carried = homography @ np.linalg.inv(warp)
                    except np.linalg.LinAlgError:
                        continue
                    if not plausible_anchor(carried, info):
                        continue
                    segments, _ = line_segments(target_frame)
                    if len(segments) < 3:
                        continue
                    error = marking_error(carried, segments, info)
                    if np.isfinite(error):
                        per_gap[gap].append(error)
                        floors[gap].append(
                            chance_floor(carried, segments, info, rng))
                        totals[gap].append(error)

        cap.release()

        # Coverage: the share of the clip lying within reach of an anchor.
        # An upper bound, since a borrow at that range still has to succeed.
        reach = max(GAPS)
        covered = np.zeros(total, dtype=bool)
        for anchor_idx, _ in anchors:
            covered[max(0, anchor_idx - reach):
                    min(total, anchor_idx + reach + 1)] = True
        coverage.append((name, len(anchors) / args.frames,
                         float(covered.mean())))

        summary = "  ".join(
            f"{gap}f {np.median(per_gap[gap]):.1f}m({len(per_gap[gap])})"
            if per_gap[gap] else f"{gap}f -" for gap in GAPS)
        print(f"{name}: {len(anchors)} anchors  ->  {summary}")

    print(f"\n  {'gap':>6s} {'seconds':>8s} {'borrows':>8s} {'tried':>6s} "
          f"{'worked':>7s} {'median error':>13s} {'chance floor':>13s}")
    for gap in GAPS:
        values = np.array(totals[gap])
        floor = np.array([f for f in floors[gap] if np.isfinite(f)])
        if not values.size:
            print(f"  {gap:6d} {0:7d}")
            continue
        rate = values.size / attempts[gap] if attempts[gap] else 0.0
        print(f"  {gap:6d} {gap / 25.0:8.1f} {values.size:8d} "
              f"{attempts[gap]:6d} {rate:7.0%} {np.median(values):13.1f} "
              f"{(np.median(floor) if floor.size else float('nan')):13.1f}")

    print("\n  An anchor that still beats its chance floor at a given gap is "
          "worth\n  borrowing at that gap. Where the two meet, the borrow is "
          "carrying nothing.\n")

    print(f"  {'clip':>14s} {'anchored alone':>15s} {'within reach':>13s}")
    for name, alone, within in coverage:
        print(f"  {name:>14s} {alone:15.0%} {within:13.0%}")
    print("\n  'Within reach' is an upper bound: a frame near an anchor can "
          "still fail to\n  borrow it, at the rate in the 'worked' column "
          "above.")


if __name__ == "__main__":
    main()
