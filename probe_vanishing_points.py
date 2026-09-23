"""Can the pitch be rectified without the ground plane? Start at the horizon.

Fitting the pitch through the ground plane failed, and the diagnosis was
specific: the line detection is sound, the projection from pixels to metres
underneath it is not. The obvious response -- which the previous attempt
skipped -- is to throw the ground plane away and fit the pitch directly in
image space.

That route begins with vanishing points. Pitch markings run in two directions
and only two, perpendicular on the ground, so their images converge on two
points. The line joining those two points is the horizon of the ground plane,
and from the horizon a projective rectification follows with no focal length,
no camera height and no assumptions from elsewhere.

Before building any of that, this measures whether the first step is even
sound, because everything after it inherits the answer:

  * do the segments separate into two convergent families at all
  * is the horizon stable from frame to frame -- the camera pans, but the
    ground plane's horizon is a property of the plane and the camera's
    orientation, so it moves smoothly rather than jumping
  * does it agree with the horizon the ground plane fitted independently,
    from player heights, by a completely different route

That last one is the valuable one. Two unrelated measurements of the same
line either agree, which is evidence for both, or disagree, which says at
least one is wrong -- and the ground plane is already the suspect.

    python probe_vanishing_points.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from probe_pitch_lines import CLIPS, line_segments
from src import ground_plane

# Segments shorter than this contribute too little direction to be worth a
# vote: a 40-pixel line's angle is uncertain by degrees, and a vanishing
# point is an extrapolation of hundreds of pixels.
MIN_VOTE_LENGTH_PX = 70

# A line supports a vanishing point when the point lies this close to it, in
# pixels, measured at the line's own scale.
VP_INLIER_PX = 3.5

VP_RANSAC_ITERATIONS = 400
MIN_FAMILY_LINES = 3

# Two segments belong to different families when their directions differ by
# more than this.
FAMILY_SPLIT_DEG = 30.0


def _homogeneous(seg):
    x1, y1, x2, y2 = seg
    return np.cross([x1, y1, 1.0], [x2, y2, 1.0])


def _split_families(segments):
    """Two groups by direction, since pitch lines run two ways only."""
    long_enough = [s for s in segments
                   if np.hypot(s[2] - s[0], s[3] - s[1]) >= MIN_VOTE_LENGTH_PX]
    if len(long_enough) < 2 * MIN_FAMILY_LINES:
        return None, None

    angles = np.array([np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])) % 180.0
                       for s in long_enough])
    # Cluster on the doubled angle so 179 and 1 degrees sit together.
    doubled = np.radians(2 * angles)
    feature = np.column_stack([np.cos(doubled), np.sin(doubled)])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, _ = cv2.kmeans(feature.astype(np.float32), 2, None,
                              criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    groups = ([s for s, k in zip(long_enough, labels) if k == 0],
              [s for s, k in zip(long_enough, labels) if k == 1])
    if min(len(g) for g in groups) < MIN_FAMILY_LINES:
        return None, None

    # Reject a split that did not actually separate anything.
    means = [np.mean([np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])) % 180.0
                      for s in g]) for g in groups]
    if min(abs(means[0] - means[1]), 180 - abs(means[0] - means[1])) \
            < FAMILY_SPLIT_DEG:
        return None, None
    return groups


def vanishing_point(family, rng):
    """The point a family of lines converges on, by RANSAC."""
    if len(family) < 2:
        return None, 0
    lines = [_homogeneous(s) for s in family]
    best, best_inliers = None, 0
    for _ in range(VP_RANSAC_ITERATIONS):
        i, j = rng.choice(len(lines), 2, replace=False)
        point = np.cross(lines[i], lines[j])
        if abs(point[2]) < 1e-9:
            continue                      # parallel in the image
        support = 0
        for seg, line in zip(family, lines):
            norm = np.hypot(line[0], line[1])
            if norm < 1e-9:
                continue
            distance = abs(line @ point) / (norm * abs(point[2]))
            if distance <= VP_INLIER_PX:
                support += 1
        if support > best_inliers:
            best, best_inliers = point, support
    if best is None:
        return None, 0
    return best / best[2], best_inliers


def horizon_row_at(vp_a, vp_b, x):
    """Where the horizon through two vanishing points crosses column x."""
    horizon = np.cross(vp_a, vp_b)
    if abs(horizon[1]) < 1e-9:
        return float("nan")
    return float(-(horizon[0] * x + horizon[2]) / horizon[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("Do the markings converge on two vanishing points, and does the "
          "horizon\nthey imply match the one the ground plane fitted from "
          "player heights?\n")
    print(f"  {'clip':>14s} {'usable':>7s} {'horizon row':>13s} "
          f"{'spread':>8s} {'plane says':>11s} {'gap':>8s}")

    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info = json.loads((path / "clip.json").read_text())
        cap = cv2.VideoCapture(info["path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        rows, attempted = [], 0
        for idx in np.linspace(0, total - 1, args.frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            attempted += 1
            segments, _ = line_segments(frame)
            family_a, family_b = _split_families(segments)
            if not family_a:
                continue
            vp_a, support_a = vanishing_point(family_a, rng)
            vp_b, support_b = vanishing_point(family_b, rng)
            if vp_a is None or vp_b is None:
                continue
            if support_a < MIN_FAMILY_LINES or support_b < MIN_FAMILY_LINES:
                continue
            row = horizon_row_at(vp_a, vp_b, info["width"] / 2.0)
            if np.isfinite(row):
                rows.append((int(idx), row))

        cap.release()
        if len(rows) < 4:
            print(f"  {name:>14s} {len(rows):7d}   too few usable frames")
            continue

        values = np.array([r for _, r in rows])
        # The ground plane's horizon, for the same frames, in raw image rows.
        plane_path = path / "ground_plane.json"
        plane_row = float("nan")
        stored = (json.loads(plane_path.read_text())
                  if plane_path.exists() else {})
        if stored and not stored.get("refused"):
            plane = ground_plane.GroundPlane(**stored)
            motion_path = path / "camera_motion.npy"
            motion = np.load(motion_path) if motion_path.exists() else None
            per_frame = []
            for idx, _ in rows:
                h = float(plane.horizon_at(np.array([idx]))[0])
                if motion is not None and len(motion) > idx:
                    h += float(motion[int(idx), 1])
                per_frame.append(h)
            plane_row = float(np.median(per_frame))

        gap = (abs(np.median(values) - plane_row)
               if np.isfinite(plane_row) else float("nan"))
        print(f"  {name:>14s} {len(rows):7d} {np.median(values):13.0f} "
              f"{values.std():8.0f} {plane_row:11.0f} {gap:8.0f}")

    print("\n  The horizon is far above the frame for a camera looking down "
          "at a pitch,\n  so large negative rows are expected. What matters "
          "is the spread across\n  frames and the gap to the independent "
          "estimate.")


if __name__ == "__main__":
    main()
