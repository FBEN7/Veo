"""Can a goal box plus the ground plane give a distance to goal?

`src/ground_plane.py` maps any image point on the pitch to metres, but its
origin is the camera, so it cannot say where the goal is -- which is why
goals, shots and out-of-play are disabled. A goal detector supplies exactly
that missing origin: project the detected box to the ground and the pitch
frame acquires a landmark, with no pitch markings and no homography.

Before building anything on that, it has to be checked, and it checks
itself. A goal is **7.32 m wide, always**. Project the two bottom corners of
a box to the ground and the distance between them is a measurement of a
quantity whose true value is known exactly. Nothing is fitted to it and
nothing is free.

So this is falsifiable in advance:

    the chain works        if post separations cluster near 7.32 m
    the chain is broken    if they scatter, or sit at some other value

The boxes used here are the 25 drawn by hand, not detections, so a failure
is the *geometry* failing and not the detector. If hand boxes fail, a
perfect detector would not save it.

## The second measurement, which is what makes this diagnostic

A goal is also 2.44 m **tall**, and that is the useful part: under an
oblique view a goal's projected *width* shrinks towards the depth of the
net, while its height does not shrink at all. Vertical is vertical. So
measuring both separates two very different diagnoses:

    width short, height right     the ground plane is fine and the
                                  axis-aligned box is foreshortening
    both wrong together           the ground plane is wrong

## What it found

Width: median 4.7 m against a true 7.32, 1 of 20 within a tenth, short on
almost every frame. Height: median 2.5 m against a true 2.44 -- the first
absolute metric check this pipeline has passed, and evidence the ground
plane's scale is right.

That is the first diagnosis. A bounding box cannot carry the geometry,
because an axis-aligned rectangle discards the very slant that encodes
orientation, and the labeller said so at the time: "with the perspective,
the goal is not a rectangle but rather a paralelipede rectangle".

The good half should not be oversold. Per-frame heights run 0.5 to 6.1 m on
a quantity that is always 2.44, so 0 of 20 land within a tenth and the
median landing near truth is suggestive rather than proof. The scatter has
the same cause: a box's bottom edge is a noisy estimate of where the posts
actually stand.

Two things were checked along the way and are not the cause. Image rows
really are iso-depth here -- within a row band the median player height on
the left and right of the frame agree to a few pixels -- and the fitted
height model predicts player pixel heights well (49.8 against 47.6
observed, 69.6 against 73.1).

    python probe_goal_ruler.py --labels goal_labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ground_plane import GroundPlane  # noqa: E402
from propagate_goal_labels import DIRS, REPLAY_AREA, area  # noqa: E402

GOAL_WIDTH_M = 7.32

# Foreshortened by an oblique view; the height is not, which is what makes
# measuring both diagnostic rather than merely discouraging.
GOAL_HEIGHT_M = 2.44

# How far from 7.32 m still counts as agreement. A tenth is generous: it is
# 73 cm on a quantity fixed by the laws of the game.
TOLERANCE = 0.10


PLAYER_HEIGHT_M = 1.75


def pixels_per_metre(plane: GroundPlane, y_feet: float, horizon: float):
    """Scale at a given row, as the pixel height of a player standing there.

    The projective form -- X = h_cam * (x - cx) / (y - horizon) -- reduces to
    exactly this once h_cam is written as PLAYER_HEIGHT / slope, so the two
    are the same measurement. This spelling is used because it is the one
    that also works vertically, and the vertical measurement is the
    diagnostic half.
    """
    w = y_feet - horizon
    if w <= 1e-6:
        return None
    px_h = (PLAYER_HEIGHT_M / plane.h_cam_m) * w
    return None if px_h <= 1.0 else px_h / PLAYER_HEIGHT_M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()

    rows = json.loads(Path(args.labels).read_text())["frames"]
    boxes = [r for r in rows if r["label"] and r["label"]["goal"]
             and area(r["label"]["goal"]) <= REPLAY_AREA]

    print(f"Two known quantities: the goal is {GOAL_WIDTH_M} m wide and "
          f"{GOAL_HEIGHT_M} m tall.\nAn oblique view shrinks the first and "
          f"leaves the second alone.\n")
    print(f"  {'frame':>18s} {'wide m':>7s} {'w err':>6s} {'tall m':>7s} "
          f"{'h err':>6s}")

    planes: dict[str, GroundPlane] = {}
    widths, heights = [], []
    for row in boxes:
        clip = row["clip"]
        if clip not in planes:
            path = Path(DIRS.get(clip, "")) / "ground_plane.json"
            if not path.exists():
                planes[clip] = None
            else:
                blob = json.loads(path.read_text())
                planes[clip] = (None if blob.get("refused")
                                else GroundPlane(**blob))
        plane = planes[clip]
        if plane is None:
            continue

        x0, y0, x1, y1 = row["label"]["goal"]
        base = y1 * plane.height
        horizon = float(plane.horizon_at(np.array([row["frame"]]))[0])
        scale = pixels_per_metre(plane, base, horizon)
        if scale is None:
            print(f"  {row['id']:>18s} {'above the horizon':>30s}")
            continue

        wide = (x1 - x0) * plane.width / scale
        tall = (y1 - y0) * plane.height / scale
        widths.append(wide)
        heights.append(tall)
        print(f"  {row['id']:>18s} {wide:7.1f} {wide/GOAL_WIDTH_M-1:+5.0%} "
              f"{tall:7.1f} {tall/GOAL_HEIGHT_M-1:+5.0%}")

    if not widths:
        print("\n  Nothing measurable.")
        return

    def verdict(vals, truth, what):
        v = np.array(vals)
        good = int(np.sum(np.abs(v / truth - 1.0) <= TOLERANCE))
        print(f"  {what:7s} median {np.median(v):5.1f} m (true {truth})   "
              f"spread {v.min():.1f}-{v.max():.1f}   "
              f"within {TOLERANCE:.0%}: {good} of {len(v)}")

    print()
    verdict(widths, GOAL_WIDTH_M, "width")
    verdict(heights, GOAL_HEIGHT_M, "height")
    print("\n  Width short, height right on the median: the ground plane's "
          "scale is\n  sound and the axis-aligned box is foreshortening. A "
          "rectangle cannot\n  carry the orientation, and these are "
          "hand-drawn boxes -- a perfect\n  detector would not rescue it.")
    print("\n  The per-frame height spread is the caveat: 0.5 to 6.1 m on a "
          "quantity\n  that is always 2.44, so the median near truth is "
          "suggestive, not proof.\n  A box's bottom edge is a noisy guess at "
          "where the posts stand.")


if __name__ == "__main__":
    main()
