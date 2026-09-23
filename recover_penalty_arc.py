"""Anchor on the penalty D deliberately, instead of throwing it away.

Raising the arc-span gate to 200 degrees fixed the tail. It also discarded
half the anchors, and most of a clip's worth on Veo footage, whose coverage
fell from 68% of frames within reach of an anchor to 25%. That is the right
trade over keeping anchors that were 41.5 m wrong, but it is not the best
available one, because the frames being thrown away are not bad frames. They
are frames looking at the penalty area, and the penalty area is where shots
happen.

A D-anchored frame is not wrong by a random amount. It is wrong by exactly
41.5 m along the pitch -- the centre spot at 52.5 against the penalty spot at
11 -- and in a direction that the arc itself gives away.

## Which end, from the shape of the arc

The D is the part of the penalty arc lying outside the penalty area, and it
always bulges *away* from the goal, toward the middle of the pitch. At the
left end the spot is at x = 11, the area's edge at x = 16.5, and the arc
reaches out to x = 20.15. At the right end the spot is at 94, the edge at
88.5, and the arc reaches back to 84.85.

So take the provisional anchor -- the one that has wrongly put the arc's
centre at (52.5, 34) -- and map the arc's own supporting pixels through it.
Their centroid sits on the far side of the centre from the goal. If it lies
at greater x, the goal is the left one and the true centre is (11, 34); if
at smaller x, it is the right one and the true centre is (94, 34). Shift by
41.5 m accordingly and the frame is anchored properly, from the D, on
purpose.

This inherits the orientation convention rather than fighting it. Which end
is "left" is fixed by `metric_from_circle` pinning the pitch to the camera's
side, and that choice is the same on every frame of the clip, so every
recovered frame lands consistently with every other. It does not claim to
know which physical goal is which -- nothing here can, and a pitch is
symmetric -- only that the clip agrees with itself.

## The test that has to come first, which the first version skipped

The danger is the opposite mistake: taking a real centre circle for a D and
introducing a 41.5 m error where there was none. The first version of this
guarded against it with the arc span alone -- short arc, therefore a D --
and that is not enough, because a centre circle with most of itself hidden
behind players or running out of frame is also a short arc. Measured, the
correction improved 54% of the frames it touched. A coin toss, and for
exactly that reason: about half of them were not Ds.

What a D has and a partial circle does not is its chord. The penalty-area
line is what cut the circle, and it lies 5.5 m from the centre -- 16.5 m out
from the goal line against the spot's 11. So the arc is only treated as a D
when a real straight line is found sitting 5.5 m from its centre, and that
distance can be measured before the rotation is known, since a rotation does
not move anything nearer or further from the centre.

That same chord then fixes the rotation. It runs parallel to the goal line,
which is the same family as the halfway line, so it plays exactly the part
the halfway line plays for the centre circle -- and it is the only line that
can, because on a D frame nothing passes through the centre at all. The
existing `halfway_line` helper, which wants a line through the centre within
22 px, is no use here and was quietly returning the chord anyway on two
frames out of three, which is its own problem.

The result is scored on markings the fit never saw, against the same chance
floor as everything else.

    python recover_penalty_arc.py [--frames 60]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fit_pitch_anchor import (CHANCE_OFFSET_M, CHANCE_TRIALS, marking_error,
                              plausible_anchor)
from probe_centre_circle import MIN_ARC_SPAN_DEG, find_circle
from src.pitch_model import (D_SPAN_MAX, D_SPAN_MIN,
                             PENALTY_SPOTS_M,
                             anchor_from_penalty_arc,
                             penalty_arc_chord)
from probe_pitch_lines import CLIPS, line_segments
from src import pitch_model as pm

# The D subtends 2*acos(5.5/9.15) = 106 degrees of its circle. Arcs are
# looked for from a little below that, to allow for a detector that clips the
# ends, and accepted as a D only up to a ceiling well short of what a centre
# circle shows -- measured at a median of 240 degrees.

# Where the two penalty spots are, along the pitch.
PENALTY_SPOTS_M = (11.0, pm.PITCH_LENGTH_M - 11.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("Frames the span gate refuses, anchored on the penalty D on "
          "purpose.\nScored on markings the fit never saw, against the same "
          "chance floor.\n")
    print(f"  {'clip':>14s} {'D frames':>9s} {'as centre circle':>17s} "
          f"{'recovered':>10s} {'chance':>7s} {'left/right':>11s}")

    pooled_before, pooled_after, pooled_floor = [], [], []
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info = json.loads((path / "clip.json").read_text())
        cap = cv2.VideoCapture(info["path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        before, after, floors, ends = [], [], [], []
        for index in np.linspace(0, total - 1, args.frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                continue
            circle = find_circle(frame, rng, min_span_deg=D_SPAN_MIN)
            if circle is None:
                continue
            span = circle["span_deg"]
            # Only frames the centre-circle gate refuses, and only arcs short
            # enough that a centre circle is not a plausible reading.
            if span >= MIN_ARC_SPAN_DEG or not (D_SPAN_MIN <= span
                                                <= D_SPAN_MAX):
                continue

            # Unrotated first, only to measure how far lines sit from the
            # arc's centre -- a distance the rotation cannot change.
            unrotated = pm.metric_from_circle(
                pm.AT_INFINITY_LINE, circle["ellipse"], None)
            if unrotated is None:
                continue
            chord = penalty_arc_chord(unrotated, circle["segments"])
            if chord is None:
                continue

            # The chord runs across the pitch, parallel to the goal line,
            # which is the same family as the halfway line -- so it fixes the
            # rotation in exactly the same way. On a D frame it is also the
            # only line that can, since nothing passes through the centre.
            direction = np.array([chord[2] - chord[0], chord[3] - chord[1]],
                                 dtype=float)
            provisional = pm.metric_from_circle(
                pm.AT_INFINITY_LINE, circle["ellipse"], direction)
            if provisional is None:
                continue
            fixed, spot = anchor_from_penalty_arc(provisional,
                                                  circle["support"])
            if fixed is None or not plausible_anchor(fixed, info):
                continue

            segments, _ = line_segments(frame)
            if len(segments) < 3:
                continue
            plain = marking_error(provisional, segments, info)
            better = marking_error(fixed, segments, info)
            if not (np.isfinite(plain) and np.isfinite(better)):
                continue

            chance = []
            for _ in range(CHANCE_TRIALS):
                displaced = np.eye(3)
                displaced[:2, 2] = rng.uniform(-CHANCE_OFFSET_M,
                                               CHANCE_OFFSET_M, 2)
                value = marking_error(displaced @ fixed, segments, info)
                if np.isfinite(value):
                    chance.append(value)
            before.append(plain)
            after.append(better)
            ends.append(spot)
            if chance:
                floors.append(float(np.median(chance)))
        cap.release()

        if not before:
            print(f"  {name:>14s} {0:9d}")
            continue
        left = sum(1 for s in ends if s == PENALTY_SPOTS_M[0])
        print(f"  {name:>14s} {len(before):9d} {np.median(before):16.1f}m "
              f"{np.median(after):9.1f}m "
              f"{(np.median(floors) if floors else np.nan):6.1f}m "
              f"{left}/{len(ends) - left:<5d}")
        pooled_before += before
        pooled_after += after
        pooled_floor += floors

    if pooled_before:
        print(f"\n  {'pooled':>14s} {len(pooled_before):9d} "
              f"{np.median(pooled_before):16.1f}m "
              f"{np.median(pooled_after):9.1f}m "
              f"{np.median(pooled_floor):6.1f}m")
        rescued = np.mean(np.array(pooled_after) < np.array(pooled_before))
        print(f"\n  The 41.5 m correction improves {rescued:.0%} of these "
              f"frames.")

    print("\n  These are frames the anchor currently refuses outright. Any "
          "of them that\n  comes back under the chance floor is coverage "
          "regained rather than error\n  reintroduced -- and they are frames "
          "of the penalty area, where shots are.")


if __name__ == "__main__":
    main()
