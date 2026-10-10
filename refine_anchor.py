"""Refine an anchor against the straight markings, and measure whether it helped.

The anchor as built uses the centre circle for everything and the straight
lines for almost nothing. The circle gives scale (its radius is 9.15 m),
position (its centre is the middle of the pitch) and, through the halfway
line, rotation. Every touchline, penalty-area edge and six-yard box in the
frame is then used only to *score* the result.

That is a strange division of labour. Those markings carry real information
about the pose -- a touchline pins the lateral offset far better than an
ellipse centre estimated from a 55-pixel arc -- and the anchor is currently
throwing it away. The reason it was built this way is sound: the lines were
what kept failing to identify, because every straight pitch line looks like
every other one. But that was line identification *from scratch*. Starting
from a circle anchor that is already within a few metres, a line only has to
be matched to the nearest candidate, which is a far easier problem.

So: keep the circle for initialisation, then let the lines correct it.

## What is allowed to move

A rotation and a translation in pitch coordinates, and nothing more. Three
parameters against, typically, five to fifteen segments -- and each of those
three is a quantity the circle is genuinely weak on: the rotation comes from
a halfway line that is often not detected at all, and the position from an
ellipse centre estimated off a short arc. Scale is excluded because the
circle measures it metrically and a nearest-line loss does not measure it
honestly; that is recorded below.

The temptation is to refine the full homography, all eight parameters. That
is rejected on principle rather than measured: eight parameters fitted to a
handful of segments, all of which lie in the small part of the pitch the
camera is looking at, is a fit that can trade a genuine pose error against a
projective distortion that makes the visible markings agree and everything
outside the view worse. A similarity cannot do that. Whatever it gets right
in the middle of the frame it gets right at the edges too.

## Why this is not circular, and how far that goes

Refining against markings and then scoring against markings would measure
nothing. The segments are therefore split in two at random: one half moves
the anchor, the other half scores it, and the halves never mix. A refinement
that has learnt the real pose improves on markings it never saw; one that has
merely bent itself onto the segments it was given does not.

The split is by segment, which is the honest unit here but not a perfect one:
two segments detected on the same touchline are not independent, so a
refinement that gets that touchline right is rewarded twice. That inflates
the improvement somewhat and cannot be removed without knowing which line is
which, which is the problem being solved. The cross-frame check below has no
such weakness and is the one to believe.

## What it measures: not enough to ship

    clip            circle only   refined   chance
    SoccerNet w1           1.0m      1.0m     6.5m
    SoccerNet w2           2.9m      2.4m     6.0m
    SoccerNet w3           1.0m      0.9m     4.9m
    reading                1.0m      0.9m     5.7m
    Veo                    2.2m      2.3m     4.3m
    pooled                 1.2m      1.3m     5.8m

It improves the held-out markings on 61% of frames and the pooled median not
at all, and it moves each anchor by 0.5 degrees and 0.8 m -- which is to say
it does almost nothing, carefully. On the cross-frame agreement check, which
never looks at a marking, it is slightly worse than leaving the anchor alone:
1.3 m against 1.8 m.

So this is not enabled anywhere. It is kept because the two failures behind
that number are worth not repeating, and because the conclusion -- that the
straight markings have little left to add once the circle has spoken --
is itself a finding, and one that `probe_anchor_bias.py` explains.

    python refine_anchor.py [--frames 40]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from fit_pitch_anchor import (CHANCE_OFFSET_M, CHANCE_TRIALS, anchored_frames,
                              marking_error, plausible_anchor)
from probe_pitch_lines import CLIPS, line_segments
from src import pitch_model as pm

# How far the refinement may move the anchor. These are deliberately tight.
# The pitch lines across the length sit 11 m apart around the penalty area,
# so a search free to slide 15 m can land a marking on the wrong line and
# report a confident, badly wrong pose. The circle anchor is already within a
# few metres; the refinement's job is the last of it, not a fresh search.
MAX_ROTATION_DEG = 10.0
MAX_SHIFT_M = 5.0

# The coarse grid that initialises the local search, since nearest-line
# assignment makes the loss bumpy and a local method started at zero settles
# into whichever basin it began in.
GRID_ROTATION_DEG = (-8.0, -4.0, 0.0, 4.0, 8.0)
GRID_SHIFT_M = (-4.0, 0.0, 4.0)

# How much the anchor is expected to be wrong by, and so how hard the
# refinement is pulled back toward leaving it alone. This is the circle
# anchor acting as a prior rather than merely as a starting point, and the
# first version of this file omitted it -- with the result that the
# refinement improved the two clips whose anchors were poor and made the
# three that were already good worse, for no net gain at all. A handful of
# segments and four free parameters will always find something to correct;
# the prior is what makes it correct only what it has the evidence for.
PRIOR_ROTATION_DEG = 4.0
PRIOR_SHIFT_M = 2.0

# Residuals beyond this are treated as a marking that was never a pitch line
# -- a shadow, a boot, the edge of a technical area -- rather than as evidence
# about the pose. Without it a single spurious segment drags the whole fit.
HUBER_M = 2.5

# A refinement is only accepted where there is something to refine against.
MIN_FIT_SEGMENTS = 3


def rigid(params) -> np.ndarray:
    """A rotation and shift about the pitch centre, in metres.

    Scale is deliberately not among the parameters. The circle already
    measures it, and metrically: its radius is 9.15 m by the laws of the
    game, which is a harder fact than anything a handful of line segments
    can offer. Letting the refinement change it was measured and was worse --
    the fitted scale ran to within a fraction of its bound on all five clips,
    which is what a parameter does when the data are not deciding it.
    """
    theta, shift_x, shift_y = params
    cos, sin = np.cos(theta), np.sin(theta)
    linear = np.array([[cos, -sin], [sin, cos]])
    centre = np.array(pm.PITCH_CENTRE_M)
    out = np.eye(3)
    out[:2, :2] = linear
    out[:2, 2] = centre - linear @ centre + np.array([shift_x, shift_y])
    return out


def _mapped_markings(homography, segments):
    """Each segment as (coordinate, which axis it is constant along, weight)."""
    out = []
    for x1, y1, x2, y2 in segments:
        pts = np.array([[x1, x2], [y1, y2], [1.0, 1.0]], dtype=float)
        mapped = homography @ pts
        if np.any(np.abs(mapped[2]) < 1e-9):
            continue
        mapped = mapped[:2] / mapped[2]
        if not np.all(np.isfinite(mapped)):
            continue
        mid = mapped.mean(axis=1)
        span_x = abs(mapped[0, 0] - mapped[0, 1])
        span_y = abs(mapped[1, 0] - mapped[1, 1])
        # A long segment says more about the pose than a short one, and a
        # segment mapping to a plausible length says more than one near the
        # horizon mapping to hundreds of metres.
        length = float(np.hypot(span_x, span_y))
        if not (0.5 <= length <= 80.0):
            continue
        axis = 0 if span_x < span_y else 1
        out.append((float(mid[axis]), axis, min(length, 30.0)))
    return out


def _loss(markings, params=None) -> float:
    """Weighted Huber distance to the nearest pitch line, plus the prior."""
    if not markings:
        return float("inf")
    model = (np.array(pm.LINES_ACROSS_M), np.array(pm.LINES_ALONG_M))
    total, weight_sum = 0.0, 0.0
    for value, axis, weight in markings:
        residual = float(np.abs(model[axis] - value).min())
        if residual <= HUBER_M:
            cost = 0.5 * residual ** 2
        else:
            cost = HUBER_M * (residual - 0.5 * HUBER_M)
        total += weight * cost
        weight_sum += weight
    value = total / weight_sum
    if params is not None:
        theta, shift_x, shift_y = params
        value += (np.rad2deg(theta) / PRIOR_ROTATION_DEG) ** 2
        value += (np.hypot(shift_x, shift_y) / PRIOR_SHIFT_M) ** 2
    return float(value)


def refine(homography, segments, info):
    """The anchor, corrected by a similarity fitted to these segments."""
    if len(segments) < MIN_FIT_SEGMENTS:
        return homography, np.zeros(3)

    if len(_mapped_markings(homography, segments)) < MIN_FIT_SEGMENTS:
        return homography, np.zeros(3)

    # The segments are re-projected at every step rather than their mapped
    # coordinates being moved. A rotation mixes the two pitch axes, and which
    # axis a segment is scored along can change as it turns, so the mapped
    # midpoint alone is not enough to evaluate a candidate.
    def objective(params):
        moved = rigid(params) @ homography
        return _loss(_mapped_markings(moved, segments), params)

    best, best_cost = np.zeros(3), objective(np.zeros(3))
    for theta in np.deg2rad(GRID_ROTATION_DEG):
        for shift_x in GRID_SHIFT_M:
            for shift_y in GRID_SHIFT_M:
                start = np.array([theta, shift_x, shift_y])
                cost = objective(start)
                if cost < best_cost:
                    best, best_cost = start, cost

    bounds = [(-np.deg2rad(MAX_ROTATION_DEG), np.deg2rad(MAX_ROTATION_DEG)),
              (-MAX_SHIFT_M, MAX_SHIFT_M), (-MAX_SHIFT_M, MAX_SHIFT_M)]
    result = minimize(objective, best, method="Powell", bounds=bounds,
                      options={"xtol": 1e-3, "ftol": 1e-4, "maxiter": 2000})
    params = result.x if result.success else best
    refined = rigid(params) @ homography
    if not plausible_anchor(refined, info):
        return homography, np.zeros(3)
    return refined, params


def chance_floor(homography, segments, info, rng):
    scores = []
    for _ in range(CHANCE_TRIALS):
        shifted = np.eye(3)
        shifted[:2, 2] = rng.uniform(-CHANCE_OFFSET_M, CHANCE_OFFSET_M, 2)
        value = marking_error(shifted @ homography, segments, info)
        if np.isfinite(value):
            scores.append(value)
    return float(np.median(scores)) if scores else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("The circle anchor, corrected by a similarity fitted to half the "
          "straight\nmarkings and scored on the other half. Neither half is "
          "seen by the other.\n")

    print(f"  {'clip':>14s} {'frames':>7s} {'circle only':>12s} "
          f"{'refined':>9s} {'chance':>8s} {'moved by':>16s}")
    pooled_raw, pooled_ref, pooled_floor = [], [], []
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info, anchors = anchored_frames(path, args.frames, rng)
        if not anchors:
            continue

        cap = cv2.VideoCapture(info["path"])
        raws, refs, floors, moves = [], [], [], []
        for idx, homography in anchors:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            segments, _ = line_segments(frame)
            if len(segments) < 2 * MIN_FIT_SEGMENTS:
                continue
            order = rng.permutation(len(segments))
            half = len(order) // 2
            fit = [segments[i] for i in order[:half]]
            held = [segments[i] for i in order[half:]]

            refined, params = refine(homography, fit, info)
            raw_error = marking_error(homography, held, info)
            ref_error = marking_error(refined, held, info)
            if not (np.isfinite(raw_error) and np.isfinite(ref_error)):
                continue
            raws.append(raw_error)
            refs.append(ref_error)
            floors.append(chance_floor(homography, held, info, rng))
            moves.append(params)

        cap.release()
        if not raws:
            continue
        moved = np.array(moves)
        shift = np.hypot(moved[:, 1], moved[:, 2])
        summary = (f"{np.rad2deg(np.abs(moved[:, 0])).mean():4.1f}deg "
                   f"{shift.mean():4.1f}m")
        floor = np.array([f for f in floors if np.isfinite(f)])
        print(f"  {name:>14s} {len(raws):7d} {np.median(raws):11.1f}m "
              f"{np.median(refs):8.1f}m "
              f"{(np.median(floor) if floor.size else np.nan):7.1f}m "
              f"{summary:>16s}")
        pooled_raw += raws
        pooled_ref += refs
        pooled_floor += list(floor)

    if pooled_raw:
        print(f"\n  {'pooled':>14s} {len(pooled_raw):7d} "
              f"{np.median(pooled_raw):11.1f}m {np.median(pooled_ref):8.1f}m "
              f"{np.median(pooled_floor):7.1f}m")
        wins = np.mean(np.array(pooled_ref) < np.array(pooled_raw))
        print(f"\n  The refinement improves the held-out markings on "
              f"{wins:.0%} of frames.")

    print("\n  'Moved by' is how far the refinement had to correct the circle "
          "anchor.\n  A correction near the search bounds means the bounds, "
          "not the markings,\n  are deciding -- which would show up as no "
          "improvement held out.")


if __name__ == "__main__":
    main()
