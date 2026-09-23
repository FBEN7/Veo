"""Find the centre circle, the one marking that says where you are.

Every straight pitch line looks like every other straight pitch line, which
is why identification kept failing: a frame carrying three to five lines
against seven candidate positions is underdetermined, and no tightening of
the match fixed it.

The centre circle is not like that. There is exactly one, its radius is 9.15
metres, and its centre is the middle of the pitch. That is scale and origin
in a single object.

It also composes well with the rectification already built. Mapping two
vanishing points to the axes leaves the two world directions axis-aligned but
their scales independent and unknown, so a world circle lands as an
*axis-aligned ellipse*. Its two semi-axes are therefore the two scales -- 9.15
metres each -- and its centre is the pitch centre. Four unknowns, one conic,
no straight lines needed at all.

And the frames it works on are the ones everything else discards. The camera
sits on midfield for most of a match, which is precisely where the circle is.

This probe does the detection half: marking pixels, minus everything already
explained by a straight line, fitted as an ellipse and checked for being
one.

One structural finding falls out of it, and it shapes what comes next. The
circle and the horizon almost never arrive in the same frame:

    clip            frames   circle   horizon   both   either
    SoccerNet w1        30        9         3      1       11
    SoccerNet w2        30        8         9      1       16
    SoccerNet w3        30        3         2      1        4
    reading             30       14         2      0       16
    Veo                 30        2         0      0        2

which is not bad luck. The circle is at midfield and the vanishing-point gate
needs two well-populated families of straight lines, which is the penalty
area. They are complementary views, not simultaneous ones.

That does not block the anchor, because the horizon belongs to the camera
rather than to the frame, and the camera is bolted down. A horizon measured
on a penalty-area frame transfers to a midfield frame through the tilt the
motion estimator already tracks. So the shape of the solution is: take the
horizon where the straight lines give it, take the scale and origin where the
circle gives them, and carry each to the frames that lack it.

    python probe_centre_circle.py [--save]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from probe_pitch_lines import (CLIPS, LINE_MIN_BRIGHTNESS, TOPHAT_KERNEL,
                               line_segments, pitch_surface)

# How thick a straight segment is painted out before looking for the arc,
# when that is done at all. It is off by default, and the reason is measured.
#
# Erasing the straight lines first seemed obviously right: the conic fit
# should not have to contend with a touchline. It is destructive, because the
# line detector fits straight *chords* to a large arc -- a 55-pixel chord of a
# big circle is nearly straight -- so painting the lines out deletes the
# circle itself. The larger the circle in frame, the worse it gets.
#
# Measured over 24 frames a clip, detections with erasing against without:
#
#     w1        7  ->  14        w3        2  ->   9
#     w2        6  ->  10        reading  12  ->  11
#     Veo       0  ->   8
#
# The Veo clip is the extreme case and the one that exposed it -- its centre
# circle spans most of the frame, so erasing removed all of it and the
# detector found nothing in 24 frames. But it was costing detections
# everywhere, and the ones it left were worse: median residual 0.56-0.77 px
# with erasing against 0.54-0.70 without, over arcs of 120-170 degrees
# against 135-235. RANSAC rejects the straight lines perfectly well on its
# own, which is what it is for.
LINE_ERASE_PX = 11

# An arc needs this many pixels to be worth fitting a conic to.
MIN_ARC_PIXELS = 120

# A component that fits a straight line this well is a straight line, however
# it was missed earlier. Measured as the smaller eigenvalue of the pixel
# scatter over the larger.
MAX_STRAIGHTNESS = 0.02

# The fitted ellipse is believed when its pixels sit this close to it, in
# pixels, at the median.
MAX_ELLIPSE_RESIDUAL_PX = 2.5

# A centre circle is a substantial object; these bound it away from scuffs
# and from the whole-frame nonsense a degenerate fit produces.
MIN_ELLIPSE_AXIS_PX = 25
MAX_ELLIPSE_AXIS_RATIO = 6.0

# How much of the ellipse the supporting pixels must actually cover, in
# degrees.
#
# This is the constraint that matters and the first version had nothing like
# it. A short arc admits a whole family of ellipses, so fitting one to a
# fragment produces a confident answer with no information in it. Inspected,
# that is exactly what happened: on one frame the fit covered the left half
# of the centre circle and stopped, on another it sat on a small blob at the
# frame edge while the real circle was plainly visible in the middle.
#
# Two things caused the fragments. Players stand in front of the circle, and
# painting out the halfway line -- which crosses the circle -- cuts it in
# half. So the conic is now fitted by RANSAC across ALL arc pixels at once
# rather than per connected component, which lets the two halves of a cut
# circle support one another.
# How much of a circle must be visible before the arc is believed to be the
# centre circle. This is not a noise threshold, though it started as one at
# 120 degrees. It is what tells the centre circle from the penalty D.
#
# Both arcs have a radius of 9.15 m -- they come from the same measurement in
# the laws of the game -- so an ellipse fitted to the D has the right scale,
# a good residual and nothing wrong with it except which arc it is. Anchoring
# on it puts the whole map out by 52.5 - 11 = 41.5 m, the centre spot against
# the penalty spot, and `marking_error` sees a frame sitting 7 m from the
# markings rather than anything obviously broken.
#
# What separates them is fixed by geometry. The D is only the part of its
# circle outside the penalty area; the penalty spot is 11 m from the goal
# line and the area's edge is 16.5, so the chord lies 5.5 m from a 9.15 m
# centre and the arc spans 2*acos(5.5/9.15) = 106 degrees. Never more. At a
# threshold of 120 the D walked straight through.
#
# Measured over 129 anchored frames on five clips:
#
#     min span   anchors kept   median   p90    over 5 m
#         120d        129/129     1.0m   7.3m        19%
#         160d         94/129     0.8m   6.9m        12%
#         180d         78/129     0.7m   3.3m         8%
#         200d         70/129     0.6m   2.1m         4%
#         220d         60/129     0.6m   1.5m         3%
#
# 200 takes the tail down by three and a half times and improves the median
# as well, which is the sign that what it is refusing was wrong rather than
# merely marginal. It costs 46% of the anchors; propagation covers the frames
# they would have anchored.
MIN_ARC_SPAN_DEG = 200.0
RANSAC_ITERATIONS = 300
RANSAC_INLIER_PX = 2.0


def marking_pixels(frame):
    """Bright thin structure on the playing surface."""
    surface = pitch_surface(frame)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(
        grey, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT,
                                  (TOPHAT_KERNEL, TOPHAT_KERNEL)))
    bright = (tophat > LINE_MIN_BRIGHTNESS).astype(np.uint8) * 255
    return cv2.bitwise_and(bright, surface)


def arc_pixels(frame, erase_lines: bool = False):
    """Marking pixels, optionally with every straight segment painted out.

    Erasing helps when the circle is small in frame and the straight lines
    around it would dominate a conic fit. It is actively destructive when the
    circle is large, because the line detector then fits straight *chords* to
    the arc -- a 55-pixel chord of a big circle is nearly straight -- and
    painting those out deletes the circle before the ellipse fit sees it.
    That is the Veo case exactly: the clip shows a centre circle spanning most
    of the frame, and the detector found one in none of 24 frames until this
    became optional.
    """
    mask = marking_pixels(frame)
    segments, _ = line_segments(frame)
    if not erase_lines:
        return mask, segments
    erased = mask.copy()
    for x1, y1, x2, y2 in segments:
        cv2.line(erased, (x1, y1), (x2, y2), 0, LINE_ERASE_PX)
    return erased, segments


def _ellipse_distances(points, ellipse):
    """First-order distance from each point to the ellipse, in pixels."""
    (cx, cy), (major, minor), angle = ellipse
    a, b = major / 2.0, minor / 2.0
    if a < 1e-6 or b < 1e-6:
        return np.full(len(points), np.inf)
    theta = np.radians(angle)
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    radial = np.hypot(u / a, v / b)
    scale = np.hypot(u / (a * a), v / (b * b))
    return np.abs(radial - 1.0) / np.maximum(scale, 1e-9)


def _ellipse_residual(points, ellipse):
    """Median distance from points to the fitted ellipse, in pixels."""
    (cx, cy), (major, minor), angle = ellipse
    return float(np.median(_ellipse_distances(points, ellipse)))


def _arc_span_deg(points, ellipse) -> float:
    """How many degrees of the ellipse the supporting points cover."""
    (cx, cy), (major, minor), angle = ellipse
    theta = np.radians(angle)
    dx, dy = points[:, 0] - cx, points[:, 1] - cy
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    phi = np.degrees(np.arctan2(v / max(minor / 2.0, 1e-6),
                                u / max(major / 2.0, 1e-6))) % 360.0
    occupied = np.zeros(36, dtype=bool)          # ten-degree buckets
    occupied[(phi / 10.0).astype(int) % 36] = True
    return float(occupied.sum() * 10)


def find_circle(frame, rng=None, erase_lines: bool = False):
    """One conic fitted across every arc pixel, by RANSAC.

    Not per connected component: the circle arrives in pieces, cut by the
    halfway line and by the players standing on it, and a piece on its own
    does not determine an ellipse.
    """
    rng = rng or np.random.default_rng(0)
    erased, segments = arc_pixels(frame, erase_lines)
    ys, xs = np.nonzero(erased > 0)
    if len(xs) < MIN_ARC_PIXELS:
        return None
    points = np.column_stack([xs, ys]).astype(np.float32)

    # Thin the candidates so RANSAC samples spread out rather than clustering
    # in whichever blob happens to be densest.
    if len(points) > 4000:
        points = points[rng.choice(len(points), 4000, replace=False)]

    best = None
    for _ in range(RANSAC_ITERATIONS):
        sample = points[rng.choice(len(points), 6, replace=False)]
        try:
            candidate = cv2.fitEllipse(sample)
        except cv2.error:
            continue
        (_, _), (major, minor), _ = candidate
        if min(major, minor) / 2.0 < MIN_ELLIPSE_AXIS_PX:
            continue
        if max(major, minor) / max(min(major, minor), 1e-6) > \
                MAX_ELLIPSE_AXIS_RATIO:
            continue

        distances = _ellipse_distances(points, candidate)
        inliers = distances <= RANSAC_INLIER_PX
        if inliers.sum() < MIN_ARC_PIXELS:
            continue
        if best is None or inliers.sum() > best[0]:
            best = (int(inliers.sum()), candidate, points[inliers])

    if best is None:
        return None

    # Refit on the inliers, then judge the refit.
    support = best[2]
    if len(support) < 5:
        return None
    ellipse = cv2.fitEllipse(support)
    residual = _ellipse_residual(support, ellipse)
    if residual > MAX_ELLIPSE_RESIDUAL_PX:
        return None
    span = _arc_span_deg(support, ellipse)
    if span < MIN_ARC_SPAN_DEG:
        return None

    return dict(ellipse=ellipse, residual_px=residual,
                pixels=int(len(support)), span_deg=span, segments=segments)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--out", default="/tmp/centre_circle")
    args = ap.parse_args()

    print("Marking pixels, minus every straight line, fitted as an ellipse.\n")
    print(f"  {'clip':>14s} {'frames':>7s} {'found':>7s} {'rate':>6s} "
          f"{'residual':>9s} {'axes px':>16s}")

    for name, out_dir in CLIPS:
        info_path = Path(out_dir) / "clip.json"
        if not info_path.exists():
            continue
        info = json.loads(info_path.read_text())
        cap = cv2.VideoCapture(info["path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        hits, residuals, axes, spans, saved = 0, [], [], [], 0
        for idx in np.linspace(0, total - 1, args.frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            found = find_circle(frame)
            if found is None:
                continue
            hits += 1
            residuals.append(found["residual_px"])
            spans.append(found["span_deg"])
            (_, _), (major, minor), _ = found["ellipse"]
            axes.append((major / 2.0, minor / 2.0))

            if args.save and saved < 3:
                canvas = frame.copy()
                cv2.ellipse(canvas, found["ellipse"], (0, 0, 255), 2)
                for x1, y1, x2, y2 in found["segments"]:
                    cv2.line(canvas, (x1, y1), (x2, y2), (0, 200, 0), 1)
                Path(args.out).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(args.out) /
                                f"{out_dir}_f{int(idx)}.png"), canvas)
                saved += 1

        cap.release()
        if not hits:
            print(f"  {name:>14s} {args.frames:7d} {0:7d} {0:6.0%}")
            continue
        mean_axes = np.mean(axes, axis=0)
        print(f"  {name:>14s} {args.frames:7d} {hits:7d} "
              f"{hits / args.frames:6.0%} {np.median(residuals):9.2f} "
              f"{mean_axes[0]:7.0f} x{mean_axes[1]:6.0f}"
              f"  span {np.median(spans):3.0f} deg")

    print("\n  A centre circle is visible in a minority of frames, so a high "
          "rate here\n  would mean something else is being fitted. The "
          "residual is the check that\n  what was found is actually an "
          "ellipse.")


if __name__ == "__main__":
    main()
