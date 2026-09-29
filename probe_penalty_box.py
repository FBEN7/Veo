"""Are the penalty-area markings even there, in the frames that matter?

Shot recall is 0 of 10 and goal detection 0 of 2, and the cause is settled:
the anchor is fitted from the centre circle, which is at the halfway line,
and a camera following play into the box does not show it. The proposed fix
is an anchor fitted from the markings that *are* in the box -- the goal
line, the six-yard box, the penalty area, and the sides joining them.

Before writing a fitter, a cheaper question: are those markings detectable
in the frames where the anchor fails? If they are not, the fitter is
pointless and the honest answer is that broadcast footage of the box cannot
be mapped at all.

## What "enough" means

A homography needs four point correspondences. On a pitch the lines come in
two families -- parallel to the goal line, and perpendicular to it -- and
two lines from each family give four intersections. So a frame is usable
when it shows at least two lines of each family.

That is the bar this measures. It deliberately does not ask which line is
which; identifying them is the fitter's job and a harder one. This asks only
whether there is anything to identify.

## Why the families are found by direction and not by position

Perspective makes parallel lines converge, so their image angles differ.
They still cluster: the spread within a family is far smaller than the
angle between families, because the camera looks along the pitch rather
than across it. Clustering on angle is crude and sufficient for a triage --
a frame that passes may still defeat a fitter, but a frame that fails has
nothing for one to work with.

    python probe_penalty_box.py [--frames 40] [--clip 1302]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from probe_pitch_lines import line_segments

# Windows cut around labelled shots: the frames where the anchor fails.
SHOT_WINDOWS = (
    ("Stoke 13:02", "output_stoke_1302"),
    ("Stoke 42:07", "output_stoke_4207"),
    ("Stoke 70:01", "output_stoke_7001"),
    ("Reading 07:37", "output_reading_0737"),
    ("Reading 11:55", "output_reading_1155"),
    ("Reading 25:19", "output_reading_2519"),
)

# A segment shorter than this is noise, a stud mark or part of a player.
MIN_LENGTH_PX = 45.0

# Two segments are the same line if their directions agree this closely and
# each lies this near the other's infinite line.
SAME_ANGLE_DEG = 6.0
SAME_OFFSET_PX = 18.0

# Two lines are usefully transverse if they meet at more than this angle.
# Well below 90 because perspective shears everything.
CROSS_ANGLE_DEG = 25.0


def direction(segment):
    x1, y1, x2, y2 = segment
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
    return angle


def length(segment):
    x1, y1, x2, y2 = segment
    return float(np.hypot(x2 - x1, y2 - y1))


def angle_gap(a: float, b: float) -> float:
    """Smallest angle between two undirected directions, in degrees."""
    gap = abs(a - b) % 180.0
    return min(gap, 180.0 - gap)


def merge_segments(segments):
    """Collapse collinear segments into lines, longest first.

    A painted line arrives as a handful of separate detections -- players
    stand on it, the mower leaves it patchy -- and counting those as
    separate lines would make any frame look rich in markings.
    """
    kept = []
    for segment in sorted(segments, key=length, reverse=True):
        if length(segment) < MIN_LENGTH_PX:
            continue
        x1, y1, x2, y2 = segment
        theta = direction(segment)
        mid = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        merged = False
        for other in kept:
            if angle_gap(theta, other["angle"]) > SAME_ANGLE_DEG:
                continue
            # Distance from this segment's midpoint to the other's line.
            ox, oy = other["point"]
            dx, dy = np.cos(np.radians(other["angle"])), np.sin(
                np.radians(other["angle"]))
            away = abs((mid[0] - ox) * dy - (mid[1] - oy) * dx)
            if away <= SAME_OFFSET_PX:
                other["length"] += length(segment)
                merged = True
                break
        if not merged:
            kept.append({"angle": theta, "point": mid,
                         "length": length(segment), "segment": segment})
    return kept


def families(lines):
    """Split lines into two transverse groups, or return None.

    The split is taken at the widest gap in the sorted angles rather than by
    clustering to a fixed k: a frame showing only one family should come
    back as one family, not be forced into two.
    """
    if len(lines) < 2:
        return None
    ordered = sorted(lines, key=lambda l: l["angle"])
    angles = [l["angle"] for l in ordered]
    # Gaps around the circle of directions, including the wrap.
    gaps = [(angles[i + 1] - angles[i], i) for i in range(len(angles) - 1)]
    gaps.append((angles[0] + 180.0 - angles[-1], len(angles) - 1))
    widest, at = max(gaps)
    if widest < CROSS_ANGLE_DEG:
        return None                      # everything points one way
    first = ordered[at + 1:] + ordered[:at + 1] if at + 1 < len(ordered) \
        else ordered
    # Rotate so the split is at the start, then cut at the second widest gap.
    rotated = ordered[at + 1:] + ordered[:at + 1]
    rot_angles = [l["angle"] for l in rotated]
    inner = [(angle_gap(rot_angles[i + 1], rot_angles[i]), i)
             for i in range(len(rot_angles) - 1)]
    if not inner:
        return None
    second, cut = max(inner)
    if second < CROSS_ANGLE_DEG:
        return None                      # one family only
    return rotated[:cut + 1], rotated[cut + 1:]


def look(frame):
    """What one frame offers a fitter."""
    segments, _ = line_segments(frame)
    lines = merge_segments(segments)
    split = families(lines)
    if split is None:
        return {"lines": len(lines), "across": len(lines), "along": 0,
                "usable": False}
    a, b = split
    return {"lines": len(lines), "across": max(len(a), len(b)),
            "along": min(len(a), len(b)),
            "usable": min(len(a), len(b)) >= 2 and max(len(a), len(b)) >= 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--clip", default=None)
    args = ap.parse_args()

    print("Lines found in the frames where the centre-circle anchor fails.\n"
          "A frame is 'usable' when two families of two lines meet, which "
          "is four\nintersections -- the minimum a homography needs.\n")
    print(f"  {'window':>14s} {'frames':>7s} {'median lines':>13s} "
          f"{'usable':>8s}")

    for name, out_dir in SHOT_WINDOWS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        if args.clip and args.clip not in out_dir:
            continue
        info = json.loads((path / "clip.json").read_text())
        cap = cv2.VideoCapture(info["path"])
        total = int(info["n_frames"])
        counts, usable = [], 0
        for idx in np.linspace(0, total - 1, args.frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            seen = look(frame)
            counts.append(seen["lines"])
            usable += int(seen["usable"])
        cap.release()
        if not counts:
            continue
        print(f"  {name[-14:]:>14s} {len(counts):7d} "
              f"{np.median(counts):13.0f} {usable / len(counts):8.0%}",
              flush=True)

    print("\n  A usable frame is not a solved frame: this counts lines "
          "without asking\n  which line is which, and telling the goal line "
          "from the six-yard line is\n  the harder half. But a frame that "
          "fails here has nothing for a fitter to\n  work with, whatever "
          "the fitter is.")


if __name__ == "__main__":
    main()
