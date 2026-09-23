"""Tell the centre circle from the penalty arc, and pick the threshold by measuring.

`diagnose_tail.py` found what the tail is. Of 24 anchored frames scoring
worse than 5 m, 21 come back to 2.0-2.6 m when the map is shifted by exactly
41.5 m along the pitch -- which is 52.5 minus 11, the centre spot minus the
penalty spot. Those frames had locked onto the D outside the penalty area
instead of the centre circle.

It is an easy mistake to make and a hard one to notice. Both arcs are struck
with a radius of 9.15 m, because both come from the same measurement in the
laws of the game, so the scale comes out right, the residual comes out right
and the ellipse is a perfectly good ellipse. Everything about the fit is
sound except which arc it is.

What separates them is how much of the circle there is. The centre circle is
a whole one. The D is only the part of its circle lying outside the penalty
area, and that is fixed by geometry: the penalty spot is 11 m from the goal
line and the penalty-area line is 16.5 m, so the chord sits 5.5 m from the
centre of a 9.15 m circle and the arc spans

    2 * acos(5.5 / 9.15) = 106 degrees

and no more, ever. Measured, the tail frames have a median span of 150
degrees against 240 for the rest -- the D plus whatever noise the detector
swept in, against a real centre circle.

So the gate is a minimum arc span, and it is already there: `find_circle`
requires 120 degrees. That was set to reject noise, and 120 is below 106
plus any reasonable slack, so the D goes straight through it.

Raising it costs anchors, because a centre circle at the edge of frame shows
less of itself. This sweeps the threshold and measures both sides of that
trade -- how many anchors survive, and what happens to the tail they were
polluting. The circles are detected once and re-filtered, so the sweep costs
one pass rather than one per threshold.

    python tune_circle_gate.py [--frames 60]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fit_pitch_anchor import halfway_line, marking_error, plausible_anchor
from probe_centre_circle import find_circle
from probe_pitch_lines import CLIPS, line_segments
from src import pitch_model as pm

# The largest arc the penalty D can possibly subtend, from the laws of the
# game: a 9.15 m radius struck from the penalty spot, cut by the penalty-area
# line 5.5 m away.
D_MAX_SPAN_DEG = 2.0 * np.degrees(np.arccos(5.5 / 9.15))

SPAN_THRESHOLDS = (120.0, 140.0, 160.0, 180.0, 200.0, 220.0)

TAIL_M = 5.0


def collect(out_dir: Path, n_frames: int, rng):
    """Detect once; every threshold is then a filter over the same circles."""
    info = json.loads((out_dir / "clip.json").read_text())
    cap = cv2.VideoCapture(info["path"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    found = []
    for index in np.linspace(0, total - 1, n_frames).astype(int):
        index = int(index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        circle = find_circle(frame, rng)
        if circle is None:
            continue
        (cx, cy), _, _ = circle["ellipse"]
        line = halfway_line(circle["segments"], (cx, cy))
        direction = (None if line is None
                     else np.array([line[2] - line[0], line[3] - line[1]],
                                   dtype=float))
        homography = pm.metric_from_circle(
            np.array([0.0, 0.0, 1.0]), circle["ellipse"], direction)
        if homography is None or not plausible_anchor(homography, info):
            continue
        segments, _ = line_segments(frame)
        if len(segments) < 3:
            continue
        error = marking_error(homography, segments, info)
        if not np.isfinite(error):
            continue
        found.append((circle["span_deg"], error))
    cap.release()
    return info, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print(f"The penalty D cannot span more than "
          f"{D_MAX_SPAN_DEG:.0f} degrees. The centre circle can\n"
          f"span anything up to 360. Where to put the line between them:\n")

    per_clip = {}
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        _, found = collect(path, args.frames, rng)
        if found:
            per_clip[name] = found

    everything = [row for rows in per_clip.values() for row in rows]
    if not everything:
        print("no anchors")
        return

    print(f"  {'min span':>9s} {'anchors kept':>13s} {'median':>7s} "
          f"{'p90':>6s} {'over 5 m':>9s}")
    for threshold in SPAN_THRESHOLDS:
        kept = [error for span, error in everything if span >= threshold]
        if not kept:
            print(f"  {threshold:8.0f}d {0:13d}")
            continue
        kept = np.array(kept)
        print(f"  {threshold:8.0f}d "
              f"{kept.size}/{len(everything):<9d} "
              f"{np.median(kept):6.1f}m {np.percentile(kept, 90):5.1f}m "
              f"{np.mean(kept > TAIL_M):8.0%}")

    print(f"\n  Per clip, at the current 120 degrees and at 200:\n")
    print(f"  {'clip':>14s} {'at 120':>18s} {'at 200':>18s}")
    for name, rows in per_clip.items():
        line = f"  {name:>14s}"
        for threshold in (120.0, 200.0):
            kept = np.array([e for s, e in rows if s >= threshold])
            if kept.size:
                line += (f" {kept.size:3d} anchors {np.median(kept):4.1f}m"
                         f" {np.mean(kept > TAIL_M):3.0%}")
            else:
                line += f" {0:3d} anchors     -    -"
        print(line)

    print("\n  Raising the threshold buys a smaller tail and costs anchors. "
          "The anchors\n  it costs are the ones that were wrong, so long as "
          "the median holds up.")


if __name__ == "__main__":
    main()
