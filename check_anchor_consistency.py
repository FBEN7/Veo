"""Do two anchors agree with each other? A check that touches no markings.

Every accuracy number for the anchor so far is a distance from a mapped
marking to the nearest real pitch line. That metric has a weakness which has
been called out but not removed: the refinement is fitted to markings and
scored on markings, and holding half of them out only partly fixes it,
because two segments on the same touchline are not independent.

This check has nothing to do with markings. The pitch does not move. So if
frame a and frame b are related by the warp W that maps a's pixels onto b's,
and both frames carry a correct anchor, then

    H_b  ==  H_a W^-1

must hold -- not approximately for one point but everywhere in the overlap.
W is measured by matching features between the two images, so it knows
nothing about pitch coordinates and cannot be nudged by either anchor.

The disagreement is then reported where it matters: in metres on the ground,
at points spread across the part of the frame the pitch occupies.

## What it can and cannot catch

It catches error that differs between the two frames -- a mis-centred
ellipse, a rotation taken from a badly located halfway line, a circle
confused with the penalty arc. Those move one anchor and not the other, and
show up directly.

It cannot catch error the two share. If the scale is 3% wrong on every frame
of a clip, both anchors are wrong by the same 3% and they agree perfectly.
So this is a lower bound on error, and the marking distance -- which does
have an absolute reference, the real pitch -- remains the measure of record.
The two fail in different directions, which is the reason for having both.

## What it found

    clip            pairs   disagreement   worst   over 5 m
    SoccerNet w1       31           1.1m   80.6m        26%
    SoccerNet w2       10           5.7m   53.8m        50%
    SoccerNet w3       12           1.2m   36.5m        17%
    reading            17           0.4m   46.9m        29%
    Veo                 2          12.3m   20.9m        50%

The median is reassuring and beside the point. What matters is that a
quarter to a half of all pairs disagree by more than five metres, with worst
cases of twenty, fifty and eighty -- and those frames do not look wrong. The
median marking error on this same anchor is one to two metres, so the tail is
invisible to the measure the project had been using.

Splitting by how far apart the two frames are:

    gap (frames)  pairs   disagreement   over 5 m
    25-60            41           1.3m        27%
    60-150           31           2.1m        32%

Nearly flat, which locates the blame. If the warp between the frames were
the problem it would climb steeply with the gap, and it does not. These are
the anchors disagreeing, not the propagation between them.

    python check_anchor_consistency.py [--frames 40]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from fit_pitch_anchor import anchored_frames
from probe_pitch_lines import CLIPS, line_segments
from propagate_anchor import frame_to_frame
from refine_anchor import MIN_FIT_SEGMENTS, refine

# Pairs further apart than this are not compared: the views stop overlapping
# and the warp between them is fitted on too little.
MAX_PAIR_GAP = 150

# Disagreement is reported by how far apart the two frames are, because it
# has two sources and they separate this way. Anchor error is the same
# whatever the gap; warp error grows with it, as the two views share less.
# A figure that is flat across these bands is anchor error; one that climbs
# is the warp, and is a limit on propagation rather than on the anchor.
GAP_BANDS = ((0, 25), (25, 60), (60, 150))

# Where on the image the disagreement is measured. The top of a football
# frame is stands and sky, which is not on the ground plane and where a
# ground-plane map means nothing.
SAMPLE_ROWS = (0.55, 0.70, 0.85, 0.97)
SAMPLE_COLS = (0.15, 0.35, 0.65, 0.85)


def disagreement(homography_a, homography_b, warp, width, height):
    """Median distance, in metres, between two anchors' idea of the ground."""
    points = np.array([[c * width for c in SAMPLE_COLS for _ in SAMPLE_ROWS],
                       [r * height for _ in SAMPLE_COLS for r in SAMPLE_ROWS],
                       [1.0] * (len(SAMPLE_COLS) * len(SAMPLE_ROWS))])
    try:
        carried = homography_a @ np.linalg.inv(warp)
    except np.linalg.LinAlgError:
        return float("nan")
    here, there = homography_b @ points, carried @ points
    if np.any(np.abs(here[2]) < 1e-9) or np.any(np.abs(there[2]) < 1e-9):
        return float("nan")
    here, there = here[:2] / here[2], there[:2] / there[2]
    gaps = np.hypot(*(here - there))
    gaps = gaps[np.isfinite(gaps)]
    return float(np.median(gaps)) if gaps.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("Two anchored frames, related by a warp measured from image "
          "features alone.\nIf both anchors are right they must describe the "
          "same ground.\n")
    print(f"  {'clip':>14s} {'pairs':>6s} {'circle only':>12s} "
          f"{'refined':>9s} {'worst raw':>10s} {'over 5 m':>9s}")

    pooled_raw, pooled_ref = [], []
    banded = {band: [] for band in GAP_BANDS}
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info, anchors = anchored_frames(path, args.frames, rng)
        if len(anchors) < 2:
            continue

        cap = cv2.VideoCapture(info["path"])
        frames, raw, refined = {}, {}, {}
        for idx, homography in anchors:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frames[idx] = frame
            raw[idx] = homography
            segments, _ = line_segments(frame)
            if len(segments) >= MIN_FIT_SEGMENTS:
                refined[idx], _ = refine(homography, segments, info)
            else:
                refined[idx] = homography
        cap.release()

        keys = sorted(frames)
        raw_gaps, ref_gaps = [], []
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if b - a > MAX_PAIR_GAP:
                    continue
                warp, _ = frame_to_frame(frames[a], frames[b])
                if warp is None:
                    continue
                one = disagreement(raw[a], raw[b], warp,
                                   info["width"], info["height"])
                two = disagreement(refined[a], refined[b], warp,
                                   info["width"], info["height"])
                if np.isfinite(one) and np.isfinite(two):
                    raw_gaps.append(one)
                    ref_gaps.append(two)
                    for band in GAP_BANDS:
                        if band[0] <= b - a < band[1]:
                            banded[band].append(one)

        if not raw_gaps:
            print(f"  {name:>14s} {0:6d}   no overlapping pairs")
            continue
        bad = float(np.mean(np.array(raw_gaps) > 5.0))
        print(f"  {name:>14s} {len(raw_gaps):6d} {np.median(raw_gaps):11.1f}m "
              f"{np.median(ref_gaps):8.1f}m {max(raw_gaps):9.1f}m "
              f"{bad:8.0%}")
        pooled_raw += raw_gaps
        pooled_ref += ref_gaps

    if pooled_raw:
        print(f"\n  {'pooled':>14s} {len(pooled_raw):6d} "
              f"{np.median(pooled_raw):11.1f}m {np.median(pooled_ref):8.1f}m")
        better = np.mean(np.array(pooled_ref) < np.array(pooled_raw))
        print(f"\n  The refinement brings two anchors into closer agreement "
              f"on {better:.0%} of pairs.")

    print(f"\n  {'gap (frames)':>14s} {'pairs':>6s} {'disagreement':>13s} "
          f"{'over 5 m':>9s}")
    for band in GAP_BANDS:
        values = np.array(banded[band])
        if not values.size:
            continue
        print(f"  {f'{band[0]}-{band[1]}':>14s} {values.size:6d} "
              f"{np.median(values):12.1f}m "
              f"{np.mean(values > 5.0):8.0%}")
    print("\n  Flat across the bands means the disagreement is the anchors; "
          "climbing\n  means it is the warp between them.")

    print("\n  Shared error cancels here, so this is a lower bound. A clip "
          "that agrees\n  with itself to half a metre may still sit a metre "
          "off the real pitch.")


if __name__ == "__main__":
    main()
