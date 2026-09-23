"""Which way round is the pitch? The anchor has been deciding by coin toss.

A circle looks the same from every angle, so `metric_from_circle` takes the
rotation from the halfway line. That fixes it to within 180 degrees and no
further, because the halfway line looks the same from both ends too. The code
resolves the remainder with

    angle = arctan2(mapped[0], mapped[1])

on the line's direction vector -- and that vector is built from a detected
segment's two endpoints, in the order the detector happened to list them.
`HoughLinesP` gives no guarantee about that order. Reverse it and the angle
moves by 180 degrees, and the whole pitch turns end for end.

So the orientation of the map has been decided, frame by frame, by which end
of a line segment came first in an array.

## Why nothing caught it

Because a football pitch is exactly symmetric under that rotation. The lines
across it sit at 0, 5.5, 16.5, 52.5, 88.5, 99.5 and 105 m, and subtracting
each from 105 gives the same seven numbers. The lines along it sit at 0,
13.84, 24.84, 43.16, 54.16 and 68, and subtracting each from 68 gives those
six. A flipped anchor puts every marking onto a real pitch line -- the wrong
one, but a real one -- so `marking_error` scores it *identically*.

Every accuracy figure in this project is therefore blind to the flip. It is
also the one error that matters most for what the pipeline is for: a shot
flipped end for end is attributed to the other goal, and a shot from six
yards becomes a shot from ninety-nine.

What is not blind to it is comparing two frames to each other, and that is
where the 20, 50 and 80 metre disagreements in `check_anchor_consistency.py`
were always likely to be coming from.

## What can and cannot be fixed

The absolute orientation cannot be recovered from markings, by the same
symmetry -- there is genuinely no way to tell one end of a bare pitch from
the other. It does not need to be. What matters is that every frame of a
clip agrees, so that a shot at one end is always attributed to that end.

That is fixable, because the camera does not orbit the pitch. It sits on one
side, so the near touchline stays near. Choosing the orientation in which
moving DOWN the image moves toward the same touchline pins the flip to
something physical and frame-independent, rather than to an array order.

## Measured, before the fix

    clip            pairs   as is   turned   disagree   better turned
    SoccerNet w1       31    1.1m    32.1m        26%             13%
    SoccerNet w2       10    5.7m    43.0m        50%              0%
    SoccerNet w3       12    1.2m    35.6m        17%              0%
    reading            17    0.4m    25.0m        29%             18%
    Veo                 2   12.3m    62.0m        50%              0%
    pooled             72    1.3m    31.8m        29%             10%

    pairs that disagree about the orientation   median 34.5 m apart
    pairs that agree about it                   median  1.1 m apart

One pair in ten was flipped relative to its partner, and those pairs sit
**34.5 m apart against 1.1 m** for the pairs that agree. That is the whole
of the extreme tail the consistency check reported -- the 20, 50 and 80
metre disagreements were not a wobbly anchor, they were two frames that had
tossed the coin differently.

`turned` is the control and behaves: deliberately flipping one anchor of a
pair takes the median disagreement from 1.3 m to 31.8 m, which is the size
of the error this bug produces when it fires.

## The fix, and what it does not fix

`pitch_model.metric_from_circle` now settles the orientation by requiring
that moving down the image moves toward increasing y on the pitch. The
camera sits on one side of the pitch and stays there, so the near touchline
is always the near one, and that is the same choice on every frame.

It does not recover the absolute orientation, and nothing can from markings
alone -- a pitch is symmetric, so there is no telling one end from the other
without something outside the geometry. Which goal is which still has to
come from elsewhere: the direction of play, the team in possession, or
simply being told. What is fixed is that a clip no longer contradicts
itself, so a shot at one end is attributed to that end throughout.

    python check_orientation.py [--frames 40]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from check_anchor_consistency import MAX_PAIR_GAP, disagreement
from fit_pitch_anchor import anchored_frames
from probe_pitch_lines import CLIPS
from propagate_anchor import frame_to_frame
from src import pitch_model as pm


def turned() -> np.ndarray:
    """The pitch, end for end: (x, y) -> (105 - x, 68 - y)."""
    out = np.eye(3)
    out[:2, :2] = -np.eye(2)
    out[0, 2] = pm.PITCH_LENGTH_M
    out[1, 2] = pm.PITCH_WIDTH_M
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("For each pair of anchored frames: do they agree about which way "
          "round\nthe pitch is? Turning one of them end for end should make "
          "things worse.\n")
    print(f"  {'clip':>14s} {'pairs':>6s} {'as is':>7s} {'turned':>7s} "
          f"{'disagree':>9s} {'better turned':>14s}")

    pooled_plain, pooled_turn, pooled_flipped = [], [], []
    flip = turned()
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info, anchors = anchored_frames(path, args.frames, rng)
        if len(anchors) < 2:
            continue

        cap = cv2.VideoCapture(info["path"])
        images = {}
        for index, _ in anchors:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                images[index] = frame
        cap.release()

        maps = dict(anchors)
        keys = sorted(images)
        plain, turn, flipped = [], [], []
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if b - a > MAX_PAIR_GAP:
                    continue
                warp, _ = frame_to_frame(images[a], images[b])
                if warp is None:
                    continue
                one = disagreement(maps[a], maps[b], warp,
                                   info["width"], info["height"])
                two = disagreement(flip @ maps[a], maps[b], warp,
                                   info["width"], info["height"])
                if np.isfinite(one) and np.isfinite(two):
                    plain.append(one)
                    turn.append(two)
                    # If turning one of them HELPS, the two frames were
                    # already disagreeing about which way round the pitch is.
                    flipped.append(two < one)

        if not plain:
            continue
        print(f"  {name:>14s} {len(plain):6d} {np.median(plain):6.1f}m "
              f"{np.median(turn):6.1f}m "
              f"{np.mean(np.array(plain) > 5.0):8.0%} "
              f"{np.mean(flipped):13.0%}")
        pooled_plain += plain
        pooled_turn += turn
        pooled_flipped += flipped

    if pooled_plain:
        plain = np.array(pooled_plain)
        flipped = np.array(pooled_flipped)
        print(f"\n  {'pooled':>14s} {len(plain):6d} {np.median(plain):6.1f}m "
              f"{np.median(pooled_turn):6.1f}m "
              f"{np.mean(plain > 5.0):8.0%} {np.mean(flipped):13.0%}")
        if flipped.any():
            print(f"\n  Pairs that disagree about the orientation: median "
                  f"{np.median(plain[flipped]):.1f} m apart.")
            print(f"  Pairs that agree about it:                   median "
                  f"{np.median(plain[~flipped]):.1f} m apart.")

    print("\n  Two frames of a fixed pitch cannot disagree about which way "
          "round it is.\n  Every pair that does is a bug, and the metre "
          "figure it produces is the\n  size of that bug rather than of any "
          "measurement error.")


if __name__ == "__main__":
    main()
