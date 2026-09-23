"""How much pitch geometry is visible, and is it enough to anchor on?

Shots, goals, assists, chances created and xG all need to know where the goal
is. That means anchoring the ground plane to the pitch, which means finding
markings whose real dimensions are known -- the penalty area is 16.5 by 40.32
metres, the six-yard box 5.5 by 18.32, and the goal mouth 7.32 across.

`probe_homography.py` measured markings at 1.1 to 3.4 percent of the visible
pitch and concluded they were too thin. That measurement was taken over
frames sampled uniformly through the clip, which is the wrong sample for this
question: the camera spends most of a match on midfield, where there is
nothing but a halfway line, and swings to the box only when play does. Shots
happen in the box. So the question is not how much geometry is visible on
average, it is how much is visible *when it matters*.

This measures that. Line segments are found on the pitch surface only, and
reported per frame along with how many distinct orientations they span --
because one line is a constraint and two crossing lines are a corner, and a
corner is what anchoring needs.

    python probe_pitch_lines.py [--save-frames]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.detect_track_hybrid import _grass_mask

CLIPS = (("SoccerNet w1", "output_soccernet"),
         ("SoccerNet w2", "output_soccernet_w2"),
         ("SoccerNet w3", "output_soccernet_w3"),
         ("reading", "output_soccernet_reading"),
         ("Veo", "output_veo"))

SAMPLES = 40

# Pitch markings are brighter than the grass around them and only a few
# pixels wide. A white top-hat keeps exactly that: thin bright structure,
# while rejecting the broad bright regions -- mown stripes, sunlit grass --
# that a plain threshold picks up.
TOPHAT_KERNEL = 11
LINE_MIN_BRIGHTNESS = 22

# A segment shorter than this is a scuff, a boot, or part of a player.
MIN_SEGMENT_PX = 55

# Two segments count as differently oriented when they differ by more than
# this. A corner needs two orientations; a single touchline gives one.
ORIENTATION_TOLERANCE_DEG = 25.0

# Erode the grass inward so the touchline itself, where grass meets
# advertising hoardings, is not mistaken for a marking.
#
# 13 was not enough and the error was specific: the hoardings at these
# grounds are *green* -- bet365 boards at Stoke, ajgroup at Reading -- so
# they survive a grass mask and connect to the pitch through the touchline,
# carrying their lettering into the largest component. Pulling 35 pixels
# inside the boundary drops segments per frame from 60/55/59/19 to
# 44/37/46/12 across the four broadcast clips. Most of that difference was
# advertising.
GRASS_ERODE = 35


def pitch_surface(frame: np.ndarray) -> np.ndarray:
    """The playing surface: the largest connected region of grass, pulled in.

    Taking the largest component matters. "Anything green" also selects the
    grass banks behind the goal and the green of the hoardings, and an
    earlier version of this measurement reported 28 segments per frame that
    were sitting on roofs and advertising boards.
    """
    grass = _grass_mask(frame)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (grass > 0).astype(np.uint8), 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        grass = ((labels == biggest) * 255).astype(np.uint8)
    return cv2.erode(grass, np.ones((GRASS_ERODE, GRASS_ERODE), np.uint8), 1)


def line_segments(frame: np.ndarray):
    """Marking segments lying on the playing surface."""
    surface = pitch_surface(frame)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(
        grey, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT,
                                  (TOPHAT_KERNEL, TOPHAT_KERNEL)))
    bright = (tophat > LINE_MIN_BRIGHTNESS).astype(np.uint8) * 255
    bright = cv2.bitwise_and(bright, surface)

    found = cv2.HoughLinesP(bright, 1, np.pi / 180, threshold=45,
                            minLineLength=MIN_SEGMENT_PX, maxLineGap=12)
    # HoughLinesP returns (N, 1, 4) on some OpenCV builds and (N, 4) on
    # others. Reshaping covers both rather than indexing one of them.
    segments = ([] if found is None
                else [tuple(int(v) for v in s)
                      for s in np.asarray(found).reshape(-1, 4)])
    return segments, surface


def orientations(segments) -> int:
    """How many distinct directions the segments span."""
    angles = []
    for x1, y1, x2, y2 in segments:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        if not any(min(abs(angle - a), 180 - abs(angle - a))
                   < ORIENTATION_TOLERANCE_DEG for a in angles):
            angles.append(angle)
    return len(angles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--out", default="/tmp/pitch_lines")
    args = ap.parse_args()

    print("Markings found on the playing surface, sampled across each clip.\n")
    print(f"  {'clip':>14s} {'frames':>7s} {'segments/frame':>15s} "
          f"{'>=1 seg':>8s} {'>=2 orient':>11s}  (a corner needs two)")

    for name, out_dir in CLIPS:
        info = Path(out_dir) / "clip.json"
        if not info.exists():
            continue
        path = json.loads(info.read_text())["path"]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        picks = np.linspace(0, total - 1, SAMPLES).astype(int)

        counts, multi, any_seg = [], 0, 0
        best = (-1, None, None, None)
        for idx in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            segments, _ = line_segments(frame)
            counts.append(len(segments))
            any_seg += len(segments) >= 1
            n_orient = orientations(segments)
            multi += n_orient >= 2
            # Deliberately not the frame with the most segments. That was
            # the first choice here and it is the worst one: the frame with
            # the most segments is the frame where the mask leaked furthest,
            # so inspecting it shows the detector at its least typical.
            if best[1] is None and 0.1 < idx / max(total - 1, 1) < 0.2:
                best = (len(segments), frame, segments, int(idx))

        cap.release()
        if not counts:
            continue
        n = len(counts)
        print(f"  {name:>14s} {n:7d} {np.mean(counts):15.1f} "
              f"{any_seg / n:8.0%} {multi / n:11.0%}")

        if args.save_frames and best[1] is not None:
            canvas = best[1].copy()
            for x1, y1, x2, y2 in best[2]:
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            Path(args.out).mkdir(parents=True, exist_ok=True)
            dest = Path(args.out) / f"{out_dir}_f{best[3]}.png"
            cv2.imwrite(str(dest), canvas)

    print("\n  '>=2 orient' is the column that matters. One line constrains "
          "the plane\n  partially; two crossing lines give a corner, which "
          "is what a pitch model\n  can be fitted against.")


if __name__ == "__main__":
    main()
