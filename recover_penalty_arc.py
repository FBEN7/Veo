"""Anchor on the penalty D, with the end of the pitch settled from outside.

Raising the arc-span gate to 200 degrees fixed the tail and cost half the
anchors -- on Veo footage, coverage fell from 68% of frames within reach of
an anchor to 25%. The discarded frames are not bad frames. They are frames
looking at the penalty area, which is where shots are.

A D-anchored frame is wrong by exactly 41.5 m, the centre spot at 52.5
against the penalty spot at 11. Everything else about it is right: both arcs
are struck at 9.15 m from the same measurement in the laws of the game, so
the scale is right, and the chord -- the penalty-area line that cut the
circle, lying 5.5 m from its centre -- identifies the arc as a D and fixes
the rotation, since nothing crosses a D's centre the way the halfway line
crosses the circle's.

## The one bit that the pitch cannot supply

Which of the two penalty spots. The markings are symmetric about the halfway
line, so a map placing the arc at x = 11 and one placing it at x = 94 put
every visible marking onto a real pitch line and score *identically*.

That is not a theoretical worry. An earlier version of this decided the end
from the arc's bulge alone and was checked against the markings, where it
improved 94% of the frames it touched. Switched on, it put two anchors on
the same clip 114 m apart. The 94% had confirmed the size of the correction
and was blind to its sign, which was the whole question -- the same
blindness that hid the orientation flip for the life of this project.

## Two signs, and they have to agree

**Where the camera is looking.** It follows play, so its accumulated pan
says which half of the pitch is in view, measured against the frames whose
centre circle pins them to the middle. This asks very little of the
measurement: the two candidate spots are 41.5 m apart, and the pan drifts
about 4 m over a clip. Ten times the margin, for one bit.

**Which way the arc bulges.** The D is the part of its circle lying outside
the penalty area, so it swells away from its goal. With the rotation pinned
by the camera sitting on one side of the pitch, that direction names the end
too.

They are independent -- one is about the clip, the other about the frame --
so where they agree the answer is decided rather than guessed, and where
they disagree the frame is refused. Refusing is the point. A D anchor at the
wrong end is a shot at the wrong end of the pitch, and nothing about it
looks wrong.

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
from fit_pitch_anchor import (anchored_frames, penalty_arc_candidate,
                              resolve_end)
from probe_centre_circle import find_circle
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

    print("Penalty-D frames, with the end of the pitch settled by two "
          "independent\nsigns: where the camera is looking, and which way "
          "the arc bulges.\n")
    print(f"  {'clip':>14s} {'D found':>8s} {'agreed':>7s} {'accepted':>9s} "
          f"{'error':>7s} {'chance':>7s}")

    pooled_error, pooled_floor, agreed_all = [], [], []
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info = json.loads((path / "clip.json").read_text())
        motion_path = path / "camera_motion.npy"
        motion = np.load(motion_path) if motion_path.exists() else None

        # The centre-circle anchors are the reference the pan is measured
        # against, so they have to exist before any D can be placed.
        _, references = anchored_frames(path, args.frames, rng,
                                        use_penalty_arc=False)
        references = dict(references)

        cap = cv2.VideoCapture(info["path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        found, agreed, errors, floors = 0, 0, [], []
        for index in np.linspace(0, total - 1, args.frames).astype(int):
            index = int(index)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            if find_circle(frame, rng) is not None:
                continue                      # a real centre circle, not a D
            candidate = penalty_arc_candidate(frame, rng, info)
            if candidate is None:
                continue
            found += 1
            settled = resolve_end(candidate[0], candidate[1], index,
                                  references, motion, info)
            if settled is None or not plausible_anchor(settled, info):
                continue
            agreed += 1

            segments, _ = line_segments(frame)
            if len(segments) < 3:
                continue
            error = marking_error(settled, segments, info)
            if not np.isfinite(error):
                continue
            errors.append(error)
            chance = []
            for _ in range(CHANCE_TRIALS):
                displaced = np.eye(3)
                displaced[:2, 2] = rng.uniform(-CHANCE_OFFSET_M,
                                               CHANCE_OFFSET_M, 2)
                value = marking_error(displaced @ settled, segments, info)
                if np.isfinite(value):
                    chance.append(value)
            if chance:
                floors.append(float(np.median(chance)))
        cap.release()

        if not found:
            print(f"  {name:>14s} {0:8d}")
            continue
        print(f"  {name:>14s} {found:8d} {agreed / found:6.0%} "
              f"{agreed:9d} "
              f"{(np.median(errors) if errors else np.nan):6.1f}m "
              f"{(np.median(floors) if floors else np.nan):6.1f}m")
        pooled_error += errors
        pooled_floor += floors
        agreed_all.append((found, agreed))

    if pooled_error:
        found = sum(f for f, _ in agreed_all)
        agreed = sum(a for _, a in agreed_all)
        print(f"\n  {'pooled':>14s} {found:8d} {agreed / found:6.0%} "
              f"{agreed:9d} {np.median(pooled_error):6.1f}m "
              f"{np.median(pooled_floor):6.1f}m")

    print("\n  The two signs are independent, so the frames where they "
          "disagree are the\n  ones that would have been guesses. Refusing "
          "them is the point: a D anchor\n  placed at the wrong end is a "
          "shot at the wrong end, and it looks fine.")


if __name__ == "__main__":
    main()
