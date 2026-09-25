"""Are the penalty-D anchors at the wrong end, or merely less precise?

Switching the D anchors on moves the frame-to-frame disagreement from 0.6 m
to 0.9 m and the share of pairs over 5 m from 4% to 15%. That is far better
than the version that read the end off the arc's bulge alone, which reached
2.7 m and 114 m at worst, but it is still worse than not having them.

Two quite different things could be behind it, and they call for opposite
decisions.

**Weaker anchors.** A D anchor scores 2.0 m against the markings where a
centre-circle anchor scores 0.6 m. It sees less of the pitch, and its scale
comes off an arc covering a hundred degrees rather than most of a circle. A
pair involving one should disagree more, and that is a cost to weigh against
the coverage, not a bug.

**Wrong ends.** If the camera's pan and the arc's bulge agree on a frame and
are both wrong, the anchor lands 83 m away and takes any pair it is in with
it. That is not a cost to weigh; it is the failure the whole exercise exists
to avoid.

The two are easy to tell apart, because they live at different scales.
Splitting the pairs by whether a D anchor is involved, and looking at the
worst case rather than the median, says which is happening: a spread of a
metre or two is imprecision, and anything near 83 m is an end.

    python check_penalty_end.py [--frames 40]
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

# What a wrong end costs: the two penalty spots are 83 m apart.
WRONG_END_M = 83.0

# Anything past this is not imprecision by any reading.
BLUNDER_M = 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("Pairs of anchors, split by whether a penalty-D anchor is "
          "involved.\nImprecision shows up as metres; a wrong end shows up "
          "as tens of them.\n")
    print(f"  {'clip':>14s} {'circle':>7s} {'D':>4s} "
          f"{'circle-circle pairs':>21s} {'pairs with a D':>20s}")

    pooled = {"plain": [], "mixed": []}
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info, every = anchored_frames(path, args.frames, rng,
                                      use_penalty_arc=True, with_kind=True)
        if len(every) < 2:
            continue
        # Tagged where it was built. Running this twice and diffing does not
        # work -- the circle detector's RANSAC draws from `rng`, so the two
        # runs find different frames and the difference is mostly that.
        from_circle = {index for index, _, kind in every if kind == "circle"}
        maps = {index: homography for index, homography, _ in every}

        cap = cv2.VideoCapture(info["path"])
        images = {}
        for index, _, _ in every:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                images[index] = frame
        cap.release()

        keys = sorted(images)
        plain, mixed = [], []
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if b - a > MAX_PAIR_GAP:
                    continue
                warp, _ = frame_to_frame(images[a], images[b])
                if warp is None:
                    continue
                value = disagreement(maps[a], maps[b], warp,
                                     info["width"], info["height"])
                if not np.isfinite(value):
                    continue
                if a in from_circle and b in from_circle:
                    plain.append(value)
                else:
                    mixed.append(value)

        def describe(values):
            if not values:
                return f"{'-':>21s}"
            array = np.array(values)
            return (f"{array.size:3d} med {np.median(array):5.1f} "
                    f"worst {array.max():6.1f}")

        print(f"  {name:>14s} {len(from_circle):7d} "
              f"{len(every) - len(from_circle):4d} "
              f"{describe(plain):>21s} {describe(mixed):>20s}")
        pooled["plain"] += plain
        pooled["mixed"] += mixed

    for label, values in pooled.items():
        if not values:
            continue
        array = np.array(values)
        print(f"\n  {label}: {array.size} pairs, median "
              f"{np.median(array):.1f} m, worst {array.max():.1f} m, "
              f"{np.mean(array > BLUNDER_M):.0%} past {BLUNDER_M:.0f} m")

    if pooled["mixed"]:
        array = np.array(pooled["mixed"])
        near_end = np.mean(np.abs(array - WRONG_END_M) < 25.0)
        print(f"\n  {near_end:.0%} of the pairs involving a D sit within "
              f"25 m of {WRONG_END_M:.0f} m,\n  which is what a wrong end "
              f"costs. If that is near zero, the ends are right\n  and what "
              f"is left is the D anchors simply being less precise.")


if __name__ == "__main__":
    main()
