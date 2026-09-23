"""Throw away the anchors that disagree with their neighbours.

The anchor's median error is not what stands between it and being usable.
That is 1 to 2 metres on every clip. The problem is the tail: the consistency
check found 27% to 50% of anchor pairs disagreeing by more than 5 metres, and
worst cases of 20, 50 and 80. A shot placed one metre out is a shot in
roughly the right part of the box. A shot placed fifty metres out is in the
other half, and it is worse than useless because nothing about it looks
wrong.

So the question is not how to make the median better. It is how to know,
without labels and without markings, which anchors to refuse.

Two anchors on different frames can be compared to each other. The pitch does
not move, so if the warp between two frames is measured from image features,
each anchor implies where the other's frame sits on the pitch, and they
either agree or they do not. A frame whose anchor contradicts its
neighbours is wrong -- not probably wrong, wrong, because they cannot both be
right about a pitch that stayed still.

This is worth more than a better average, because it is available at run
time. Marking distance needs the markings to be visible and identifiable,
which is the whole difficulty. Agreement between frames needs neither. It
works on a frame showing nothing but grass, and it works on the Veo clip,
which has the fewest markings of any footage here.

## Why the test is not circular

The gate uses only frame-to-frame agreement. The score is distance from
mapped markings to real pitch lines. Nothing crosses between them, so if the
anchors the gate refuses turn out to be the ones scoring badly on markings
it never looked at, that is a real result and not the measurement admiring
itself.

The floor to beat is that a gate throwing away anchors at random would also
improve the average of what remains, simply by removing some bad ones. So
the comparison is against refusing the same NUMBER of anchors at random,
measured rather than assumed.

## It does not work

    clip            judged  kept        kept        refused
    SoccerNet w1        21  14     1.0m  p90 6.0    1.0m  p90 4.4
    SoccerNet w2         5   2     6.6m      6.7    2.1m      5.1
    SoccerNet w3         6   5     1.7m      2.1    7.6m      7.6
    reading             10   7     0.8m      1.0    1.8m      5.7
    pooled              42  28     1.0m      6.5    1.4m      7.3
    at random                      1.1m      6.7

The anchors the gate refuses score 1.4 m where the ones it keeps score 1.0 --
a difference in the right direction and far too small to matter. Refusing
the same number at random scores 1.1 m, which is the whole of the effect:
throw away any 14 of 42 anchors and the rest look slightly better. On w2 the
gate is actively inverted, keeping the two worst frames on the clip.

The Veo clip could not be judged at all. It needs two anchored frames close
enough together to compare, and it has too few.

This is kept because the reasoning is sound and the result is not, which is
the useful kind of negative: it says the bad anchors are not the ones that
disagree with their neighbours. What that leaves is anchors that are wrong
together, which is what `probe_anchor_bias.py` goes after.

    python gate_anchors.py [--frames 40]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from check_anchor_consistency import MAX_PAIR_GAP, disagreement
from fit_pitch_anchor import anchored_frames, marking_error
from probe_pitch_lines import CLIPS, line_segments
from propagate_anchor import chance_floor, frame_to_frame

# How far two anchors may disagree, in metres, before one of them is refused.
# This is a metre or two above the typical disagreement between two good
# anchors, so that ordinary noise does not trip it.
MAX_DISAGREEMENT_M = 3.0

# An anchor with fewer peers than this is not judged either way -- one
# neighbour is an opinion, not a quorum, and two anchors wrong in the same
# way would vouch for each other.
MIN_PEERS = 2


def agreement(anchors, images, info):
    """For each anchor, how far it sits from what its neighbours say.

    The median across peers rather than the best of them: an anchor only has
    to find one accomplice to look respectable under a minimum, and a wrong
    anchor near another wrong anchor is exactly the case that matters.
    """
    scores = {}
    for index, homography in anchors:
        if index not in images:
            continue
        gaps = []
        for other, other_map in anchors:
            if other == index or other not in images:
                continue
            if abs(other - index) > MAX_PAIR_GAP:
                continue
            warp, _ = frame_to_frame(images[other], images[index])
            if warp is None:
                continue
            value = disagreement(other_map, homography, warp,
                                 info["width"], info["height"])
            if np.isfinite(value):
                gaps.append(value)
        if len(gaps) >= MIN_PEERS:
            scores[index] = float(np.median(gaps))
    return scores


def summarise(values):
    if not values:
        return "     -        -"
    array = np.array(values)
    return f"{np.median(array):6.1f}m {np.percentile(array, 90):7.1f}m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("An anchor is refused when it contradicts its neighbours. The gate "
          "never\nlooks at a marking; the score is nothing but markings.\n")
    print(f"  {'clip':>14s} {'judged':>7s} {'kept':>5s} "
          f"{'kept: median   p90':>21s} {'refused: median   p90':>24s}")

    pooled_keep, pooled_drop, pooled_random = [], [], []
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info, anchors = anchored_frames(path, args.frames, rng)
        if len(anchors) < MIN_PEERS + 1:
            continue

        cap = cv2.VideoCapture(info["path"])
        images, errors = {}, {}
        for index, homography in anchors:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            images[index] = frame
            segments, _ = line_segments(frame)
            if len(segments) >= 3:
                value = marking_error(homography, segments, info)
                if np.isfinite(value):
                    errors[index] = value
        cap.release()

        scores = agreement(anchors, images, info)
        judged = [i for i in scores if i in errors]
        if len(judged) < 3:
            continue
        kept = [errors[i] for i in judged
                if scores[i] <= MAX_DISAGREEMENT_M]
        dropped = [errors[i] for i in judged
                   if scores[i] > MAX_DISAGREEMENT_M]

        # The floor: refuse the same number at random, many times over.
        if dropped:
            all_errors = np.array([errors[i] for i in judged])
            for _ in range(200):
                order = rng.permutation(all_errors.size)
                pooled_random += list(all_errors[order[:len(kept)]])

        print(f"  {name:>14s} {len(judged):7d} "
              f"{len(kept)}/{len(judged):<3d} {summarise(kept):>21s} "
              f"{summarise(dropped):>24s}")
        pooled_keep += kept
        pooled_drop += dropped

    if pooled_keep:
        print(f"\n  {'pooled':>14s} {len(pooled_keep) + len(pooled_drop):7d} "
              f"{len(pooled_keep)}/{len(pooled_keep) + len(pooled_drop):<3d} "
              f"{summarise(pooled_keep):>21s} "
              f"{summarise(pooled_drop):>24s}")
        if pooled_random:
            print(f"  {'at random':>14s} {'':7s} {'':5s} "
                  f"{summarise(pooled_random):>21s}")

    print("\n  The gate is worth having if what it refuses scores worse than "
          "what it\n  keeps, and if keeping the same number at random does "
          "not do as well.")


if __name__ == "__main__":
    main()
