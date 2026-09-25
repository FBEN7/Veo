"""One anchor for a whole clip, where the footage allows it. It mostly does not.

`probe_veo_panorama.py` settles a question this was built on, and settles it
in favour: a Veo camera does not turn. It films wide and still, and the view
that follows the ball is a window sliding over that fixed image. Matched
features say so, though not in the form first guessed. A pure translation
explains a pair of Veo frames to 1.0 px at a gap of 25 frames but is out by 8
to 19 px at 100 and beyond, while a translation *with a scale* holds at 0.9
to 1.3 px all the way to 2000 frames -- as accurate as a full
eight-parameter homography over the same pairs. The crop slides and it
zooms. The camera behind it never moves.

That is worth having on its own, because it means the clip has ONE pose
relative to the pitch. Every anchored frame measures the same quantity, so
they can be combined; a frame with no markings in it at all could still be
placed, because the answer would not depend on the frame; and reach would no
longer be limited by two views overlapping.

## It does not work, and the reason is not the geometry

Tying the anchored frames to a common panorama needs each of them matched,
directly or through a chain of keyframes, to the same reference. That
matching fails on the footage this was for. Chaining keyframes 100 frames
apart and letting each fall back to an earlier one when its hop fails, the
anchors that end up placed are:

    SoccerNet w1   24 of 25        SoccerNet w3    1 of 16
    SoccerNet w2   12 of 14        reading         0 of 14
    Veo             2 of 12

and the failure is abrupt rather than gradual. Counting good matches per hop
on the Veo clip gives 245, 0, 0, 0, 0, 0, 133, 542, 645, 1114, 0, 0, ...:
either hundreds of features match or none do. Runs of consecutive zeros mean
stretches of the clip where nothing matches anything, which is what
texture-poor grass at 640x360 looks like to a corner detector. Tightening the
keyframe spacing to 50 frames made it worse, not better, which rules out the
hop distance as the cause.

So the premise holds and the plumbing does not. What would fix it is a
matcher that works where ORB does not -- a denser or learned one, or matching
against the pitch markings themselves rather than against corners, since
markings are the one thing this footage reliably has. That is a different
piece of work and it is not attempted here.

## What the vote measured anyway

On the two clips where enough anchors were placed, the vote does not beat a
single anchor: w1 scores 1.2 m voted against 1.0 m from the frame's own
circle fit, with twenty voters. Twenty independent measurements of one
quantity should comfortably beat one. That they do not is evidence the
errors are not independent, which `probe_anchor_bias.py` follows up.

Scoring is leave-one-out: a frame is never scored by a vote it took part in.

    python global_anchor.py [--frames 40]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fit_pitch_anchor import anchored_frames, marking_error, plausible_anchor
from probe_pitch_lines import CLIPS, line_segments
from probe_veo_panorama import matched_points
from propagate_anchor import MIN_INLIERS, RANSAC_PX, chance_floor



# How far apart the keyframes are that the panorama map is chained between.
# The pipeline's per-frame motion estimate is accurate but accumulates:
# measured against direct matching, the Veo offsets agree to 7-20 px out to
# 300 frames and then part company -- 141 px by 600 and worse after. Four and
# a half thousand small steps drift; eighteen long ones do not, because each
# is measured against the scene rather than against its predecessor.
KEYFRAME_GAP = 250

# The grid the candidates vote on, as fractions of the frame. Kept to the
# lower part of the image, which is where the ground is.
VOTE_ROWS = (0.55, 0.72, 0.88, 1.0)
VOTE_COLS = (0.05, 0.35, 0.65, 0.95)

# Fewer than this and a median is not a median.
MIN_VOTERS = 3


def vote_grid(width, height):
    cols = [c * width for c in VOTE_COLS for _ in VOTE_ROWS]
    rows = [r * height for _ in VOTE_COLS for r in VOTE_ROWS]
    return np.array([cols, rows, [1.0] * len(cols)])


def consensus(candidates, width, height):
    """The homography agreeing with the median of what the candidates say."""
    if len(candidates) < MIN_VOTERS:
        return None
    grid = vote_grid(width, height)
    votes = []
    for homography in candidates:
        mapped = homography @ grid
        if np.any(np.abs(mapped[2]) < 1e-9):
            continue
        mapped = mapped[:2] / mapped[2]
        if np.all(np.isfinite(mapped)):
            votes.append(mapped)
    if len(votes) < MIN_VOTERS:
        return None
    agreed = np.median(np.stack(votes), axis=0)
    fitted, _ = cv2.findHomography(grid[:2].T.astype(np.float32),
                                   agreed.T.astype(np.float32), 0)
    return fitted


def direct_similarity(frame_a, frame_b):
    """The crop move from one frame to the other: a slide and a zoom.

    A translation is not enough, and the measurement says so plainly. On the
    Veo clip a pure translation explains a pair of frames to 1.0 px at a gap
    of 25 but is out by 8 to 19 px at 100 frames and beyond, while a
    similarity holds at 0.9 to 1.3 px all the way out to 2000 frames -- as
    accurate as a full eight-parameter homography over the same pairs.

    That is the signature of a fixed camera whose crop both slides and
    changes size, which is what a Veo view following play does. The camera
    has one pose; the window onto it does not.
    """
    src, dst = matched_points(frame_a, frame_b)
    if src is None:
        return None
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if homography is None or mask is None or int(mask.sum()) < MIN_INLIERS:
        return None
    keep = mask.ravel().astype(bool)
    partial, _ = cv2.estimateAffinePartial2D(src[keep], dst[keep],
                                             method=cv2.LMEDS)
    if partial is None:
        return None
    return np.vstack([partial, [0.0, 0.0, 1.0]])


def panorama_maps(cap, total, wanted):
    """For each wanted frame, the map from its pixels to the fixed panorama.

    Built in two stages so that error grows with the number of keyframes
    rather than the number of frames. Keyframes are chained to each other,
    each hop measured directly against the scene; every other frame is then
    matched to its nearest keyframe in a single hop. Nothing is accumulated
    over more than one step from something that was itself measured.

    The first version of this chained the per-frame motion estimate instead,
    over four and a half thousand steps, and drifted 141 px by frame 600.
    """
    keys = list(range(0, total, KEYFRAME_GAP))
    frames, maps = {}, {}
    for key in keys:
        cap.set(cv2.CAP_PROP_POS_FRAMES, key)
        ok, frame = cap.read()
        if ok:
            frames[key] = frame

    available = sorted(frames)
    if not available:
        return {}
    # Each keyframe is matched to the most recent mapped keyframe that will
    # match it, nearest first. Chaining only from the last SUCCESSFUL hop was
    # tried and is much worse: one failure pins the chain to a frame that
    # then has to match everything after it across a growing gap, so it fails
    # again, and the chain never recovers. Tightening the spacing does not
    # help, because the problem is the distance to the frame being matched
    # FROM, not the spacing itself -- at 50 frames apart the coverage fell
    # rather than rose.
    maps[available[0]] = np.eye(3)
    for position, key in enumerate(available[1:], start=1):
        for back in range(position - 1, -1, -1):
            previous = available[back]
            if previous not in maps:
                continue
            step = direct_similarity(frames[previous], frames[key])
            if step is None:
                continue
            try:
                maps[key] = maps[previous] @ np.linalg.inv(step)
            except np.linalg.LinAlgError:
                continue
            break

    out = {}
    for index in wanted:
        if not maps:
            break
        near = min(maps, key=lambda k: abs(k - index))
        if near == index:
            out[index] = maps[near]
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        step = direct_similarity(frames[near], frame)
        if step is None:
            continue
        try:
            out[index] = maps[near] @ np.linalg.inv(step)
        except np.linalg.LinAlgError:
            continue
    return out


def candidates_for(frame_index, anchors, maps, skip=None):
    """Every anchor's opinion about this frame, carried through the panorama."""
    if frame_index not in maps:
        return []
    here = maps[frame_index]
    out = []
    for source, homography in anchors:
        if skip is not None and source == skip:
            continue
        if source not in maps:
            continue
        # The anchor maps ITS image to the pitch. Going back through its
        # panorama map gives panorama-to-pitch, which is the quantity that
        # does not depend on the frame; this frame's map then brings it here.
        try:
            panorama = homography @ np.linalg.inv(maps[source])
        except np.linalg.LinAlgError:
            continue
        out.append(panorama @ here)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("Every anchor in the clip, referred to panorama coordinates and "
          "made to\nvote on one camera pose. Leave-one-out: no frame is "
          "scored by its own vote.\n")
    print(f"  {'clip':>14s} {'frames':>7s} {'voters':>7s} {'own anchor':>11s} "
          f"{'nearest':>8s} {'all of them':>12s} {'chance':>7s}")

    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info = json.loads((path / "clip.json").read_text())
        _, anchors = anchored_frames(path, args.frames, rng)
        if len(anchors) < MIN_VOTERS + 1:
            print(f"  {name:>14s}   {len(anchors)} anchors, too few to vote")
            continue

        cap = cv2.VideoCapture(info["path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        maps = panorama_maps(cap, total, [a[0] for a in anchors])
        own, near, allof, floors, counts = [], [], [], [], []
        for target, homography in anchors:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if not ok:
                continue
            segments, _ = line_segments(frame)
            if len(segments) < 3:
                continue

            others = candidates_for(target, anchors, maps, skip=target)
            others = [h for h in others if plausible_anchor(h, info)]
            if len(others) < MIN_VOTERS:
                continue
            voted = consensus(others, info["width"], info["height"])
            if voted is None or not plausible_anchor(voted, info):
                continue

            # The nearest anchor carried the same way, which is what
            # propagation does today, as the thing to beat.
            nearest = min(
                (a for a in anchors if a[0] != target),
                key=lambda a: abs(a[0] - target))
            carried = candidates_for(target, [nearest], maps)
            if not carried:
                continue
            carried = carried[0]

            own_error = marking_error(homography, segments, info)
            near_error = marking_error(carried, segments, info)
            all_error = marking_error(voted, segments, info)
            if not np.isfinite(all_error):
                continue
            own.append(own_error)
            near.append(near_error)
            allof.append(all_error)
            floors.append(chance_floor(voted, segments, info, rng))
            counts.append(len(others))
        cap.release()

        if not allof:
            continue
        finite = lambda v: np.median([x for x in v if np.isfinite(x)])
        floor = [f for f in floors if np.isfinite(f)]
        print(f"  {name:>14s} {len(allof):7d} {np.mean(counts):7.1f} "
              f"{finite(own):10.1f}m {finite(near):7.1f}m "
              f"{np.median(allof):11.1f}m "
              f"{(np.median(floor) if floor else np.nan):6.1f}m")

    print("\n  'Own anchor' is the frame's own circle fit, which is the "
          "current method.\n  'All of them' is the clip-wide vote. It should "
          "win on Veo, where the\n  camera really is fixed, and lose on "
          "broadcast, where it is not.")


if __name__ == "__main__":
    main()
