"""Is the Veo camera really a window sliding over a fixed panorama?

A Veo unit does not pan. It films wide and still, and the "camera" that
follows the ball is a crop taken out of that fixed image. If that is literally
what the footage is, then any two frames of a clip are related by a
*translation* -- the crop moved -- and not by the general homography that
relates two views from a camera that has turned.

That distinction is worth settling rather than assuming, because a great deal
follows from it. If the frames really are crops of one image, then all of
them are views of a single camera with a single pose relative to the pitch.
There is then one homography from the panorama to the pitch for the entire
clip, every anchored frame is a measurement of that same homography, and a
frame with no markings in it at all can still be placed -- because the answer
does not depend on the frame.

The test compares models on the same evidence. ORB features are matched
between a pair of frames and then fitted three ways: as a pure translation,
as a translation with a scale, and as a full homography. All three are scored
by where they put the same matched points, in pixels.

The answer is the middle one, and the first version of this probe missed it
by not asking. Out to a gap of 25 frames a pure translation looks sufficient
on the Veo clip -- 1.0 px, against 0.8 for a homography -- and that is what
the first run concluded from. Over longer gaps it falls apart: 8.4 px at 100
frames, 11.6 at 900, 18.7 at 2000. A translation with a scale holds at 0.92
to 1.29 px across every one of those gaps, matching the homography exactly.

So the crop slides AND zooms, which is what a Veo view following play does,
and the camera behind it is fixed. The broadcast clips are the control and
behave differently: a translation is out by 45 px at a gap of 300 on w1, and
while a similarity fits them tolerably -- a camera panning across a narrow
field of view is nearly a similarity -- it is the homography that fits them
best, and their pose genuinely changes.

The distinction matters because a fixed camera has one pose relative to the
pitch for the whole clip. What that buys, and why it could not be collected,
is in `global_anchor.py`.

    python probe_veo_panorama.py [--pairs 12]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from probe_pitch_lines import CLIPS
from propagate_anchor import (MIN_INLIERS, MIN_MATCHES, ORB_FEATURES,
                              RANSAC_PX)

# Gaps at which to compare the two models. A translation and a homography
# agree over a small gap whatever the camera is doing, so the question is
# only interesting once the view has moved appreciably.
GAPS = (25, 100, 300, 900, 2000)


def matched_points(frame_a, frame_b):
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    kp_a, desc_a = orb.detectAndCompute(
        cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY), None)
    kp_b, desc_b = orb.detectAndCompute(
        cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY), None)
    if desc_a is None or desc_b is None:
        return None, None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    good = [p[0] for p in pairs
            if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_MATCHES:
        return None, None
    src = np.float32([kp_a[m.queryIdx].pt for m in good])
    dst = np.float32([kp_b[m.trainIdx].pt for m in good])
    return src, dst


def compare_models(src, dst):
    """Median pixel error of a pure translation against a full homography.

    Both are scored on the homography's inliers, so the two models are asked
    to explain exactly the same points. Scoring the translation on its own
    inliers would let it choose an easier question.
    """
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if homography is None or mask is None or int(mask.sum()) < MIN_INLIERS:
        return None
    keep = mask.ravel().astype(bool)
    src_in, dst_in = src[keep], dst[keep]

    # The best translation for these points is the median displacement --
    # median rather than mean so that a few surviving mismatches, or a
    # player moving under the crop, do not set it.
    shift = np.median(dst_in - src_in, axis=0)
    translated = src_in + shift

    # A third model, between the two: the crop slid AND changed size. A Veo
    # view follows play by zooming as well as panning, so if a translation
    # fails at long range while a similarity holds, the footage is still one
    # fixed camera -- just not one fixed crop.
    similarity, _ = cv2.estimateAffinePartial2D(
        src_in, dst_in, method=cv2.LMEDS)
    if similarity is None:
        scaled_error = float("nan")
    else:
        scaled = (similarity[:, :2] @ src_in.T).T + similarity[:, 2]
        scaled_error = float(
            np.median(np.linalg.norm(scaled - dst_in, axis=1)))

    points = np.hstack([src_in, np.ones((src_in.shape[0], 1))])
    mapped = (homography @ points.T)
    mapped = (mapped[:2] / mapped[2]).T

    return (float(np.median(np.linalg.norm(translated - dst_in, axis=1))),
            float(np.median(np.linalg.norm(mapped - dst_in, axis=1))),
            scaled_error, int(keep.sum()), shift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=12)
    args = ap.parse_args()

    print("Two frames, matched by ORB, then explained twice: once by a crop "
          "that\nslid, once by a camera that turned. Both scored on the same "
          "points.\n")
    print(f"  {'clip':>14s} {'gap':>5s} {'pairs':>6s} {'translation':>12s} "
          f"{'+ zoom':>8s} {'homography':>11s} {'ratio':>6s}")

    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        import json
        info = json.loads((path / "clip.json").read_text())
        cap = cv2.VideoCapture(info["path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for gap in GAPS:
            starts = np.linspace(0, max(0, total - gap - 1),
                                 args.pairs).astype(int)
            rows = []
            for start in starts:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
                ok_a, frame_a = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(start) + gap)
                ok_b, frame_b = cap.read()
                if not (ok_a and ok_b):
                    continue
                src, dst = matched_points(frame_a, frame_b)
                if src is None:
                    continue
                result = compare_models(src, dst)
                if result is not None:
                    rows.append(result)
            if not rows:
                print(f"  {name:>14s} {gap:5d} {0:6d}   no usable pairs")
                continue
            shift_err = np.median([r[0] for r in rows])
            full_err = np.median([r[1] for r in rows])
            zoom_err = np.nanmedian([r[2] for r in rows])
            ratio = shift_err / full_err if full_err > 1e-6 else np.inf
            print(f"  {name:>14s} {gap:5d} {len(rows):6d} "
                  f"{shift_err:11.2f}px {zoom_err:7.2f}px {full_err:10.2f}px "
                  f"{ratio:6.1f}")
        cap.release()

    print("\n  A ratio near 1 means a sliding crop explains the footage as "
          "well as a\n  turning camera does, and the clip is one camera pose "
          "throughout. A large\n  ratio means the view really is changing "
          "shape, and each frame needs its own.")


if __name__ == "__main__":
    main()
