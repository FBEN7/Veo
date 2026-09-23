"""Is what is left of the anchor's error noise, or is it bias?

Three separate attempts to improve the anchor this session all came to
nothing, and they failed in the same way:

  * refining each anchor against the straight markings moved it by half a
    degree and 0.8 m and changed the held-out error not at all;
  * refusing the anchors that disagree most with their neighbours left the
    survivors no better than refusing the same number at random;
  * letting up to twenty anchors vote on one frame did not beat that
    frame's own single anchor.

Every one of those is a way of averaging away *noise*. If the error varied
frame to frame, all three would have helped, and the third especially --
twenty independent measurements of one quantity beat one measurement, and it
is not close. That none of them helped says the error does not vary frame to
frame. It is shared: a systematic offset belonging to the clip.

This measures that directly. One correction -- rotation, shift, scale -- is
fitted for a whole clip, on half its anchored frames, and applied to the
other half. A correction that transfers is bias, and bias can be removed.
One that does not is noise pretending to be bias.

Which is a more useful thing to know than it sounds, because it says where
the remaining error must come from. A per-clip offset cannot be the ellipse
fit wobbling or the halfway line being missed on some frames; those vary.
It has to be something the clip holds constant -- the horizon taken at
infinity when the camera does have some tilt, a circle detected slightly
small because its outer edge is where the grass wins, a principal point
assumed at the image centre.

    python probe_anchor_bias.py [--frames 40] [--splits 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from fit_pitch_anchor import anchored_frames, marking_error, plausible_anchor
from probe_pitch_lines import CLIPS, line_segments
from refine_anchor import _loss, _mapped_markings

# Wider than the per-frame refinement allows, because a bias is a different
# claim from a wobble and there is far more evidence behind it: a clip-level
# fit sees every marking on every anchored frame at once, so a hundred-odd
# markings determine three or four parameters instead of a handful doing it.
MAX_ROTATION_DEG = 15.0
MAX_SHIFT_M = 10.0
MAX_SCALE = 0.20

GRID_ROTATION_DEG = (-10.0, -5.0, 0.0, 5.0, 10.0)
GRID_SHIFT_M = (-6.0, -3.0, 0.0, 3.0, 6.0)
GRID_SCALE = (0.9, 1.0, 1.1)

MIN_FRAMES = 6


def correction(params) -> np.ndarray:
    theta, shift_x, shift_y, log_scale = params
    scale = float(np.exp(log_scale))
    cos, sin = np.cos(theta), np.sin(theta)
    linear = scale * np.array([[cos, -sin], [sin, cos]])
    centre = np.array([52.5, 34.0])
    out = np.eye(3)
    out[:2, :2] = linear
    out[:2, 2] = centre - linear @ centre + np.array([shift_x, shift_y])
    return out


def fit_correction(frames):
    """One rotation, shift and scale for all of these frames together."""
    def objective(params):
        moved = correction(params)
        total = []
        for homography, segments in frames:
            here = moved @ homography
            markings = _mapped_markings(here, segments)
            if markings:
                total.append(_loss(markings))
        return float(np.mean(total)) if total else float("inf")

    best, best_cost = np.zeros(4), objective(np.zeros(4))
    for theta in np.deg2rad(GRID_ROTATION_DEG):
        for shift_x in GRID_SHIFT_M:
            for shift_y in GRID_SHIFT_M:
                for scale in GRID_SCALE:
                    start = np.array([theta, shift_x, shift_y, np.log(scale)])
                    cost = objective(start)
                    if cost < best_cost:
                        best, best_cost = start, cost

    bounds = [(-np.deg2rad(MAX_ROTATION_DEG), np.deg2rad(MAX_ROTATION_DEG)),
              (-MAX_SHIFT_M, MAX_SHIFT_M), (-MAX_SHIFT_M, MAX_SHIFT_M),
              (np.log(1 - MAX_SCALE), np.log(1 + MAX_SCALE))]
    result = minimize(objective, best, method="Powell", bounds=bounds,
                      options={"xtol": 1e-3, "ftol": 1e-4, "maxiter": 4000})
    return result.x if result.success else best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--splits", type=int, default=20)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("One correction for a whole clip, fitted on half its anchored "
          "frames and\napplied to the other half. If it transfers, the error "
          "is bias.\n")
    print(f"  {'clip':>14s} {'frames':>7s} {'before':>7s} {'after':>7s} "
          f"{'the correction it found':>32s}")

    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info, anchors = anchored_frames(path, args.frames, rng)
        if len(anchors) < MIN_FRAMES:
            continue

        cap = cv2.VideoCapture(info["path"])
        usable = []
        for index, homography in anchors:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            segments, _ = line_segments(frame)
            if len(segments) >= 3:
                usable.append((homography, segments))
        cap.release()
        if len(usable) < MIN_FRAMES:
            continue

        before, after, found = [], [], []
        for _ in range(args.splits):
            order = rng.permutation(len(usable))
            half = len(order) // 2
            fit = [usable[i] for i in order[:half]]
            held = [usable[i] for i in order[half:]]
            params = fit_correction(fit)
            moved = correction(params)
            found.append(params)
            for homography, segments in held:
                plain = marking_error(homography, segments, info)
                here = moved @ homography
                fixed = (marking_error(here, segments, info)
                         if plausible_anchor(here, info) else plain)
                if np.isfinite(plain) and np.isfinite(fixed):
                    before.append(plain)
                    after.append(fixed)

        if not before:
            continue
        params = np.median(np.array(found), axis=0)
        shape = (f"{np.rad2deg(params[0]):+5.1f}deg "
                 f"{params[1]:+5.1f}m {params[2]:+5.1f}m "
                 f"{100 * np.expm1(params[3]):+5.1f}%")
        print(f"  {name:>14s} {len(usable):7d} {np.median(before):6.1f}m "
              f"{np.median(after):6.1f}m {shape:>32s}")

    print("\n  A correction that helps on frames it was not fitted to is a "
          "real bias, and\n  the numbers beside it say what kind. One that "
          "helps only where it was\n  fitted is the same overfitting that "
          "sank the per-frame refinement.")


if __name__ == "__main__":
    main()
