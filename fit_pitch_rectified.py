"""Rectify by vanishing points, then try to say which line is which.

This is the route that uses no ground plane. It has two halves and they have
different verdicts, which is the point of running it.

The first half works. Two vanishing points give the horizon, the horizon
gives a rectification, and in the rectified frame both families of markings
become axis-aligned -- confirmed by the segments' rectified positions staying
constant along their own length, which is not something a wrong rectification
produces. Gated on support, the horizon is stable to 5 pixels on one clip and
102 on another, from markings alone.

The second half does not. Identifying *which* pitch line each detected line
is remains underdetermined from a single frame, because a frame carries three
to five distinct lines against seven candidate positions. Tightening the
match to 0.6 m, demanding a third inlier beyond the two that any two-point
solve fits by construction, and rejecting fits that imply more pitch on
screen than a broadcast frame can hold, all help and none is enough:

  * the family-to-axis assignment still flips between frames of one clip,
    which a fixed camera cannot do
  * the implied centre of view still scatters by 21 to 32 metres
  * on one frame both assignments survive, at 8 and 6 inliers

So the rectification is a result and the identification is not. What the
failure says is specific and it is not "try another heuristic": a single
frame does not carry enough distinct lines, and the fix is to stop asking it
to. Either use the centre circle, which is the one marking whose position is
unique and which pins scale and origin by itself, or solve all gated frames
together, which is now worth doing where it was not before -- the earlier
joint attempt failed on a rectification that wandered, and this one does not.

    python fit_pitch_rectified.py [--frames 30]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from probe_pitch_lines import CLIPS, line_segments
from src import pitch_model as pm
import probe_vanishing_points as vp


def gated_frames(out_dir: Path, n_frames: int, rng):
    """Frames whose markings converge well enough to rectify."""
    info = json.loads((out_dir / "clip.json").read_text())
    cap = cv2.VideoCapture(info["path"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = []
    for idx in np.linspace(0, total - 1, n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        segments, _ = line_segments(frame)
        family_a, family_b = vp._split_families(segments)
        if not family_a:
            continue
        vp_a, support_a = vp.vanishing_point(family_a, rng)
        vp_b, support_b = vp.vanishing_point(family_b, rng)
        if vp_a is None or vp_b is None:
            continue
        if min(support_a, support_b) < vp.GATE_MIN_SUPPORT_LINES:
            continue
        if (support_a / len(family_a) < vp.GATE_MIN_SUPPORT_FRACTION
                or support_b / len(family_b) < vp.GATE_MIN_SUPPORT_FRACTION):
            continue
        out.append((int(idx), family_a, family_b, vp_a, vp_b))
    cap.release()
    return info, out


def fit_frame(info, family_a, family_b, vp_a, vp_b):
    """Every surviving interpretation of one frame."""
    width, height = info["width"], info["height"]
    rectification = pm.rectify_from_vanishing_points(
        vp_a, vp_b, (width / 2.0, height * 0.75))

    across = pm.cluster_positions(
        pm.rectified_positions(family_a, rectification, 1))
    along = pm.cluster_positions(
        pm.rectified_positions(family_b, rectification, 0))
    if len(across) < 3 or len(along) < 3:
        return []

    results = []
    for model_a, model_b, label in (
            (pm.LINES_ACROSS_M, pm.LINES_ALONG_M, "A=across"),
            (pm.LINES_ALONG_M, pm.LINES_ACROSS_M, "A=along")):
        fit_a = pm.match_line_positions(across, model_a)
        fit_b = pm.match_line_positions(along, model_b)
        if not fit_a or not fit_b:
            continue
        if label == "A=across":
            affine = np.array([[0, fit_a[2], fit_a[3]],
                               [fit_b[2], 0, fit_b[3]], [0, 0, 1.0]])
        else:
            affine = np.array([[fit_b[2], 0, fit_b[3]],
                               [0, fit_a[2], fit_a[3]], [0, 0, 1.0]])
        homography = affine @ rectification

        corners = np.array([[0, width, width, 0],
                            [height * 0.45, height * 0.45, height, height],
                            [1, 1, 1, 1]], dtype=float)
        mapped = homography @ corners
        if np.any(np.abs(mapped[2]) < 1e-9):
            continue
        mapped = mapped[:2] / mapped[2]
        visible_x, visible_y = float(np.ptp(mapped[0])), float(np.ptp(mapped[1]))
        if visible_x > pm.MAX_VISIBLE_X_M or visible_y > pm.MAX_VISIBLE_Y_M:
            continue
        results.append(dict(label=label, homography=homography,
                            visible=(visible_x, visible_y),
                            centre=(float(mapped[0].mean()),
                                    float(mapped[1].mean())),
                            inliers=fit_a[0] + fit_b[0]))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("Rectifying by vanishing points, then identifying the lines.\n"
          "The rectification is checked by the horizon's stability; the "
          "identification\nby whether a fixed camera gets one answer.\n")

    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info, frames = gated_frames(path, args.frames, rng)
        if not frames:
            print(f"{name}: no frames clear the vanishing-point gate\n")
            continue

        rows = []
        for idx, family_a, family_b, vp_a, vp_b in frames:
            for fit in fit_frame(info, family_a, family_b, vp_a, vp_b):
                rows.append((idx, fit))

        print(f"{name}: {len(frames)} frames gated, {len(rows)} surviving fits")
        for idx, fit in rows:
            print(f"   f{idx:5d} {fit['label']:>9s}  visible "
                  f"{fit['visible'][0]:5.1f} x {fit['visible'][1]:5.1f} m  "
                  f"centre ({fit['centre'][0]:6.1f},{fit['centre'][1]:6.1f})  "
                  f"inliers {fit['inliers']}")
        if len(rows) >= 2:
            labels = {fit["label"] for _, fit in rows}
            cx = np.array([fit["centre"][0] for _, fit in rows])
            cy = np.array([fit["centre"][1] for _, fit in rows])
            verdict = "CONSISTENT" if len(labels) == 1 else "FLIPS"
            print(f"   -> assignment {verdict}, centre of view scatters "
                  f"{cx.std():.1f} x {cy.std():.1f} m")
        print()

    print("A fixed camera cannot have its line families swap roles, and its "
          "centre of\nview cannot scatter by tens of metres. Where those "
          "happen the frame did not\ncarry enough distinct lines to be "
          "identified, whatever the residual says.")


if __name__ == "__main__":
    main()
