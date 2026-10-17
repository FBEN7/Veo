"""Fit the pitch to the detected markings, three ways, and fail all three.

There is no labelled pitch to score against, so the checks are physical and
none of them is something the fit is told:

  1. A broadcast camera is bolted to a gantry, so the camera position the fit
     implies must be the same on every frame. The pose translation *is* that
     position -- the camera sits at the origin of the ground frame -- so the
     spread across frames measures whether the fit found the pitch.

  2. The pan is already known, measured frames ago by a different method. It
     rotates ground coordinates and nothing else, so fitted orientation has
     to track it.

  3. Splitting the frames in half and fitting each half must give the same
     camera twice.

All three fail, and the way they fail says where the problem is. Residuals
around a metre with three quarters of points as inliers look like a working
fit; the camera moving 30 to 40 metres between frames says it is not. A
pitch is a repetitive grid of parallel lines, so putting most points near
*some* line is cheap, and the residual is measuring that rather than a
correct pose.

Two attempts to rescue it, both recorded here because their failure is the
informative part.

Correcting the depth scale does not. An error of k in the focal length
multiplies depth by k and leaves lateral alone -- `Z = f*h/v` scales,
`X = (x - cx)*Z/f` does not -- so it is an anisotropic stretch that no rigid
pose can absorb, which would produce exactly this signature. Swept from 0.6
to 2.0, the best setting still leaves the camera wandering 20 to 26 metres.

Fitting all frames jointly does not either, and it is the more telling
failure. The camera being fixed means frame i differs from frame 0 by the
pan alone, so de-rotating each frame by its own measured pan should bring
every observation into one frame where a single pose explains all of them.
It makes things worse: residual rises from about 1.0 m to 1.9, inliers fall
from 74-84% to 40-48%, and the two halves of a clip disagree by 33 to 77
metres. Frames that should overlay do not.

That is the finding. The line detection is sound -- inspected, it traces the
penalty area, the six-yard box and the goal area. What is not sound is the
ground plane underneath it: the projection that turns pixels into metres is
not consistent enough between frames for a rigid pitch to match. Which is
the same conclusion the ground plane reached on its own terms, where
projecting onto it made every measurable instrument worse.

    python fit_pitch_model.py [--frames 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from probe_pitch_lines import CLIPS, line_segments
from src import ground_plane, pitch_model

MIN_SEGMENTS = 4
MIN_GROUND_POINTS = 40


def collect(out_dir: Path, n_frames: int):
    """Ground-projected marking points per frame, with each frame's pan."""
    info = json.loads((out_dir / "clip.json").read_text())
    plane_path = out_dir / "ground_plane.json"
    if not plane_path.exists():
        return None, None, None, "no ground plane cached for this clip"
    stored = json.loads(plane_path.read_text())
    if stored.get("refused"):
        return None, None, None, "no ground plane: focal length refused here"
    plane = ground_plane.GroundPlane(**stored)

    motion_path = out_dir / "camera_motion.npy"
    motion = np.load(motion_path) if motion_path.exists() else None

    cap = cv2.VideoCapture(info["path"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    per_frame, pans = [], []
    for idx in np.linspace(0, total - 1, n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        segments, _ = line_segments(frame)
        if len(segments) < MIN_SEGMENTS:
            continue

        # The plane's horizon was fitted in camera-compensated coordinates.
        # These segments are on the raw frame, where the horizon has moved
        # with the camera by exactly the vertical motion.
        horizon = float(plane.horizon_at(np.array([idx]))[0])
        if motion is not None and len(motion) > idx:
            horizon += float(motion[int(idx), 1])

        points = pitch_model.segments_to_ground(segments, plane, horizon)
        if len(points) < MIN_GROUND_POINTS:
            continue
        per_frame.append(points)
        pans.append(float(motion[int(idx), 0]) / plane.focal_px
                    if motion is not None and len(motion) > idx
                    else float("nan"))

    cap.release()
    return plane, per_frame, pans, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20)
    args = ap.parse_args()

    field, origin, resolution = pitch_model.distance_field()
    print("Fitting the pitch to the detected markings.\n"
          "The checks are physical: a fixed camera must stay put, the pan is "
          "already\nknown, and two halves of a clip must agree.")

    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        plane, per_frame, pans, problem = collect(path, args.frames)
        if problem:
            print(f"\n{name}: {problem}")
            continue
        if len(per_frame) < 6:
            print(f"\n{name}: only {len(per_frame)} frames yielded markings")
            continue

        print(f"\n{name}: {len(per_frame)} frames, "
              f"{sum(len(p) for p in per_frame)} ground points")

        # 1. Each frame on its own.
        poses = []
        for points in per_frame:
            pose = pitch_model.fit_pose(points, [points], field, origin,
                                        resolution)
            if pose and pose["usable"]:
                poses.append(pose)
        if len(poses) >= 4:
            tx = np.array([p["tx"] for p in poses])
            ty = np.array([p["ty"] for p in poses])
            res = np.median([p["residual_m"] for p in poses])
            inl = np.median([p["inlier_fraction"] for p in poses])
            print(f"  per frame : residual {res:.2f} m, inliers {inl:.0%}, "
                  f"camera spread {tx.std():.1f} x {ty.std():.1f} m "
                  f"(should be 0)")

        # 2. Every frame at once, de-rotated by its own pan.
        joint = pitch_model.fit_pose_multiframe(per_frame, pans, field,
                                                origin, resolution)
        if joint:
            half_a = pitch_model.fit_pose_multiframe(
                per_frame[0::2], pans[0::2], field, origin, resolution)
            half_b = pitch_model.fit_pose_multiframe(
                per_frame[1::2], pans[1::2], field, origin, resolution)
            gap = (np.hypot(half_a["tx"] - half_b["tx"],
                            half_a["ty"] - half_b["ty"])
                   if half_a and half_b else float("nan"))
            print(f"  joint     : residual {joint['residual_m']:.2f} m, "
                  f"inliers {joint['inlier_fraction']:.0%}, "
                  f"halves disagree by {gap:.1f} m (should be 0)")

    print("\nAll three checks fail, and they fail the same way: the fit puts "
          "points near\nlines without finding the pitch. The markings are "
          "detected correctly -- the\nprojection that turns them into metres "
          "is not consistent enough between\nframes for a rigid pitch to "
          "match. The ground plane is the thing to fix.")


if __name__ == "__main__":
    main()
