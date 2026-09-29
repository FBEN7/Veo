"""Place a ball on the pitch using the goal, not the centre circle.

`detect_shots.py` finds shots by putting the ball into pitch metres and
looking for a fast one heading at a goal. Until now the only way to do that
was the centre-circle anchor, and shot recall is **0 of 10** because of it:
the anchor is fitted at midfield, and a camera following play into the box
does not show the centre circle. The landmark that *is* in frame when shots
happen is the goal.

This provides the same service from the goal instead, so the detector's
logic is untouched and only its geometry changes.

## How a frame gets a pose

Two pieces, in order:

1. **Once per clip**, hand-clicked goal corners give the camera's position
   through `goal_pose.bundle`. A broadcast camera is bolted to a gantry, so
   that position holds for every frame of the clip -- the bundle reprojects
   all of a clip's corner frames from one position to under a pixel.
2. **Per frame**, the goal detector returns a box. With position already
   fixed, rotation and focal are the only unknowns -- four numbers, and a
   box is four numbers -- so `goal_pose.pose_from_box` solves it.

The cost of the labelling is therefore per clip, not per frame.

## Goal-frame to pitch-frame

`ground_point` answers in goal-frame metres: X across the goal line from the
left post, Z out onto the pitch. The pitch frame the detector works in puts
a goal at x = 0 and the pitch centre line at y = WIDTH/2, so

    pitch_x = Z
    pitch_y = X + (PITCH_WIDTH - GOAL_WIDTH) / 2

and the calibrated goal is always the "left" one as far as the detector is
concerned.

That is a real limitation and not a rounding of one: only the goal that was
calibrated can be shot at. A window showing play at both ends would need
both ends clicked. The labelled windows here are cut around a shot, so the
goal in question is the one in view.

## What it cannot fix

The ground intersection assumes the ball is on the grass, and a struck ball
is in the air. Taking the shooter's feet avoids that for a known shot
moment, but a *detector* has to work from the ball, because the ball's speed
and heading are what make a shot a shot. So the placed position is right
where the ball is low and increasingly wrong as it rises. The anchor
homography has exactly the same flaw, so this is not a regression -- but it
is a ceiling, and it belongs here rather than in a footnote.
"""

from __future__ import annotations

import numpy as np

from . import pitch_model as pm
from . import shot_geometry as sg
from .goal_pose import GOAL_WIDTH_M, ground_point, pose_from_box

# Detections below this are not trusted to carry geometry. Higher than the
# 0.15 the detector runs at for "is there a goal here", because a box that
# is only just a goal is not a box worth solving a camera from.
MIN_BOX_CONFIDENCE = 0.35

# A box fit worse than this is not describing the calibrated goal -- most
# likely the goal at the other end, or a false positive on a hoarding.
MAX_BOX_RESIDUAL_PX = 40.0

# Solve a pose every this many frames and reuse it in between. The camera
# moves smoothly and a pose costs an optimisation, so this is the same
# bargain `detect_shots.ANCHOR_GRID` strikes for anchors.
POSE_GRID = 4


def to_pitch(x: float, z: float):
    """Goal-frame metres to pitch metres, with the calibrated goal at x = 0."""
    return z, x + (pm.PITCH_WIDTH_M - GOAL_WIDTH_M) / 2.0


class GoalPlacer:
    """Ball pixels to pitch metres, through a goal pose per frame."""

    #: Which goal the calibration belongs to, in the detector's frame.
    side = "left"

    def __init__(self, poses: dict, grid: int = POSE_GRID):
        self.poses = poses
        self.grid = grid

    def __len__(self) -> int:
        return len(self.poses)

    def pose_at(self, frame: int):
        index = int(frame)
        pose = self.poses.get(index)
        if pose is not None:
            return pose
        for offset in range(1, self.grid // 2 + 1):
            for candidate in (self.poses.get(index - offset),
                              self.poses.get(index + offset)):
                if candidate is not None:
                    return candidate
        return None

    def place(self, frame, px, py):
        """Pitch coordinates for a pixel, or None where there is no pose."""
        pose = self.pose_at(frame)
        if pose is None:
            return None
        hit = ground_point(pose, float(px), float(py))
        if hit is None:
            return None
        x, y = to_pitch(hit[0], hit[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        # Off the pitch by more than a goal's width is a grazing ray, not a
        # ball: near the horizon a few pixels become tens of metres.
        if not (-sg.GOAL_HALF_M <= x <= pm.PITCH_LENGTH_M + sg.GOAL_HALF_M):
            return None
        if not (-pm.PITCH_WIDTH_M <= y <= 2 * pm.PITCH_WIDTH_M):
            return None
        return np.array([x, y])


def build(video_path: str, frames_wanted, camera_position, focal_seed: float,
          width: int, height: int, detector, grid: int = POSE_GRID,
          conf: float = MIN_BOX_CONFIDENCE, verbose: bool = False):
    """Solve a pose on a grid of frames, from detected goal boxes.

    `detector` is anything with ultralytics' `predict` interface, kept as an
    argument rather than loaded here: the weights are trained on footage
    under a non-commercial agreement and cannot live in this repository, so
    the caller supplies them.
    """
    import cv2

    wanted = sorted({int(f) - int(f) % grid for f in frames_wanted})
    if not wanted:
        return GoalPlacer({}, grid)

    cap = cv2.VideoCapture(video_path)
    poses, tried, detected = {}, 0, 0
    for index in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        tried += 1
        result = detector.predict(frame, conf=conf, verbose=False)[0]
        if not len(result.boxes):
            continue
        scores = result.boxes.conf.cpu().numpy()
        box = result.boxes.xyxy.cpu().numpy()[int(np.argmax(scores))]
        detected += 1
        pose = pose_from_box(box, camera_position, focal_seed,
                             width / 2.0, height / 2.0)
        if pose is None or pose.reprojection_px > MAX_BOX_RESIDUAL_PX:
            continue
        poses[index] = pose
    cap.release()

    if verbose:
        print(f"  [goal] {detected} goals found on {tried} sampled frames, "
              f"{len(poses)} gave a pose")
    return GoalPlacer(poses, grid)
