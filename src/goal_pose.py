"""Distance and angle to goal, from four corners of the goal mouth.

The pipeline has never been able to say where a shot was taken from. The
centre-circle anchor is fitted at midfield and the camera following play into
the box does not show the centre circle, so shot recall is 0 of 10 and goals
0 of 2. A learned goal detector finds the goal at 6 of 6 shot moments, which
fixes *where the goal is* but not *how far away anything is*: put a bounding
box through the ground plane and it measures a goal 4.7 m wide where the laws
of the game say 7.32. An axis-aligned rectangle has no orientation in it.

The goal mouth is a rectangle of known size -- 7.32 by 2.44 m, fixed by the
laws -- so four of its image points determine the camera's pose completely.
That is a textbook PnP problem with an exact planar solver, and it needs no
pitch markings, no homography and no anchor.

## The ground comes free

The two post bases stand on the pitch, so in a frame with its origin at the
left post base and Y up, the pitch *is* the plane Y = 0. Any image point can
then be back-projected as a ray and intersected with it. A ball on the
ground at image (u, v) lands at a known (X, Z) in metres from the goal, and
distance and angle follow directly.

The assumption to keep in view: the ball must be on the ground. A ball in
flight is above the plane, and intersecting its ray with Y = 0 puts it
further away than it is. `ball_tracking.kinematics` already knows when the
ball is moving too fast to be held, and the same signal marks the frames
where this number should not be trusted.

## What three and two corners give

Four corners is the full pose. Three still determines it -- three points is
the classical minimum, with the fourth resolving the ambiguity -- and the
solver is given the fourth when it exists. With two, pose is not recoverable
and this returns None rather than a number that looks like a measurement.
Two are still worth labelling: they train a keypoint model, which is a
separate use.

## Checked against a camera whose answer is known

`selftest` builds a synthetic camera, projects the goal and a ball at a
measured distance, and asks this module to recover it. That is the only
check available before real corner labels exist, and it tests the algebra
rather than the footage -- a real-data check follows when the labels land.
Noiseless, it recovers the camera to 0.000 m and the distance exactly.

## How much a mis-click costs

Exact recovery on exact input says nothing about real labels, so
`noise_sensitivity` perturbs the corners and reports what survives. On a
goal spanning 288 px, for a shot at a true 16.0 m:

    click error    median distance error    90th pct    refused
      0.5 px             0.21 m              0.52 m        0%
      1   px             0.41 m              1.07 m        0%
      2   px             0.79 m              2.07 m        8%
      3   px             1.12 m              3.22 m       17%

Steep, and it is why the labelling page crops to the known box and
magnifies. Clicking to within three screen pixels at the page's median 3.4x
is under a pixel on the original, so about 0.4 m -- inside the bin width any
xG model would use. Clicking the same three pixels on an unmagnified frame
costs 1.1 m and loses one frame in six.

The refusals are the flip guard earning its place: with noisy corners the
planar solver will happily return a mirrored pose that reprojects perfectly,
and a camera below the pitch is the tell.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# The laws of the game. Not measured, not fitted, not a parameter.
GOAL_WIDTH_M = 7.32
GOAL_HEIGHT_M = 2.44

# Goal-frame model points, in the labeller's click order:
#   left post base, left post top, right post top, right post base.
# Origin at the left post base, X along the goal line towards the right
# post, Y up, Z out onto the pitch. A right-handed frame.
MODEL = np.array([
    [0.0,           0.0,            0.0],
    [0.0,           GOAL_HEIGHT_M,  0.0],
    [GOAL_WIDTH_M,  GOAL_HEIGHT_M,  0.0],
    [GOAL_WIDTH_M,  0.0,            0.0],
], dtype=np.float64)

# Fewer than this many clicked corners and the pose is not determined.
MIN_CORNERS = 4

# Beyond this, the answer is not a shot and probably not a position either.
#
# The ground intersection is a ray meeting a plane, and a ray that grazes
# the plane meets it arbitrarily far away: a few pixels near the horizon
# become tens of metres. Run on real clicks, two of four shot moments came
# back at 171.7 m and 50.9 m, both with the detected "ball" high in the
# frame -- in flight, above the plane, or a false positive. A pitch is about
# 105 m end to end and nobody shoots from 60, so a number past this is
# evidence the input was not a ball resting on the grass.
#
# Refusing is the point. A distance is worth having only when it is a
# measurement, and the failure mode here returns something that looks like
# one.
MAX_PITCH_DISTANCE_M = 60.0

# A reprojection worse than this means the corners do not describe a goal:
# a mis-click, a corner put on the net rather than the frame, or a label
# whose order was scrambled. In pixels.
MAX_REPROJECTION_PX = 6.0


@dataclass
class GoalPose:
    """Where the camera is, relative to the goal."""

    rvec: np.ndarray
    tvec: np.ndarray
    focal_px: float
    cx: float
    cy: float
    reprojection_px: float
    n_corners: int

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([[self.focal_px, 0.0, self.cx],
                         [0.0, self.focal_px, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def summary(self) -> str:
        eye = self.camera_position()
        return (f"  camera at ({eye[0]:+.1f}, {eye[1]:+.1f}, {eye[2]:+.1f}) m "
                f"from the left post base\n"
                f"  reprojection {self.reprojection_px:.2f} px over "
                f"{self.n_corners} corners")

    def camera_position(self) -> np.ndarray:
        """Where the camera sits, in goal-frame metres."""
        rot, _ = cv2.Rodrigues(self.rvec)
        return (-rot.T @ self.tvec).ravel()


def solve(corners, focal_px: float, cx: float, cy: float) -> GoalPose | None:
    """Camera pose from clicked goal corners. None if underdetermined.

    `corners` is four entries in the labeller's order, each an (x, y) pixel
    pair or None for a corner that was not visible.
    """
    if corners is None or len(corners) != 4:
        return None
    have = [i for i, c in enumerate(corners) if c is not None]
    if len(have) < MIN_CORNERS:
        return None

    image_pts = np.array([corners[i] for i in have], dtype=np.float64)
    object_pts = MODEL[have]
    camera = np.array([[focal_px, 0.0, cx], [0.0, focal_px, cy],
                       [0.0, 0.0, 1.0]], dtype=np.float64)

    # IPPE is the planar solver: the goal mouth is a plane, and the general
    # iterative method is both slower here and happy to return a pose that
    # reprojects well while facing the wrong way.
    ok, rvec, tvec = cv2.solvePnP(
        object_pts, image_pts, camera, None,
        flags=cv2.SOLVEPNP_IPPE if len(have) == 4 else cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    projected, _ = cv2.projectPoints(object_pts, rvec, tvec, camera, None)
    err = float(np.sqrt(np.mean(np.sum(
        (projected.reshape(-1, 2) - image_pts) ** 2, axis=1))))

    pose = GoalPose(rvec=rvec, tvec=tvec, focal_px=float(focal_px),
                    cx=float(cx), cy=float(cy), reprojection_px=err,
                    n_corners=len(have))
    # A camera below the pitch means the solver has flipped the goal. The
    # reprojection cannot see this -- a mirrored pose fits the same points.
    if pose.camera_position()[1] <= 0.0:
        return None
    return pose


def ground_point(pose: GoalPose, x: float, y: float):
    """Image point on the pitch -> (X, Z) metres in the goal frame.

    The post bases stand on the pitch, so the pitch is the plane Y = 0 in
    this frame. The ray through the pixel is intersected with it.
    """
    rot, _ = cv2.Rodrigues(pose.rvec)
    eye = (-rot.T @ pose.tvec).ravel()
    ray = rot.T @ (np.linalg.inv(pose.camera_matrix)
                   @ np.array([x, y, 1.0], dtype=np.float64))
    if abs(ray[1]) < 1e-9:
        return None
    t = -eye[1] / ray[1]
    if t <= 0:
        return None          # the plane is behind the camera
    hit = eye + t * ray
    return float(hit[0]), float(hit[2])


def shot_geometry(pose: GoalPose, x: float, y: float):
    """Distance in metres and angle in degrees, from a ball on the pitch.

    Distance is to the centre of the goal line. The angle is the one xG
    models use: how much of the goal the shooter can see, the angle the two
    posts subtend at the ball. A tap-in on the line is near 180 degrees; a
    shot from the corner flag is near zero.
    """
    hit = ground_point(pose, x, y)
    if hit is None:
        return None
    ball = np.array(hit, dtype=np.float64)
    centre = np.array([GOAL_WIDTH_M / 2.0, 0.0])
    if float(np.linalg.norm(ball - centre)) > MAX_PITCH_DISTANCE_M:
        return None          # a grazing ray, not a ball on the grass
    left = np.array([0.0, 0.0]) - ball
    right = np.array([GOAL_WIDTH_M, 0.0]) - ball
    cosine = float(np.dot(left, right)
                   / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-9))
    return (float(np.linalg.norm(ball - centre)),
            float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))


def selftest(verbose: bool = True) -> bool:
    """Project a goal from a known camera, then recover the known answer."""
    focal, cx, cy = 1800.0, 640.0, 360.0
    camera = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])

    ok = True
    for name, eye, ball in (
        ("behind and above", np.array([3.66, 18.0, 42.0]),
         np.array([3.66, 0.0, 16.0])),
        ("off to one side", np.array([-22.0, 14.0, 30.0]),
         np.array([10.0, 0.0, 11.0])),
        ("high and close", np.array([3.66, 25.0, 20.0]),
         np.array([1.0, 0.0, 6.0])),
    ):
        # Look at the goal centre; build the rotation from that.
        target = np.array([GOAL_WIDTH_M / 2.0, GOAL_HEIGHT_M / 2.0, 0.0])
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rot = np.vstack([right, down, forward])
        rvec, _ = cv2.Rodrigues(rot)
        tvec = (-rot @ eye).reshape(3, 1)

        corners, _ = cv2.projectPoints(MODEL, rvec, tvec, camera, None)
        corners = [tuple(p) for p in corners.reshape(-1, 2)]
        ball_px, _ = cv2.projectPoints(ball.reshape(1, 3), rvec, tvec,
                                       camera, None)
        ball_px = ball_px.reshape(2)

        pose = solve(corners, focal, cx, cy)
        if pose is None:
            print(f"  {name:>18s}  FAILED to solve")
            ok = False
            continue
        got = shot_geometry(pose, float(ball_px[0]), float(ball_px[1]))
        truth_d = float(np.linalg.norm(
            np.array([ball[0], ball[2]])
            - np.array([GOAL_WIDTH_M / 2.0, 0.0])))
        err_pos = float(np.linalg.norm(pose.camera_position() - eye))
        err_d = abs(got[0] - truth_d)
        good = err_pos < 0.05 and err_d < 0.05
        ok &= good
        if verbose:
            print(f"  {name:>18s}  camera err {err_pos:6.3f} m   "
                  f"distance {got[0]:5.2f} m (true {truth_d:5.2f}, err "
                  f"{err_d:.3f})   angle {got[1]:5.1f} deg   "
                  f"{'ok' if good else 'WRONG'}")

    # Two corners must refuse rather than guess.
    if solve([(0.0, 0.0), None, None, (10.0, 0.0)], focal, cx, cy) is not None:
        print("  two corners returned a pose -- it must refuse")
        ok = False
    elif verbose:
        print(f"  {'two corners':>18s}  refused, as it should")
    return ok


def _synthetic(eye, ball, focal=1800.0, cx=640.0, cy=360.0):
    """Project the goal and a ball from a camera looking at the goal."""
    camera = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    target = np.array([GOAL_WIDTH_M / 2.0, GOAL_HEIGHT_M / 2.0, 0.0])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    rot = np.vstack([right, np.cross(forward, right), forward])
    rvec, _ = cv2.Rodrigues(rot)
    tvec = (-rot @ eye).reshape(3, 1)
    corners, _ = cv2.projectPoints(MODEL, rvec, tvec, camera, None)
    ball_px, _ = cv2.projectPoints(np.asarray(ball).reshape(1, 3), rvec, tvec,
                                   camera, None)
    return corners.reshape(-1, 2), ball_px.reshape(2)


def noise_sensitivity(trials: int = 400, seed: int = 7) -> None:
    """What a mis-click costs, in metres. The case for magnifying the page."""
    focal, cx, cy = 1800.0, 640.0, 360.0
    eye = np.array([GOAL_WIDTH_M / 2.0, 18.0, 42.0])
    ball = np.array([GOAL_WIDTH_M / 2.0, 0.0, 16.0])
    truth = 16.0
    corners, ball_px = _synthetic(eye, ball, focal, cx, cy)
    rng = np.random.default_rng(seed)

    span = float(np.hypot(*(corners[0] - corners[3])))
    print(f"\n  A goal spanning {span:.0f} px, a shot at a true "
          f"{truth:.2f} m.\n")
    print(f"  {'click err':>10s} {'median':>10s} {'90th pct':>9s} "
          f"{'refused':>8s}")
    for sd in (0.5, 1.0, 2.0, 3.0, 5.0):
        errors, refused = [], 0
        for _ in range(trials):
            noisy = [tuple(p + rng.normal(0, sd, 2)) for p in corners]
            pose = solve(noisy, focal, cx, cy)
            if pose is None:
                refused += 1
                continue
            moved = ball_px + rng.normal(0, sd, 2)
            got = shot_geometry(pose, float(moved[0]), float(moved[1]))
            if got is None:
                refused += 1
                continue
            errors.append(abs(got[0] - truth))
        errors = np.array(errors)
        print(f"  {sd:8.1f}px {np.median(errors):8.2f} m "
              f"{np.percentile(errors, 90):7.2f} m {refused/trials:7.0%}")


if __name__ == "__main__":
    print("Recovering a known camera from a projected goal.\n")
    good = selftest()
    noise_sensitivity()
    raise SystemExit(0 if good else 1)
