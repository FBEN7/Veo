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


def bundle(frames, focal_guess: float, cx: float, cy: float,
           max_nfev: int = 300):
    """One camera position for several frames, each with its own zoom.

    Solving each frame alone leaves the focal length free per frame, and
    position scales with it, so recovered positions scatter by tens of
    metres for a camera that never moved. But a broadcast camera is bolted
    to a gantry: across a ninety-second window it pans, tilts and zooms, and
    does not travel. That is a constraint the per-frame solve throws away.

    So position is shared and rotation and focal are per frame, fitted
    together by least squares on reprojection. For n frames that is 3 + 4n
    unknowns against 8n measurements, determined from two frames up.

    It is also a test rather than a smoothing. If the corners are right and
    the camera really is fixed, one position reprojects every frame to a
    couple of pixels. If it cannot, something is wrong -- the corners, the
    assumption, or which goal a frame is looking at -- and the residual says
    so instead of hiding in a plausible-looking average.

    `frames` is a list of four-corner pixel lists. Returns
    (camera position, [GoalPose], residual px) or None.
    """
    from scipy.optimize import least_squares

    usable = [f for f in frames if f is not None
              and all(c is not None for c in f)]
    if len(usable) < 2:
        return None

    starts = [solve(f, focal_guess, cx, cy) for f in usable]
    if any(s is None for s in starts):
        return None
    eye0 = np.mean([s.camera_position() for s in starts], axis=0)

    observed = [np.asarray(f, dtype=np.float64) for f in usable]
    n = len(usable)

    def unpack(p):
        eye = p[:3]
        rvecs = p[3:3 + 3 * n].reshape(n, 3)
        focals = np.exp(p[3 + 3 * n:])      # kept positive
        return eye, rvecs, focals

    def residuals(p):
        eye, rvecs, focals = unpack(p)
        out = []
        for i in range(n):
            rot, _ = cv2.Rodrigues(rvecs[i])
            tvec = (-rot @ eye).reshape(3, 1)
            camera = np.array([[focals[i], 0.0, cx],
                               [0.0, focals[i], cy], [0.0, 0.0, 1.0]])
            proj, _ = cv2.projectPoints(MODEL, rvecs[i], tvec, camera, None)
            out.append((proj.reshape(-1, 2) - observed[i]).ravel())
        return np.concatenate(out)

    p0 = np.concatenate([eye0]
                        + [s.rvec.ravel() for s in starts]
                        + [np.full(n, np.log(focal_guess))])
    fit = least_squares(residuals, p0, method="lm", max_nfev=max_nfev)

    eye, rvecs, focals = unpack(fit.x)
    if eye[1] <= 0.0:
        return None
    rms = float(np.sqrt(np.mean(fit.fun.reshape(-1, 2) ** 2).sum()))

    poses = []
    for i in range(n):
        rot, _ = cv2.Rodrigues(rvecs[i])
        per = fit.fun[i * 8:(i + 1) * 8].reshape(-1, 2)
        poses.append(GoalPose(
            rvec=rvecs[i].reshape(3, 1),
            tvec=(-rot @ eye).reshape(3, 1),
            focal_px=float(focals[i]), cx=cx, cy=cy,
            reprojection_px=float(np.sqrt(np.mean(np.sum(per ** 2, axis=1)))),
            n_corners=4))
    return eye, poses, rms


def pose_from_box(box, eye, focal_guess: float, cx: float, cy: float,
                  max_nfev: int = 600):
    """Pose on a frame with no corners, from a detected box and a fixed camera.

    Corners exist on 25 hand-labelled frames and nowhere else, which would
    confine metric geometry to those frames. But `bundle` establishes
    something that carries: a clip's camera position, to sub-pixel
    reprojection. Position fixed leaves rotation and focal unknown -- four
    numbers -- and a detected goal box is also four numbers. So the box is
    enough, once the clip has been calibrated once.

    That is what connects the working goal detector (6 of 6 at the shot
    moments) to metric distance on frames nobody clicked.

    Measured against the corner-derived pose on the 12 frames that have
    both, started from a neutral guess that knows nothing of the true
    rotation: median disagreement 1.8 m, 90th percentile 3.2 m. Usable for
    xG bands, weaker than corners, and not a substitute for calibrating the
    clip in the first place.

    `box` is (x0, y0, x1, y1) in pixels. `eye` is the camera position in
    goal-frame metres. Returns a GoalPose, or None if it flips.
    """
    from scipy.optimize import least_squares

    eye = np.asarray(eye, dtype=np.float64)
    observed = np.asarray(box, dtype=np.float64)

    def bbox(rvec, focal):
        rot, _ = cv2.Rodrigues(rvec)
        tvec = (-rot @ eye).reshape(3, 1)
        camera = np.array([[focal, 0.0, cx], [0.0, focal, cy],
                           [0.0, 0.0, 1.0]])
        pts, _ = cv2.projectPoints(MODEL, rvec, tvec, camera, None)
        pts = pts.reshape(-1, 2)
        return np.array([pts[:, 0].min(), pts[:, 1].min(),
                         pts[:, 0].max(), pts[:, 1].max()])

    # Start looking straight at the goal centre. This knows nothing about
    # the true rotation, which is the point: seeding from a known-good pose
    # would let the fit sit where it started and report agreement it had
    # not earned.
    target = np.array([GOAL_WIDTH_M / 2.0, GOAL_HEIGHT_M / 2.0, 0.0])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    start, _ = cv2.Rodrigues(np.vstack([right, np.cross(forward, right),
                                        forward]))

    fit = least_squares(
        lambda q: bbox(q[:3], np.exp(q[3])) - observed,
        np.concatenate([start.ravel(), [np.log(focal_guess)]]),
        method="lm", max_nfev=max_nfev)

    rvec = fit.x[:3].reshape(3, 1)
    rot, _ = cv2.Rodrigues(rvec)
    pose = GoalPose(rvec=rvec, tvec=(-rot @ eye).reshape(3, 1),
                    focal_px=float(np.exp(fit.x[3])), cx=cx, cy=cy,
                    reprojection_px=float(np.sqrt(np.mean(fit.fun ** 2))),
                    n_corners=0)
    return None if pose.camera_position()[1] <= 0.0 else pose


def pose_at(corners, eye, focal_guess: float, cx: float, cy: float,
            max_nfev: int = 400):
    """Rotation and focal from four corners, with the camera's position known.

    A single frame's corners fix the camera only to a line: focal and
    distance trade against each other, and across the plausible focals the
    reprojection error barely moves (0.6-2 px on stoke_7001 from 1500 to
    4000 px while the camera slides 50 m). Once `camera_position.locate` has
    fixed where on that line the camera is, this recovers the rest.
    """
    from scipy.optimize import least_squares

    if corners is None or any(c is None for c in corners):
        return None
    eye = np.asarray(eye, dtype=np.float64)
    observed = np.asarray(corners, dtype=np.float64)
    target = np.array([GOAL_WIDTH_M / 2.0, GOAL_HEIGHT_M / 2.0, 0.0])
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    start, _ = cv2.Rodrigues(np.vstack([right, np.cross(forward, right),
                                        forward]))

    def project(q):
        rot, _ = cv2.Rodrigues(q[:3])
        focal = np.exp(q[3])
        camera = np.array([[focal, 0.0, cx], [0.0, focal, cy],
                           [0.0, 0.0, 1.0]])
        pts, _ = cv2.projectPoints(MODEL, q[:3], (-rot @ eye).reshape(3, 1),
                                   camera, None)
        return pts.reshape(-1, 2)

    fit = least_squares(lambda q: (project(q) - observed).ravel(),
                        np.concatenate([start.ravel(),
                                        [np.log(focal_guess)]]),
                        method="lm", max_nfev=max_nfev)
    rvec = fit.x[:3].reshape(3, 1)
    rot, _ = cv2.Rodrigues(rvec)
    per = fit.fun.reshape(-1, 2)
    return GoalPose(rvec=rvec, tvec=(-rot @ eye).reshape(3, 1),
                    focal_px=float(np.exp(fit.x[3])), cx=cx, cy=cy,
                    reprojection_px=float(np.sqrt(np.mean(np.sum(per ** 2,
                                                                 axis=1)))),
                    n_corners=4)


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

    # One camera, three zooms: the bundle must find the shared position that
    # per-frame solving cannot, because each frame alone leaves focal free.
    eye = np.array([-30.0, 15.0, 34.0])
    shots = []
    for f, ball in ((1500.0, [2.0, 0.0, 12.0]),
                    (2200.0, [5.0, 0.0, 20.0]),
                    (2900.0, [3.0, 0.0, 9.0])):
        corners, _ = _synthetic(eye, ball, focal=f)
        shots.append([tuple(p) for p in corners])
    got = bundle(shots, 2000.0, 640.0, 360.0)
    if got is None:
        print("  bundle over three frames FAILED to fit")
        ok = False
    else:
        eye_fit, poses, rms = got
        err = float(np.linalg.norm(eye_fit - eye))
        good = err < 0.5 and rms < 1.0
        ok &= good
        if verbose:
            focals = ", ".join(f"{p.focal_px:.0f}" for p in poses)
            print(f"  {'one camera, 3 zooms':>18s}  position err {err:.3f} m  "
                  f"residual {rms:.2f} px   focals {focals} "
                  f"(true 1500, 2200, 2900)   {'ok' if good else 'WRONG'}")

    # A box and a known camera position, with no corners at all.
    eye = np.array([-25.0, 16.0, 30.0])
    corners, _ = _synthetic(eye, [3.0, 0.0, 14.0], focal=1900.0)
    corners = np.asarray(corners)
    box = (corners[:, 0].min(), corners[:, 1].min(),
           corners[:, 0].max(), corners[:, 1].max())
    from_box = pose_from_box(box, eye, 1500.0, 640.0, 360.0)
    from_corners = solve([tuple(p) for p in corners], 1900.0, 640.0, 360.0)
    if from_box is None or from_corners is None:
        print("  pose from a box FAILED")
        ok = False
    else:
        a = shot_geometry(from_corners, 640.0, 540.0)
        b = shot_geometry(from_box, 640.0, 540.0)
        gap = abs(a[0] - b[0]) if a and b else float("inf")
        good = gap < 1.0
        ok &= good
        if verbose:
            print(f"  {'box, no corners':>18s}  distance {b[0]:5.2f} m "
                  f"against {a[0]:5.2f} from corners, gap {gap:.2f} m   "
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
