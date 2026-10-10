"""How high the ball is, from the shape of its flight.

Placing the ball assumes it is on the grass. In the air it is not, and its
ray meets the grass far beyond it: an in-flight ball twice turned a cross
into a 45-57 m/s "shot", and cost a real out-of-play crossing whose ball
read 19 m in 0.24 s as it went over the bar.

A ball in flight is not free to be anywhere along its rays, though. Between
touches it follows a parabola under gravity, and gravity is a known
acceleration in metres. So a run of rays -- one per frame, from a camera
whose pose is known on each -- pins the flight in 3-D: six unknowns
(position and velocity at the start) against two per frame. The rolling
ball is the same with height held at zero and four unknowns.

Each short window is fitted both ways. The flight is believed only where it
fits the pixels clearly better than rolling does and stays above the grass;
otherwise the ball is on the ground, which is what the placer already
assumed.

Coordinates are the calibrated goal's frame: X across from the left post,
Y up, Z out onto the pitch.
"""

from __future__ import annotations

import numpy as np

GRAVITY_MS2 = 9.81

# How long a stretch of flight each fit sees, in seconds, and the fewest
# detections it needs inside that. Measured on synthetic flights from a
# located camera with a pixel of noise, 40 draws each:
#
#     window     flights recognised    median height error    rolling balls
#     0.48 s     16-17 / 30            2.4-2.7 m              30 / 30
#     0.64 s     25-26 / 40            0.8-1.0 m              40 / 40
#     0.80 s     34-36 / 40            0.4 m                  40 / 40
#     1.00 s     40 / 40               0.22-0.26 m            40 / 40
#
# Over half a second gravity bends the path by less than the detector's
# noise can resolve; over a second it is unmistakable. Windows are by time,
# not by frame count, because real tracks have gaps.
WINDOW_S = 1.0
MIN_DETECTIONS = 12

# How much better flight must fit than rolling before the ball is believed
# to be in the air: its residual at most this share of rolling's. A ratio,
# not a pixel count, because the noise it has to beat is the camera pose's,
# and that differs by clip -- on stoke_4207 a still ball misfits by 6 px
# through the per-frame box poses alone. Measured on synthetic 1 s windows
# with that much pose wobble, a flight misfits rolling by 9-12 px against
# its own 5.5, and a grass pass by 5.6 against 5.5.
MAX_AIR_RATIO = 0.7

# A flight whose fitted height dips below this, in metres, is not a flight
# above this pitch: it has fitted the noise with a trajectory through the
# ground.
MIN_HEIGHT_M = -0.5

# Neither model is believed on a window it fits worse than this many times
# the clip's noise floor. A window that straddles a kick -- a still ball,
# then a struck one -- is no single parabola, and the first version fitted
# one anyway: on stoke_4207 it put the ball 20-55 m up, by the camera,
# rather than admit the window held two motions. At 2.5 times the floor a
# rolling fit still passed on a window half still ball and half flight,
# spreading its error thinly, and called a ball 1.8 m up on the grass; at
# 1.5 those frames are left unknown, which is the honest answer for them.
MAX_FIT_NOISE = 1.5

# The noise floor is estimated per clip as this quantile of the rolling
# fit's residual over all windows: most windows hold a ball on the grass,
# so the low end of that distribution is what pose and detector noise
# alone produce.
NOISE_QUANTILE = 25

# What a flight may do. Higher than the tallest clearance, faster than the
# hardest shot, or off beyond the hoardings is not a football's flight.
MAX_FLIGHT_HEIGHT_M = 30.0
MAX_FLIGHT_SPEED_MS = 45.0
PITCH_REACH_M = 15.0


def ray(pose, u: float, v: float):
    """Camera centre and unit direction through a pixel, in the goal frame."""
    import cv2

    rot, _ = cv2.Rodrigues(pose.rvec)
    eye = (-rot.T @ pose.tvec).ravel()
    direction = rot.T @ (np.linalg.inv(pose.camera_matrix)
                         @ np.array([u, v, 1.0]))
    return eye, direction / np.linalg.norm(direction)


def _positions(params, t, airborne: bool):
    """Ball positions at times `t` for a flight or a roll."""
    t = np.asarray(t, dtype=float)[:, None]
    if airborne:
        p0, vel = params[:3], params[3:6]
        pos = p0 + vel * t
        pos[:, 1] -= 0.5 * GRAVITY_MS2 * t[:, 0] ** 2
        return pos
    x0, z0, vx, vz = params
    return np.column_stack([x0 + vx * t[:, 0], np.zeros(len(t)),
                            z0 + vz * t[:, 0]])


def _project(pose, points):
    import cv2

    img, _ = cv2.projectPoints(np.asarray(points, dtype=float), pose.rvec,
                               pose.tvec, pose.camera_matrix, None)
    return img.reshape(-1, 2)


def fit(times, poses, pixels, airborne: bool):
    """Fit one model to a window. (params, rms px, positions, px per point).

    Solved linearly first -- a point on a ray satisfies d x (p - eye) = 0,
    which is linear in the unknowns because gravity's term is known -- then
    refined on pixel residuals, which is the error the detector makes.
    """
    from scipy.optimize import least_squares

    t = np.asarray(times, dtype=float) - float(times[0])
    rows, rhs = [], []
    for ti, pose, (u, v) in zip(t, poses, pixels):
        eye, d = ray(pose, u, v)
        cross = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]],
                          [-d[1], d[0], 0]])
        if airborne:
            basis = np.hstack([np.eye(3), np.eye(3) * ti])
            known = np.array([0.0, -0.5 * GRAVITY_MS2 * ti ** 2, 0.0])
        else:
            basis = np.array([[1, 0, ti, 0], [0, 0, 0, 0], [0, 1, 0, ti]],
                             dtype=float)
            known = np.zeros(3)
        rows.append(cross @ basis)
        rhs.append(cross @ (eye - known))
    A, b = np.vstack(rows), np.concatenate(rhs)
    try:
        start = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    observed = np.asarray(pixels, dtype=float)

    def residuals(params):
        pos = _positions(params, t, airborne)
        return np.concatenate([_project(pose, p[None])[0] - o
                               for pose, p, o in zip(poses, pos, observed)])

    got = least_squares(residuals, start, method="lm", max_nfev=400)
    rms = float(np.sqrt(np.mean(got.fun ** 2)))
    per_point = np.hypot(got.fun[0::2], got.fun[1::2])
    return got.x, rms, _positions(got.x, t, airborne), per_point


def classify(times, poses, pixels, noise_px: float = 1.0):
    """Flight or roll for one window. (airborne, positions, report) or None.

    `noise_px` is the clip's noise floor (`track_heights` estimates it).
    """
    ground = fit(times, poses, pixels, airborne=False)
    if ground is None:
        return None
    report = {"ground_px": ground[1]}
    air = fit(times, poses, pixels, airborne=True)
    if air is not None:
        report["air_px"] = air[1]
        report["height_m"] = (float(air[2][:, 1].min()),
                              float(air[2][:, 1].max()))
        if (air[1] <= MAX_AIR_RATIO * ground[1]
                and air[1] <= MAX_FIT_NOISE * noise_px
                and plausible_flight(air[0], air[2])):
            report["per_point"] = air[3]
            return True, air[2], report
    if ground[1] <= MAX_FIT_NOISE * noise_px:
        report["per_point"] = ground[3]
        return False, ground[2], report
    return None                          # two motions, or neither


def plausible_flight(params, positions) -> bool:
    """Could a football do this: height, speed, and where it is."""
    from . import pitch_model as pm
    from .goal_placer import to_pitch

    heights = positions[:, 1]
    if heights.min() < MIN_HEIGHT_M or heights.max() > MAX_FLIGHT_HEIGHT_M:
        return False
    if float(np.linalg.norm(params[3:6])) > MAX_FLIGHT_SPEED_MS:
        return False
    for x, _, z in positions:
        px, py = to_pitch(x, z)
        if not (-PITCH_REACH_M <= px <= pm.PITCH_LENGTH_M + PITCH_REACH_M
                and -PITCH_REACH_M <= py <= pm.PITCH_WIDTH_M + PITCH_REACH_M):
            return False
    return True


def track_heights(frames, times, pixels, pose_at, segments=None,
                  window_s: float = WINDOW_S,
                  min_detections: int = MIN_DETECTIONS):
    """Height and 3-D position per detection, where a window can be fitted.

    `pose_at(frame)` gives the camera pose on a frame, or None. A window is
    every detection within `window_s` centred on this one, in the same
    tracked segment -- across a break the tracker lost the ball, and there
    is no single flight to fit. Returns {frame: (x, y, z, airborne)} in the
    goal frame.
    """
    frames, times = list(frames), np.asarray(times, dtype=float)
    segments = list(segments) if segments is not None else [0] * len(frames)

    def window(i, lo_s, hi_s):
        near = [k for k in range(len(frames))
                if lo_s <= times[k] - times[i] <= hi_s
                and segments[k] == segments[i]]
        if len(near) < min_detections:
            return None
        if times[near[-1]] - times[near[0]] < 0.8 * window_s:
            return None                  # the detections bunch at one end
        poses = [pose_at(frames[k]) for k in near]
        if any(p is None for p in poses):
            return None
        return near, poses

    # The clip's noise floor, from the rolling fit on centred windows.
    rolls = []
    for i in range(0, len(frames), 3):
        got = window(i, -window_s / 2, window_s / 2)
        if got is None:
            continue
        near, poses = got
        g = fit(times[near], poses, [pixels[k] for k in near], False)
        if g is not None:
            rolls.append(g[1])
    if not rolls:
        return {}
    noise = max(1.0, float(np.percentile(rolls, NOISE_QUANTILE)))

    out = {}
    for i in range(len(frames)):
        # A window ending here, centred here, and starting here. A frame
        # just after a kick is only in one clean window -- the one that
        # starts at the kick -- and a centred window alone would always mix
        # the kick in.
        best = None
        for lo_s, hi_s in ((-window_s, 0.0), (-window_s / 2, window_s / 2),
                           (0.0, window_s)):
            got = window(i, lo_s, hi_s)
            if got is None:
                continue
            near, poses = got
            got = classify(times[near], poses, [pixels[k] for k in near],
                           noise)
            if got is None:
                continue
            airborne, positions, report = got
            # Judged where it matters: at this frame and its neighbours. A
            # window mostly of a still ball fits well on average while
            # missing the flight that starts at its end -- which is the
            # frame being asked about.
            at = near.index(i)
            local = float(np.mean(report["per_point"][max(0, at - 2):at + 3]))
            if local > MAX_FIT_NOISE * noise:
                continue
            if best is None or local < best[0]:
                best = (local, airborne, positions[at])
        if best is None:
            continue
        _, airborne, (x, y, z) = best
        out[frames[i]] = (float(x), float(y), float(z), bool(airborne))
    return out


def selftest(verbose: bool = True) -> bool:
    """A cross in the air and a pass on the grass, seen by a known camera.

    The camera is where the Reading clips located theirs; the frames carry
    a pixel of detector noise.
    """
    from .goal_pose import GoalPose
    import cv2

    from .midfield_pose import look_at

    rng = np.random.default_rng(11)
    eye = np.array([-64.0, 18.8, 52.3])
    focal, cx, cy = 2800.0, 640.0, 360.0

    def pose_looking_at(target):
        rot = look_at(eye, target)
        rvec, _ = cv2.Rodrigues(rot)
        return GoalPose(rvec=rvec, tvec=(-rot @ eye).reshape(3, 1),
                        focal_px=focal, cx=cx, cy=cy, reprojection_px=0.0,
                        n_corners=0)

    ok = True
    frames_all = np.arange(int(WINDOW_S * 25) + 1)
    cases = (
        # A cross: 17 m/s across the box, rising at 7 m/s from 0.3 m.
        ("cross in the air", np.array([20.0, 0.3, 14.0]),
         np.array([-17.0, 7.0, -5.0]), True),
        # A long ball, rising at 12 m/s.
        ("lofted ball", np.array([40.0, 0.3, 30.0]),
         np.array([-15.0, 12.0, -10.0]), True),
        # A pass along the grass at 14 m/s.
        ("pass on the grass", np.array([15.0, 0.11, 25.0]),
         np.array([-6.0, 0.0, -12.5]), False),
    )
    for name, p0, vel, airborne in cases:
        right, errs = 0, []
        for draw in range(20):
            # Half the frames missing, as on a real track.
            keep = np.sort(rng.choice(frames_all, size=len(frames_all) // 2
                                      + 1, replace=False))
            t = keep / 25.0
            truth = p0 + vel * t[:, None]
            if airborne:
                truth[:, 1] -= 0.5 * GRAVITY_MS2 * t ** 2
            pose = pose_looking_at(truth[len(t) // 2])
            pixels = [_project(pose, p[None])[0] + rng.normal(0, 1.0, 2)
                      for p in truth]
            got = classify(t, [pose] * len(t), pixels)
            if got is None:
                errs.append(np.inf)
                continue
            said, positions, _ = got
            right += said == airborne
            errs.append(float(np.max(np.abs(positions[:, 1]
                                            - truth[:, 1]))))
        med = float(np.median(errs))
        good = right >= 18 and med < (0.5 if airborne else 0.2)
        ok &= good
        if verbose:
            print(f"  {name:>18s}  right on {right}/20 draws with half the "
                  f"frames missing, median height error {med:.2f} m   "
                  f"{'ok' if good else 'WRONG'}")
    # A still ball for a second, then struck into the air: the kick must not
    # be explained as a flight 20-55 m up, as the first version did on real
    # footage. Every frame is either left unknown or given a height within
    # a metre of the truth.
    t = np.arange(0, 2.0, 0.04)
    kick = 1.0
    truth = np.tile([22.0, 0.11, 30.0], (len(t), 1)).astype(float)
    after = t > kick
    dt = t[after] - kick
    truth[after] += np.column_stack([-16.0 * dt, 6.0 * dt, -8.0 * dt])
    truth[after, 1] -= 0.5 * GRAVITY_MS2 * dt ** 2
    truth[:, 1] = np.maximum(truth[:, 1], 0.11)
    pose = pose_looking_at(truth[len(t) // 2])
    pixels = [_project(pose, p[None])[0] + rng.normal(0, 1.0, 2)
              for p in truth]
    got = track_heights(range(len(t)), t, pixels, lambda f: pose)
    worst = max((abs(got[f][1] - truth[f, 1]) for f in got), default=0.0)
    good = worst < 1.0
    ok &= good
    if verbose:
        print(f"  {'still, then kicked':>18s}  {len(got)} of {len(t)} frames "
              f"given a height, worst {worst:.2f} m off   "
              f"{'ok' if good else 'WRONG'}")
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
