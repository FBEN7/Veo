"""Where the broadcast camera is, from the goal and the centre circle together.

## Why neither landmark is enough alone

Four clicked goal corners fix the camera only to a line. Focal length and
distance trade against each other: on stoke_7001's one usable corner frame,
assuming 1500, 2500 and 4000 px puts the camera at

    (-32.2, 10.3, 24.6)   (-55.0, 16.2, 41.1)   (-89.5, 25.1, 65.7)

with reprojection errors of 2.1, 1.0 and 0.6 px, so nothing in the corners
picks one. The pipeline took the ground plane's focal (1677 px) and
got (-36.2, 11.4, 27.5). A bundle over several corner frames pins the
point only if the framings differ enough; reading_2519's three landed at
z = 48.8, the others had one frame each.

The centre circle has the same problem the other way round. Freed
completely, the circle-plus-halfway fit on stoke_7001 passed 71 of 73
frames, with every freed camera at z = 51-53, level with the halfway line,
but scattered 98 m along the line of sight -- the same dolly-zoom
ambiguity.

## Two lines cross at a point

So the camera is placed where they meet. The corners give one line; each
centre-circle frame, fitted with the camera free, gives another from the
centre spot; the clip's camera is the median of where they cross, over the
frames whose lines pass within `MAX_MISS_M` of each other.

Measured on the six labelled clips:

    clip          voting    camera (x, up, z)      lines miss   spread
    reading_0737  29 / 30   (-63.4, 18.4, 52.4)    0.1 m        0.2 m
    reading_1155  26 / 30   (-64.1, 18.8, 52.3)    0.3 m        0.5 m
    reading_2519  21 / 30   (-63.0, 19.1, 52.4)    0.5 m        0.2 m
    stoke_4207    11 / 11   ( 76.2, 19.8, 52.7)    1.3 m        0.3 m
    stoke_7001    21 / 30   (-70.8, 20.3, 52.2)    1.6 m        0.7 m
    stoke_1302    refused, 2 frames

Three things nothing forced: every camera lands on the halfway line
(z = 52.2-52.7, the line at 52.5), where a main broadcast camera sits; the
three Reading clips, fitted independently, agree to 1.1 m; and the two
Stoke clips, calibrated from opposite goals, sit 74.5 and 72.5 m out from
the goal's centreline at 20.3 and 19.8 m up. Player heights predicted by
the refitted midfield poses move to 1.01 and 1.07 of observed.

One assumption carries through: the circle is placed half a 105 m pitch
from the calibrated goal. On a 100-102 m pitch the camera moves 1-2.5 m
along the pitch with it.

## What it cannot see

The goal and the centre spot both sit on the pitch's long axis. A second
camera displaced within the plane through that axis and the first -- the
same height and distance out, further along the touchline -- gives lines
that still cross, so circle frames from it would be accepted and averaged
in. `selftest` pins that as known behaviour. Out of that plane (a lower
gantry, a camera nearer the touchline) the lines miss by metres and the
frames are refused.

Coordinates are the calibrated goal's frame: X across the goal line from
the left post, Y up, Z out onto the pitch.
"""

from __future__ import annotations

import numpy as np

from .goal_pose import MAX_REPROJECTION_PX, solve

# Focal lengths swept to trace each corner frame's line of possible cameras.
# The span of a broadcast lens at 1280 wide, as in `midfield_pose`.
RAY_FOCALS = np.geomspace(1000.0, 9000.0, 16)

# A circle frame whose fit is worse than this, in pixels, does not vote.
# Same gate as the midfield placer.
MAX_CIRCLE_RMS_PX = 8.0

# How closely a circle frame's line must pass the corners' line, in metres,
# for it to vote. Two lines in space almost never meet exactly; lines from
# one camera come within noise of each other, and lines from a different
# camera miss by roughly how far apart the two cameras are.
MAX_MISS_M = 3.0

# Fewer voting frames than this and the clip keeps its old position. Five
# is enough for a median to shrug off one bad frame.
MIN_VOTES = 5

# How much the voting frames may disagree along the line, as an
# interquartile range in metres, before the answer is refused.
MAX_SPREAD_M = 5.0

# Below this angle between the two lines, in degrees, where they cross is
# poorly determined. The measured clips sit near 35.
MIN_CROSSING_DEG = 10.0

# Circle frames fitted to locate the camera. Each fit takes a second or so,
# and the answer settles long before this; spread evenly over the clip.
MAX_LOCATE_FRAMES = 30


def corner_ray(corner_frames, cx: float, cy: float):
    """The line of cameras the corners allow. (origin, direction, spread m).

    `corner_frames` is a list of four-corner pixel lists. The direction
    points out onto the pitch (+Z). Several frames from one camera give
    lines through the same point; for the near-parallel views of one goal
    from one gantry, a single line fitted through all of them is close, and
    `spread` says how close.
    """
    points = []
    for corners in corner_frames:
        if corners is None or any(c is None for c in corners):
            continue
        for focal in RAY_FOCALS:
            pose = solve(corners, float(focal), cx, cy)
            if pose is not None and pose.reprojection_px <= MAX_REPROJECTION_PX:
                points.append(pose.camera_position())
    if len(points) < 3:
        return None
    points = np.asarray(points)
    origin = points.mean(axis=0)
    direction = np.linalg.svd(points - origin)[2][0]
    if direction[2] < 0:
        direction = -direction
    rel = points - origin
    off = np.linalg.norm(rel - np.outer(rel @ direction, direction), axis=1)
    return origin, direction, float(np.median(off))


def closest_approach(origin_a, dir_a, origin_b, dir_b):
    """Where two lines pass nearest. (along a, along b, miss distance m)."""
    w = origin_a - origin_b
    a, b, c = dir_a @ dir_a, dir_a @ dir_b, dir_b @ dir_b
    d, e = dir_a @ w, dir_b @ w
    denom = a * c - b * b
    if denom < 1e-12:
        return None
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    miss = float(np.linalg.norm((origin_a + s * dir_a) - (origin_b + t * dir_b)))
    return float(s), float(t), miss


def locate(circles, ray, start, focal_seed: float, cx: float, cy: float,
           limit: int = MAX_LOCATE_FRAMES):
    """The camera, where the centre circle's lines cross the corners' line.

    `circles` is a list of (arc pixels, halfway endpoints) from frames
    showing the centre circle with its halfway line. Each is fitted with
    the camera free; that pins the line from the centre spot to the camera
    though not the point on it, and where that line passes the corners'
    line is this frame's vote. How far apart the two lines pass is the
    check: a circle seen from a different camera misses.

    The first version slid the camera along the corners' line and fitted
    each circle there. It could not refuse anything: a synthetic second
    camera 30 m down the touchline still fitted at 1.6 px, pulled onto the
    line at a wrong point. Fitting free and intersecting keeps the miss
    distance, which is what it lacked.

    `start` only seeds the free fits. Returns (position, report), with
    position None when too few frames agree.
    """
    from .midfield_pose import CENTRE, fit_free

    origin, direction, _ = ray
    if len(circles) > limit:
        pick = np.linspace(0, len(circles) - 1, limit).round().astype(int)
        circles = [circles[i] for i in sorted(set(pick))]

    along, misses, fitted = [], [], 0
    for arc, half in circles:
        eye, _, _, rms = fit_free(arc, half, start, focal_seed, cx, cy)
        if rms > MAX_CIRCLE_RMS_PX:
            continue
        fitted += 1
        sight = eye - CENTRE
        sight /= np.linalg.norm(sight)
        crossing = np.degrees(np.arccos(min(1.0, abs(sight @ direction))))
        if crossing < MIN_CROSSING_DEG:
            continue
        got = closest_approach(origin, direction, CENTRE, sight)
        if got is None or got[1] <= 0:
            continue
        misses.append(got[2])
        if got[2] <= MAX_MISS_M:
            along.append(got[0])

    report = {"tried": len(circles), "fitted": fitted, "agreeing": len(along),
              "miss_m": float(np.median(misses)) if misses else None}
    if len(along) < MIN_VOTES:
        report["refused"] = (f"{len(along)} frames cross the corners' line, "
                             f"{MIN_VOTES} needed")
        return None, report
    along = np.asarray(along)
    spread = float(np.percentile(along, 75) - np.percentile(along, 25))
    report["spread_m"] = spread
    if spread > MAX_SPREAD_M:
        report["refused"] = (f"frames disagree by {spread:.1f} m along the "
                             f"line, {MAX_SPREAD_M:.0f} m allowed")
        return None, report
    return origin + float(np.median(along)) * direction, report


def scan_circles(video_path: str, frames, find_circle, halfway_line,
                 max_arc: int = 300):
    """Centre circles with their halfway line, on the given frames.

    Returns {frame: (arc pixels, halfway endpoints or None)}, only for
    frames where a circle was found. `find_circle` and `halfway_line` are
    passed in for the same reason as in `goal_placer.build_midfield`.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    out = {}
    for index in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        circle = find_circle(frame, np.random.default_rng(int(index)))
        if circle is None:
            continue
        arc = np.asarray(circle["support"], dtype=float)
        if len(arc) > max_arc:
            arc = arc[np.linspace(0, len(arc) - 1, max_arc).astype(int)]
        (ex, ey), _, _ = circle["ellipse"]
        half = halfway_line(circle["segments"], (ex, ey))
        out[int(index)] = (arc, None if half is None else np.array(
            [[half[0], half[1]], [half[2], half[3]]], dtype=float))
    cap.release()
    return out


def selftest(verbose: bool = True) -> bool:
    """A known camera, recovered from exact corners and a noisy circle.

    Also that a circle seen from somewhere off the corners' line is refused
    rather than averaged into a position.
    """
    import cv2

    from .goal_pose import GOAL_HEIGHT_M, GOAL_WIDTH_M, MODEL, pose_at
    from .midfield_pose import CENTRE, CIRCLE_RADIUS_M, camera, look_at

    cx, cy = 640.0, 360.0
    truth = np.array([-64.0, 18.7, 52.3])
    rng = np.random.default_rng(3)

    def shoot(eye, target, focal, points):
        rot = look_at(eye, target)
        cam = rot @ (points - eye).T
        uv = (camera(focal, cx, cy) @ cam)
        return (uv[:2] / uv[2]).T

    goal_mid = np.array([GOAL_WIDTH_M / 2, GOAL_HEIGHT_M / 2, 0.0])
    corners = [list(shoot(truth, goal_mid + shift, focal, MODEL))
               for shift, focal in ((np.zeros(3), 3200.0),
                                    (np.array([2.0, 0.0, 3.0]), 2900.0))]
    ray = corner_ray(corners, cx, cy)

    theta = np.linspace(0.0, 2 * np.pi, 400, endpoint=False)
    ring = np.column_stack([CENTRE[0] + CIRCLE_RADIUS_M * np.cos(theta),
                            np.zeros_like(theta),
                            CENTRE[2] + CIRCLE_RADIUS_M * np.sin(theta)])
    line = np.array([[CENTRE[0] - 12, 0, CENTRE[2]],
                     [CENTRE[0] + 12, 0, CENTRE[2]]])

    def circle_views(eye, n=6):
        views = []
        for k in range(n):
            target = CENTRE + np.array([rng.uniform(-4, 4), 0,
                                        rng.uniform(-4, 4)])
            focal = rng.uniform(1600, 2600)
            arc = shoot(eye, target, focal, ring)
            inside = ((arc[:, 0] > 0) & (arc[:, 0] < 2 * cx)
                      & (arc[:, 1] > 0) & (arc[:, 1] < 2 * cy))
            arc = arc[inside][::2]
            arc = arc + rng.normal(0, 0.7, arc.shape)
            views.append((arc, shoot(eye, target, focal, line)))
        return views

    wrong_start = np.array([-36.2, 11.4, 27.5])
    ok = True
    found, report = locate(circle_views(truth), ray, wrong_start, 1700.0,
                           cx, cy)
    err = np.inf if found is None else float(np.linalg.norm(found - truth))
    good = err < 1.0
    ok &= good
    if verbose:
        print(f"  corner line off by {ray[2]:.2f} m; camera from a start "
              f"{np.linalg.norm(wrong_start - truth):.0f} m away: "
              f"{'refused' if found is None else np.round(found, 2)}, "
              f"error {err:.2f} m, {report['agreeing']}/{report['tried']} "
              f"frames   {'ok' if good else 'FAIL'}")

    pose = pose_at(corners[0], truth, 1500.0, cx, cy)
    good = pose is not None and abs(pose.focal_px - 3200.0) < 5.0
    ok &= good
    if verbose:
        print(f"  corner pose with the position held: focal "
              f"{pose.focal_px:.0f} (true 3200), "
              f"{pose.reprojection_px:.2f} px   {'ok' if good else 'FAIL'}")

    # A second camera, for the circle frames only. The check can catch one
    # displaced out of the plane through the pitch's long axis and the true
    # camera -- lower, or nearer the touchline -- and cannot catch one
    # displaced within it, such as along the touchline at the same height:
    # the goal and the centre spot both sit on that axis, so both lines stay
    # in one plane and cross. That blind case is asserted too, so the day
    # it changes is noticed.
    for label, shift, want_refused in (
            ("8 m lower", np.array([0.0, -8.0, 0.0]), True),
            ("15 m nearer the touchline", np.array([15.0, 0.0, 0.0]), True),
            ("30 m along the touchline (blind)",
             np.array([0.0, 0.0, -30.0]), False)):
        found, report = locate(circle_views(truth + shift), ray, truth,
                               2000.0, cx, cy)
        good = (found is None) == want_refused
        ok &= good
        if verbose:
            miss = report.get("miss_m")
            print(f"  circle frames from a camera {label}: "
                  f"{report.get('refused', 'accepted')} (lines miss by "
                  f"{'-' if miss is None else f'{miss:.1f} m'})   "
                  f"{'ok' if good else 'FAIL'}")
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
