"""Shots read where they cross the goal line, not where they meet the grass.

Placing the ball assumes it is on the grass. A shot is often not: traced
through every labelled shot and goal on six windows, the detector's fastest
movement around five of the twelve read 56-112 m/s, because the ball was in
the air. (What first prompted this -- a sighting "in the top corner" of
reading_1155's goal -- turned out to be an object in the stand behind the
net; see `MIN_FLIGHT_SIGHTINGS`.)

A shot on goal has to pass through the goal mouth, a vertical rectangle
7.32 by 2.44 m on the goal line. That plane is known exactly in the goal's
frame, so a sighting's ray can be met with it instead of with the grass:
where it crosses is where the ball is across the mouth **and how high**,
if the ball is at the line on that frame. Whether it is, is checked against
where the ball last was on the grass and how long it took: a speed a
struck ball can have.

Coordinates are the calibrated goal's frame: X across from the left post,
Y up, Z out onto the pitch. Pitch metres are x = Z, y = X + (68 - 7.32)/2.
"""

from __future__ import annotations

import numpy as np

from .goal_pose import GOAL_HEIGHT_M, GOAL_WIDTH_M

# How far outside the frame of the goal a crossing still counts as a shot,
# off target: wide of a post, and over the bar. A shot is an attempt on
# goal; one that misses by more than this is a clearance or a cross.
WIDE_M = 3.0
OVER_M = 2.0

# A ray that meets the goal plane below the grass met the grass first: the
# ball is on the ground in front of the goal, not at the line. A little
# below zero is allowed for the ball's radius and pose noise.
MIN_HEIGHT_M = -0.3

# Speeds a struck ball can have between the strike and the line. Slower is
# a pass or a ball rolling to the keeper; faster is a different object.
MIN_SPEED_MS = 10.0
MAX_SPEED_MS = 45.0

# How far back to look for the strike: where the ball last was on the grass,
# at least a few metres out and inside shooting range.
LOOKBACK_S = 1.5
MIN_STRIKE_M = 3.0
MAX_STRIKE_M = 35.0

# One shot is reported once.
MERGE_S = 2.0

# The struck ball's own flight. Reading the crossing from whatever sighting
# happens to lie in the goal mouth's line of sight was measured to read
# stewards: every shot and goal the first version matched on six windows
# was a yellow vest or bib in the stand behind the goal, seen through the
# mouth at the right moment (EVENT_ACCURACY.md, "Full resolution and the
# classifier together"). So a crossing now has to be the strike's own
# ball: sightings on the same tracked segment as the strike, after it,
# fitted as one flight from the strike's spot under gravity -- three
# unknowns, the velocity, against two per sighting -- and carried to the
# goal line where the ball itself was not seen.
#
# At least two sightings: one fits any flight exactly -- three unknowns,
# and the plane supplies the third equation -- so nothing can refuse it.
# Measured: with one allowed, the three crossings the reader found on six
# windows were all stewards or a yellow object in the stand behind the
# goal, linked into the strike's own track. With two, a flight through
# something behind the goal reaches the line before it and is refused.
MIN_FLIGHT_SIGHTINGS = 2
# A flight that misfits its sightings by more than this, in pixels, is not
# one flight: the ball was touched, or a sighting is something else. Pose
# noise alone reaches about 6 px on stoke_4207 (`ball_height.py`).
MAX_FLIGHT_RMS_PX = 8.0
BALL_RADIUS_M = 0.11

# An attempt seen only briefly. A shot blocked in a crowded box is seen for a
# few frames and then sent elsewhere; four sightings in 0.16 s fit almost
# any speed along the line of sight, so its flight -- and whether it was on
# target -- cannot be read from one camera. Its direction can: every
# straight flight from the strike through those sightings lies in the one
# plane through the camera, the strike and the ball. If that plane cuts
# the goal mouth (with the off-target margins), the ball moves towards the
# goal on the grass at a struck ball's pace, and the strike is inside the
# width of the penalty area, it is counted as an attempt, outcome unknown.
AIM_WINDOW_S = 0.3
AIM_MIN_SIGHTINGS = 3
# Ground speed from the sightings placed on the grass: exact for a low shot,
# an overestimate for a lofted one. A dribble or a short pass is slower.
AIM_MIN_GROUND_SPEED_MS = 12.0
# Towards the goal: at least this share of the ground velocity is towards
# the goal line. A pass across the box is not.
AIM_TOWARDS_SHARE = 0.6
# Half the width of the penalty area: a strike from wider is a cross.
AIM_MAX_OFF_CENTRE_M = 20.0
# The sightings must be one ball: on a straight path in the picture, at a
# steady pace, within this many pixels. Measured on six windows, every false
# attempt the rule first found was sightings jumping between players, or
# sitting on the hoardings, which a straight-line fit refuses.
AIM_MAX_PATH_PX = 6.0
# Faster than any struck ball even allowing for a lofted one placed beyond
# itself: a steward moving with the camera pan read 121 m/s.
AIM_MAX_GROUND_SPEED_MS = 60.0
# Placed sightings stay on the pitch, give or take placement error.
AIM_PITCH_MARGIN_M = 2.0

# What happens after an on-target crossing decides whether it was a goal.
# Within this long the ball is either seen in the net -- behind the line and
# inside the mouth -- or back in play in front of it: parried, held, or
# cleared off the line.
AFTERMATH_S = 3.0

# In front of the line by more than this is back in play rather than
# in the goal mouth with placement noise.
BACK_IN_PLAY_M = 1.0

# How far behind the line the ball can be and still be in the net, in
# metres: a net is about 2 m deep at the grass, plus placement error. A
# sighting further back is the stand -- a steward's vest seen through the
# goal reads as "behind the line, inside the mouth" too, but placed on the
# grass it lands metres beyond the net.
NET_DEPTH_M = 3.0

# After a goal nothing is a shot until play restarts from the centre, which
# the laws make a stoppage: the celebration, the walk back, the kick-off.
# Twenty seconds is short of any real restart.
RESTART_S = 20.0

# A goal is also confirmed by what follows it, without the ball being seen
# in the net: play restarted from the centre spot after a shot or attempt
# that could have gone in, before any other shot. Measured on the labelled
# windows, neither goal is ever seen in the net -- reading_1155's ball sits
# low in the corner for under a second behind the keeper -- while both are
# found as attempts. Within this long: the celebration, the replays and the
# walk back take about a minute on these broadcasts.
KICKOFF_WITHIN_S = 150.0

# The kick-off that follows a goal: the ball still on the centre spot. Within
# this radius of the spot, allowing for midfield placement error, and still
# -- moving less than `KICKOFF_DRIFT_M` -- for at least `KICKOFF_STILL_S`
# with `KICKOFF_READINGS` readings, which a ball merely rolling through the
# centre circle does not do.
CENTRE_SPOT = (52.5, 34.0)       # pitch metres: halfway, mid-width
KICKOFF_RADIUS_M = 3.0
KICKOFF_DRIFT_M = 1.5
KICKOFF_STILL_S = 1.0
KICKOFF_READINGS = 10

PITCH_WIDTH_M = 68.0
LEFT_POST_Y = (PITCH_WIDTH_M - GOAL_WIDTH_M) / 2.0


def mouth_crossing(pose, u: float, v: float):
    """Where the ray through a pixel meets the goal plane: (X, Y) or None."""
    from .ball_height import ray

    eye, d = ray(pose, u, v)
    if abs(d[2]) < 1e-9:
        return None
    t = -eye[2] / d[2]
    if t <= 0:
        return None
    hit = eye + t * d
    return float(hit[0]), float(hit[1])


def classify(X: float, Y: float):
    """'on target', 'off target', or None for a crossing nowhere near."""
    if Y < MIN_HEIGHT_M:
        return None
    if 0.0 <= X <= GOAL_WIDTH_M and Y <= GOAL_HEIGHT_M:
        return "on target"
    if (-WIDE_M <= X <= GOAL_WIDTH_M + WIDE_M
            and Y <= GOAL_HEIGHT_M + OVER_M):
        return "off target"
    return None


def find_shots(ball, placer, fps: float, trace: list | None = None):
    """Shots at the calibrated goal, from sightings at its plane.

    `placer` needs `place(frame, px, py)` for the grass and `pose_at(frame)`
    for the camera. Returns a list of dicts with the strike and the
    crossing, earliest first. With `trace`, a list, every strike considered
    is appended to it with what each fitted flight made of it.
    """
    grass = []
    for row in ball.itertuples():
        point = placer.place(row.frame, row.px, row.py)
        if point is not None:
            grass.append((float(row.time_s), float(point[0]),
                          float(point[1]), int(row.frame)))

    rows = list(ball.itertuples())
    has_segments = "segment" in ball.columns
    found = []
    for k, row in enumerate(rows):
        point = placer.place(row.frame, row.px, row.py)
        if point is None:
            if trace is not None:
                trace.append({"frame": int(row.frame), "skipped": "not placed",
                              "pose": placer.pose_at(int(row.frame))
                              is not None})
            continue
        t0, x0, y0, f0 = (float(row.time_s), float(point[0]),
                          float(point[1]), int(row.frame))
        if not MIN_STRIKE_M <= x0 <= MAX_STRIKE_M:
            if trace is not None:
                trace.append({"frame": f0, "skipped": "out of range",
                              "at_m": [round(x0, 1), round(y0, 1)]})
            continue
        segment = int(row.segment) if has_segments else 0
        after = []
        for later in rows[k + 1:]:
            if float(later.time_s) - t0 > LOOKBACK_S:
                break
            if has_segments and int(later.segment) != segment:
                continue
            pose = placer.pose_at(int(later.frame))
            if pose is not None:
                after.append((float(later.time_s), int(later.frame), pose,
                              float(later.px), float(later.py)))
        start = np.array([y0 - LEFT_POST_Y, BALL_RADIUS_M, x0])
        crossing = read_crossing(start, t0, after, fps)
        if trace is not None:
            trace.append(_trace_strike(start, t0, f0, after, crossing))
        outcome = None
        if crossing is not None:
            X, Y, when, frame, speed, how, used = crossing
            outcome = classify(X, Y)
            if not MIN_SPEED_MS <= speed <= MAX_SPEED_MS:
                outcome = None
        if outcome is None:
            aim = aimed_attempt(start, t0, after)
            if aim is None:
                continue
            speed, when, used = aim
            X = Y = float("nan")
            frame = used[-1]
            how = f"aimed at the goal, {len(used)} sightings"
            outcome = "attempt"
        # `frame` is the strike's, as for the ground detector: it is where
        # the shooter is looked for.
        found.append({"frame": f0, "time_s": t0, "crossing_frame": frame,
                      "crossing_s": when, "x": x0, "y": y0,
                      "across_m": X, "height_m": Y, "speed_ms": speed,
                      "outcome": outcome, "goal": "left", "read": how,
                      "sightings": used,
                      "distance_m": float(np.hypot(
                          x0, y0 - PITCH_WIDTH_M / 2.0))})

    merged = []
    for shot in found:
        if merged and shot["time_s"] - merged[-1]["time_s"] < MERGE_S:
            rank = {"on target": 2, "off target": 1, "attempt": 0}
            if rank[shot["outcome"]] > rank[merged[-1]["outcome"]]:
                merged[-1] = shot
            continue
        merged.append(shot)

    sightings = []
    for row in ball.itertuples():
        pose = placer.pose_at(int(row.frame))
        if pose is None:
            continue
        sightings.append((float(row.time_s), pose, float(row.px),
                          float(row.py)))

    out, restart_after, pending = [], -np.inf, None
    for shot in merged:
        if shot["time_s"] < restart_after:
            continue                     # the ball is in the net, or walking back
        if pending is not None:
            # Something happened at the goal before the kick-off that should
            # have followed it: it was not a goal.
            pending["outcome"] = "on target"
            pending["goal_check"] = "no kick-off before the next shot"
            pending = None
        if shot["outcome"] == "on target" and went_in(shot, sightings):
            shot["outcome"] = "goal"
            kickoff = kickoff_after(shot["crossing_s"], grass)
            if kickoff is not None:
                shot["goal_check"] = "kick-off"
                shot["kickoff_s"] = kickoff
                restart_after = kickoff
            else:
                shot["goal_check"] = "unconfirmed"
                restart_after = shot["crossing_s"] + RESTART_S
                pending = shot
        else:
            # The ball was not seen in the net. A kick-off before the next
            # shot says it went in all the same -- whatever its brief flight
            # was read as: four sightings in 0.16 s can read a goal as wide,
            # and a shot that did go wide is restarted with a goal kick.
            kickoff = kickoff_after(shot["crossing_s"], grass)
            later = [s["time_s"] for s in merged
                     if s["time_s"] > shot["time_s"] + MERGE_S]
            if (kickoff is not None
                    and kickoff - shot["crossing_s"] <= KICKOFF_WITHIN_S
                    and not any(t < kickoff for t in later)):
                shot["outcome"] = "goal"
                shot["goal_check"] = "kick-off (not seen in the net)"
                shot["kickoff_s"] = kickoff
                restart_after = kickoff
        out.append(shot)
    if pending is not None:
        pending["goal_check"] = (
            "unconfirmed: no kick-off seen"
            if grass_after(pending["crossing_s"] + RESTART_S, grass)
            else "unconfirmed: the clip ends")
    return out


def _trace_strike(start, t0, f0, after, crossing):
    """What became of one strike: each flight fitted, and the crossing."""
    fits = []
    for n in range(MIN_FLIGHT_SIGHTINGS, len(after) + 1):
        got = struck_flight(start, t0, after[:n])
        if got is None:
            fits.append({"n": n, "fit": None})
            continue
        vel, rms = got
        dt = (-start[2] / vel[2]) if vel[2] < 0 else None
        fits.append({"n": n, "rms_px": round(rms, 1),
                     "vel": [round(float(v), 1) for v in vel],
                     "to_line_s": None if dt is None else round(float(dt), 2),
                     "last_s": round(after[n - 1][0] - t0, 2)})
    return {"strike_frame": f0, "strike_s": round(t0, 2),
            "strike_m": [round(float(start[2]), 1),
                         round(float(start[0] + LEFT_POST_Y), 1)],
            "sightings_after": [s[1] for s in after],
            "fits": fits[:6],
            "crossing": None if crossing is None else
            [round(float(v), 2) for v in crossing[:5]]}


def aimed_attempt(start, t0: float, after):
    """A brief strike towards the goal mouth: (ground speed, time, frames).

    See `AIM_WINDOW_S`. None when the sightings do not show one.
    """
    from .ball_height import ray
    from .goal_pose import ground_point

    if abs(start[0] + LEFT_POST_Y - PITCH_WIDTH_M / 2.0) > AIM_MAX_OFF_CENTRE_M:
        return None
    seen = [s for s in after if s[0] - t0 <= AIM_WINDOW_S]
    if len(seen) < AIM_MIN_SIGHTINGS:
        return None
    normals, ground = [], []
    for when, frame, pose, u, v in seen:
        eye, d = ray(pose, u, v)
        n = np.cross(d, start - eye)
        if np.linalg.norm(n) < 1e-9:
            return None
        n = n / np.linalg.norm(n)
        if normals and float(np.dot(n, normals[0][0])) < 0:
            n = -n
        normals.append((n, eye))
        g = ground_point(pose, u, v)
        if g is not None:
            ground.append((when - t0, g[0] - start[0], g[1] - start[2]))
    if len(ground) < AIM_MIN_SIGHTINGS:
        return None
    # One ball on one path: a straight line at a steady pace in the picture,
    # for the longest run from the strike that stays one -- a blocked shot
    # turns at the block, and only the run before it is the shot.
    keep = 0
    for n in range(AIM_MIN_SIGHTINGS, len(seen) + 1):
        if _path_rms(seen[:n], t0) > AIM_MAX_PATH_PX:
            break
        keep = n
    if keep < AIM_MIN_SIGHTINGS:
        return None
    seen = seen[:keep]
    ground = [g for g in ground if g[0] <= seen[-1][0] - t0 + 1e-9]
    if len(ground) < AIM_MIN_SIGHTINGS:
        return None
    # On the pitch, not the hoardings or the stand.
    for _, across, out in ground:
        x = across + start[0]
        if not (-LEFT_POST_Y - AIM_PITCH_MARGIN_M <= x
                <= PITCH_WIDTH_M - LEFT_POST_Y + AIM_PITCH_MARGIN_M
                and out + start[2] >= -AIM_PITCH_MARGIN_M):
            return None
    t = np.array([g[0] for g in ground])
    vel = np.array([np.sum(t * np.array([g[i] for g in ground]))
                    / np.sum(t * t) for i in (1, 2)])
    speed = float(np.hypot(*vel))
    if not AIM_MIN_GROUND_SPEED_MS <= speed <= AIM_MAX_GROUND_SPEED_MS:
        return None
    if -vel[1] < AIM_TOWARDS_SHARE * speed:
        return None
    # The plane through the camera holding every straight flight from the
    # strike through the sightings, and where it meets the goal plane Z = 0.
    n = np.mean([m for m, _ in normals], axis=0)
    eye = normals[0][1]
    c = float(np.dot(n, eye))
    corners = [(x, y) for x in (-WIDE_M, GOAL_WIDTH_M + WIDE_M)
               for y in (MIN_HEIGHT_M, GOAL_HEIGHT_M + OVER_M)]
    sides = [np.sign(n[0] * x + n[1] * y - c) for x, y in corners]
    if all(side > 0 for side in sides) or all(side < 0 for side in sides):
        return None
    when = t0 + float(start[2]) / max(-vel[1], 1e-6)
    return speed, when, [s[1] for s in seen]


def _path_rms(seen, t0: float) -> float:
    """Pixels off a straight, steady path through the sightings."""
    ts = np.array([s[0] - t0 for s in seen])
    design = np.column_stack([np.ones_like(ts), ts])
    residual = []
    for axis in (3, 4):
        values = np.array([s[axis] for s in seen])
        coef = np.linalg.lstsq(design, values, rcond=None)[0]
        residual.append(values - design @ coef)
    return float(np.sqrt(np.mean(np.square(residual)) * 2))


def struck_flight(start, t0: float, after):
    """The velocity of a ball struck from `start` at `t0`, fitted to its
    sightings `after` (time, frame, pose, u, v). (velocity, rms px) or None.

    Solved linearly first -- a point on a ray satisfies d x (p - eye) = 0,
    linear in the velocity because the start and gravity are known -- then
    refined on pixel residuals.
    """
    from scipy.optimize import least_squares

    from .ball_height import GRAVITY_MS2, _project, ray

    rows, rhs = [], []
    for when, _, pose, u, v in after:
        dt = when - t0
        eye, d = ray(pose, u, v)
        cross = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]],
                          [-d[1], d[0], 0]])
        fall = np.array([0.0, -0.5 * GRAVITY_MS2 * dt ** 2, 0.0])
        rows.append(cross * dt)
        rhs.append(cross @ (eye - start - fall))
    try:
        guess = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs),
                                rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    def residuals(vel):
        out = []
        for when, _, pose, u, v in after:
            dt = when - t0
            p = start + vel * dt
            p[1] -= 0.5 * GRAVITY_MS2 * dt ** 2
            out.append(_project(pose, p[None])[0] - (u, v))
        return np.concatenate(out)

    # No faster than a struck ball, and not downward: it leaves the grass.
    # From a camera 70 m away and 19 m up, a ball moving away along the line
    # of sight looks much like one moving down it, and without these bounds
    # real flights fitted the speed cap with the ball heading into the
    # ground (traced on stoke_7001 and reading_1155: vertical -12 m/s).
    bound = MAX_SPEED_MS
    lower = np.array([-bound, 0.0, -bound])
    upper = np.array([bound, bound, bound])
    start_guess = np.clip(guess, lower + 1e-3, upper - 1e-3)
    got = least_squares(residuals, start_guess, bounds=(lower, upper),
                        method="trf", max_nfev=200)
    return got.x, float(np.sqrt(np.mean(got.fun ** 2)))


def read_crossing(start, t0: float, after, fps: float):
    """Where the ball struck from `start` crosses the goal line.

    (X, Y, time, frame, speed, how, frames used) or None. The flight is
    fitted to the first n sightings for n = 2, 3, ... and the crossing is
    read from the longest run that is still one flight short of the line.
    A run that misfits by more than `MAX_FLIGHT_RMS_PX` ends the search:
    the ball was touched, reached the net, or a sighting is something else.
    A short run whose fit puts the line before its last sighting does not:
    two sightings 0.04 s apart fit almost any speed along the line of
    sight, and the first version, which stopped there, never got past two
    sightings of flights the fine-tuned detector saw on every frame.
    """
    best = None
    for n in range(MIN_FLIGHT_SIGHTINGS, len(after) + 1):
        got, rms = _crossing_from(start, t0, after[:n], fps)
        if rms is None or rms > MAX_FLIGHT_RMS_PX:
            break
        if got is not None:
            best = got
    return best


def _crossing_from(start, t0: float, seen, fps: float):
    """(crossing or None, rms px or None) for a flight fitted to `seen`."""
    from .ball_height import GRAVITY_MS2

    fitted = struck_flight(start, t0, seen)
    if fitted is None:
        return None, None
    vel, rms = fitted
    if vel[2] >= 0.0 or np.linalg.norm(vel) > MAX_SPEED_MS:
        return None, rms
    dt = -start[2] / vel[2]
    if dt > LOOKBACK_S or seen[-1][0] - t0 > dt + 0.5 / fps:
        return None, rms
    X = start[0] + vel[0] * dt
    Y = start[1] + vel[1] * dt - 0.5 * GRAVITY_MS2 * dt ** 2
    frame = int(seen[0][1] + round((t0 + dt - seen[0][0]) * fps))
    return ((float(X), float(Y), t0 + dt, frame, float(np.linalg.norm(vel)),
             f"flight fitted to {len(seen)} sightings, {rms:.1f} px",
             [s[1] for s in seen]), rms)


def kickoff_after(when: float, grass):
    """When play restarted from the centre spot after `when`, or None.

    `grass` is (time, x, y, frame) readings in pitch metres. A kick-off is the
    ball still on the spot: within `KICKOFF_RADIUS_M` of it for at least
    `KICKOFF_STILL_S`, drifting less than `KICKOFF_DRIFT_M`.
    """
    near = [(t, x, y) for t, x, y, _ in grass
            if t > when and np.hypot(x - CENTRE_SPOT[0],
                                     y - CENTRE_SPOT[1]) <= KICKOFF_RADIUS_M]
    for i, (t0, x0, y0) in enumerate(near):
        run = [(t, x, y) for t, x, y in near[i:]
               if t - t0 <= KICKOFF_STILL_S + 0.5]
        span = run[-1][0] - t0
        if span < KICKOFF_STILL_S or len(run) < KICKOFF_READINGS:
            continue
        xs = np.array([r[1] for r in run]); ys = np.array([r[2] for r in run])
        if np.max(np.hypot(xs - xs.mean(), ys - ys.mean())) <= KICKOFF_DRIFT_M:
            return float(t0)
    return None


def grass_after(when: float, grass) -> bool:
    """Is the ball placed anywhere after `when`?"""
    return any(t > when for t, _, _, _ in grass)


def went_in(shot, sightings) -> bool:
    """Was an on-target crossing followed by the ball in the net?

    In the net: behind the goal line on the grass and inside the mouth at
    the goal plane. Back in play: on the pitch, more than a metre in front
    of the line. Within `AFTERMATH_S` the ball must be seen in the net, and
    more often than back in play.

    A majority rather than a veto because the tracker is not reliable while
    the ball sits still in the net. On reading_1155's goal it was seen in
    the net for half a second and then hopped to a hoarding beyond the
    touchline and, for three frames, to something by the post; the first
    version took the first stray reading in front of the line as the ball
    back in play and called the goal a save. A crossing with nothing seen
    after it is not a goal: a save the camera turned away from looks the
    same.
    """
    from .goal_pose import ground_point

    in_net = in_play = 0
    for when, pose, u, v in sightings:
        if when <= shot["crossing_s"]:
            continue
        if when > shot["crossing_s"] + AFTERMATH_S:
            break
        grass = ground_point(pose, u, v)
        if grass is None:
            continue
        across, out_from_line = grass
        on_pitch = -LEFT_POST_Y <= across <= PITCH_WIDTH_M - LEFT_POST_Y
        if out_from_line > BACK_IN_PLAY_M:
            in_play += on_pitch
            continue
        hit = mouth_crossing(pose, u, v)
        if (-NET_DEPTH_M <= out_from_line < 0.0 and hit is not None
                and 0.0 <= hit[0] <= GOAL_WIDTH_M
                and MIN_HEIGHT_M <= hit[1] <= GOAL_HEIGHT_M):
            in_net += 1
    return in_net > 0 and in_net > in_play


def selftest(verbose: bool = True) -> bool:
    """A shot into the top corner, one wide, a cross, and a ball at rest.

    Seen by a located camera, with the ball undetected in flight -- struck,
    then sighted once at or near the line -- as on the real goal that
    prompted this.
    """
    import cv2
    import pandas as pd

    from .goal_pose import GoalPose
    from .midfield_pose import look_at

    eye = np.array([-64.0, 18.8, 52.3])
    rot = look_at(eye, np.array([5.0, 0.0, 12.0]))
    rvec, _ = cv2.Rodrigues(rot)
    pose = GoalPose(rvec=rvec, tvec=(-rot @ eye).reshape(3, 1),
                    focal_px=2400.0, cx=640.0, cy=360.0,
                    reprojection_px=0.0, n_corners=0)

    def pixel(p):
        img, _ = cv2.projectPoints(np.asarray(p, float).reshape(1, 3), rvec,
                                   pose.tvec, pose.camera_matrix, None)
        return img.reshape(2)

    class Placer:
        def pose_at(self, frame):
            return pose

        def place(self, frame, px, py):
            from .goal_pose import ground_point
            hit = ground_point(pose, px, py)
            return None if hit is None else np.array(
                [hit[1], hit[0] + LEFT_POST_Y])

    def track(points, noise_px=0.0):
        jitter = np.random.default_rng(1)
        rows = [{"frame": k, "time_s": k / 25.0,
                 "px": pixel(p)[0] + jitter.normal(0, noise_px),
                 "py": pixel(p)[1] + jitter.normal(0, noise_px),
                 "segment": rest[0] if rest else 0}
                for k, p, *rest in points]
        return pd.DataFrame(rows)

    def flight(target, frames, strike=9, at=(3.0, 0.11, 18.0), arrive=28):
        """Points on the parabola from `at` at frame `strike` to `target`
        at frame `arrive`, at `frames`."""
        dt = (arrive - strike) / 25.0
        p0, p1 = np.array(at), np.array(target)
        vel = (p1 - p0) / dt
        vel[1] += 0.5 * 9.81 * dt
        out = []
        for k in frames:
            t = (k - strike) / 25.0
            p = p0 + vel * t
            p[1] -= 0.5 * 9.81 * t * t
            out.append((k, tuple(p)))
        return out

    # Goal frame points: (X across, Y up, Z out).
    still = [(k, (3.0, 0.11, 18.0)) for k in range(10)]
    # Struck from `still`, seen twice just before the line, top corner.
    corner = flight((6.8, 2.2, 0.0), (25, 28))
    cases = (
        ("shot into the top corner",
         still + corner, "on target"),
        ("shot wide of the post",
         still + flight((9.5, 0.6, 0.0), (23, 26), arrive=26),
         "off target"),
        ("ball rolling to the keeper",
         still + [(60, (3.6, 0.11, 0.0))], None),
        ("ball at rest in the box",
         [(k, (3.0, 0.11, 11.0)) for k in range(40)], None),
    )
    in_net = [(k, (6.5, 0.4, -1.2)) for k in range(30, 60, 3)]
    saved = [(k, (3.5, 0.11, 4.0 + 0.2 * (k - 30))) for k in range(30, 60, 3)]
    cases = cases + (
        ("top corner, then in the net",
         still + corner + in_net, "goal"),
        ("top corner, then parried out",
         still + corner + saved, "on target"),
        # The ball still in the net seconds later reads as a second shot
        # from the goalmouth; play has not restarted, so it is not one.
        # The tracker leaving the ball in the net: a hoarding beyond the
        # touchline, and a three-frame blip by the post, as on the real goal.
        ("a goal, then the tracker strays",
         still + corner + in_net[:5]
         + [(k, (-38.0, 0.11, 7.0)) for k in range(46, 52)]
         + [(k, (7.4, 0.11, 1.5)) for k in range(52, 55)], "goal"),
        ("a goal, then the ball in the net",
         still + corner + in_net
         + [(k, (3.0, 0.11, 6.0)) for k in range(120, 126)]
         + flight((4.0, 1.5, 0.0), (129, 132), strike=125, at=(3.0, 0.11, 6.0),
                  arrive=132), "goal only"),
    )
    # The ball lost before the line, as it is on most real shots: seen three
    # times early in its flight and never again. And a steward in the stand
    # behind the goal, still, on a track of its own -- what every shot the
    # first version found turned out to be.
    early = flight((6.8, 2.2, 0.0), (12, 15, 18))
    steward = [(k, (10.52, 1.09, -4.18), 1) for k in range(24, 48)]
    cases = cases + (
        ("seen only early in flight", still + early, "on target"),
        ("seen early, going wide",
         still + flight((10.0, 1.0, 0.0), (12, 15, 18)), "off target"),
        ("struck, lost, a steward behind", still + steward, None),
        # The same steward linked into the strike's own track, as on the
        # real windows: a flight through it would reach the line first.
        ("a steward on the strike's track",
         still + [(k, p) for k, p, _ in steward], None),
        ("seen early, then a steward",
         still + early + [(k, p) for k, p, _ in steward], "on target"),
    )
    # Seen on every frame after the strike, with a pixel of detector noise,
    # as the fine-tuned detector sees real shots: two sightings 0.04 s
    # apart then fit almost any speed along the line of sight, and the
    # first version stopped there.
    dense = [(k, p) for k, p in flight((6.8, 2.2, 0.0), range(10, 26))]
    rolling = [(k, (3.0 - 0.2 * k, 0.11, 18.0 + 0.1 * k)) for k in range(10)]
    noisy = (
        ("every frame of the flight, noisy", rolling + dense, "on target", 1.0),
        ("every frame, going wide, noisy",
         rolling + flight((10.0, 1.0, 0.0), range(10, 26)), "off target", 1.0),
    )
    # Blocked: struck at goal, seen for four frames, then a defender sends
    # it back out. The flight is read from the sightings before the block
    # and carried to the line, so the attempt still counts as a shot.
    struck = flight((5.0, 1.0, 0.0), range(10, 14))
    rebound = [(k, (4.0 - 0.1 * (k - 14), 0.5, 14.0 + 0.4 * (k - 14)))
               for k in range(14, 26)]
    rolling_, struck_, rebound_ = rolling, struck, rebound
    # And two that must not count: a hard pass across the box, and one back
    # out towards midfield, each seen for the same few frames.
    across = [(k, (3.0 + 0.9 * (k - 9), 0.11, 18.0 - 0.1 * (k - 9)))
              for k in range(10, 22)]
    back = [(k, (3.0 + 0.1 * (k - 9), 0.3, 18.0 + 0.8 * (k - 9)))
            for k in range(10, 22)]
    # Sightings that are not one ball: alternating between two players
    # towards goal, as the false attempts on the real windows did.
    jumping = [(k, (2.0 + (6.0 if k % 2 else 0.0), 0.11,
                    17.0 - 0.6 * (k - 9)))
               for k in range(10, 18)]
    noisy = noisy + (
        ("a shot blocked after 0.16 s", rolling + struck + rebound,
         "attempt", 1.0),
        ("sightings jumping between two players", rolling + jumping,
         None, 1.0),
        ("a hard pass across the box", rolling + across, None, 1.0),
        ("a hard pass back out", rolling + back, None, 1.0),
    )
    ok = True
    for name, points, want, noise in noisy:
        got = find_shots(track(points, noise), Placer(), 25.0)
        said = got[0]["outcome"] if got else None
        extra = (f", {got[0]['speed_ms']:.0f} m/s, {got[0]['height_m']:.1f} m "
                 f"up at the line" if got else "")
        good = said == want
        ok &= good
        if verbose:
            print(f"  {name:>28s}  {said or 'no shot'}{extra}   "
                  f"{'ok' if good else 'WRONG'}")
    for name, points, want in cases:
        got = find_shots(track(points), Placer(), 25.0)
        said = got[0]["outcome"] if got else None
        if want == "goal only":
            said = ("goal only" if [g["outcome"] for g in got] == ["goal"]
                    else ", ".join(g["outcome"] for g in got))
        extra = (f", {got[0]['speed_ms']:.0f} m/s, {got[0]['height_m']:.1f} m "
                 f"up at the line" if got else "")
        good = said == want
        ok &= good
        if verbose:
            print(f"  {name:>28s}  {said or 'no shot'}{extra}   "
                  f"{'ok' if good else 'WRONG'}")
    # The kick-off that confirms a goal, and its absence that refutes one.
    goal = still + corner + in_net
    spot = [(k, (3.66, 0.11, 52.5)) for k in range(1000, 1040)]
    another = ([(k, (3.0, 0.11, 18.0)) for k in range(700, 710)]
               + flight((6.8, 2.2, 0.0), (722, 725), strike=709, arrive=725))
    # Goals never seen in the net, as on both labelled ones: a brief attempt
    # or an on-target crossing, then a kick-off -- or another shot first,
    # or a wide one, neither of which is a goal.
    attempt = rolling_ + struck_ + rebound_
    # Ball through the centre circle without stopping: not a kick-off.
    passing = [(k, (3.66 + 0.4 * (k - 1000), 0.11, 52.5))
               for k in range(1000, 1040)]
    for name, points, want, noise in (
            ("attempt, then a kick-off", attempt + spot,
             ("goal", "kick-off (not seen in the net)"), 1.0),
            ("attempt, another shot, kick-off",
             attempt + another + spot, ("attempt", None), 0.0),
            ("attempt, ball through centre", attempt + passing,
             ("attempt", None), 1.0)):
        got = find_shots(track(points, noise), Placer(), 25.0)
        said = (got[0]["outcome"], got[0].get("goal_check")) if got else None
        good = said == want
        ok &= good
        if verbose:
            print(f"  {name:>28s}  {said[0] if said else 'no shot'}"
                  f"{' (' + said[1] + ')' if said and said[1] else ''}   "
                  f"{'ok' if good else 'WRONG'}")
    for name, points, want in (
            ("goal, then a kick-off", goal + spot, ("goal", "kick-off")),
            ("goal, then another shot first", goal + another,
             ("on target", "no kick-off before the next shot")),
            ("goal, and the clip ends", goal,
             ("goal", "unconfirmed: the clip ends"))):
        got = find_shots(track(points), Placer(), 25.0)
        said = (got[0]["outcome"], got[0].get("goal_check")) if got else None
        good = said == want
        ok &= good
        if verbose:
            print(f"  {name:>28s}  {said[0] if said else 'no shot'}"
                  f"{' (' + said[1] + ')' if said and said[1] else ''}   "
                  f"{'ok' if good else 'WRONG'}")
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
