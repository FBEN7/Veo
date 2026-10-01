"""Shots read where they cross the goal line, not where they meet the grass.

Placing the ball assumes it is on the grass. A shot is often not: traced
through every labelled shot and goal on six windows, the detector's fastest
movement around five of the twelve read 56-112 m/s, because the ball was in
the air. Rendered, reading_1155's goal is a strike at the edge of the box,
no detections in flight, and the next sighting in the top corner of the net
-- which, projected to the grass, lands 2.9 m wide of the post and makes
the strike 72 m/s. The ground-based detector rejects it twice over.

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

# What happens after an on-target crossing decides whether it was a goal.
# Within this long the ball is either seen in the net -- behind the line and
# inside the mouth -- or back in play in front of it: parried, held, or
# cleared off the line.
AFTERMATH_S = 3.0

# In front of the line by more than this is back in play rather than
# in the goal mouth with placement noise.
BACK_IN_PLAY_M = 1.0

# After a goal nothing is a shot until play restarts from the centre, which
# the laws make a stoppage: the celebration, the walk back, the kick-off.
# Twenty seconds is short of any real restart.
RESTART_S = 20.0

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


def find_shots(ball, placer, fps: float):
    """Shots at the calibrated goal, from sightings at its plane.

    `placer` needs `place(frame, px, py)` for the grass and `pose_at(frame)`
    for the camera. Returns a list of dicts with the strike and the
    crossing, earliest first.
    """
    grass = []
    for row in ball.itertuples():
        point = placer.place(row.frame, row.px, row.py)
        if point is not None:
            grass.append((float(row.time_s), float(point[0]),
                          float(point[1]), int(row.frame)))

    found = []
    for row in ball.itertuples():
        pose = placer.pose_at(int(row.frame))
        if pose is None:
            continue
        hit = mouth_crossing(pose, float(row.px), float(row.py))
        if hit is None:
            continue
        X, Y = hit
        outcome = classify(X, Y)
        if outcome is None:
            continue
        when = float(row.time_s)
        # The strike: the latest grass reading in shooting range before this.
        strike = None
        for t, x, y, f in reversed(grass):
            if t >= when:
                continue
            if when - t > LOOKBACK_S:
                break
            if MIN_STRIKE_M <= x <= MAX_STRIKE_M:
                strike = (t, x, y, f)
                break
        if strike is None:
            continue
        t0, x0, y0, f0 = strike
        travel = float(np.linalg.norm([x0 - 0.0, y0 - (LEFT_POST_Y + X),
                                       0.11 - Y]))
        speed = travel / (when - t0)
        if not MIN_SPEED_MS <= speed <= MAX_SPEED_MS:
            continue
        # `frame` is the strike's, as for the ground detector: it is where
        # the shooter is looked for.
        found.append({"frame": f0, "time_s": t0, "crossing_frame":
                      int(row.frame), "crossing_s": when, "x": x0, "y": y0,
                      "across_m": X, "height_m": Y, "speed_ms": speed,
                      "outcome": outcome, "goal": "left",
                      "distance_m": float(np.hypot(
                          x0, y0 - PITCH_WIDTH_M / 2.0))})

    merged = []
    for shot in found:
        if merged and shot["time_s"] - merged[-1]["time_s"] < MERGE_S:
            if (merged[-1]["outcome"] != "on target"
                    and shot["outcome"] == "on target"):
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

    out, restart_after = [], -np.inf
    for shot in merged:
        if shot["time_s"] < restart_after:
            continue                     # the ball is in the net, or walking back
        if shot["outcome"] == "on target" and went_in(shot, sightings):
            shot["outcome"] = "goal"
            restart_after = shot["crossing_s"] + RESTART_S
        out.append(shot)
    return out


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
        if (out_from_line < 0.0 and hit is not None
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

    def track(points):
        rows = [{"frame": k, "time_s": k / 25.0,
                 "px": pixel(p)[0], "py": pixel(p)[1]}
                for k, p in points]
        return pd.DataFrame(rows)

    # Goal frame points: (X across, Y up, Z out).
    still = [(k, (3.0, 0.11, 18.0)) for k in range(10)]
    cases = (
        ("shot into the top corner",
         still + [(28, (6.8, 2.2, 0.0))], "on target"),
        ("shot wide of the post",
         still + [(26, (9.5, 0.6, 0.0))], "off target"),
        ("ball rolling to the keeper",
         still + [(60, (3.6, 0.11, 0.0))], None),
        ("ball at rest in the box",
         [(k, (3.0, 0.11, 11.0)) for k in range(40)], None),
    )
    in_net = [(k, (6.5, 0.4, -1.2)) for k in range(30, 60, 3)]
    saved = [(k, (3.5, 0.11, 4.0 + 0.2 * (k - 30))) for k in range(30, 60, 3)]
    cases = cases + (
        ("top corner, then in the net",
         still + [(28, (6.8, 2.2, 0.0))] + in_net, "goal"),
        ("top corner, then parried out",
         still + [(28, (6.8, 2.2, 0.0))] + saved, "on target"),
        # The ball still in the net seconds later reads as a second shot
        # from the goalmouth; play has not restarted, so it is not one.
        # The tracker leaving the ball in the net: a hoarding beyond the
        # touchline, and a three-frame blip by the post, as on the real goal.
        ("a goal, then the tracker strays",
         still + [(28, (6.8, 2.2, 0.0))] + in_net[:5]
         + [(k, (-38.0, 0.11, 7.0)) for k in range(46, 52)]
         + [(k, (7.4, 0.11, 1.5)) for k in range(52, 55)], "goal"),
        ("a goal, then the ball in the net",
         still + [(28, (6.8, 2.2, 0.0))] + in_net
         + [(k, (3.0, 0.11, 6.0)) for k in range(120, 126)]
         + [(132, (4.0, 1.5, 0.0))], "goal only"),
    )
    ok = True
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
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
