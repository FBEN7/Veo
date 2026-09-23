"""The two numbers every expected-goals model is built on.

An xG model is a function of where the shot was taken from, and essentially
all of them take the same two geometric inputs:

    distance    how far the shot is from the centre of the goal
    angle       how wide the goal mouth appears from there

Everything else a model might use -- body part, defensive pressure, whether
it followed a cross -- refines a prediction that these two already carry most
of. They are also the part that is pure geometry: no coefficients, no fitting,
no data. So they can be built and checked now, exactly, and whichever
published model is eventually chosen drops on top without rework.

What this module deliberately does not do is produce an xG number. That needs
coefficients from a specific published model, and inventing plausible-looking
ones would produce a figure that reads like Opta and is not.

## It is not usable yet, and the code says so rather than the docs

Both inputs are positions on a real pitch: distance from a goal that has to
be *somewhere*. This pipeline has no absolute pitch coordinates. What it has
is metric but free-floating -- `prepare_tracks_for_events` returns
`absolute=False` for exactly this reason, and goals, shots and out-of-play
are switched off downstream of that flag.

So `shot_features_from_events` refuses when handed relative coordinates
rather than computing confident nonsense from them. The functions below it
are pure geometry and take pitch coordinates directly, which is what makes
them testable today: `check_shot_geometry.py` checks the closed form against
an independent vector derivation and against cases whose answers are fixed by
construction.

## Which goal is being attacked is a separate unsolved problem

A shot is toward the opponent's goal, and nothing here knows which end that
is. Teams swap at half time, and the pipeline has no pitch landmarks to
orient itself by. Every function therefore takes `goal` explicitly and none
of them guesses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The pitch this project's coordinates are expressed in. These mirror the
# constants in `events.py`; `check_shot_geometry.py` asserts they still agree,
# rather than importing them, so that wiring this module into events later
# cannot create a circular import.
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_WIDTH_M = 7.32
GOAL_HALF_M = GOAL_WIDTH_M / 2.0

# Penalty area: 16.5 m deep, 40.32 m wide.
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32

# The penalty spot, for reference and for the self-check.
PENALTY_SPOT_M = 11.0

GOALS = ("left", "right")


def _goal_centre(goal: str) -> tuple[float, float]:
    if goal == "left":
        return 0.0, PITCH_WIDTH_M / 2.0
    if goal == "right":
        return PITCH_LENGTH_M, PITCH_WIDTH_M / 2.0
    raise ValueError(f"goal must be one of {GOALS}, not {goal!r}")


def to_goal_frame(x, y, goal: str):
    """Pitch coordinates as (forward, lateral) from the goal being attacked.

    `forward` is distance out from the goal line, positive into the pitch.
    `lateral` is offset from the middle of the goal, signed. Expressing
    everything this way means the geometry below is written once instead of
    once per end.
    """
    gx, gy = _goal_centre(goal)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    forward = (x - gx) if goal == "left" else (gx - x)
    return forward, y - gy


def distance_to_goal(x, y, goal: str):
    """Straight-line distance to the centre of the goal, in metres."""
    forward, lateral = to_goal_frame(x, y, goal)
    return np.hypot(forward, lateral)


def goal_mouth_angle(x, y, goal: str):
    """Angle the goal mouth subtends at the shot, in radians.

    With the goal half-width w and the shot at (f, l),

        tan(angle) = 2 * w * f / (f^2 + l^2 - w^2)

    which is the cross product over the dot product of the vectors to the two
    posts. It is written as an `arctan2` of those two terms rather than as an
    `arctan` of their ratio because the dot product goes negative close in
    and between the posts, where the goal subtends more than a right angle.
    Taking `arctan` there would silently fold a 150-degree view down to 30.

    On the goal line between the posts the answer is pi -- the goal occupies
    the entire half-plane -- and `arctan2(0, negative)` returns exactly that.
    Behind the goal line the quantity is meaningless and comes back zero.
    """
    forward, lateral = to_goal_frame(x, y, goal)
    numerator = GOAL_WIDTH_M * forward
    denominator = forward ** 2 + lateral ** 2 - GOAL_HALF_M ** 2
    angle = np.arctan2(numerator, denominator)
    return np.where(forward >= 0.0, angle, 0.0)


def angle_to_goal_centre(x, y, goal: str):
    """How far off a straight-on shot is, in radians.

    Zero when the shot faces the middle of the goal square on, and plus or
    minus a right angle out on the goal line. Distinct from
    `goal_mouth_angle`, which is how *wide* the goal looks rather than which
    direction it lies in; xGHub records both, as `angle_to_goal` and
    `angle_to_posts`.

    Signed, so the two sides of the pitch are distinguishable. Datasets that
    mirror their clips to a single side -- xGHub has a `mirror_flag` for
    exactly that -- have folded the sign away, so compare magnitudes.
    """
    forward, lateral = to_goal_frame(x, y, goal)
    return np.arctan2(lateral, forward)


def in_penalty_area(x, y, goal: str):
    """Whether the shot was taken inside the box."""
    forward, lateral = to_goal_frame(x, y, goal)
    return ((forward >= 0.0) & (forward <= PENALTY_AREA_DEPTH_M)
            & (np.abs(lateral) <= PENALTY_AREA_WIDTH_M / 2.0))


def defenders_in_cone(shot_x, shot_y, defender_x, defender_y, goal: str):
    """How many defenders stand between the shot and the goal mouth.

    The triangle with the shot at one corner and a post at each of the others
    is the region a shot on target must pass through, so bodies inside it are
    the ones that actually block. This is the one pressure feature available
    from what the pipeline already produces -- player positions and teams --
    and it inherits their limits: team assignment measures 0.93, and a
    defender who was not detected is not counted.

    A point is inside the triangle when it falls on the same side of all
    three edges, tested by the sign of the cross product.
    """
    gx, gy = _goal_centre(goal)
    post_a = np.array([gx, gy - GOAL_HALF_M])
    post_b = np.array([gx, gy + GOAL_HALF_M])
    shot = np.array([float(shot_x), float(shot_y)])

    dx = np.asarray(defender_x, dtype=float)
    dy = np.asarray(defender_y, dtype=float)

    def side(a, b):
        return ((b[0] - a[0]) * (dy - a[1]) - (b[1] - a[1]) * (dx - a[0]))

    s1, s2, s3 = side(shot, post_a), side(post_a, post_b), side(post_b, shot)
    inside = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | \
             ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
    return int(np.count_nonzero(inside))


def shot_features(x, y, goal: str) -> dict:
    """Every geometric input, for one shot or an array of them."""
    forward, lateral = to_goal_frame(x, y, goal)
    angle = goal_mouth_angle(x, y, goal)
    bearing = angle_to_goal_centre(x, y, goal)
    return dict(
        distance_m=distance_to_goal(x, y, goal),
        angle_rad=angle,
        angle_deg=np.degrees(angle),
        bearing_rad=bearing,
        bearing_deg=np.degrees(bearing),
        forward_m=forward,
        lateral_m=lateral,
        in_penalty_area=in_penalty_area(x, y, goal),
    )


def shot_features_from_events(events, absolute_pitch: bool,
                              attacking_goal: dict[str, str] | None = None,
                              ) -> pd.DataFrame:
    """Geometric inputs for detected shots, or a refusal.

    `absolute_pitch` is the flag `pixel_scale.prepare_tracks_for_events`
    returns. It is False on every clip this project has processed, because
    there is no homography and so no goal line. Distance to a goal that is
    not located anywhere is not a smaller version of the right answer; it is
    a number with no meaning, and it would flow straight into an xG figure
    that looks authoritative. So this raises rather than returning something.

    `attacking_goal` maps team to the end it attacks. There is no default:
    nothing in this pipeline can yet tell which way a team is playing.
    """
    if not absolute_pitch:
        raise ValueError(
            "shot geometry needs absolute pitch coordinates and this clip has "
            "none. Distance and angle are measured from a goal, and without a "
            "homography there is no goal line to measure from. See "
            "VEO_FOOTAGE.md; the ground plane is metric but its origin is the "
            "camera, not a corner flag.")

    shots = [e for e in events if e.get("event_type") == "shot"]
    if not shots:
        return pd.DataFrame(columns=["timestamp_s", "team", "distance_m",
                                     "angle_deg", "in_penalty_area"])
    if not attacking_goal:
        raise ValueError(
            "attacking_goal is required: a shot is toward the opponent's "
            "goal, and which end that is cannot be inferred here.")

    rows = []
    for shot in shots:
        team = shot.get("team")
        goal = attacking_goal.get(team)
        if goal is None:
            raise ValueError(f"no attacking goal given for team {team!r}")
        feats = shot_features(shot["x"], shot["y"], goal)
        rows.append(dict(
            timestamp_s=shot.get("timestamp_s"),
            team=team,
            attacking=goal,
            distance_m=float(feats["distance_m"]),
            angle_deg=float(feats["angle_deg"]),
            in_penalty_area=bool(feats["in_penalty_area"]),
        ))
    return pd.DataFrame(rows)
