"""Check the shot geometry against a second derivation and against fixed cases.

These two numbers will sit under every expected-goals figure this project
ever reports, and unlike most of what it computes they have exact right
answers. So they are checked exactly, three ways:

  * the closed form against an independent vector derivation, over a grid
  * cases whose answers are fixed by construction, not by measurement
  * invariances the geometry must have: mirror symmetry, monotonicity

The closed form is the one worth checking hardest. Writing the angle as
`arctan` of a ratio rather than `arctan2` of its two parts is a real and
easy mistake, and it is silent: it agrees everywhere except close to goal,
where it folds an obtuse view of the goal mouth down to its supplement and
makes the best chances on the pitch look like the worst.

    python check_shot_geometry.py
"""

from __future__ import annotations

import numpy as np

from src import shot_geometry as sg

TOLERANCE_RAD = 1e-9


def angle_by_vectors(x: float, y: float, goal: str) -> float:
    """The same angle, derived without the closed form.

    Vectors from the shot to each post, then the angle between them from the
    dot product. Slower, uglier, and independent -- which is the point.
    """
    gx, gy = sg._goal_centre(goal)
    shot = np.array([x, y], dtype=float)
    to_a = np.array([gx, gy - sg.GOAL_HALF_M]) - shot
    to_b = np.array([gx, gy + sg.GOAL_HALF_M]) - shot
    na, nb = np.linalg.norm(to_a), np.linalg.norm(to_b)
    if na == 0 or nb == 0:
        return float("nan")
    cosine = np.clip(np.dot(to_a, to_b) / (na * nb), -1.0, 1.0)
    return float(np.arccos(cosine))


def check_constants():
    from src import events

    pairs = [("GOAL_WIDTH_M", sg.GOAL_WIDTH_M, events.GOAL_WIDTH_M),
             ("goal half", sg.GOAL_HALF_M, events.GOAL_HALF),
             ("pitch length", sg.PITCH_LENGTH_M, events.RIGHT_GOAL_X_MIN
              + events.GOAL_DEPTH_M),
             ("goal centre y", sg.PITCH_WIDTH_M / 2.0,
              (events.LEFT_GOAL_Y_MIN + events.LEFT_GOAL_Y_MAX) / 2.0)]
    print("1. The pitch this module assumes is the one events.py assumes\n")
    bad = 0
    for name, ours, theirs in pairs:
        ok = abs(ours - theirs) < 1e-9
        bad += not ok
        print(f"   {name:<16s} {ours:8.2f} vs {theirs:8.2f}  "
              f"{'ok' if ok else 'MISMATCH'}")
    return bad


def check_against_vectors(n: int = 4000, seed: int = 0):
    """Closed form against the vector derivation, everywhere on the pitch."""
    print("\n2. Closed form against an independent derivation\n")
    rng = np.random.default_rng(seed)
    worst, worst_at = 0.0, None
    for goal in sg.GOALS:
        for _ in range(n):
            x = rng.uniform(0.0, sg.PITCH_LENGTH_M)
            y = rng.uniform(0.0, sg.PITCH_WIDTH_M)
            forward, _ = sg.to_goal_frame(x, y, goal)
            if forward <= 0.05:          # on the line itself the vector form
                continue                 # is degenerate, not wrong
            a = float(sg.goal_mouth_angle(x, y, goal))
            b = angle_by_vectors(x, y, goal)
            if not np.isfinite(b):
                continue
            if abs(a - b) > worst:
                worst, worst_at = abs(a - b), (goal, x, y)
    print(f"   worst disagreement {worst:.2e} rad over {2 * n} points")
    if worst_at:
        print(f"   at {worst_at[0]} goal, ({worst_at[1]:.1f}, "
              f"{worst_at[2]:.1f})")
    ok = worst < TOLERANCE_RAD
    print(f"   {'ok' if ok else 'THE TWO DERIVATIONS DISAGREE'}")
    return not ok


def check_fixed_cases():
    """Answers set by construction rather than by measurement."""
    print("\n3. Cases with answers fixed by construction\n")
    half = sg.GOAL_HALF_M
    centre_y = sg.PITCH_WIDTH_M / 2.0
    cases = [
        ("penalty spot, distance",
         float(sg.distance_to_goal(sg.PENALTY_SPOT_M, centre_y, "left")),
         sg.PENALTY_SPOT_M, 1e-9),
        ("on the line, between the posts",
         float(np.degrees(sg.goal_mouth_angle(0.0, centre_y, "left"))),
         180.0, 1e-6),
        ("on the line, well outside the posts",
         float(np.degrees(sg.goal_mouth_angle(0.0, centre_y + 10.0, "left"))),
         0.0, 1e-9),
        ("level with a post, half a goal-width out",
         float(np.degrees(sg.goal_mouth_angle(half, centre_y, "left"))),
         90.0, 1e-6),
        ("behind the goal line",
         float(np.degrees(sg.goal_mouth_angle(-5.0, centre_y, "left"))),
         0.0, 1e-9),
        ("mirrored across the halfway line",
         float(np.degrees(sg.goal_mouth_angle(18.0, centre_y + 6.0, "left"))),
         float(np.degrees(sg.goal_mouth_angle(sg.PITCH_LENGTH_M - 18.0,
                                              centre_y + 6.0, "right"))),
         1e-9),
        ("mirrored across the pitch",
         float(np.degrees(sg.goal_mouth_angle(18.0, centre_y + 6.0, "left"))),
         float(np.degrees(sg.goal_mouth_angle(18.0, centre_y - 6.0, "left"))),
         1e-9),
    ]
    bad = 0
    for name, got, want, tol in cases:
        ok = abs(got - want) <= tol
        bad += not ok
        print(f"   {name:<38s} {got:8.3f}  expected {want:8.3f}  "
              f"{'ok' if ok else 'WRONG'}")
    return bad


def check_monotonic():
    """Walking straight back from goal must never widen the goal mouth."""
    print("\n4. The angle narrows as the shot gets further out\n")
    centre_y = sg.PITCH_WIDTH_M / 2.0
    forwards = np.linspace(0.0, 60.0, 400)
    angles = np.degrees(sg.goal_mouth_angle(forwards, centre_y, "left"))
    falling = bool(np.all(np.diff(angles) <= 1e-9))
    print(f"   from the goal line to 60 m: {angles[0]:.1f} deg down to "
          f"{angles[-1]:.1f} deg")
    print(f"   monotonic: {'ok' if falling else 'NOT MONOTONIC'}")
    return not falling


def check_cone():
    """Defenders in the shooting triangle."""
    print("\n5. Defenders between the shot and the goal\n")
    centre_y = sg.PITCH_WIDTH_M / 2.0
    shot_x, shot_y = 95.0, centre_y
    cases = [
        ("one directly in the way", [100.0], [centre_y], 1),
        ("one behind the shooter", [90.0], [centre_y], 0),
        ("one wide of the cone", [100.0], [centre_y + 11.0], 0),
        ("two in the way, one wide", [100.0, 102.0, 99.0],
         [centre_y - 1.0, centre_y + 1.0, centre_y + 15.0], 2),
    ]
    bad = 0
    for name, dx, dy, want in cases:
        got = sg.defenders_in_cone(shot_x, shot_y, dx, dy, "right")
        ok = got == want
        bad += not ok
        print(f"   {name:<28s} {got} defenders, expected {want}  "
              f"{'ok' if ok else 'WRONG'}")
    return bad


def check_refusal():
    """Relative coordinates must be refused, not silently used."""
    print("\n6. Relative coordinates are refused\n")
    try:
        sg.shot_features_from_events([], absolute_pitch=False)
    except ValueError as exc:
        print(f"   raised, as it should: {str(exc)[:66]}...")
        return 0
    print("   RETURNED A RESULT from a clip with no goal line")
    return 1


def main():
    failures = (check_constants() + check_against_vectors()
                + check_fixed_cases() + check_monotonic() + check_cone()
                + check_refusal())

    spot = float(np.degrees(sg.goal_mouth_angle(
        sg.PENALTY_SPOT_M, sg.PITCH_WIDTH_M / 2.0, "left")))
    print(f"\n   For reference, the penalty spot subtends {spot:.1f} degrees "
          f"at {sg.PENALTY_SPOT_M:.0f} m.")

    if failures:
        raise SystemExit(f"\n{failures} check(s) failed")
    print("\n   All checks passed.")


if __name__ == "__main__":
    main()
