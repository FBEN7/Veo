"""The pitch map must not depend on which end of a line segment came first.

This locks closed a bug that was live for the whole life of the anchor and
that no accuracy measurement could have caught.

`metric_from_circle` fixes the rotation from the halfway line, and a circle
plus a line determine it only to within 180 degrees -- both look the same
from either end. The resolution was `arctan2` on the line's direction
vector, built from a detected segment's two endpoints in the order
`HoughLinesP` happened to list them, which is not an order it promises
anything about. Reverse it and the angle moves by 180 degrees and the whole
pitch turns end for end.

Nothing caught it because a football pitch is exactly symmetric under that
rotation: 105 minus each line across it gives back the same seven numbers,
68 minus each line along it the same six. A flipped map puts every marking
on a real pitch line, so `marking_error` scores it identically -- and so
would any other check built on distance to the markings.

The consequence is not cosmetic. A shot flipped end for end is attributed to
the other goal, and a shot from six yards becomes a shot from ninety-nine.

    python test_pitch_orientation.py
"""

from __future__ import annotations

import numpy as np

from src import pitch_model as pm


def a_circle(centre=(320.0, 260.0), axes=(180.0, 70.0), angle=8.0):
    """An ellipse standing in for a centre circle seen at an angle."""
    return (centre, axes, angle)


def test_reversing_the_halfway_line_changes_nothing():
    """The direction vector's sign is an artefact; it must not reach output."""
    circle = a_circle()
    for degrees in range(0, 180, 15):
        radians = np.deg2rad(degrees)
        direction = np.array([np.cos(radians), np.sin(radians)])
        forward = pm.metric_from_circle(pm.AT_INFINITY_LINE, circle, direction)
        backward = pm.metric_from_circle(pm.AT_INFINITY_LINE, circle,
                                         -direction)
        assert forward is not None and backward is not None
        assert np.allclose(forward, backward, rtol=1e-9, atol=1e-9), (
            f"reversing the halfway line at {degrees} deg changed the map")


def test_down_the_image_means_down_the_pitch():
    """The convention the orientation is pinned to, stated as a test.

    The camera sits beside the pitch rather than orbiting it, so the near
    touchline stays near. Whatever the halfway line's direction, moving down
    the image must move toward increasing y.
    """
    circle = a_circle()
    for degrees in range(0, 360, 15):
        radians = np.deg2rad(degrees)
        direction = np.array([np.cos(radians), np.sin(radians)])
        metric = pm.metric_from_circle(pm.AT_INFINITY_LINE, circle, direction)
        assert metric is not None
        probe = np.array([[circle[0][0], circle[0][0]],
                          [circle[0][1], circle[0][1] + 40.0],
                          [1.0, 1.0]])
        mapped = metric @ probe
        here, below = (mapped[:2] / mapped[2]).T
        assert below[1] >= here[1] - 1e-9, (
            f"at {degrees} deg, moving down the image moved up the pitch")


def test_the_pitch_model_is_symmetric_end_for_end():
    """Why no marking-based check could ever have found the bug."""
    across = np.array(pm.LINES_ACROSS_M)
    along = np.array(pm.LINES_ALONG_M)
    assert np.allclose(np.sort(pm.PITCH_LENGTH_M - across), np.sort(across))
    assert np.allclose(np.sort(pm.PITCH_WIDTH_M - along), np.sort(along))


if __name__ == "__main__":
    failures = 0
    for name, case in sorted(globals().items()):
        if not name.startswith("test_") or not callable(case):
            continue
        try:
            case()
            print(f"  ok    {name}")
        except AssertionError as problem:
            failures += 1
            print(f"  FAIL  {name}: {problem}")
    print(f"\n{'all passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
