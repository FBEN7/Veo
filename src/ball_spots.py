"""Ball candidates on the pitch's marked spots (protocol v5, step 3b).

On a fixed, calibrated view the centre spot and the two penalty spots are
at known pitch points. A ball detector's candidate there is a mark, or a
ball placed for a restart -- not the match ball in play (the judges' S).
v3 picks a spot when no ball is visible (the blind check of its picks on
M6), and tracking keeps those picks (its fallback takes the raw top-1).

`on_spot` tells, for candidate rows, whether their ground point (the
ball's bottom, (u, v + size / 2), through the match calibration) lies
within SPOT_R of a spot. SPOT_R = 0.75 m was set on the 14 B windows
(training data, not M6): of their candidates within 0.5 m of a spot, all
212 are static (the same place in 2 or more other frames), none judged the
ball; between 0.5 and 0.75 m 40% are static; beyond, about 20% (the
pitch's background). `drop_spots` removes them from a dense run's rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPOTS = np.array([[0.0, 0.0], [-41.5, 0.0], [41.5, 0.0]])    # centre, penalty spots (m)
SPOT_R = 0.75


def spot_distance(u, v, size, cal) -> np.ndarray:
    """Distance (m) of each candidate's ground point to the nearest spot;
    inf where the point does not project onto the ground."""
    from src import soccertrack_v2 as st

    u, v = np.asarray(u, float), np.asarray(v, float)
    if not len(u):
        return np.zeros(0)
    size = np.nan_to_num(np.asarray(size, float), nan=0.0)
    xy = st.image_to_pitch(np.c_[u, v + size / 2], cal)
    d = np.hypot(xy[:, None, 0] - SPOTS[None, :, 0], xy[:, None, 1] - SPOTS[None, :, 1]).min(1)
    return np.where(np.isfinite(d), d, np.inf)


def on_spot(u, v, size, cal, r: float = SPOT_R) -> np.ndarray:
    return spot_distance(u, v, size, cal) <= r


def drop_spots(rows: pd.DataFrame, cal, r: float = SPOT_R) -> pd.DataFrame:
    """A dense run's rows (px, py, crop_h) less those on a spot."""
    keep = ~on_spot(rows.px.to_numpy(), rows.py.to_numpy(), rows.crop_h.to_numpy(), cal, r)
    return rows[keep].reset_index(drop=True)


def _check():
    class Cal(dict):
        pass

    import src.soccertrack_v2 as st

    orig = st.image_to_pitch
    st.image_to_pitch = lambda uv, cal: np.asarray(uv, float)       # identity "calibration"
    try:
        d = spot_distance([0.0, 41.0, 10.0], [0.0, 0.0, 10.0], [0.0, 0.0, 0.0], Cal())
        assert np.allclose(d[:2], [0.0, 0.5]) and d[2] > 14
        rows = pd.DataFrame({"px": [0.0, 41.0, 10.0], "py": [0.0, 0.0, 10.0], "crop_h": [0.0] * 3})
        assert drop_spots(rows, Cal()).px.tolist() == [10.0]
    finally:
        st.image_to_pitch = orig
    print("  ball_spots: distance to the spots, drop: ok")


if __name__ == "__main__":
    _check()
