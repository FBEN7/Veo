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


# Amendment 1 (the radius above failed on M6: its calibration puts the far
# penalty spot about 25 px -- some 7 m on the far side -- from the painted
# one, where v3 fires). The painted spot is a fixed pixel on a fixed view:
# find it from the window's own candidates, near the projected spot, and
# drop the candidates on it. No labels: the same at test time.
SEARCH_PX = 60.0       # around a projected spot: room for the calibration's error
MARK_SHARE = 0.05      # a mark holds a candidate on 5% of the frames (9 s of 3 min):
                       # a ball in play never stays that long on one point
MARK_PX = 4.0          # a candidate within this of a mark is on it (about the ball's size)
MAX_MARKS = 2          # per spot


def spot_marks(rows: pd.DataFrame, cal, n_frames: int, search_px: float = SEARCH_PX,
               share: float = MARK_SHARE, r_px: float = MARK_PX,
               max_marks: int = MAX_MARKS) -> list[tuple[float, float]]:
    """Image points of the painted spots in a dense run's `rows` (frame,
    px, py): near each projected spot, the point with candidates (any rank)
    in the most distinct frames within `r_px`, kept when they are at least
    `share` of the `n_frames` processed; then the next, at most
    `max_marks` a spot."""
    from src.ball_heatmap import pitch_to_image

    marks = []
    proj = pitch_to_image(SPOTS, cal)
    uv = rows[["px", "py"]].to_numpy(float)
    fr = rows.frame.to_numpy()
    for su, sv in proj:
        if not np.isfinite([su, sv]).all():
            continue
        near = np.flatnonzero(np.hypot(uv[:, 0] - su, uv[:, 1] - sv) <= search_px)
        for _ in range(max_marks):
            if not len(near):
                break
            p = uv[near]
            best, best_n, best_sel = None, 0, None
            for i in range(len(near)):
                sel = np.hypot(p[:, 0] - p[i, 0], p[:, 1] - p[i, 1]) <= r_px
                n = len(set(fr[near[sel]]))
                if n > best_n:
                    best, best_n, best_sel = i, n, sel
            if best_n < share * n_frames:
                break
            m = tuple(p[best_sel].mean(0))
            if all(np.hypot(m[0] - q[0], m[1] - q[1]) > r_px for q in marks):
                marks.append(m)
            near = near[~best_sel]
    return marks


def drop_marks(rows: pd.DataFrame, marks, r_px: float = MARK_PX) -> pd.DataFrame:
    """`rows` less the candidates within `r_px` of a mark."""
    if not marks:
        return rows.reset_index(drop=True)
    m = np.asarray(marks, float)
    d = np.hypot(rows.px.to_numpy()[:, None] - m[None, :, 0],
                 rows.py.to_numpy()[:, None] - m[None, :, 1]).min(1)
    return rows[d > r_px].reset_index(drop=True)


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
    import src.ball_heatmap as bh

    orig2 = bh.pitch_to_image
    bh.pitch_to_image = lambda xy, cal: np.asarray(xy, float) + [100.0, 100.0]
    try:
        rng = np.random.default_rng(0)
        n = 1000
        mark = pd.DataFrame({"frame": np.arange(n), "px": 141.5 + 12 + rng.normal(0, 0.5, n),
                             "py": 100.0 + 16 + rng.normal(0, 0.5, n)})          # 20 px off the projection
        ball = pd.DataFrame({"frame": np.arange(200), "px": 153.5 + np.linspace(-100, 100, 200),
                             "py": np.full(200, 116.0)})              # through it at 1 px a frame
        rows = pd.concat([mark, ball], ignore_index=True)
        marks = spot_marks(rows, Cal(), n)
        assert len(marks) == 1 and abs(marks[0][0] - 153.5) < 1 and abs(marks[0][1] - 116) < 1
        kept = drop_marks(rows, marks)
        assert (np.hypot(kept.px - marks[0][0], kept.py - marks[0][1]) > MARK_PX).all()
        assert len(kept) >= len(ball) - 10
        assert spot_marks(ball, Cal(), n) == []
    finally:
        bh.pitch_to_image = orig2
    print("  ball_spots: distance to the spots, drop, marks found from the candidates: ok")


if __name__ == "__main__":
    _check()
