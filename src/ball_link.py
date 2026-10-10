"""One ball per frame from a detector's candidates, chosen in pitch metres
(`link`), for a calibrated fixed view.

`ball_selection.select_single_ball` scores motion in pixels with one
pixels-per-metre for the whole picture. A SoccerTrack v2 panorama spans
about 2-72 px/m between its near touchline and its far corners, so the same
step in pixels is a walk on one side and an impossible shot on the other.
Here every candidate is put on the pitch first, at its ground point
image_to_pitch(u, v + size/2) as `pixel_scale.to_pitch_metres` does, and the
chain is chosen there by a Viterbi pass that may skip frames.

Costs along a chain, lower is better:

- **Candidate:** -logit(p) + logit(tau), p its (calibrated) confidence. A
  candidate at tau costs nothing; a confident one is a gain.
- **No ball in a frame:** c_null. Two consecutive picks may be up to
  MAX_SKIP detection frames apart, each skipped frame costing c_null; a
  longer gap ends the chain, and the next may start anywhere. A jump to an
  unreachable candidate therefore costs at least (MAX_SKIP + 1) c_null.
- **Motion** between consecutive picks a, b: their distance less a slack of
  2 (sigma_a + sigma_b), over the time between them, is a speed v. It is
  free up to 35 m/s, costs 4 (v/35 - 1)^2 above that and is forbidden
  above 60 m/s. sigma is what 1.5 px in the picture is on the ground there:
  1.5 px over the smaller of the local px/m along and across the pitch,
  clipped to [0.05, 1.0] m.
- **Static:** c_static on a candidate that has stayed within 0.5 m for 2 s
  or more and is 3 m or more from every player the pipeline detected at
  that frame: a spare ball by a goal, a penalty spot.

Players come only from the pipeline's own detections, never from ground
truth.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

MAX_SKIP = 5                 # detection frames a chain may skip between picks
V_FREE_MS = 35.0             # free speed (m/s)
V_MAX_MS = 60.0              # forbidden above (m/s)
SPEED_WEIGHT = 4.0
SIGMA_PX = 1.5               # ground-point error, in picture pixels
SIGMA_CLIP_M = (0.05, 1.0)
STATIC_RADIUS_M = 0.5
STATIC_MIN_S = 2.0
STATIC_GAP_S = 1.0           # a stay may be interrupted this long
STATIC_CLEAR_M = 3.0         # ...and counts only this far from every player
PLAYER_NEAR_FRAMES = 2       # players read at the nearest frame within this


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ground_points(rows: pd.DataFrame, cal: dict) -> np.ndarray:
    """(N, 2) pitch metres (centre origin) of rows' picture points: a
    ball's box bottom (v + crop_h/2), a player's feet; NaN where unmapped."""
    from .soccertrack_v2 import image_to_pitch

    if rows.empty:
        return np.zeros((0, 2))
    u = (rows.px_raw if "px_raw" in rows else rows.px).to_numpy(float)
    v = (rows.py_raw if "py_raw" in rows else rows.py).to_numpy(float)
    if "crop_h" in rows:
        v = np.where(rows.cls.to_numpy() == "ball",
                     v + rows.crop_h.fillna(0).to_numpy(float) / 2, v)
    return np.asarray(image_to_pitch(np.c_[u, v], cal), float).reshape(-1, 2)


def ground_sigma(xy: np.ndarray, cal: dict) -> np.ndarray:
    """Per point, SIGMA_PX on the ground: over the smaller of the local
    px/m along and across the pitch (`ball_heatmap.local_scale`,
    vectorised), clipped to SIGMA_CLIP_M."""
    from .ball_heatmap import pitch_to_image

    n = len(xy)
    if n == 0:
        return np.zeros(0)
    step = 0.5
    p = pitch_to_image(np.vstack([xy, xy + (step, 0.0), xy + (0.0, step)]),
                       cal).reshape(3, n, 2)
    sx = np.hypot(*(p[1] - p[0]).T) / step
    sy = np.hypot(*(p[2] - p[0]).T) / step
    with np.errstate(divide="ignore", invalid="ignore"):
        s = SIGMA_PX / np.minimum(sx, sy)
    return np.clip(np.nan_to_num(s, nan=SIGMA_CLIP_M[1]), *SIGMA_CLIP_M)


def static_mask(frames: np.ndarray, xy: np.ndarray, fps: float) -> np.ndarray:
    """Candidates that have stayed: those with candidates within
    STATIC_RADIUS_M of them over a run of frames lasting STATIC_MIN_S or
    more (gaps up to STATIC_GAP_S) that contains their own frame."""
    out = np.zeros(len(frames), bool)
    if not len(frames):
        return out
    cell = np.floor(xy / STATIC_RADIUS_M).astype(int)
    buckets: dict[tuple, list] = defaultdict(list)
    for i, c in enumerate(map(tuple, cell)):
        buckets[c].append(i)
    buckets = {k: np.array(v) for k, v in buckets.items()}
    gap, need = STATIC_GAP_S * fps, STATIC_MIN_S * fps
    for i in range(len(frames)):
        cx, cy = cell[i]
        cand = np.concatenate([buckets.get((cx + a, cy + b), np.zeros(0, int))
                               for a in (-1, 0, 1) for b in (-1, 0, 1)])
        near = cand[np.hypot(*(xy[cand] - xy[i]).T) <= STATIC_RADIUS_M]
        fr = np.unique(frames[near])
        if fr[-1] - fr[0] < need:
            continue
        breaks = np.flatnonzero(np.diff(fr) > gap)
        k = int(np.searchsorted(fr, frames[i]))
        r = int(np.searchsorted(breaks, k, side="left"))
        start = 0 if r == 0 else breaks[r - 1] + 1
        end = breaks[r] if r < len(breaks) else len(fr) - 1
        out[i] = fr[end] - fr[start] >= need
    return out


def clear_of_players(frames: np.ndarray, xy: np.ndarray,
                     players_xy: dict[int, np.ndarray]) -> np.ndarray:
    """Candidates STATIC_CLEAR_M or more from every player at their frame
    (or the nearest player frame within PLAYER_NEAR_FRAMES; none -> clear)."""
    out = np.ones(len(frames), bool)
    for i, f in enumerate(frames):
        for d in sorted(range(-PLAYER_NEAR_FRAMES, PLAYER_NEAR_FRAMES + 1),
                        key=abs):
            p = players_xy.get(int(f) + d)
            if p is not None:
                out[i] = bool(np.all(np.hypot(*(p - xy[i]).T)
                                     >= STATIC_CLEAR_M))
                break
    return out


def motion_cost(xa, sa, fa, xb, sb, fb, fps: float) -> np.ndarray:
    """Transition cost between candidate sets a (rows) and b (columns):
    0 to V_FREE_MS, SPEED_WEIGHT (v/V_FREE_MS - 1)^2 to V_MAX_MS, inf above."""
    d = np.hypot(xb[None, :, 0] - xa[:, None, 0], xb[None, :, 1] - xa[:, None, 1])
    dt = (fb[None, :] - fa[:, None]) / fps
    v = np.maximum(d - 2 * (sa[:, None] + sb[None, :]), 0.0) / dt
    cost = SPEED_WEIGHT * (np.maximum(v / V_FREE_MS - 1, 0.0)) ** 2
    return np.where(v > V_MAX_MS, np.inf, cost)


def link(rows: pd.DataFrame, cal: dict, players: pd.DataFrame,
         c_null: float = 1.0, c_static: float = 2.0, fps: float = 25,
         tau: float = 0.5, frames=None) -> pd.DataFrame:
    """At most one ball row per frame: the subset of `rows` (ball
    candidates; px/py or px_raw/py_raw pixels, crop_h the ball's size,
    confidence its probability) on the cheapest chain, same columns and
    index. `players`: the pipeline's player rows (pixels), for the static
    penalty. The detection frames, which skips count, are `frames` or the
    frames of `rows` and `players`. Rows that do not map onto the pitch
    are never chosen."""
    if rows.empty:
        return rows.copy()
    xy = ground_points(rows, cal)
    ok = np.isfinite(xy).all(1)
    cand = rows[ok]
    xy = xy[ok]
    if cand.empty:
        return cand.copy()
    fr = cand.frame.to_numpy(int)
    p = (cand.confidence.fillna(tau).to_numpy(float)
         if "confidence" in cand else np.full(len(cand), tau))
    sigma = ground_sigma(xy, cal)

    pxy = ground_points(players, cal) if len(players) else np.zeros((0, 2))
    keep = np.isfinite(pxy).all(1)
    pf = players.frame.to_numpy(int)[keep] if len(players) else np.zeros(0, int)
    players_xy = {int(f): pxy[keep][pf == f] for f in np.unique(pf)}
    static = static_mask(fr, xy, fps) & clear_of_players(fr, xy, players_xy)
    unary = -_logit(p) + _logit(tau) + c_static * static

    grid = np.unique(np.r_[fr, pf if frames is None
                           else np.asarray(frames, int)])
    g = np.searchsorted(grid, fr)
    order = np.argsort(g, kind="stable")
    js, starts = np.unique(g[order], return_index=True)
    by_g = dict(zip(js.tolist(), np.split(order, starts[1:])))

    n = len(cand)
    best = np.full(n, np.inf)
    back = np.full(n, -1)
    # Chains that ended at least MAX_SKIP + 1 frames before: the cheapest
    # best[c] - c_null * g[c], so a restart at frame j costs it plus
    # c_null * (j - 1) for the frames in between.
    run_min, run_arg = c_null, -1     # the empty chain, at g = -1
    for j in range(len(grid)):
        ended = j - MAX_SKIP - 2
        if ended >= 0:
            for c in by_g.get(ended, ()):
                v = best[c] - c_null * g[c]
                if v < run_min:
                    run_min, run_arg = v, c
        cur = by_g.get(j)
        if cur is None:
            continue
        cost = np.full(len(cur), run_min + c_null * (j - 1))
        arg = np.full(len(cur), run_arg)
        for lag in range(1, MAX_SKIP + 2):
            prev = by_g.get(j - lag)
            if prev is None:
                continue
            m = (best[prev][:, None] + c_null * (lag - 1)
                 + motion_cost(xy[prev], sigma[prev], fr[prev],
                               xy[cur], sigma[cur], fr[cur], fps))
            k = np.argmin(m, axis=0)
            mk = m[k, np.arange(len(cur))]
            better = mk < cost
            cost[better] = mk[better]
            arg[better] = prev[k[better]]
        best[cur] = unary[cur] + cost
        back[cur] = arg

    end = best + c_null * (len(grid) - 1 - g)
    if not np.isfinite(end).any() or end.min() >= c_null * len(grid):
        return cand.iloc[:0].copy()
    chain, c = [], int(np.argmin(end))
    while c >= 0:
        chain.append(c)
        c = int(back[c])
    return cand.iloc[sorted(chain)].copy()


def _check():
    """Synthetic tracks on a stand-in calibration (20 px/m everywhere),
    checking that the linker follows a ball across 4 missing detection
    frames, refuses a confident one-frame decoy 15 m off its path, prefers
    a moving ball near a player to a brighter resting one far from
    everyone, and keeps at most one row per frame."""
    # A fisheye camera with a very long focal length is a pinhole one to
    # within 1e-5 px here; the homography is 20 px/m about the centre.
    K = np.array([[1e6, 0, 2048], [0, 1e6, 540], [0, 0, 1]])
    s = 20.0
    Hp = np.array([[s, 0, 2048 - 52.5 * s], [0, s, 540 - 34 * s], [0, 0, 1]])
    cal = {"K": K, "D": np.zeros(4), "Knew": K, "Hinv": np.linalg.inv(Hp)}
    from . import soccertrack_v2 as st

    probe = st.image_to_pitch([[2048.0, 540.0]], cal)[0]
    assert np.allclose(probe, (0, 0), atol=1e-3), probe

    def px(x, y):
        return 2048 + x * s, 540 + y * s

    rows, players = [], []
    for f in range(0, 200, 2):
        t = f / 25
        # The ball moves at 10 m/s along x from (-20, 0), unseen 32-38.
        if not 32 <= f <= 38:
            u, v = px(-20 + 10 * t, 0)
            rows.append(dict(frame=f, time_s=t, track_id=-1, cls="ball",
                             px=u, py=v, crop_h=0.0, confidence=0.6))
        # A brighter resting ball at (30, 20), far from everyone.
        u, v = px(30, 20)
        rows.append(dict(frame=f, time_s=t, track_id=-1, cls="ball", px=u,
                         py=v, crop_h=0.0, confidence=0.9))
        # A confident one-frame decoy 15 m ahead of it at frame 60.
        if f == 60:
            u, v = px(-20 + 10 * t + 15, 0)
            rows.append(dict(frame=f, time_s=t, track_id=-1, cls="ball",
                             px=u, py=v, crop_h=0.0, confidence=0.99))
        u, v = px(-20 + 10 * t + 1, 1)
        players.append(dict(frame=f, time_s=t, track_id=1, cls="player",
                            px=u, py=v, crop_h=40.0))
    rows, players = pd.DataFrame(rows), pd.DataFrame(players)

    xy = ground_points(rows, cal)
    assert np.allclose(xy[0], (-20, 0), atol=1e-6)
    sig = ground_sigma(xy[:3], cal)
    assert np.allclose(sig, 1.5 / s), sig
    st_mask = static_mask(rows.frame.to_numpy(), xy, 25)
    assert st_mask[rows.confidence.to_numpy() == 0.9].all()
    assert not st_mask[rows.confidence.to_numpy() == 0.6].any()
    mc = motion_cost(np.array([[0.0, 0]]), np.array([0.0]), np.array([0]),
                     np.array([[1.4, 0], [2.0, 0], [3.0, 0]]),
                     np.zeros(3), np.array([1, 1, 1]), 25)
    assert mc[0, 0] == 0 and 0 < mc[0, 1] < np.inf and mc[0, 2] == np.inf

    out = link(rows, cal, players, c_null=1.0, c_static=2.0, tau=0.5)
    assert out.frame.is_unique
    moving = out[out.confidence == 0.6]
    assert len(moving) == (rows.confidence == 0.6).sum(), len(moving)
    assert not (out.confidence == 0.99).any()
    assert not (out.confidence == 0.9).any(), "resting ball chosen"
    # Without the static penalty the brighter resting ball wins.
    free = link(rows, cal, players, c_null=1.0, c_static=0.0, tau=0.5)
    assert (free.confidence == 0.9).sum() > len(free) // 2
    # No candidates, or none worth a frame: nothing.
    assert link(rows.iloc[:0], cal, players).empty
    weak = rows.assign(confidence=0.01)
    assert link(weak, cal, players, c_null=1.0, tau=0.5).empty
    print("ball_link check ok:", len(out), "of", len(rows), "rows kept")


if __name__ == "__main__":
    _check()
