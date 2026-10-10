"""What the footage says around every moment that could be a strike.

`goal_plane.find_shots` decides with hand-set rules: a flight that fits, or
a straight, fast run aimed at the mouth. Measured on the labelled windows
it finds half the shots, and the misses are each refused by one rule or
another -- a blocked shot moving at 9 m/s, a path that bends at the block,
three sightings where the rule wants four (EVENT_ACCURACY.md). A rule
refuses on its weakest reading; a learned classifier can weigh them
together.

This keeps the readings and drops the thresholds. Every placed ball
sighting inside shooting range is a candidate strike, and for each the
same quantities the rules look at are recorded as numbers -- where it is,
how fast and which way the ball leaves, how straight, whether a flight
fits and where it crosses, whether the plane of its path cuts the mouth,
how crowded it is, how near the goal the ball gets and whether play
restarts from the centre afterwards. `train_shot_classifier.py` learns
from them against the labels.

Undefined readings are NaN: gradient boosted trees take them as their own
branch, which is what "too few sightings to say" should be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import goal_plane as gp

# Windows after the strike that the readings are taken over.
SHORT_S = 0.3
LONG_S = 0.6
# Before the strike: is the ball still, or already moving?
BEFORE_S = 0.3
# How long afterwards to look for the ball reaching the goal.
REACH_S = 2.0
# Flight fits are tried on at most this many sightings: a shot is decided
# within half a second and fits are the costly part.
MAX_FIT_SIGHTINGS = 12
# Players this close to the ball, in pixels at the footage's scale, crowd it.
CROWD_PX = 60.0


def _ground_velocity(ground):
    """Least-squares velocity through the strike, (across, out), m/s."""
    if len(ground) < 2:
        return np.nan, np.nan
    t = np.array([g[0] for g in ground])
    if np.sum(t * t) <= 0:
        return np.nan, np.nan
    return tuple(float(np.sum(t * np.array([g[i] for g in ground]))
                       / np.sum(t * t)) for i in (1, 2))


def _image_speed(seen, t0):
    """Pixels per second of a straight fit through the sightings."""
    if len(seen) < 2:
        return np.nan
    ts = np.array([s[0] - t0 for s in seen])
    if np.ptp(ts) <= 0:
        return np.nan
    vx = np.polyfit(ts, [s[3] for s in seen], 1)[0]
    vy = np.polyfit(ts, [s[4] for s in seen], 1)[0]
    return float(np.hypot(vx, vy))


def _plane_reading(start, seen):
    """Where the plane through camera, strike and sightings meets the goal
    plane at mid-height, across from the goal's centre (m), and whether it
    cuts the mouth with the off-target margins (1/0)."""
    from .ball_height import ray

    normals, eye = [], None
    for _, _, pose, u, v in seen:
        e, d = ray(pose, u, v)
        n = np.cross(d, start - e)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        if normals and float(np.dot(n, normals[0])) < 0:
            n = -n
        normals.append(n)
        eye = e if eye is None else eye
    if len(normals) < 2:
        return np.nan, np.nan
    n = np.mean(normals, axis=0)
    c = float(np.dot(n, eye))
    corners = [(x, y) for x in (-gp.WIDE_M, gp.GOAL_WIDTH_M + gp.WIDE_M)
               for y in (gp.MIN_HEIGHT_M, gp.GOAL_HEIGHT_M + gp.OVER_M)]
    sides = [np.sign(n[0] * x + n[1] * y - c) for x, y in corners]
    cuts = float(not (all(s > 0 for s in sides) or all(s < 0 for s in sides)))
    mid = gp.GOAL_HEIGHT_M / 2.0
    across = ((c - n[1] * mid) / n[0] - gp.GOAL_WIDTH_M / 2.0
              if abs(n[0]) > 1e-6 else np.nan)
    return float(np.clip(across, -60.0, 60.0)), cuts


def _straight_run(seen, t0):
    """The longest run from the strike on one straight, steady path."""
    keep = 0
    for n in range(3, len(seen) + 1):
        if gp._path_rms(seen[:n], t0) > gp.AIM_MAX_PATH_PX:
            break
        keep = n
    return keep


def _flight(start, t0, after, fps):
    """Flight-fit readings: rms on the first three sightings, how many fit
    as one flight, and the crossing if one is read."""
    after = after[:MAX_FIT_SIGHTINGS]
    out = {"fit_rms3": np.nan, "fit_n": 0.0, "cross_found": 0.0,
           "cross_x": np.nan, "cross_y": np.nan, "cross_speed": np.nan,
           "cross_dt": np.nan}
    if len(after) >= 3:
        got = gp.struck_flight(start, t0, after[:3])
        if got is not None:
            out["fit_rms3"] = got[1]
    best = None
    for n in range(gp.MIN_FLIGHT_SIGHTINGS, len(after) + 1):
        got, rms = gp._crossing_from(start, t0, after[:n], fps)
        if rms is None or rms > gp.MAX_FLIGHT_RMS_PX:
            break
        out["fit_n"] = float(n)
        if got is not None:
            best = got
    if best is not None:
        X, Y, when, _, speed, _, _ = best
        out.update(cross_found=1.0, cross_x=X - gp.GOAL_WIDTH_M / 2.0,
                   cross_y=Y, cross_speed=speed, cross_dt=when - t0)
    return out


def candidates(ball: pd.DataFrame, placer, fps: float,
               players: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row of readings per candidate strike."""
    from .goal_pose import ground_point

    rows = list(ball.itertuples())
    placed = []
    for row in rows:
        point = placer.place(row.frame, row.px, row.py)
        placed.append(None if point is None else
                      (float(point[0]), float(point[1])))
    grass = [(float(r.time_s), p[0], p[1], int(r.frame))
             for r, p in zip(rows, placed) if p is not None]
    kickoffs, when = [], -np.inf
    while True:
        k = gp.kickoff_after(when, grass)
        if k is None:
            break
        kickoffs.append(k)
        when = k + gp.KICKOFF_STILL_S + 5.0
    by_frame = ({int(f): g[["px", "py"]].to_numpy()
                 for f, g in players.groupby("frame")}
                if players is not None and len(players) else {})
    has_segments = "segment" in ball.columns
    times = np.array([float(r.time_s) for r in rows])

    out = []
    for k, (row, point) in enumerate(zip(rows, placed)):
        if point is None:
            continue
        x0, y0 = point
        if not gp.MIN_STRIKE_M <= x0 <= gp.MAX_STRIKE_M:
            continue
        t0, f0 = float(row.time_s), int(row.frame)
        segment = int(row.segment) if has_segments else 0
        start = np.array([y0 - gp.LEFT_POST_Y, gp.BALL_RADIUS_M, x0])

        after, before_grass = [], []
        for later in rows[k + 1:]:
            if float(later.time_s) - t0 > gp.LOOKBACK_S:
                break
            if has_segments and int(later.segment) != segment:
                continue
            pose = placer.pose_at(int(later.frame))
            if pose is not None:
                after.append((float(later.time_s), int(later.frame), pose,
                              float(later.px), float(later.py)))
        lo = np.searchsorted(times, t0 - BEFORE_S)
        before = [r for r in rows[lo:k]
                  if not has_segments or int(r.segment) == segment]
        for r, p in zip(rows[lo:k], placed[lo:k]):
            if p is not None and (not has_segments
                                  or int(r.segment) == segment):
                before_grass.append((float(r.time_s) - t0, p[1] - y0,
                                     p[0] - x0))

        feats = {"frame": f0, "time_s": t0, "x_m": x0, "off_centre_m":
                 y0 - gp.PITCH_WIDTH_M / 2.0,
                 "distance_m": float(np.hypot(x0, y0 - gp.PITCH_WIDTH_M / 2)),
                 "confidence": float(row.confidence),
                 "fine_tuned": float(row.detection_method == "ball_detector")}
        # The angle the goal mouth subtends from the strike.
        left = np.arctan2(gp.LEFT_POST_Y - y0, x0)
        right = np.arctan2(gp.LEFT_POST_Y + gp.GOAL_WIDTH_M - y0, x0)
        feats["mouth_angle"] = float(abs(right - left))

        for name, span in (("s", SHORT_S), ("l", LONG_S)):
            seen = [s for s in after if s[0] - t0 <= span]
            ground = []
            for when, _, pose, u, v in seen:
                g = ground_point(pose, u, v)
                if g is not None:
                    ground.append((when - t0, g[0] - start[0],
                                   g[1] - start[2]))
            vx, vz = _ground_velocity(ground)
            speed = float(np.hypot(vx, vz)) if not np.isnan(vx) else np.nan
            feats[f"n_{name}"] = float(len(seen))
            feats[f"placed_{name}"] = float(len(ground))
            feats[f"ground_speed_{name}"] = speed
            feats[f"towards_{name}"] = (-vz / speed if speed and speed > 0
                                        else np.nan)
            feats[f"img_speed_{name}"] = _image_speed(seen, t0)
            feats[f"path_rms_{name}"] = (gp._path_rms(seen, t0)
                                         if len(seen) >= 3 else np.nan)
            feats[f"straight_{name}"] = float(_straight_run(seen, t0))
            if ground:
                xs = [g[1] + start[0] for g in ground]
                zs = [g[2] + start[2] for g in ground]
                off = [not (-gp.LEFT_POST_Y - gp.AIM_PITCH_MARGIN_M <= x
                            <= gp.PITCH_WIDTH_M - gp.LEFT_POST_Y
                            + gp.AIM_PITCH_MARGIN_M
                            and z >= -gp.AIM_PITCH_MARGIN_M)
                       for x, z in zip(xs, zs)]
                feats[f"off_pitch_{name}"] = float(np.mean(off))
            else:
                feats[f"off_pitch_{name}"] = np.nan
            across, cuts = _plane_reading(start, seen)
            feats[f"plane_across_{name}"] = across
            feats[f"plane_cuts_{name}"] = cuts

        bvx, bvz = _ground_velocity(before_grass)
        feats["speed_before"] = (float(np.hypot(bvx, bvz))
                                 if not np.isnan(bvx) else np.nan)
        feats["img_speed_before"] = _image_speed(
            [(float(r.time_s), 0, None, float(r.px), float(r.py))
             for r in before], t0)
        feats["speed_gain"] = (feats["img_speed_s"]
                               - feats["img_speed_before"])
        feats.update(_flight(start, t0, after, fps))

        later = [(t, x) for t, x, _, _ in grass if t0 < t <= t0 + REACH_S]
        feats["min_x_after"] = (min(x for _, x in later) if later
                                else np.nan)
        feats["approach_m"] = (x0 - feats["min_x_after"]
                               if later else np.nan)
        nxt = [t for t in kickoffs if t > t0]
        feats["kickoff_after_s"] = (nxt[0] - t0 if nxt else np.nan)

        near = by_frame.get(f0)
        if near is not None and len(near):
            d = np.hypot(near[:, 0] - row.px, near[:, 1] - row.py)
            feats["nearest_player_px"] = float(d.min())
            feats["crowd"] = float(np.sum(d < CROWD_PX))
        else:
            feats["nearest_player_px"] = np.nan
            feats["crowd"] = np.nan
        out.append(feats)
    return pd.DataFrame(out)
