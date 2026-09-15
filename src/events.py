"""Rule-based football event detection from tracking data.

Events detected:
    - pass: successful possession transfer between teammates
    - interception: possession transfer to the opposing team
    - shot: fast ball action moving toward goal
    - goal: ball enters a goal rectangle
    - out_of_play: ball leaves pitch bounds
    - carry: controlled ball progression (conduite de balle)
    - recovery: team regains control after a loose-ball phase
    - tackle: duel/tackle proxy on direct possession wins

All coordinates are in pitch metres (0-105 x 0-68).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import ball_tracking

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSSESSION_RADIUS_M = 8.0        # Larger radius for loose ball detection (limited view)
POSSESSION_GAP_FILL_S = 3.0      # Longer gaps tolerated in limited visibility
# How far the ball must travel to count as a pass.
#
# This looked like the obvious place to fix pass precision: the emitted passes
# had a median travel of 0.6 m over 0.12 s, which is not a pass. Raising the
# gate was measured and rejected -- at 3.0 m, precision on the first window
# reached 1.00 but only five passes survived, F1 falling from 0.64 to 0.25,
# because the gate deleted the true passes along with the false ones.
#
# That is what identified the real fault. The distance was not a pass length
# at all: possession was gap-filled through the ball's flight, so it only
# changed once the ball had arrived and both endpoints were measured at the
# receiver's feet. With that fixed (see _possession_per_frame), the median
# travel is 1.3 m on one window and 4.6 m on the other, and 0.3 m is once
# again the right floor -- a sweep over 0.3 to 8.0 m on both windows puts the
# best cross-window result at this value.
MIN_PASS_DISTANCE_M = 0.3

# Kept at zero deliberately. A minimum interval is a plausible-sounding filter
# and it measured as a straight loss: raising it to 0.1 s cost F1 0.70 -> 0.62
# on one window and 0.55 -> 0.51 on the other. Real short passes complete
# quickly, so the interval does not separate them from noise.
MIN_PASS_INTERVAL_S = 0.0

MAX_PASS_INTERVAL_S = 8.0        # More time to complete passes across limited view
PASS_UNKNOWN_BRIDGE_S = 2.5      # Longer unknown bridges for partial visibility
SHOT_MIN_KMH = 10.0              # Lower speed for visible shots
SHOT_MAX_DIST_FROM_GOAL_M = 50.0 # Shots from further (limited angle coverage)
SHOT_COOLDOWN_S = 0.5            # Faster shot detection in limited play
CARRY_MIN_TIME_S = 0.8           # Shorter carries (limited distance visible)
CARRY_MIN_DISTANCE_M = 2.0       # Lower distance threshold (partial moves only)
RECOVERY_MIN_LOOSE_S = 0.3       # Quick recovery detection
TACKLE_MAX_PLAYER_DIST_M = 4.0   # Larger tackle radius for proximity estimation
TACKLE_MAX_BALL_SPEED_KMH = 40.0 # Higher speed tolerance

# Goal rectangles (centre of pitch width = y = 34 m)
GOAL_WIDTH_M = 7.32
GOAL_HALF = GOAL_WIDTH_M / 2.0  # 3.66 m
GOAL_DEPTH_M = 2.0              # how deep into goal we allow the ball

LEFT_GOAL_X_MAX = GOAL_DEPTH_M
LEFT_GOAL_Y_MIN = 34.0 - GOAL_HALF
LEFT_GOAL_Y_MAX = 34.0 + GOAL_HALF

RIGHT_GOAL_X_MIN = 105.0 - GOAL_DEPTH_M
RIGHT_GOAL_Y_MIN = 34.0 - GOAL_HALF
RIGHT_GOAL_Y_MAX = 34.0 + GOAL_HALF

# Pitch bounds with tolerance
PITCH_X_MIN, PITCH_X_MAX = -1.0, 106.0
PITCH_Y_MIN, PITCH_Y_MAX = -1.0, 69.0

VEL_SMOOTH_WINDOW = 5           # frames over which to smooth ball velocity

# --- Possession quality -----------------------------------------------------
# Nearest-player assignment flickers between two players standing close
# together, and every flicker used to emit a pass. These three rules make
# possession something a player holds rather than something the geometry
# reassigns frame by frame.

# A possession spell shorter than this is assignment noise, not control.
MIN_POSSESSION_HOLD_S = 0.20

# A challenger must be this much closer than the incumbent to take possession.
# Without hysteresis two players a few centimetres apart trade the ball back
# and forth for as long as they run together.
POSSESSION_MARGIN_M = 0.5

# Nobody controls a ball travelling this fast; it is in flight, and whoever
# happens to be nearest to its path does not possess it. Without this the
# nearest player to a moving ball "possesses" it at every frame of its
# trajectory, turning one pass into a chain of them.
POSSESSION_MAX_BALL_SPEED_KMH = 60.0

# The ball is under control, rather than still arriving, below this speed.
BALL_CONTROL_SPEED_KMH = 20.0

# How long after a pass to wait before deciding who received it.
RECEIVER_SETTLE_WINDOW_S = 1.5

# A failed pass is one event with an outcome, not a pass plus an interception.
# Emitting both double-counts every turnover.
EMIT_INTERCEPTION_AS_SEPARATE_EVENT = False

# Merging near-duplicate detections of the same type. Measured and rejected.
#
# The reasoning was sound and the evidence looked strong. Every false positive
# the detector produced sat within five seconds of a real labelled event and
# most within one -- 44 of 71 inside a second, none beyond five -- so it was
# reporting one action twice rather than inventing football. Collapsing
# near-duplicates addressed exactly that, and on the two windows available at
# the time it improved precision by 0.05 on all four measurements with recall
# untouched and every p-value strengthening.
#
# A third window, held out and scored only after these values were fixed,
# withdrew it. Mean F1 over all three windows:
#
#   merge      pass    carry   windows above chance
#   0.0 s      0.642   0.576   3/3 and 3/3
#   0.8 s      0.635   0.576   3/3 and 3/3
#   1.0 s      0.637   0.585   3/3 and 2/3
#   1.5 s      0.593   0.590   2/3 and 3/3
#
# Pass is best with merging off. Carry's gain at 1.0 s is +0.009 mean F1 --
# noise across three samples -- and costs a window its significance. On the
# held-out window alone, merging took pass from p 0.001 to p 0.029 and carry
# from p 0.059 to p 0.094.
#
# So the two values were fitted to two windows, which is what the commit
# adding them warned they might be. Merging stays in the code because the
# diagnosis behind it still stands and a larger sample may yet support it;
# it is off because three windows say it does not help.
MERGE_WINDOW_S = 0.0

MERGE_WINDOW_BY_TYPE: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth_series(s: pd.Series, window: int = VEL_SMOOTH_WINDOW) -> pd.Series:
    # Use expanding window at edges to avoid NaN propagation early on
    rolled = s.rolling(window, min_periods=1, center=True).median()
    # For first few frames, use simpler 3-point smoothing
    if len(s) > 0:
        rolled_early = s.rolling(3, min_periods=1, center=True).median()
        mask = s.index < min(window, 10)
        rolled[mask] = rolled_early[mask]
    return rolled


def _ball_kinematics(tracks: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per frame that has a ball detection.

    Columns: frame, time_s, bx, by, vel_x, vel_y, speed_kmh

    Uses Kalman filtering for smoothing (no gap-filling to keep frame consistency).
    """
    # Use improved ball tracking (Kalman filter + validation, NO gap-filling)
    # Gap-filling creates frames that don't exist in player tracking, breaking possession detection
    ball = ball_tracking.extract_ball_tracking(tracks, smooth_window=3, fill_gaps=False)

    if ball.empty:
        return pd.DataFrame(columns=["frame", "time_s", "bx", "by", "vel_x", "vel_y", "speed_kmh"])

    # Rename columns for compatibility
    ball = ball.rename(columns={"x": "bx", "y": "by", "vx": "vel_x", "vy": "vel_y"})

    # Calculate speed from velocity components
    speed = np.sqrt(ball["vel_x"]**2 + ball["vel_y"]**2) * 3.6  # m/s to km/h
    ball["speed_kmh"] = speed.clip(upper=120.0)

    return ball[["frame", "time_s", "bx", "by", "vel_x", "vel_y", "speed_kmh"]]


def _detect_ball_movement_events(ball: pd.DataFrame, events: list[dict[str, Any]]) -> None:
    """Detect ball movement patterns that indicate passes/shots (limited FOV mode).

    DISABLED: This function was creating 138 false positives per 2 minutes.
    High-speed ball movements alone are insufficient to reliably detect events.

    Requires: Ball movement + visible player validation to work correctly.
    Status: Disabled until proper player validation is implemented.
    """
    # Disabled - creates too many false positives without player validation
    return


def _possession_per_frame(
    ball: pd.DataFrame, tracks: pd.DataFrame,
    release_on_flight: bool = False,
) -> pd.DataFrame:
    """Return frame-wise possession.

    Nearest-player assignment is used first, with hysteresis so two players
    running together do not trade the ball, and a minimum hold so assignment
    flicker does not read as control.

    ``release_on_flight`` decides what happens while the ball is travelling
    too fast for anyone to control, and the two answers are both right for
    different questions:

      False  the holder keeps the ball through its flight. Possession spells
             stay unbroken, which is what carry detection needs -- a carry is
             a continuous spell of 0.8 s or more, and a ball that briefly
             exceeds the speed gate mid-dribble would otherwise chop it in
             two. Ball speed is noisy enough that 21.6% of frames cross the
             gate, so the fragmentation is severe: carry detection drops from
             p 0.001 to p 0.179 on one window and p 0.026 to p 0.896 on the
             other.

      True   possession ends where the ball was struck. The spell boundary is
             then the pass itself rather than the moment the ball arrives, so
             a pass is measured from the passer to the receiver instead of
             from the receiver's feet to the receiver's feet. Median measured
             travel goes from 0.6 m to 1.3 and 4.6 m on the two windows.

    Passes and carries therefore want opposite timelines, which is why
    ``detect_events`` builds both rather than compromising on one.
    """
    players = tracks[
        (tracks.cls == "player")
        & (tracks.team.isin(["team_A", "team_B"]))
    ][["frame", "track_id", "team", "px", "py"]]

    if players.empty:
        return pd.DataFrame(
            {
                "frame": ball["frame"],
                "time_s": ball["time_s"],
                "possessor_id": -1,
                "possessor_team": "unknown",
                "dist": np.nan,
            }
        )

    # Only compute distances for frames where we actually have player data
    merged = ball[["frame", "time_s", "bx", "by"]].merge(players, on="frame", how="inner")
    merged["dist"] = np.sqrt(
        (merged["px"] - merged["bx"]) ** 2 + (merged["py"] - merged["by"]) ** 2
    )

    valid = merged.dropna(subset=["dist"])
    if valid.empty:
        return pd.DataFrame(
            {
                "frame": ball["frame"],
                "time_s": ball["time_s"],
                "possessor_id": -1,
                "possessor_team": "unknown",
                "dist": np.nan,
            }
        )

    idx = valid.groupby("frame")["dist"].idxmin()
    nearest = valid.loc[idx, ["frame", "time_s", "track_id", "team", "dist"]].rename(
        columns={"track_id": "raw_id", "team": "raw_team", "dist": "raw_dist"}
    )

    ball_cols = ["frame", "time_s"]
    if "speed_kmh" in ball.columns:
        ball_cols.append("speed_kmh")
    full = ball[ball_cols].merge(nearest, on=["frame", "time_s"], how="left")
    if "speed_kmh" not in full.columns:
        full["speed_kmh"] = np.nan

    # Every player's distance to the ball, per frame. The nearest player is
    # not enough: hysteresis has to compare a challenger against where the
    # *incumbent* is now, and an incumbent who has drifted away is no longer
    # the nearest, so their current distance would otherwise be unavailable.
    dist_by_frame: dict[int, dict[int, tuple[float, str]]] = {}
    for frame, group in valid.groupby("frame", sort=False):
        dist_by_frame[int(frame)] = {
            int(tid): (float(d), str(tm))
            for tid, d, tm in zip(group.track_id, group.dist, group.team)
        }

    poss_ids: list[int] = []
    poss_teams: list[str] = []
    poss_dist: list[float] = []

    current_id = -1
    current_team = "unknown"
    gap_time = 0.0
    prev_t: float | None = None

    for _, row in full.iterrows():
        t = float(row["time_s"])
        dt = 0.0 if prev_t is None else max(0.0, t - prev_t)

        rid = int(row["raw_id"]) if pd.notna(row["raw_id"]) else -1
        rteam = str(row["raw_team"]) if pd.notna(row["raw_team"]) else "unknown"
        rdist = float(row["raw_dist"]) if pd.notna(row["raw_dist"]) else np.nan

        # A ball in flight belongs to nobody. Whoever is nearest to its path
        # is not in control of it, and treating them as the possessor turns a
        # single pass into a chain of handovers along its trajectory.
        in_flight = (pd.notna(row.get("speed_kmh"))
                     and float(row["speed_kmh"]) > POSSESSION_MAX_BALL_SPEED_KMH)

        # Where the incumbent is *now*, not where they were when they gained
        # the ball. Holding the distance at which possession was taken lets a
        # player who has run twenty metres away keep it, because a challenger
        # has to beat a stale number.
        here = dist_by_frame.get(int(row["frame"]), {})
        incumbent_dist = here.get(current_id, (np.nan, ""))[0] if current_id != -1 else np.nan
        incumbent_holding = (pd.notna(incumbent_dist)
                             and incumbent_dist <= POSSESSION_RADIUS_M)

        # Hysteresis: an incumbent still near the ball keeps it unless a
        # challenger is clearly closer, so two players running together stop
        # trading it frame by frame.
        takes_over = (
            rid != -1 and pd.notna(rdist) and rdist <= POSSESSION_RADIUS_M
            and (rid == current_id
                 or current_id == -1
                 or not incumbent_holding
                 or rdist <= incumbent_dist - POSSESSION_MARGIN_M)
        )

        if takes_over and not in_flight:
            current_id = rid
            current_team = rteam
            gap_time = 0.0
        elif not in_flight and incumbent_holding:
            gap_time = 0.0
        elif in_flight and release_on_flight:
            current_id = -1
            current_team = "unknown"
            gap_time = 0.0
        else:
            gap_time += dt
            if current_id != -1 and gap_time <= POSSESSION_GAP_FILL_S:
                pass
            else:
                current_id = -1
                current_team = "unknown"

        poss_ids.append(current_id)
        poss_teams.append(current_team)
        poss_dist.append(rdist)
        prev_t = t

    out = full[["frame", "time_s"]].copy()
    out["possessor_id"] = poss_ids
    out["possessor_team"] = poss_teams
    out["dist"] = poss_dist
    if "speed_kmh" in full.columns:
        out["speed_kmh"] = full["speed_kmh"].to_numpy()
    return _stabilise_possession(out)


def _stabilise_possession(poss: pd.DataFrame) -> pd.DataFrame:
    """Erase possession spells too short to be control.

    Nearest-player assignment produces one- and two-frame spells whenever two
    players converge on the ball, and each one reads downstream as a change of
    possession -- which is to say, as a pass. Requiring a spell to last
    MIN_POSSESSION_HOLD_S before it counts removes the flicker without
    touching genuine quick touches, which last several frames at 25 fps.

    A spell that is dropped becomes unassigned rather than being handed to a
    neighbour: the honest statement is that nobody was in control, not that
    somebody else was.
    """
    if poss.empty:
        return poss

    out = poss.copy()
    ids = out.possessor_id.to_numpy()
    times = out.time_s.to_numpy(dtype=float)

    start = 0
    for i in range(1, len(ids) + 1):
        if i < len(ids) and ids[i] == ids[start]:
            continue
        if ids[start] != -1:
            duration = times[i - 1] - times[start]
            if duration < MIN_POSSESSION_HOLD_S:
                out.iloc[start:i, out.columns.get_loc("possessor_id")] = -1
                out.iloc[start:i, out.columns.get_loc("possessor_team")] = "unknown"
        start = i

    return out


def _settled_possessor(poss: pd.DataFrame, after_time: float,
                       window_s: float = RECEIVER_SETTLE_WINDOW_S) -> tuple[int, str]:
    """Who holds the ball once it settles after a pass.

    Crediting the receiver to whoever is nearest at the instant the ball
    leaves credits whoever the ball happened to pass close to -- often an
    opponent it went by at speed. The receiver is whoever is in possession
    once the ball is under control again.
    """
    window = poss[(poss.time_s > after_time)
                  & (poss.time_s <= after_time + window_s)]
    if window.empty:
        return -1, "unknown"

    if "speed_kmh" in window.columns:
        settled = window[window.speed_kmh.fillna(0.0) <= BALL_CONTROL_SPEED_KMH]
        if not settled.empty:
            window = settled

    held = window[window.possessor_id != -1]
    if held.empty:
        return -1, "unknown"

    row = held.iloc[0]
    return int(row.possessor_id), str(row.possessor_team)


def _merge_adjacent_events(events: list[dict[str, Any]],
                           window_s: float = MERGE_WINDOW_S) -> list[dict[str, Any]]:
    """Collapse detections of the same type that describe one action.

    Only events of the same type are candidates: a pass and a carry at the
    same instant are two different claims about the same moment, and one of
    them being wrong is not fixed by deleting the other.

    Clusters chain -- A merges with B, B with C, so A, B and C become one --
    which is the right shape for a single action smeared across several
    detections, and the wrong shape for a genuine rapid exchange. The window
    is what separates those two cases, and it is swept rather than chosen.

    Within a cluster the survivor is the most substantial member: the pass
    that travelled furthest, the carry that lasted longest. Keeping the first
    would keep whichever fragment happened to fire earliest, which on a
    smeared detection is usually the least complete one.
    """
    if len(events) < 2:
        return events

    def substance(e: dict) -> float:
        if e.get("event_type") == "pass":
            return float(e.get("pass_distance_m") or 0.0)
        if e.get("event_type") == "carry":
            return float(e.get("carry_duration_s")
                         or e.get("carry_distance_m") or 0.0)
        return 0.0

    by_type: dict[str, list[dict]] = {}
    for e in events:
        by_type.setdefault(e.get("event_type", "?"), []).append(e)

    kept: list[dict] = []
    for event_type, group in by_type.items():
        # An extended action's fragments sit further apart than a brief one's,
        # so the window is per type.
        w = MERGE_WINDOW_BY_TYPE.get(event_type, window_s)
        if w <= 0:
            kept.extend(group)
            continue
        group = sorted(group, key=lambda e: e["timestamp_s"])
        cluster = [group[0]]
        for e in group[1:]:
            if e["timestamp_s"] - cluster[-1]["timestamp_s"] <= w:
                cluster.append(e)
            else:
                kept.append(max(cluster, key=substance))
                cluster = [e]
        kept.append(max(cluster, key=substance))

    return sorted(kept, key=lambda e: e["timestamp_s"])


def _in_goal(bx: float, by: float) -> str | None:
    """Return 'left' or 'right' if (bx, by) is inside a goal rect, else None."""
    if bx <= LEFT_GOAL_X_MAX and LEFT_GOAL_Y_MIN <= by <= LEFT_GOAL_Y_MAX:
        return "left"
    if bx >= RIGHT_GOAL_X_MIN and RIGHT_GOAL_Y_MIN <= by <= RIGHT_GOAL_Y_MAX:
        return "right"
    return None


def _out_of_pitch(bx: float, by: float) -> bool:
    return not (PITCH_X_MIN <= bx <= PITCH_X_MAX and PITCH_Y_MIN <= by <= PITCH_Y_MAX)


def _dist_to_nearest_goal(bx: float, by: float) -> float:
    left_dist = np.sqrt(bx**2 + (by - 34.0) ** 2)
    right_dist = np.sqrt((bx - 105.0) ** 2 + (by - 34.0) ** 2)
    return float(min(left_dist, right_dist))


def _toward_goal(vx: float, vy: float, bx: float, by: float) -> bool:
    """Return True when velocity vector points materially toward nearest goal."""
    gx, gy = (0.0, 34.0) if bx < 52.5 else (105.0, 34.0)
    gvx, gvy = gx - bx, gy - by
    v_norm = np.sqrt(vx * vx + vy * vy)
    g_norm = np.sqrt(gvx * gvx + gvy * gvy)
    if v_norm < 1e-6 or g_norm < 1e-6:
        return False
    cosang = (vx * gvx + vy * gvy) / (v_norm * g_norm)
    return bool(cosang >= 0.35)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def _player_positions_by_frame(tracks: pd.DataFrame) -> dict[int, dict[int, tuple[float, float]]]:
    players = tracks[(tracks.cls == "player") & (tracks.team.isin(["team_A", "team_B"]))][
        ["frame", "track_id", "px", "py"]
    ]
    out: dict[int, dict[int, tuple[float, float]]] = {}
    for frame, g in players.groupby("frame"):
        out[int(frame)] = {int(r.track_id): (float(r.px), float(r.py)) for _, r in g.iterrows()}
    return out


def _emit_possession_transition_event(
    events: list[dict[str, Any]],
    *,
    start_time: float,
    end_time: float,
    from_id: int,
    from_team: str,
    to_id: int,
    to_team: str,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    inferred: bool,
    player_xy: dict[int, dict[int, tuple[float, float]]],
    frame: int,
    ball_speed_kmh: float,
) -> None:
    travel = _distance(start_xy, end_xy)
    dt_change = max(0.0, end_time - start_time)

    # A player cannot pass to themselves. This fires when a track is lost and
    # reissued to the same player, or when re-identification rejoins two
    # fragments across the moment possession was recomputed -- neither is a
    # pass, and both were being emitted as one.
    if from_id == to_id:
        return

    if (travel < MIN_PASS_DISTANCE_M
            or dt_change > MAX_PASS_INTERVAL_S
            or dt_change < MIN_PASS_INTERVAL_S):
        return

    if to_team == from_team:
        events.append(
            dict(
                event_type="pass",
                timestamp_s=start_time,
                team=from_team,
                player_track_id=from_id,
                location_x=start_xy[0],
                location_y=start_xy[1],
                end_location_x=end_xy[0],
                end_location_y=end_xy[1],
                outcome="success",
                receiver_track_id=int(to_id),
                pass_distance_m=round(travel, 1),
                pass_interval_s=round(dt_change, 2),
                inferred_from_unknown_bridge=bool(inferred),
            )
        )
        return

    events.append(
        dict(
            event_type="pass",
            timestamp_s=start_time,
            team=from_team,
            player_track_id=from_id,
            location_x=start_xy[0],
            location_y=start_xy[1],
            end_location_x=end_xy[0],
            end_location_y=end_xy[1],
            outcome="intercepted",
            receiver_track_id=int(to_id),
            intercepted_by_track_id=int(to_id),
            intercepting_team=to_team,
            pass_distance_m=round(travel, 1),
            pass_interval_s=round(dt_change, 2),
            inferred_from_unknown_bridge=bool(inferred),
        )
    )
    # One turnover is one event. Emitting the failed pass and an interception
    # separately counts the same moment twice: every failed pass inflates the
    # event total, and any per-minute rate computed from it. The interception
    # is recorded as the pass's outcome and the player who made it, which
    # carries the same information without the double count.
    if EMIT_INTERCEPTION_AS_SEPARATE_EVENT:
        events.append(
            dict(
                event_type="interception",
                timestamp_s=end_time,
                team=to_team,
                player_track_id=to_id,
                location_x=end_xy[0],
                location_y=end_xy[1],
                end_location_x=end_xy[0],
                end_location_y=end_xy[1],
                outcome="won",
                from_player_track_id=from_id,
                from_team=from_team,
            )
        )

    pmap = player_xy.get(frame, {})
    p_old = pmap.get(from_id)
    p_new = pmap.get(to_id)
    if p_old is not None and p_new is not None:
        duel_dist = _distance(p_old, p_new)
        if duel_dist <= TACKLE_MAX_PLAYER_DIST_M and ball_speed_kmh <= TACKLE_MAX_BALL_SPEED_KMH:
            events.append(
                dict(
                    event_type="tackle",
                    timestamp_s=end_time,
                    team=to_team,
                    player_track_id=to_id,
                    location_x=end_xy[0],
                    location_y=end_xy[1],
                    outcome="won",
                    target_player_track_id=from_id,
                    duel_distance_m=round(duel_dist, 2),
                )
            )


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_events(tracks: pd.DataFrame, H: np.ndarray | None = None,
                  absolute_pitch: bool = True) -> list[dict[str, Any]]:
    """Detect football events from tracking data in metres.

    Parameters
    ----------
    tracks : pd.DataFrame
        Must contain columns: frame, time_s, cls ('player'/'ball'), track_id,
        team, x, y -- in metres. Use ``pixel_scale.prepare_tracks_for_events``
        to produce them, with or without a homography.

    H : np.ndarray, optional
        Homography matrix, kept for API compatibility. Projection happens
        upstream.

    absolute_pitch : bool
        Whether those metres sit on a known pitch. ``True`` means a position
        can be compared against the goal line, so goals, shots and
        out-of-play are decidable. ``False`` means the coordinates are metric
        but float freely -- distances, speeds and possession are still right,
        but nothing about *where* the ball is can be trusted.

        This is the flag that keeps the pipeline honest without a working
        homography. The alternative is to run the goal test against a pitch
        whose origin is wherever the scale estimate happened to put it, which
        produces goals and throw-ins at arbitrary moments and no way to tell
        them from real ones. Emitting nothing is a worse product and a
        truthful one.

    Returns
    -------
    list[dict]
        Each dict is ready for ``MatchDatabase.insert_events()``.
    """
    events: list[dict[str, Any]] = []

    # ---- 1. Ball kinematics ------------------------------------------------
    ball = _ball_kinematics(tracks)
    if ball.empty:
        print("[Events] No ball detections — cannot detect events.")
        return events

    # ---- 2. Possession timelines --------------------------------------------
    # Two of them, because passes and carries need opposite answers to the
    # same question: does a player still possess the ball while it is in
    # flight? For a carry, yes -- the spell must stay whole or an 0.8 s
    # dribble is chopped in two by a noisy speed estimate. For a pass, no --
    # the spell has to end where the ball was struck, or the pass is measured
    # from the receiver's feet to the receiver's feet.
    #
    # Trying to serve both from one timeline was the fault behind every failed
    # attempt at pass precision: each fix that corrected the pass geometry
    # destroyed carry detection, and each that preserved carries left passes
    # measuring 0.6 m.
    poss_hold = _possession_per_frame(ball, tracks, release_on_flight=False)
    poss_flight = _possession_per_frame(ball, tracks, release_on_flight=True)

    timeline = ball.merge(
        poss_hold[["frame", "possessor_id", "possessor_team", "dist"]],
        on="frame",
        how="left",
    ).merge(
        poss_flight[["frame", "possessor_id", "possessor_team"]].rename(
            columns={"possessor_id": "flight_possessor_id",
                     "possessor_team": "flight_possessor_team"}),
        on="frame",
        how="left",
    ).sort_values("frame").reset_index(drop=True)

    # ---- 2.5. Inferred ball movement events (for limited FOV) ----------------
    _detect_ball_movement_events(ball, events)

    # ---- 3. State machine over timeline -------------------------------------
    prev_possessor = -1
    prev_team = "unknown"
    prev_flight = -1
    prev_flight_team = "unknown"
    prev_bx: float | None = None
    prev_by: float | None = None
    prev_time = 0.0

    seg_start_t: float | None = None
    seg_start_xy: tuple[float, float] | None = None

    loose_start_t: float | None = None
    pending_from_id: int = -1
    pending_from_team: str = "unknown"
    pending_start_time: float | None = None
    pending_start_xy: tuple[float, float] | None = None

    last_goal_time: float = -10.0
    last_shot_time: float = -10.0
    GOAL_COOLDOWN_S = 5.0
    was_out: bool = False
    player_xy = _player_positions_by_frame(tracks)

    # Where and when the ball was last under control during the current
    # possession spell -- i.e. where it was struck from, once it leaves.
    release_time: float | None = None
    release_xy: tuple[float, float] | None = None
    ball_struck: bool = False
    last_release_time: float | None = None
    last_release_xy: tuple[float, float] | None = None

    for _, brow in timeline.iterrows():
        frame = int(brow["frame"])
        t = float(brow["time_s"])
        bx = float(brow["bx"])
        by = float(brow["by"])
        speed = float(brow["speed_kmh"])
        vx = float(brow["vel_x"])
        vy = float(brow["vel_y"])
        cur_possessor = int(brow["possessor_id"]) if pd.notna(brow["possessor_id"]) else -1
        cur_team = str(brow["possessor_team"]) if pd.notna(brow["possessor_team"]) else "unknown"

        # The flight-aware view, used only where a pass is decided. Everything
        # else -- carries, recoveries, goal and shot attribution -- reads the
        # held view above, where a spell survives the ball leaving the ground.
        cur_flight = (int(brow["flight_possessor_id"])
                      if pd.notna(brow["flight_possessor_id"]) else -1)
        cur_flight_team = (str(brow["flight_possessor_team"])
                           if pd.notna(brow["flight_possessor_team"])
                           else "unknown")

        # Where the ball sat before it was struck. The mark follows the ball
        # while it is slow, and freezes the moment it is not: everything after
        # that is flight, and the point of interest is where the flight began.
        #
        # Freezing is what makes this work. Marking every slow frame instead
        # tracks the ball all the way down its trajectory, because possession
        # is carried through the flight and a ball decelerating into its
        # receiver is slow while still attributed to the passer. Requiring
        # proximity does not help either -- `dist` is the distance to the
        # nearest player, and on a crowded pitch someone is near the path.
        if cur_flight != prev_flight:
            # Hand the outgoing spell's mark to the transition handling below,
            # which runs later in this same iteration -- resetting in place
            # would wipe it before the pass that needs it is emitted.
            last_release_time, last_release_xy = release_time, release_xy
            release_time, release_xy, ball_struck = None, None, False
        if speed > BALL_CONTROL_SPEED_KMH:
            ball_struck = True
        elif cur_flight != -1 and not ball_struck:
            release_time, release_xy = t, (bx, by)

        # ---- Goal detection ------------------------------------------------
        goal_side = _in_goal(bx, by) if absolute_pitch else None
        if goal_side and (t - last_goal_time) > GOAL_COOLDOWN_S:
            last_goal_time = t
            # Scoring team: the team that last possessed before the goal
            scoring_team = prev_team if prev_team != "unknown" else cur_team
            events.append(
                dict(
                    event_type="goal",
                    timestamp_s=t,
                    team=scoring_team,
                    player_track_id=prev_possessor if prev_possessor != -1 else cur_possessor,
                    location_x=bx,
                    location_y=by,
                    end_location_x=bx,
                    end_location_y=by,
                    outcome="goal",
                    goal_side=goal_side,
                )
            )

        # ---- Out-of-play detection -----------------------------------------
        if absolute_pitch and _out_of_pitch(bx, by):
            if not was_out:
                events.append(
                    dict(
                        event_type="out_of_play",
                        timestamp_s=t,
                        team=prev_team,
                        player_track_id=prev_possessor,
                        location_x=bx,
                        location_y=by,
                        outcome="out",
                    )
                )
            was_out = True
            prev_possessor, prev_team = cur_possessor, cur_team
            prev_flight, prev_flight_team = cur_flight, cur_flight_team
            prev_bx, prev_by = bx, by
            prev_time = t
            continue
        else:
            was_out = False

        # ---- Shot detection ------------------------------------------------
        if (
            absolute_pitch
            and speed >= SHOT_MIN_KMH
            and prev_team != "unknown"
            and (t - last_shot_time) >= SHOT_COOLDOWN_S
            and _toward_goal(vx, vy, bx, by)
        ):
            dist_goal = _dist_to_nearest_goal(bx, by)
            if dist_goal <= SHOT_MAX_DIST_FROM_GOAL_M:
                last_shot_time = t
                # Determine shot outcome
                if goal_side:
                    shot_outcome = "goal"
                elif (bx <= 2.0 or bx >= 103.0) and abs(by - 34.0) <= (GOAL_HALF + 2.0):
                    shot_outcome = "on_target"
                else:
                    shot_outcome = "off_target"

                events.append(
                    dict(
                        event_type="shot",
                        timestamp_s=t,
                        team=prev_team,
                        player_track_id=prev_possessor if prev_possessor != -1 else cur_possessor,
                        location_x=prev_bx if prev_bx is not None else bx,
                        location_y=prev_by if prev_by is not None else by,
                        end_location_x=bx,
                        end_location_y=by,
                        outcome=shot_outcome,
                        ball_speed_kmh=round(speed, 1),
                    )
                )

        # ---- Recovery after loose-ball phase ------------------------------
        if prev_possessor != -1 and cur_possessor == -1 and loose_start_t is None:
            loose_start_t = t
        if prev_possessor == -1 and cur_possessor != -1 and loose_start_t is not None:
            loose_dur = t - loose_start_t
            if loose_dur >= RECOVERY_MIN_LOOSE_S:
                events.append(
                    dict(
                        event_type="recovery",
                        timestamp_s=t,
                        team=cur_team,
                        player_track_id=cur_possessor,
                        location_x=bx,
                        location_y=by,
                        outcome="won",
                        loose_duration_s=round(loose_dur, 2),
                    )
                )
            loose_start_t = None

        # ---- Pass / Interception / Tackle proxy ---------------------------
        if (
            cur_flight != prev_flight
            and cur_flight != -1
            and prev_flight != -1
            and prev_bx is not None
            and prev_by is not None
        ):
            # A pass starts where the ball was struck, not at the frame before
            # possession changed hands. Possession is carried through the
            # ball's flight -- deliberately, because releasing it fragments
            # the spells that carry detection depends on -- so it only changes
            # once the ball has arrived, and the previous frame's position is
            # at the receiver's feet. Measured that way the median "pass"
            # travelled 0.6 m in 0.12 s.
            #
            # The last moment the ball was under control during the passer's
            # spell is where it left them. Using that as the start gives a
            # median travel of 1.3 and 4.6 m on the two test windows, without
            # touching the possession timeline.
            start_time = (last_release_time if last_release_time is not None
                          else prev_time)
            start_xy = (last_release_xy if last_release_xy is not None
                        else (prev_bx, prev_by))
            _emit_possession_transition_event(
                events,
                start_time=start_time,
                end_time=t,
                from_id=prev_flight,
                from_team=prev_flight_team,
                to_id=cur_flight,
                to_team=cur_flight_team,
                start_xy=start_xy,
                end_xy=(bx, by),
                inferred=False,
                player_xy=player_xy,
                frame=frame,
                ball_speed_kmh=speed,
            )

        if prev_flight != -1 and cur_flight == -1 and pending_start_time is None and prev_bx is not None and prev_by is not None:
            pending_from_id = prev_flight
            pending_from_team = prev_flight_team
            pending_start_time = prev_time
            pending_start_xy = (prev_bx, prev_by)

        if pending_start_time is not None:
            if cur_flight != -1 and cur_flight_team != "unknown" and pending_from_id != -1:
                bridge_dt = max(0.0, t - pending_start_time)
                if bridge_dt <= PASS_UNKNOWN_BRIDGE_S and pending_start_xy is not None:
                    _emit_possession_transition_event(
                        events,
                        start_time=pending_start_time,
                        end_time=t,
                        from_id=pending_from_id,
                        from_team=pending_from_team,
                        to_id=cur_flight,
                        to_team=cur_flight_team,
                        start_xy=pending_start_xy,
                        end_xy=(bx, by),
                        inferred=True,
                        player_xy=player_xy,
                        frame=frame,
                        ball_speed_kmh=speed,
                    )
                pending_from_id = -1
                pending_from_team = "unknown"
                pending_start_time = None
                pending_start_xy = None
            elif (t - pending_start_time) > PASS_UNKNOWN_BRIDGE_S:
                pending_from_id = -1
                pending_from_team = "unknown"
                pending_start_time = None
                pending_start_xy = None

        # ---- Carry (conduite) --------------------------------------------
        if cur_possessor != prev_possessor:
            if prev_possessor != -1 and seg_start_t is not None and seg_start_xy is not None and prev_bx is not None and prev_by is not None:
                duration = max(0.0, prev_time - seg_start_t)
                progression = _distance(seg_start_xy, (prev_bx, prev_by))
                if duration >= CARRY_MIN_TIME_S and progression >= CARRY_MIN_DISTANCE_M:
                    events.append(
                        dict(
                            event_type="carry",
                            timestamp_s=seg_start_t,
                            team=prev_team,
                            player_track_id=prev_possessor,
                            location_x=seg_start_xy[0],
                            location_y=seg_start_xy[1],
                            end_location_x=prev_bx,
                            end_location_y=prev_by,
                            outcome="complete",
                            carry_distance_m=round(progression, 1),
                            carry_duration_s=round(duration, 2),
                        )
                    )

            if cur_possessor != -1:
                seg_start_t = t
                seg_start_xy = (bx, by)
            else:
                seg_start_t = None
                seg_start_xy = None

        if cur_possessor != -1 and seg_start_t is None:
            seg_start_t = t
            seg_start_xy = (bx, by)

        prev_possessor = cur_possessor
        prev_team = cur_team
        prev_flight = cur_flight
        prev_flight_team = cur_flight_team
        prev_bx = bx
        prev_by = by
        prev_time = t

    # Finalize last carry segment
    if (
        prev_possessor != -1
        and seg_start_t is not None
        and seg_start_xy is not None
        and prev_bx is not None
        and prev_by is not None
    ):
        duration = max(0.0, prev_time - seg_start_t)
        progression = _distance(seg_start_xy, (prev_bx, prev_by))
        if duration >= CARRY_MIN_TIME_S and progression >= CARRY_MIN_DISTANCE_M:
            events.append(
                dict(
                    event_type="carry",
                    timestamp_s=seg_start_t,
                    team=prev_team,
                    player_track_id=prev_possessor,
                    location_x=seg_start_xy[0],
                    location_y=seg_start_xy[1],
                    end_location_x=prev_bx,
                    end_location_y=prev_by,
                    outcome="complete",
                    carry_distance_m=round(progression, 1),
                    carry_duration_s=round(duration, 2),
                )
            )

    clean = _merge_adjacent_events(events, MERGE_WINDOW_S)
    clean.sort(key=lambda e: e["timestamp_s"])

    _summarize(clean)
    return clean


def _summarize(events: list[dict]) -> None:
    from collections import Counter
    counts = Counter(ev["event_type"] for ev in events)
    print("[Events] Detected:", dict(counts))
    goals = [e for e in events if e["event_type"] == "goal"]
    for g in goals:
        mm = int(g["timestamp_s"] // 60)
        ss = int(g["timestamp_s"] % 60)
        print(f"  GOAL  {mm:02d}:{ss:02d}  team={g['team']}")


# ---------------------------------------------------------------------------
# Convenience aggregators used by main.py / player_rating.py
# ---------------------------------------------------------------------------

def events_per_player(events: list[dict]) -> dict[int, dict[str, int]]:
    """Return {track_id: {n_passes, n_shots, n_goals}} for each player."""
    result: dict[int, dict[str, int]] = {}
    for ev in events:
        pid = ev.get("player_track_id", -1)
        if pid is None or pid < 0:
            continue
        pid = int(pid)
        s = result.setdefault(pid, {"n_passes": 0, "n_shots": 0, "n_goals": 0})
        if ev["event_type"] == "pass":
            if ev.get("outcome") == "success":
                s["n_passes"] += 1
        elif ev["event_type"] == "shot":
            s["n_shots"] += 1
        elif ev["event_type"] == "goal":
            s["n_goals"] += 1
    return result


def team_event_totals(events: list[dict]) -> dict[str, dict[str, int]]:
    """Return {team: {n_passes, n_shots, n_goals}} aggregated across all players."""
    result: dict[str, dict[str, int]] = {}
    for ev in events:
        team = ev.get("team", "unknown") or "unknown"
        s = result.setdefault(team, {"n_passes": 0, "n_passes_failed": 0, "n_shots": 0, "n_goals": 0})
        if ev["event_type"] == "pass":
            if ev.get("outcome") == "success":
                s["n_passes"] += 1
            else:
                s["n_passes_failed"] += 1
        elif ev["event_type"] == "shot":
            s["n_shots"] += 1
        elif ev["event_type"] == "goal":
            s["n_goals"] += 1
    return result
