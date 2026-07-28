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

POSSESSION_RADIUS_M = 4.4
POSSESSION_GAP_FILL_S = 1.4
MIN_PASS_DISTANCE_M = 2.2
MAX_PASS_INTERVAL_S = 4.5
PASS_UNKNOWN_BRIDGE_S = 1.4
MIN_POSSESSION_STABLE_S = 0.85
SHOT_MIN_KMH = 16.0
SHOT_MAX_DIST_FROM_GOAL_M = 40.0
SHOT_COOLDOWN_S = 1.8
CARRY_MIN_TIME_S = 2.0
CARRY_MIN_DISTANCE_M = 8.0
RECOVERY_MIN_LOOSE_S = 0.8
TACKLE_MAX_PLAYER_DIST_M = 2.5
TACKLE_MAX_BALL_SPEED_KMH = 28.0

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth_series(s: pd.Series, window: int = VEL_SMOOTH_WINDOW) -> pd.Series:
    return s.rolling(window, min_periods=1, center=True).median()


def _ball_kinematics(tracks: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per frame that has a ball detection.

    Columns: frame, time_s, bx, by, vel_x, vel_y, speed_kmh
    """
    if tracks.empty or not (tracks.cls == "ball").any():
        return pd.DataFrame(columns=["frame", "time_s", "bx", "by", "vel_x", "vel_y", "speed_kmh"])

    # Use improved ball tracking with Kalman filtering
    ball = ball_tracking.extract_ball_tracking(tracks, smooth_window=3, fill_gaps=True)
    if ball.empty:
        return pd.DataFrame(columns=["frame", "time_s", "bx", "by", "vel_x", "vel_y", "speed_kmh"])

    # Rename columns and compute speed from velocity
    ball["bx"] = ball["x"]
    ball["by"] = ball["y"]
    ball["vel_x"] = ball.get("vx", 0.0)
    ball["vel_y"] = ball.get("vy", 0.0)
    ball["speed_kmh"] = np.sqrt(ball["vel_x"]**2 + ball["vel_y"]**2) * 3.6

    return ball[["frame", "time_s", "bx", "by", "vel_x", "vel_y", "speed_kmh"]]


def _possession_per_frame(
    ball: pd.DataFrame, tracks: pd.DataFrame
) -> pd.DataFrame:
    """Return frame-wise possession with short-gap filling.

    Nearest-player assignment is used first. When the ball is briefly unassigned,
    the previous possessor is carried for a short period to reduce flicker.
    """
    players = tracks[
        (tracks.cls == "player")
        & (tracks.team.isin(["team_A", "team_B"]))
    ][["frame", "track_id", "team", "x", "y"]]

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

    merged = ball[["frame", "time_s", "bx", "by"]].merge(players, on="frame", how="left")
    merged["dist"] = np.sqrt(
        (merged["x"] - merged["bx"]) ** 2 + (merged["y"] - merged["by"]) ** 2
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

    full = ball[["frame", "time_s"]].merge(nearest, on=["frame", "time_s"], how="left")

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

        if rid != -1 and pd.notna(rdist) and rdist <= POSSESSION_RADIUS_M:
            current_id = rid
            current_team = rteam
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
    return out


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
        ["frame", "track_id", "x", "y"]
    ]
    out: dict[int, dict[int, tuple[float, float]]] = {}
    for frame, g in players.groupby("frame"):
        out[int(frame)] = {int(r.track_id): (float(r.x), float(r.y)) for _, r in g.iterrows()}
    return out


def _reset_transition_candidate() -> dict[str, Any]:
    return {
        "from_id": -1,
        "from_team": "unknown",
        "to_id": -1,
        "to_team": "unknown",
        "start_time": None,
        "end_time": None,
        "start_xy": None,
        "end_xy": None,
        "frame": -1,
        "ball_speed_kmh": 0.0,
        "inferred": False,
        "stabilize_start": None,
    }


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
    if travel < MIN_PASS_DISTANCE_M or dt_change > MAX_PASS_INTERVAL_S:
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
            outcome="failed",
            receiver_track_id=int(to_id),
            pass_distance_m=round(travel, 1),
            pass_interval_s=round(dt_change, 2),
            inferred_from_unknown_bridge=bool(inferred),
        )
    )
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

def detect_events(tracks: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect football events from pitch-coordinate tracking data.

    Parameters
    ----------
    tracks : pd.DataFrame
        Must contain columns: frame, time_s, cls ('player'/'ball'), track_id,
        team, x, y  (all in pitch metres after homography projection).

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

    # ---- 2. Possession timeline ---------------------------------------------
    poss = _possession_per_frame(ball, tracks)
    timeline = ball.merge(
        poss[["frame", "possessor_id", "possessor_team", "dist"]],
        on="frame",
        how="left",
    ).sort_values("frame").reset_index(drop=True)

    # ---- 3. State machine over timeline -------------------------------------
    prev_possessor = -1
    prev_team = "unknown"
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
    transition_candidate = _reset_transition_candidate()

    last_goal_time: float = -10.0
    last_shot_time: float = -10.0
    GOAL_COOLDOWN_S = 5.0
    was_out: bool = False
    player_xy = _player_positions_by_frame(tracks)

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

        # ---- Goal detection ------------------------------------------------
        goal_side = _in_goal(bx, by)
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
        if _out_of_pitch(bx, by):
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
            prev_bx, prev_by = bx, by
            prev_time = t
            continue
        else:
            was_out = False

        # ---- Shot detection ------------------------------------------------
        if (
            speed >= SHOT_MIN_KMH
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
        if transition_candidate["to_id"] != -1:
            if cur_possessor == transition_candidate["to_id"] and cur_team == transition_candidate["to_team"]:
                stable_for = t - float(transition_candidate["stabilize_start"])
                transition_candidate["end_xy"] = (bx, by)
                transition_candidate["end_time"] = t
                transition_candidate["frame"] = frame
                transition_candidate["ball_speed_kmh"] = speed
                if stable_for >= MIN_POSSESSION_STABLE_S:
                    _emit_possession_transition_event(
                        events,
                        start_time=float(transition_candidate["start_time"]),
                        end_time=float(transition_candidate["end_time"]),
                        from_id=int(transition_candidate["from_id"]),
                        from_team=str(transition_candidate["from_team"]),
                        to_id=int(transition_candidate["to_id"]),
                        to_team=str(transition_candidate["to_team"]),
                        start_xy=transition_candidate["start_xy"],
                        end_xy=transition_candidate["end_xy"],
                        inferred=bool(transition_candidate["inferred"]),
                        player_xy=player_xy,
                        frame=int(transition_candidate["frame"]),
                        ball_speed_kmh=float(transition_candidate["ball_speed_kmh"]),
                    )
                    transition_candidate = _reset_transition_candidate()
            else:
                transition_candidate = _reset_transition_candidate()

        if (
            cur_possessor != prev_possessor
            and cur_possessor != -1
            and prev_possessor != -1
            and prev_bx is not None
            and prev_by is not None
        ):
            transition_candidate = {
                "from_id": prev_possessor,
                "from_team": prev_team,
                "to_id": cur_possessor,
                "to_team": cur_team,
                "start_time": prev_time,
                "end_time": t,
                "start_xy": (prev_bx, prev_by),
                "end_xy": (bx, by),
                "frame": frame,
                "ball_speed_kmh": speed,
                "inferred": False,
                "stabilize_start": t,
            }

        if prev_possessor != -1 and cur_possessor == -1 and pending_start_time is None and prev_bx is not None and prev_by is not None:
            pending_from_id = prev_possessor
            pending_from_team = prev_team
            pending_start_time = prev_time
            pending_start_xy = (prev_bx, prev_by)

        if pending_start_time is not None:
            if cur_possessor != -1 and cur_team != "unknown" and pending_from_id != -1:
                bridge_dt = max(0.0, t - pending_start_time)
                if bridge_dt <= PASS_UNKNOWN_BRIDGE_S and pending_start_xy is not None:
                    transition_candidate = {
                        "from_id": pending_from_id,
                        "from_team": pending_from_team,
                        "to_id": cur_possessor,
                        "to_team": cur_team,
                        "start_time": pending_start_time,
                        "end_time": t,
                        "start_xy": pending_start_xy,
                        "end_xy": (bx, by),
                        "frame": frame,
                        "ball_speed_kmh": speed,
                        "inferred": True,
                        "stabilize_start": t,
                    }
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

    clean = events
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
