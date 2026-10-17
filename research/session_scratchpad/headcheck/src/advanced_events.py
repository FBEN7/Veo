"""Advanced event detection beyond basic pass/shot/goal.

Detects: dribbles, duels, pressures, clearances, crosses, through balls, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple

DRIBBLE_MIN_DISTANCE_M = 3.0
DRIBBLE_MAX_DEFENDERS_NEARBY_M = 2.5
DRIBBLE_MIN_SPEED_KMH = 8.0

PRESSURE_DISTANCE_M = 3.0
PRESSURE_MIN_DURATION_S = 0.3

CROSS_MIN_DISTANCE_M = 15.0
CROSS_MAX_ANGLE_DEG = 45.0

THROUGH_BALL_MIN_DISTANCE_M = 10.0

LONG_BALL_MIN_DISTANCE_M = 35.0
SHORT_PASS_MAX_DISTANCE_M = 10.0


def detect_dribbles(events: list[dict], tracks: pd.DataFrame) -> list[dict]:
    """Detect dribbling events (player moving with ball avoiding defenders)."""
    dribbles = []

    # Use px, py for pitch coordinates (after transformation)
    cols_to_use = ["px", "py"] if "px" in tracks.columns else ["x", "y"]
    ball = tracks[tracks.cls == "ball"][["frame", "time_s"] + cols_to_use].copy()
    if ball.empty:
        return dribbles

    ball = ball.rename(columns={cols_to_use[0]: "x", cols_to_use[1]: "y"})

    players = tracks[(tracks.cls == "player") & (tracks.team.isin(["team_A", "team_B"]))][
        ["frame", "track_id", "team"] + cols_to_use
    ]
    players = players.rename(columns={cols_to_use[0]: "x", cols_to_use[1]: "y"})

    for _, row in ball.iterrows():
        frame = int(row["frame"])
        t = float(row["time_s"])
        bx, by = float(row["x"]), float(row["y"])

        frame_players = players[players.frame == frame]
        if frame_players.empty:
            continue

        for _, p in frame_players.iterrows():
            px, py = float(p["x"]), float(p["y"])
            dist_to_ball = np.hypot(px - bx, py - by)

            if dist_to_ball > 2.0:
                continue

            opponents = frame_players[frame_players.team != p["team"]]
            nearby_defenders = opponents[
                np.hypot(opponents["x"] - px, opponents["y"] - py) <= DRIBBLE_MAX_DEFENDERS_NEARBY_M
            ]

            if len(nearby_defenders) > 0:
                dribbles.append({
                    "event_type": "dribble",
                    "timestamp_s": t,
                    "team": p["team"],
                    "player_track_id": int(p["track_id"]),
                    "location_x": bx,
                    "location_y": by,
                    "defenders_nearby": len(nearby_defenders),
                })

    return dribbles


def detect_pressures(events: list[dict], tracks: pd.DataFrame, max_duration_s: float = 2.0) -> list[dict]:
    """Detect defensive pressure (defender close to ball carrier)."""
    pressures = []

    # Use px, py for pitch coordinates (after transformation)
    cols_to_use = ["px", "py"] if "px" in tracks.columns else ["x", "y"]
    ball = tracks[tracks.cls == "ball"][["frame", "time_s"] + cols_to_use].copy()
    if ball.empty:
        return pressures

    ball = ball.rename(columns={cols_to_use[0]: "x", cols_to_use[1]: "y"})

    possession = {}
    for ev in events:
        if "possessor_id" in str(ev) and ev.get("possessor_id", -1) > 0:
            frame = int(ev.get("frame", 0))
            possession[frame] = ev.get("possessor_id")

    players = tracks[(tracks.cls == "player") & (tracks.team.isin(["team_A", "team_B"]))][
        ["frame", "track_id", "team"] + cols_to_use
    ]
    players = players.rename(columns={cols_to_use[0]: "x", cols_to_use[1]: "y"})

    for _, row in ball.iterrows():
        frame = int(row["frame"])
        t = float(row["time_s"])
        bx, by = float(row["x"]), float(row["y"])

        frame_players = players[players.frame == frame]
        if frame_players.empty:
            continue

        frame_players = frame_players.copy()
        frame_players["dist_to_ball"] = np.hypot(
            frame_players["x"] - bx, frame_players["y"] - by
        )

        near_ball = frame_players[frame_players["dist_to_ball"] <= PRESSURE_DISTANCE_M]

        if len(near_ball) > 0:
            possessing_team = near_ball.iloc[0]["team"]
            defenders = near_ball[near_ball["team"] != possessing_team]
        else:
            defenders = pd.DataFrame()

        if len(defenders) > 0:
            for _, d in defenders.iterrows():
                pressures.append({
                    "event_type": "pressure",
                    "timestamp_s": t,
                    "team": d["team"],
                    "player_track_id": int(d["track_id"]),
                    "location_x": bx,
                    "location_y": by,
                    "distance_m": float(d["dist_to_ball"]),
                })

    return pressures


def classify_pass_type(distance_m: float, angle_deg: float, receiver_location: Tuple[float, float]) -> str:
    """Classify pass into type: short, medium, long, cross, through_ball."""

    if distance_m >= LONG_BALL_MIN_DISTANCE_M:
        return "long_ball"
    elif distance_m <= SHORT_PASS_MAX_DISTANCE_M:
        return "short_pass"

    if distance_m >= CROSS_MIN_DISTANCE_M and abs(angle_deg) > 60:
        return "cross"

    if distance_m >= THROUGH_BALL_MIN_DISTANCE_M and abs(angle_deg) < 30:
        return "through_ball"

    return "medium_pass"


def add_pass_types(events: list[dict]) -> list[dict]:
    """Add pass type classification to pass events."""
    for ev in events:
        if ev.get("event_type") == "pass":
            sx, sy = ev.get("location_x", 0), ev.get("location_y", 0)
            ex, ey = ev.get("end_location_x", 0), ev.get("end_location_y", 0)

            dist = np.hypot(ex - sx, ey - sy)
            angle = np.degrees(np.arctan2(ey - sy, ex - sx))

            pass_type = classify_pass_type(dist, angle, (ex, ey))
            ev["pass_type"] = pass_type

    return events


def detect_clearances(events: list[dict], tracks: pd.DataFrame) -> list[dict]:
    """Detect defensive clearances (long kick away from goal)."""
    clearances = []

    passes = [e for e in events if e.get("event_type") == "pass"]

    for p in passes:
        if p.get("outcome") != "success":
            continue

        sx, sy = float(p.get("location_x", 0)), float(p.get("location_y", 0))
        ex, ey = float(p.get("end_location_x", 0)), float(p.get("end_location_y", 0))

        dist = np.hypot(ex - sx, ey - sy)

        if dist >= 20.0 and sy < 40.0:
            clearances.append({
                "event_type": "clearance",
                "timestamp_s": p.get("timestamp_s"),
                "team": p.get("team"),
                "player_track_id": p.get("player_track_id"),
                "location_x": sx,
                "location_y": sy,
                "end_location_x": ex,
                "end_location_y": ey,
                "clearance_distance_m": round(dist, 1),
            })

    return clearances


def enhance_events_with_advanced_detection(events: list[dict], tracks: pd.DataFrame) -> list[dict]:
    """Main function: enhance events with advanced detections."""

    enhanced = events.copy()

    enhanced.extend(detect_dribbles(events, tracks))
    enhanced.extend(detect_pressures(events, tracks))
    enhanced.extend(detect_clearances(events, tracks))
    enhanced = add_pass_types(enhanced)

    enhanced.sort(key=lambda e: e.get("timestamp_s", 0))

    return enhanced
