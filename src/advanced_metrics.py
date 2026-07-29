"""Advanced per-player metrics calculation.

Computes advanced statistics for each player including pass completion rate,
distance covered, sprints, tackles, interceptions, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List


def compute_pass_metrics(events: list[dict], players_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pass-related metrics per player."""
    pass_events = [e for e in events if e.get("event_type") == "pass"]

    metrics = {}
    for player_id in players_df["track_id"].unique():
        player_passes = [e for e in pass_events if e.get("player_track_id") == player_id]

        total_passes = len(player_passes)
        completed_passes = len([e for e in player_passes if e.get("outcome") == "success"])
        completion_rate = completed_passes / total_passes if total_passes > 0 else 0.0

        forward_passes = len([e for e in player_passes if e.get("pass_type") in ["long_ball", "through_ball"]])
        backward_passes = len([e for e in player_passes if e.get("pass_type") == "short_pass"])
        lateral_passes = len([e for e in player_passes if e.get("pass_type") == "medium_pass"])

        metrics[int(player_id)] = {
            "passes_total": total_passes,
            "passes_completed": completed_passes,
            "pass_completion_rate": completion_rate,
            "passes_forward": forward_passes,
            "passes_backward": backward_passes,
            "passes_lateral": lateral_passes,
        }

    return metrics


def compute_movement_metrics(tracks: pd.DataFrame) -> pd.DataFrame:
    """Compute movement-related metrics per player."""
    metrics = {}

    players = tracks[tracks.cls == "player"].copy()
    if players.empty:
        return {}

    # Use px, py for pitch coordinates (after transformation)
    coord_cols = ["px", "py"] if "px" in players.columns else ["x", "y"]
    if not all(col in players.columns for col in coord_cols):
        coord_cols = ["x", "y"]  # Fallback

    for player_id in players["track_id"].unique():
        player_tracks = players[players["track_id"] == player_id].sort_values("frame")

        if len(player_tracks) < 2:
            continue

        total_distance_m = 0.0
        max_speed_kmh = 0.0

        for i in range(1, len(player_tracks)):
            curr = player_tracks.iloc[i]
            prev = player_tracks.iloc[i - 1]

            dx = curr[coord_cols[0]] - prev[coord_cols[0]]
            dy = curr[coord_cols[1]] - prev[coord_cols[1]]
            distance_m = np.hypot(dx, dy)
            total_distance_m += distance_m

            dt_s = curr["time_s"] - prev["time_s"]
            if dt_s > 0:
                speed_kmh = (distance_m / dt_s) * 3.6
                max_speed_kmh = max(max_speed_kmh, speed_kmh)

        avg_speed_kmh = (total_distance_m / max(player_tracks["time_s"].max() - player_tracks["time_s"].min(), 0.1)) * 3.6

        sprints = 0
        for i in range(1, len(player_tracks)):
            curr = player_tracks.iloc[i]
            prev = player_tracks.iloc[i - 1]

            dx = curr[coord_cols[0]] - prev[coord_cols[0]]
            dy = curr[coord_cols[1]] - prev[coord_cols[1]]
            distance_m = np.hypot(dx, dy)

            dt_s = curr["time_s"] - prev["time_s"]
            if dt_s > 0 and (distance_m / dt_s) * 3.6 > 22.0:
                sprints += 1

        metrics[int(player_id)] = {
            "distance_m": round(total_distance_m, 1),
            "avg_speed_kmh": round(avg_speed_kmh, 2),
            "max_speed_kmh": round(max_speed_kmh, 2),
            "sprints": sprints,
        }

    return metrics


def compute_ball_interaction_metrics(events: list[dict], tracks: pd.DataFrame) -> Dict[int, dict]:
    """Compute ball interaction metrics per player."""
    metrics = {}

    players = tracks[(tracks.cls == "player") & (tracks.team.isin(["team_A", "team_B"]))]["track_id"].unique()

    for player_id in players:
        player_events = [e for e in events if e.get("player_track_id") == player_id]

        touches = len([e for e in player_events if e.get("event_type") in ["pass", "shot", "dribble"]])

        shots = len([e for e in player_events if e.get("event_type") == "shot"])
        shots_on_target = len([e for e in player_events if e.get("event_type") == "shot" and e.get("outcome") == "on_target"])
        shots_off_target = len([e for e in player_events if e.get("event_type") == "shot" and e.get("outcome") != "on_target"])

        goals = len([e for e in player_events if e.get("event_type") == "goal"])

        dribbles = len([e for e in player_events if e.get("event_type") == "dribble"])
        tackles = len([e for e in player_events if e.get("event_type") == "tackle"])
        interceptions = len([e for e in player_events if e.get("event_type") == "interception"])
        clearances = len([e for e in player_events if e.get("event_type") == "clearance"])

        metrics[int(player_id)] = {
            "touches": touches,
            "shots": shots,
            "shots_on_target": shots_on_target,
            "shots_off_target": shots_off_target,
            "goals": goals,
            "dribbles": dribbles,
            "tackles": tackles,
            "interceptions": interceptions,
            "clearances": clearances,
        }

    return metrics


def compute_advanced_metrics(events: list[dict], tracks: pd.DataFrame) -> pd.DataFrame:
    """Compute all advanced metrics and return as DataFrame."""

    players_df = tracks[(tracks.cls == "player") & (tracks.team.isin(["team_A", "team_B"]))][["track_id", "team"]].drop_duplicates()

    pass_metrics = compute_pass_metrics(events, players_df)
    movement_metrics = compute_movement_metrics(tracks)
    ball_metrics = compute_ball_interaction_metrics(events, tracks)

    all_metrics = {}
    for player_id in players_df["track_id"].unique():
        all_metrics[int(player_id)] = {}

        if int(player_id) in pass_metrics:
            all_metrics[int(player_id)].update(pass_metrics[int(player_id)])

        if int(player_id) in movement_metrics:
            all_metrics[int(player_id)].update(movement_metrics[int(player_id)])

        if int(player_id) in ball_metrics:
            all_metrics[int(player_id)].update(ball_metrics[int(player_id)])

    metrics_list = []
    for player_id, metrics_dict in all_metrics.items():
        row = {"player_track_id": player_id, "team": players_df[players_df["track_id"] == player_id]["team"].iloc[0]}
        row.update(metrics_dict)
        metrics_list.append(row)

    return pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()
