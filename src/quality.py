"""Quality control metrics for tracking and event reliability."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_quantile(s: pd.Series, q: float) -> float:
    if s.empty:
        return 0.0
    return float(s.quantile(q))


def compute_quality(
    tracks_pitch: pd.DataFrame,
    phys: pd.DataFrame,
    events: list[dict[str, Any]],
    ball_rate: float,
    duration_s: float,
) -> dict[str, Any]:
    """Return a compact quality report with thresholds and warnings."""
    duration_min = max(duration_s / 60.0, 1e-6)

    players = tracks_pitch[
        (tracks_pitch.cls == "player") & tracks_pitch.team.isin(["team_A", "team_B"])
    ].copy()

    if players.empty:
        return {
            "ball_detection_rate": float(ball_rate),
            "players_per_frame": {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0},
            "persistent_tracks": 0,
            "long_tracks": 0,
            "long_track_ratio": 0.0,
            "events_total": int(len(events)),
            "passes_per_min": 0.0,
            "shots_per_min": 0.0,
            "warnings": [
                "No player tracks in pitch coordinates.",
                "Ball detection too low for robust event inference.",
            ],
        }

    per_frame = players.groupby("frame")["track_id"].nunique()

    persistent_tracks = int(phys["track_id"].nunique()) if not phys.empty else 0
    long_tracks = int((phys["minutes_tracked"] >= 2.5).sum()) if not phys.empty else 0
    long_track_ratio = float(long_tracks / max(persistent_tracks, 1))

    pass_count = sum(1 for e in events if e.get("event_type") == "pass" and e.get("outcome") == "success")
    shot_count = sum(1 for e in events if e.get("event_type") == "shot")

    passes_per_min = float(pass_count / duration_min)
    shots_per_min = float(shot_count / duration_min)

    warnings: list[str] = []
    if ball_rate < 0.5:
        warnings.append("Ball detection below 50%: event detection reliability is limited.")
    if float(per_frame.mean()) < 8.0:
        warnings.append("Low players-per-frame coverage (<8): tracking misses likely.")
    if long_track_ratio < 0.6:
        warnings.append("Low long-track ratio (<60%): persistent IDs remain fragmented.")
    if passes_per_min < 4.0:
        warnings.append("Passes per minute unexpectedly low: possession transfer under-detection likely.")

    return {
        "ball_detection_rate": float(ball_rate),
        "players_per_frame": {
            "mean": round(float(per_frame.mean()), 2),
            "p50": round(_safe_quantile(per_frame, 0.5), 2),
            "p90": round(_safe_quantile(per_frame, 0.9), 2),
            "max": int(per_frame.max()) if not per_frame.empty else 0,
        },
        "persistent_tracks": persistent_tracks,
        "long_tracks": long_tracks,
        "long_track_ratio": round(long_track_ratio, 3),
        "events_total": int(len(events)),
        "passes_per_min": round(passes_per_min, 2),
        "shots_per_min": round(shots_per_min, 2),
        "warnings": warnings,
    }
