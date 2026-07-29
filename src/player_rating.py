"""Player performance rating for the post-match report.

Rating formula
--------------
Each metric is min-max normalised across all players who played at least
MIN_MINUTES_TRACKED minutes.  The composite score is a weighted sum:

    rating = 0.20 * dist_norm
           + 0.15 * sprints_norm
           + 0.25 * passes_norm
           + 0.20 * shots_norm
           + 0.10 * possession_norm
           + 1.50 * goals         (absolute bonus, not normalised)

Scores are clipped to [0, 10] for readability.

The function ``compute()`` returns an enriched copy of the physical-stats
DataFrame with added columns: n_passes, n_shots, n_goals, possession_pct,
rating, rank_in_team.

``best_worst()`` extracts the top-2 and bottom-1 players per team.
"""

from __future__ import annotations

import pandas as pd

MIN_MINUTES_TRACKED = 5.0   # ignore tracks shorter than this (bench, refs)

_WEIGHTS = {
    "distance_m":      0.20,
    "n_sprints":       0.15,
    "n_passes":        0.25,
    "n_shots":         0.20,
    "possession_pct":  0.10,
}
GOAL_BONUS = 1.50


def _minmax_norm(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def _get_event_count(track_id: int, events_per_player: dict, event_type: str) -> int:
    """Extract count for a specific event type from events_per_player dict."""
    return events_per_player.get(track_id, {}).get(event_type, 0)


def compute(
    phys: pd.DataFrame,
    events_per_player: dict[int, dict[str, int]],
    possession_proxy: dict,
) -> pd.DataFrame:
    """Compute ratings and return enriched DataFrame.

    Parameters
    ----------
    phys : pd.DataFrame
        Output of ``stats.physical_stats()`` — columns: track_id, team,
        distance_m, top_speed_kmh, n_sprints, minutes_tracked.
    events_per_player : dict
        Output of ``events.events_per_player()`` —
        {track_id: {n_passes, n_shots, n_goals}}.
    possession_proxy : dict
        Output of ``stats.possession_proxy()`` —
        {team_A: float, team_B: float, frames_used: int}.
    """
    df = phys.copy()

    df["n_passes"] = df["track_id"].map(
        lambda tid: _get_event_count(tid, events_per_player, "n_passes")
    )
    df["n_shots"] = df["track_id"].map(
        lambda tid: _get_event_count(tid, events_per_player, "n_shots")
    )
    df["n_goals"] = df["track_id"].map(
        lambda tid: _get_event_count(tid, events_per_player, "n_goals")
    )

    # ---- Possession per player (proportional share within team) ------------
    # We only have team-level possession from the proxy, so we distribute it
    # equally among tracked players of that team as a starting approximation.
    team_sizes = df.groupby("team")["track_id"].transform("count")
    df["possession_pct"] = df.apply(
        lambda row: possession_proxy.get(row["team"], 0.0) / max(team_sizes[row.name], 1),
        axis=1,
    ).fillna(0.0)

    # ---- Filter players with enough tracking time --------------------------
    eligible = df[df["minutes_tracked"] >= MIN_MINUTES_TRACKED].copy()
    if eligible.empty:
        df["rating"] = 0.0
        df["rank_in_team"] = 0
        return df

    # ---- Normalise each metric across ALL eligible players -----------------
    for col in _WEIGHTS:
        eligible[f"{col}_norm"] = _minmax_norm(eligible[col])

    # ---- Composite score ---------------------------------------------------
    score = pd.Series(0.0, index=eligible.index)
    for col, w in _WEIGHTS.items():
        score += w * eligible[f"{col}_norm"]
    score += GOAL_BONUS * eligible["n_goals"]
    score = (score * 10).clip(upper=10).round(2)  # scale to 0-10
    eligible["rating"] = score

    # ---- Rank within team (1 = best) ---------------------------------------
    eligible["rank_in_team"] = eligible.groupby("team")["rating"].rank(
        method="min", ascending=False
    ).astype(int)

    # Merge back into full df
    df = df.merge(
        eligible[["track_id", "rating", "rank_in_team"]],
        on="track_id",
        how="left",
    )
    df["rating"] = df["rating"].fillna(0.0)
    df["rank_in_team"] = df["rank_in_team"].fillna(0).astype(int)

    # Drop norm columns (they were on eligible, not on df)
    return df


def best_worst(rated: pd.DataFrame, n_best: int = 2) -> dict[str, dict]:
    """Return best and worst players per team.

    Returns
    -------
    dict with keys 'team_A' and 'team_B', each containing:
        best : list of dicts (top n_best players)
        worst: list of dicts (bottom 1 player)
    """
    result = {}
    eligible = rated[rated["minutes_tracked"] >= MIN_MINUTES_TRACKED]

    for team in ["team_A", "team_B"]:
        grp = eligible[eligible["team"] == team].sort_values(
            "rating", ascending=False
        )
        if grp.empty:
            result[team] = {"best": [], "worst": []}
            continue

        cols = ["track_id", "rating", "distance_m", "n_sprints",
                "n_passes", "n_shots", "n_goals"]
        best = grp.head(n_best)[cols].to_dict("records")
        worst = grp.tail(1)[cols].to_dict("records")
        result[team] = {"best": best, "worst": worst}

    return result
