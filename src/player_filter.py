"""Hold the tracked roster to what a football match can contain.

Detection returns people, and a frame contains more people than players:
referees and assistants stand on the pitch, substitutes warm up behind the
touchline, staff drift into view. Left alone the pipeline reports 30-plus
"players" per frame, every one of them contributing to possession, team
shape and distance covered.

Two rules, applied in order:

  * drop detections with no team assigned -- team assignment clusters torso
    colour, so a detection it cannot place is usually not a player in kit
  * cap each team at eleven per frame, keeping the longest-lived tracks

Length is the tie-breaker because it is the one signal available here that
correlates with being a real player: officials and touchline figures are
tracked in bursts as they enter and leave, while a player on the pitch is
followed continuously. It is a heuristic, and it will drop a genuine player
whose track happens to be fragmented -- which is an argument for fixing
fragmentation upstream rather than for loosening the cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PLAYERS_PER_TEAM = 11


def roster_report(tracks: pd.DataFrame) -> dict:
    """Summary of how many players are being carried per frame."""
    if tracks.empty or "cls" not in tracks.columns:
        return dict(median_per_frame=0.0, max_per_frame=0, n_tracks=0)

    players = tracks[tracks.cls == "player"]
    if players.empty:
        return dict(median_per_frame=0.0, max_per_frame=0, n_tracks=0)

    per_frame = players.groupby("frame").size()
    return dict(
        median_per_frame=float(per_frame.median()),
        max_per_frame=int(per_frame.max()),
        n_tracks=int(players.track_id.nunique()),
    )


def filter_players(tracks: pd.DataFrame, per_team: int = PLAYERS_PER_TEAM,
                   verbose: bool = False) -> pd.DataFrame:
    """Drop unassigned detections and cap each team's per-frame count."""
    if tracks.empty or "cls" not in tracks.columns:
        return tracks.copy()

    players = tracks[tracks.cls == "player"]
    others = tracks[tracks.cls != "player"]
    if players.empty:
        return tracks.copy()

    before = len(players)

    if "team" in players.columns:
        players = players[players.team.notna()]
        # Team assignment spells "unassigned" differently depending on which
        # path produced it; treat every spelling as absent rather than as a
        # third team.
        players = players[~players.team.astype(str).isin(
            ("", "-1", "unknown", "unassigned", "nan", "None"))]

    if players.empty:
        return others.copy()

    # Longest-lived tracks win a contested frame.
    lifetime = players.groupby("track_id").size().rename("track_len")
    players = players.join(lifetime, on="track_id")

    group_cols = ["frame", "team"] if "team" in players.columns else ["frame"]
    limit = per_team if "team" in players.columns else per_team * 2
    players = (players
               .sort_values("track_len", ascending=False)
               .groupby(group_cols, sort=False, group_keys=False)
               .head(limit)
               .drop(columns="track_len"))

    if verbose:
        print(f"  [roster] {before} -> {len(players)} player rows "
              f"(cap {per_team}/team/frame)")

    out = pd.concat([players, others])
    sort_cols = [c for c in ("frame", "track_id") if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)
