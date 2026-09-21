"""Team assignment measured against the kits, not against possession.

Every figure for team assignment in this project -- 0.55 to 0.81 -- was scored
at matched labelled passes: our team for the passer against SoccerNet's side
for that action. Looking at the tracks as strips of crops shows that measure
is not what it claims. Both kits appear under the same truth side, because the
comparison is contaminated by possession error: it measures

    (did we cluster this track into the right team)
      x (did we attribute the event to the right player)

and it cannot say which factor is failing. Nine colour representations and six
values of k all landed between 0.64 and 0.75 on it, which is what a metric
dominated by a different error looks like.

So the kits were read off directly. `kit_labels.json` in the window's output
directory records, for all 107 tracks of at least 15 frames in the Reading
window, whether the shirt is Fulham maroon, Reading blue-and-white hoops, or
not a player at all. It lives there rather than in the repository because it
is derived from SoccerNet video.

Against that, the two factors separate:

  * how well kit clustering assigns the tracks that *are* players;
  * how much of the tracked "roster" is not players at all, and what the
    clustering does with them.

    python score_team_assignment.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_pass_outcome import (UPLOADS, WINDOWS, TOLERANCE_S, match,
                                  truth_passes)
from score_soccernet import run_pipeline

WINDOW = "reading 5115"


def best_mapping_accuracy(assigned: dict[int, str],
                          truth: dict[int, str]) -> tuple[float, int]:
    """Agreement under the better of the two cluster-to-kit mappings."""
    shared = [t for t in truth if t in assigned]
    if len(shared) < 4:
        return float("nan"), 0
    clusters = sorted({assigned[t] for t in shared})
    if len(clusters) != 2:
        return float("nan"), len(shared)
    best = 0
    for kits in (("red", "blue"), ("blue", "red")):
        m = dict(zip(clusters, kits))
        best = max(best, sum(1 for t in shared if m[assigned[t]] == truth[t]))
    return best / len(shared), len(shared)


def main():
    name, out_dir, labels_name, offset = next(
        w for w in WINDOWS if w[0] == WINDOW)
    out = Path(out_dir)
    labels = json.loads((out / "kit_labels.json").read_text())

    kit = {int(t): "red" for t in labels["red"]}
    kit.update({int(t): "blue" for t in labels["blue"]})
    nonplayers = {int(t) for t in labels["nonplayer"]}

    raw = pd.read_parquet(out / "tracks_grass.parquet")
    players = raw[raw.cls == "player"]
    assigned = {int(k): v for k, v in
                players.drop_duplicates("track_id")
                .set_index("track_id")["team"].to_dict().items()}
    frames = players.groupby("track_id").size().to_dict()

    labelled = set(kit) | nonplayers
    print(f"{name}: {len(labelled)} tracks of >=15 frames read off the crops "
          f"-- {len(labels['red'])} maroon, {len(labels['blue'])} hooped, "
          f"{len(nonplayers)} not players")

    # --- 1. kit clustering, on the tracks that are actually players ---------
    real = {t: k for t, k in kit.items() if t in assigned
            and assigned[t] in ("team_A", "team_B")}
    acc, n = best_mapping_accuracy(assigned, real)
    print(f"\n1. kit clustering on real players only   {acc:.2f}  (n={n})")
    print("   the same tracks scored through possession at matched passes: "
          "0.55")

    # Which way each kit went.
    cross = pd.crosstab(
        pd.Series({t: kit[t] for t in real}, name="kit"),
        pd.Series({t: assigned[t] for t in real}, name="assigned"))
    print(cross.to_string().replace("\n", "\n   ").rjust(3))

    # --- 2. what happens to the non-players ---------------------------------
    placed = [t for t in nonplayers
              if assigned.get(t) in ("team_A", "team_B")]
    print(f"\n2. non-player tracks given a team: {len(placed)}/"
          f"{len(nonplayers)}")
    total_frames = sum(frames.get(t, 0) for t in labelled)
    np_frames = sum(frames.get(t, 0) for t in placed)
    print(f"   they carry {np_frames} of {total_frames} labelled player-frames "
          f"({np_frames / total_frames:.0%})")
    got = {}
    for t in placed:
        got[assigned[t]] = got.get(assigned[t], 0) + 1
    print(f"   split across the two teams: {got}")

    # --- 3. do they take possession? ----------------------------------------
    events, _ = run_pipeline(
        json.loads((out / "clip.json").read_text())["path"], out)
    truth = truth_passes(UPLOADS / labels_name, offset, True)
    ours = [e for e in events if e.get("event_type") == "pass"]
    pairs = match([e["timestamp_s"] for e in ours],
                  [t["t"] for t in truth], TOLERANCE_S)

    actors = [int(ours[pi].get("player_track_id", -1)) for pi, _ in pairs]
    actors += [int(ours[pi].get("receiver_track_id", -1)) for pi, _ in pairs]
    actors = [a for a in actors if a != -1]
    bad = [a for a in actors if a in nonplayers]
    unknown = [a for a in actors if a not in labelled]
    print(f"\n3. of {len(actors)} passer/receiver slots on matched passes:")
    print(f"   {len(bad)} ({len(bad) / len(actors):.0%}) are tracks the crops "
          "show are not players")
    print(f"   {len(unknown)} ({len(unknown) / len(actors):.0%}) are tracks "
          "too short to have been labelled")

    # --- 4. the decomposition ------------------------------------------------
    scored = [(a, kit.get(a)) for a in actors if a in kit]
    print(f"\n4. {len(scored)} slots land on a track whose kit is known, so "
          f"{len(actors) - len(scored)} of {len(actors)} "
          f"({1 - len(scored) / len(actors):.0%}) cannot be right by kit at "
          "all")


if __name__ == "__main__":
    main()
