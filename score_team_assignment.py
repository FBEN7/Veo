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

WINDOWS_LABELLED = ("reading 5115", "w1 stoke 1820")


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


def report(window):
    name, out_dir, labels_name, offset = next(
        w for w in WINDOWS if w[0] == window)
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
    placed = [t for t in labelled
              if assigned.get(t) in ("team_A", "team_B")]
    real = [t for t in placed if t in kit]

    acc, n = best_mapping_accuracy(assigned, {t: kit[t] for t in real})
    purity = len(real) / len(placed) if placed else float("nan")
    coverage = len(real) / len(kit)

    print(f"\n{name}: {len(labelled)} tracks read off the crops -- "
          f"{len(labels['red'])} one kit, {len(labels['blue'])} the other, "
          f"{len(nonplayers)} not players")
    print(f"  kit accuracy on the players it keeps   {acc:.2f}  (n={n})")
    print(f"  purity   -- kept tracks that are players   {purity:.2f}")
    print(f"  coverage -- players it keeps               {coverage:.2f}")

    kept_bad = [t for t in nonplayers if t in placed]
    total_frames = sum(frames.get(t, 0) for t in labelled)
    bad_frames = sum(frames.get(t, 0) for t in kept_bad)
    print(f"  non-players still given a team: {len(kept_bad)}/"
          f"{len(nonplayers)}, carrying {bad_frames}/{total_frames} "
          f"player-frames ({bad_frames / total_frames:.0%})")

    events, _ = run_pipeline(
        json.loads((out / "clip.json").read_text())["path"], out)
    truth = truth_passes(UPLOADS / labels_name, offset, True)
    ours = [e for e in events if e.get("event_type") == "pass"]
    pairs = match([e["timestamp_s"] for e in ours],
                  [t["t"] for t in truth], TOLERANCE_S)
    actors = [int(ours[pi].get(k, -1)) for pi, _ in pairs
              for k in ("player_track_id", "receiver_track_id")]
    actors = [a for a in actors if a != -1]
    bad = [a for a in actors if a in nonplayers]
    if actors:
        print(f"  of {len(actors)} passer/receiver slots on matched passes, "
              f"{len(bad)} ({len(bad) / len(actors):.0%}) land on a "
              "non-player")


def main():
    for w in WINDOWS_LABELLED:
        report(w)


if __name__ == "__main__":
    main()
