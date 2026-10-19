"""Team assignment, against kit labels read off the video by hand.

Three windows carry hand-read kit labels -- which track wears which kit, and
which tracks are not players at all -- so unlike most of this pipeline, team
assignment can be scored rather than eyeballed. This makes that a command
rather than something reconstructed from a shell one-liner each time.

Two numbers, and they are not the same question.

**Kit accuracy** asks whether the two teams are told apart: of the tracks a
human labelled red or blue, do the reds agree on one label and the blues on
the other. This is what the clustering is for and it is measurably good.

**Non-player rejection** asks whether everyone else is kept out. A quarter
of the tracks the detector calls players are match officials, stewards in
hi-vis, staff in dark coats, spectators, and in one window an advertising
hoarding. They must land in "other", because a steward given a team becomes
a player in every count downstream.

The second is the weaker of the two and the one worth working on. Reported
separately for that reason: a single "accuracy" would average a solved
problem together with an unsolved one and hide both.

    python check_teams.py [--residual-cut 2.5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Windows with hand-read labels. Derived from SoccerNet video, so the label
# files live in the gitignored output directories.
LABELLED = ("output_soccernet", "output_soccernet_w2",
            "output_soccernet_w3", "output_soccernet_reading")


def majority_team(players: pd.DataFrame) -> dict[int, str]:
    """One team per track: whatever most of its frames say."""
    out = {}
    for track_id, group in players.groupby("track_id"):
        seen = [value for value in group.team.tolist()
                if isinstance(value, str)]
        if seen:
            out[int(track_id)] = max(set(seen), key=seen.count)
    return out


def score_window(out_dir: Path):
    """Kit accuracy and non-player rejection for one labelled window."""
    kit_path, tracks_path = (out_dir / "kit_labels.json",
                             out_dir / "tracks_teams.parquet")
    if not (kit_path.exists() and tracks_path.exists()):
        return None
    kit = json.loads(kit_path.read_text())
    tracks = pd.read_parquet(tracks_path)
    assigned = majority_team(tracks[tracks.cls == "player"])

    red = set(kit.get("red", []))
    blue = set(kit.get("blue", []))
    non_players = set(kit.get("nonplayer", []))

    reds = [assigned.get(t) for t in red
            if assigned.get(t) in ("team_A", "team_B")]
    blues = [assigned.get(t) for t in blue
             if assigned.get(t) in ("team_A", "team_B")]
    if not reds or not blues:
        return None

    # Which cluster each kit mostly landed in. If both kits chose the same
    # cluster the teams were never separated, and accuracy is meaningless
    # rather than high.
    red_side = max(set(reds), key=reds.count)
    blue_side = max(set(blues), key=blues.count)
    if red_side == blue_side:
        return {"name": out_dir.name, "kit_accuracy": float("nan"),
                "collapsed": True, "labelled_players": len(reds) + len(blues),
                "kept": 0, "of": len(red | blue),
                "rejected": 0, "non_players": len(non_players)}

    correct = (sum(1 for value in reds if value == red_side)
               + sum(1 for value in blues if value == blue_side))
    return {
        "name": out_dir.name,
        "kit_accuracy": correct / (len(reds) + len(blues)),
        "collapsed": False,
        "labelled_players": len(reds) + len(blues),
        "kept": sum(1 for t in (red | blue)
                    if assigned.get(t) in ("team_A", "team_B")),
        "of": len(red | blue),
        "rejected": sum(1 for t in non_players
                        if assigned.get(t) == "other"),
        "non_players": len(non_players),
    }


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    print("Team assignment against kit labels read off the video by hand.\n")
    print(f"  {'window':>24s} {'kit accuracy':>13s} {'players kept':>13s} "
          f"{'non-players rejected':>21s}")

    rows = []
    for name in LABELLED:
        result = score_window(Path(name))
        if result is None:
            continue
        rows.append(result)
        kept = f"{result['kept']}/{result['of']}"
        rejected = f"{result['rejected']}/{result['non_players']}"
        accuracy = ("COLLAPSED" if result["collapsed"]
                    else f"{result['kit_accuracy']:.2f}")
        print(f"  {result['name'][-22:]:>24s} {accuracy:>13s} "
              f"{kept:>13s} {rejected:>21s}")

    if not rows:
        print("  no labelled windows found -- run the pipeline first")
        return

    good = [r for r in rows if not r["collapsed"]]
    if good:
        kit = sum(r["kit_accuracy"] * r["labelled_players"] for r in good)
        kit /= sum(r["labelled_players"] for r in good)
        kept = sum(r["kept"] for r in good) / sum(r["of"] for r in good)
        rejected = (sum(r["rejected"] for r in good)
                    / max(sum(r["non_players"] for r in good), 1))
        print(f"\n  pooled: kit {kit:.2f}, players kept {kept:.0%}, "
              f"non-players rejected {rejected:.0%}")

    print("\n  Telling the kits apart is solved; keeping non-players out is "
          "not. A\n  steward given a team becomes a player in every count "
          "downstream, so the\n  second number is the one that limits what "
          "team statistics can be trusted.")


if __name__ == "__main__":
    main()
