"""Does the chance-creation rule pick the right pass? Checked on real matches.

The rule that links a shot to the pass that made it is not perception, so it
can be checked without any of this project's detectors -- and it should be,
because the detectors cannot supply a single real shot from the footage that
exists.

The SoccerNet ball-action labels can. Two full matches, every PASS, SHOT and
GOAL marked with its team and its millisecond. Running the rule over those
asks exactly the right question: given a correct list of actions, does it
attribute the shot to the right pass, and does the answer look like football?

## What "looks like football" means here

Roughly half to two thirds of shots are assisted. The rest come from
rebounds, interceptions, dribbles and set pieces taken directly, and
football credits nobody for those either. A rule linking almost every shot
is crediting passes that created nothing; one linking almost none has been
built too strictly.

## The split

Two matches, so the rule is settled on the first and reported on the second.
Tuning a window on a number and then quoting that same number is how a
measurement stops meaning anything, and this project has done it before.

    python check_chances.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import chances as ch

UPLOADS = Path("/root/.claude/uploads/cd4d7e67-1dd4-5fa1-975c-2f5b3217663b")

# What football produces, and what this is judged against.
ASSISTED_SHARE = (0.45, 0.75)


def load_match(path: Path):
    """A labelled match as actions and shots, in this pipeline's shape."""
    blob = json.loads(path.read_text())
    actions, shots = [], []
    for row in blob["annotations"]:
        label = str(row["label"]).lower()
        item = {"timestamp_s": int(row["position"]) / 1000.0,
                "team": row.get("team"), "event_type": label}
        if label in ("shot", "goal"):
            item["is_goal"] = label == "goal"
            shots.append(item)
        actions.append(item)
    return blob.get("UrlLocal", path.name), actions, shots


def synthetic_checks():
    """The three ways the rule can be wrong, each made to happen."""
    ok = True

    # 1. The obvious case: a pass, a touch, a shot. Must link.
    actions = [{"timestamp_s": 0.0, "team": "A", "event_type": "pass"},
               {"timestamp_s": 1.0, "team": "A", "event_type": "carry"}]
    shots = [{"timestamp_s": 2.0, "team": "A", "is_goal": True}]
    links = ch.link_chances(actions, shots)
    good = links[0]["pass"] is actions[0] and links[0]["is_assist"]
    print(f"   pass, touch, goal        -> "
          f"{'assist' if links[0]['is_assist'] else 'nothing'}  "
          f"{'ok' if good else 'WRONG'}")
    ok &= good

    # 2. An opposition touch in between. Possession turned over, so the
    #    earlier pass created nothing.
    actions = [{"timestamp_s": 0.0, "team": "A", "event_type": "pass"},
               {"timestamp_s": 1.0, "team": "B", "event_type": "tackle"}]
    shots = [{"timestamp_s": 2.0, "team": "A", "is_goal": False}]
    links = ch.link_chances(actions, shots)
    good = links[0]["pass"] is None
    print(f"   pass, opposition, shot   -> "
          f"{'linked' if links[0]['pass'] else 'no chance'}  "
          f"{'ok' if good else 'WRONG'}")
    ok &= good

    # 3. A pass from the previous phase, long before. Out of reach.
    actions = [{"timestamp_s": 0.0, "team": "A", "event_type": "pass"}]
    shots = [{"timestamp_s": 60.0, "team": "A", "is_goal": False}]
    links = ch.link_chances(actions, shots)
    good = links[0]["pass"] is None
    print(f"   pass a minute earlier    -> "
          f"{'linked' if links[0]['pass'] else 'no chance'}  "
          f"{'ok' if good else 'WRONG'}")
    ok &= good

    # 4. A goal with no pass before it is a goal and not an assist.
    shots = [{"timestamp_s": 5.0, "team": "A", "is_goal": True}]
    links = ch.link_chances([], shots)
    good = not links[0]["is_assist"] and links[0]["pass"] is None
    print(f"   unassisted goal          -> "
          f"{'assist' if links[0]['is_assist'] else 'no assist'}  "
          f"{'ok' if good else 'WRONG'}")
    return ok and good


def main():
    print("The rule that links a shot to the pass that made it, checked on "
          "labelled\nmatches so no detector of ours is involved.\n")

    print("1. The cases it has to get right\n")
    passed = synthetic_checks()
    print(f"\n   {'all pass' if passed else 'SOMETHING IS WRONG'}\n")

    files = sorted(UPLOADS.glob("*Labels-ball.json"))
    if not files:
        print("2. No labelled matches available here.")
        return

    print("2. Real matches\n")
    print(f"  {'match':>34s} {'shots':>6s} {'chances':>8s} {'assists':>8s} "
          f"{'assisted':>9s} {'lookback':>9s}")
    rows = []
    for index, path in enumerate(files):
        name, actions, shots = load_match(path)
        # Settled on the first match, reported on both. The second is the
        # one that means anything.
        links = ch.link_chances(actions, shots)
        summary = ch.summarise(links)
        role = "settled on" if index == 0 else "HELD OUT"
        print(f"  {name[-32:]:>34s} {summary['shots']:6d} "
              f"{summary['chances_created']:8d} {summary['assists']:8d} "
              f"{summary['share_assisted']:8.0%} {role:>9s}")
        rows.append((name, summary, links))

    if len(rows) > 1:
        _, held, links = rows[-1]
        low, high = ASSISTED_SHARE
        share = held["share_assisted"]
        verdict = ("in the range football produces" if low <= share <= high
                   else "OUTSIDE the 45-75% football produces")
        print(f"\n  On the held-out match {share:.0%} of shots are "
              f"assisted -- {verdict}.")
        teams = ch.per_team(links)
        for team, row in sorted(teams.items()):
            print(f"    {team:>6s}: {row['chances_created']} chances, "
                  f"{row['assists']} assists")

    print("\n  A rule linking nearly every shot credits passes that created "
          "nothing; one\n  linking nearly none is too strict. Shots from "
          "rebounds, dribbles and set\n  pieces have no assist in football "
          "either, so short of 100% is correct.")


if __name__ == "__main__":
    main()
