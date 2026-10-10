"""Score who-did-it against the clicks from `make_player_labeller.py`.

For each event the person clicked, the clicked player is the detection on
that frame nearest the click (its feet), within its own box. Three things
are scored:

- **attribution**: the pipeline credited the event to that player's track
  (passes, carries, recoveries, tackles from `events.json`; for a shot, the
  player nearest the ball on the strike frame, as `detect_shots.attribute`
  does), or to another track of the same identity;
- **numbers**: the identity the clicked track belongs to carries the number
  the person typed;
- **one person, one identity**: two clicks on the same team with the same
  typed number fall in one identity (else the person was split), and two
  with different numbers in two (else two people were merged).

Events the person marked "no such event" are counted apart.

    python score_player_labels.py player_labels.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from src.track_split import player_tracks

# A click this far from a player's feet, as a share of the player's height,
# still picks that player.
CLICK_REACH = 0.6
# An event matches a pipeline event of the same kind this close in time.
EVENT_MATCH_S = 0.05


def clicked_track(merged, frame, x, y, width, height):
    players = merged[(merged.cls == "player") & (merged.frame == frame)]
    if players.empty:
        return None
    px = players.px_raw if "px_raw" in players else players.px
    py = players.py_raw if "py_raw" in players else players.py
    cx, cy = x * width, y * height
    # Distance to the box, measured from the middle of the body.
    d = np.hypot(px - cx, (py - players.crop_h / 2) - cy) / players.crop_h
    best = d.idxmin()
    if d[best] > CLICK_REACH + 0.5:
        return None
    return int(players.track_id[best])


def shooter(merged, frame):
    """The player nearest the ball on the frame, as for shots."""
    here = merged[merged.frame == frame]
    ball = here[here.cls == "ball"]
    players = here[here.cls == "player"]
    if ball.empty or players.empty:
        return None
    bx = float((ball.px_raw if "px_raw" in ball else ball.px).iloc[0])
    by = float((ball.py_raw if "py_raw" in ball else ball.py).iloc[0])
    px = players.px_raw if "px_raw" in players else players.px
    py = players.py_raw if "py_raw" in players else players.py
    return int(players.track_id[np.hypot(px - bx, py - by).idxmin()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    args = ap.parse_args()
    items = json.loads(Path(args.labels).read_text())["frames"]
    stats = defaultdict(Counter)
    typed = []                   # (clip, team of clicked track, number, identity)
    cache = {}
    for item in items:
        lab = item.get("label")
        kind = item["kind"]
        if lab is None:
            stats[kind]["can't tell / not labelled"] += 1
            continue
        if lab == "no event":
            stats[kind]["no such event"] += 1
            continue
        clip, frame = item["clip"], int(item["frame"])
        out_dir = Path(f"output_{clip}")
        if clip not in cache:
            info = json.loads((out_dir / "clip.json").read_text())
            cache[clip] = (info,
                           player_tracks(out_dir),
                           json.loads((out_dir / "events.json").read_text()),
                           json.loads((out_dir / "identities.json").read_text())
                           if (out_dir / "identities.json").exists() else None)
        info, merged, events, idents = cache[clip]
        truth = clicked_track(merged, frame, lab["x"], lab["y"],
                              info["width"], info["height"])
        if truth is None:
            stats[kind]["click on no detected player"] += 1
            continue
        if kind in ("shot", "goal"):
            got = shooter(merged, frame)
        else:
            t = frame / float(info["fps"])
            same = [e for e in events if e["event_type"] == kind
                    and abs(float(e["timestamp_s"]) - t) <= EVENT_MATCH_S]
            got = (int(same[0]["player_track_id"])
                   if same and same[0].get("player_track_id") is not None
                   else None)
        ident_of = (lambda tid: idents["tracks"].get(str(tid), {})
                    .get("identity")) if idents else (lambda tid: None)
        if got == truth:
            stats[kind]["right track"] += 1
        elif got is not None and idents and ident_of(got) == ident_of(truth):
            stats[kind]["right person, other track"] += 1
        else:
            stats[kind]["wrong player"] += 1
        if lab.get("number") and idents:
            entry = idents["tracks"].get(str(truth), {})
            typed.append((clip, entry.get("team"), str(lab["number"]),
                          entry.get("identity"), entry.get("number")))

    print("  attribution, per event type:")
    for kind, c in sorted(stats.items()):
        right = c["right track"] + c["right person, other track"]
        judged = right + c["wrong player"]
        print(f"    {kind:9s} {right}/{judged} right "
              f"({right / max(judged, 1):.0%})  " + ", ".join(
                  f"{k} {v}" for k, v in c.items()))
    if typed:
        print(f"\n  numbers typed: {len(typed)}")
        read = [t for t in typed if t[4] is not None]
        right = sum(str(t[4]) == t[2] for t in read)
        print(f"    the clicked player's identity has a number: "
              f"{len(read)}/{len(typed)}; it is the typed one: {right}/"
              f"{len(read)}")
        split = merged_pairs = same_pairs = diff_pairs = 0
        for i in range(len(typed)):
            for j in range(i + 1, len(typed)):
                a, b = typed[i], typed[j]
                if a[0] != b[0] or a[1] != b[1]:
                    continue
                if a[2] == b[2]:
                    same_pairs += 1
                    split += a[3] != b[3]
                else:
                    diff_pairs += 1
                    merged_pairs += a[3] == b[3]
        print(f"    same number, same clip and team: {same_pairs} pairs, "
              f"{same_pairs - split} in one identity, {split} split")
        print(f"    different numbers: {diff_pairs} pairs, "
              f"{merged_pairs} wrongly in one identity")


if __name__ == "__main__":
    main()
