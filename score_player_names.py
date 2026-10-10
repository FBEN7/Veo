"""Score player identities against the names from `make_number_labeller.py`.

Each click is resolved to the track whose feet it is nearest
(`at_feet`); its team and typed number say who the player is. Over
pairs of named clicks in one match:

- **within a clip**: same team and number -> the two tracks should be one
  identity (else the person was split); a different team or number -> two
  identities (else two people were merged);
- **across clips of a match**: the same, for the match-wide names
  (`match_identity.names_of`);
- **teams**: within a clip, the share of named players whose track's team
  agrees with the clip's majority pairing of our team labels to theirs.

Clicks without a number count only for the team check.

    python score_player_names.py player_names.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.match_identity import names_of
from src.track_split import player_tracks

# A click this far from a player's feet, as a share of their height, picks them.
FEET_REACH = 0.3


def at_feet(tracks, frame, x, y, width, height):
    """The player whose feet are nearest the click, within `FEET_REACH` of
    their height. The page asks for the feet; `clicked_track` measures
    from the middle of the body with a wide reach, which in close-ups,
    with players 400-500 px tall, gave a click on one player's feet to
    the player beside them."""
    p = tracks[(tracks.cls == "player") & (tracks.frame == frame)]
    if p.empty:
        return None
    d = np.hypot(p.px_raw - x * width, p.py_raw - y * height) / p.crop_h
    best = d.idxmin()
    return int(p.track_id[best]) if d[best] <= FEET_REACH else None


def resolve(items):
    """Each named click -> (clip, frame, team, number, track, our team,
    clip identity, match-wide name)."""
    out, cache, missed = [], {}, 0
    for item in items:
        clip = item["clip"]
        d = Path(f"output_{clip}")
        if clip not in cache:
            info = json.loads((d / "clip.json").read_text())
            ident = json.loads((d / "identities.json").read_text())["tracks"]
            cache[clip] = (info, player_tracks(d), ident, names_of(d))
        info, tracks, ident, names = cache[clip]
        for p in item["players"]:
            # The page reads the frame from the browser's player, which may
            # be one off the pipeline's count; the neighbours are tried next.
            f0 = int(item["frame"])
            t = None
            for f in (f0, f0 - 1, f0 + 1):
                t = at_feet(tracks, f, p["x"], p["y"],
                            info["width"], info["height"])
                if t is not None:
                    break
            if t is None:
                missed += 1
                continue
            e = ident.get(str(t), {})
            out.append({"clip": clip, "frame": int(item["frame"]),
                        "team": p["team"], "number": p.get("number"),
                        "track": t, "ours": e.get("team"),
                        "identity": e.get("identity"), "name": names.get(t)})
    return out, missed


def pairs(clicks, same_clip: bool, key: str):
    """(same number: joined, split), (different: kept apart, merged)."""
    joined = split = apart = merged = 0
    named = [c for c in clicks if c["number"]]
    for a, b in itertools.combinations(named, 2):
        if a["clip"].split("_")[0] != b["clip"].split("_")[0]:
            continue    # another match
        if (a["clip"] == b["clip"]) != same_clip:
            continue
        if a["clip"] == b["clip"] and a["track"] == b["track"]:
            continue    # one track is one identity by construction
        if a["clip"] == b["clip"] and a["frame"] == b["frame"]:
            continue    # two on one frame are kept apart by construction
        one = a[key] is not None and a[key] == b[key]
        if (a["team"], a["number"]) == (b["team"], b["number"]):
            joined += one
            split += not one
        else:
            apart += not one
            merged += one
    return joined, split, apart, merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    args = ap.parse_args()
    items = json.loads(Path(args.labels).read_text())["frames"]
    clicks, missed = resolve(items)
    print(f"  {len(clicks)} named players on a detected track"
          f" ({missed} clicks on no detection)")

    agree = total = 0
    by_clip = defaultdict(list)
    for c in clicks:
        by_clip[c["clip"]].append(c)
    for clip, cs in by_clip.items():
        pairing = Counter((c["ours"], c["team"]) for c in cs
                          if c["ours"] in ("team_A", "team_B"))
        keep = pairing[("team_A", 0)] + pairing[("team_B", 1)]
        swap = pairing[("team_A", 1)] + pairing[("team_B", 0)]
        agree += max(keep, swap)
        total += len(cs)
        print(f"    {clip}: {len(cs)} named, team right {max(keep, swap)}, "
              f"wrong {min(keep, swap)}, unplaced "
              f"{len(cs) - keep - swap}")
    print(f"  team right for {agree}/{total} ({agree / max(total, 1):.0%})")

    for label, same_clip, key in (("within a clip, identities", True, "identity"),
                                  ("within a clip, match-wide names", True, "name"),
                                  ("across clips, match-wide names", False, "name")):
        j, s, a, m = pairs(clicks, same_clip, key)
        print(f"  {label}: same player {j}/{j + s} joined; "
              f"different players {a}/{a + m} kept apart; of the links, "
              f"{j}/{j + m} right")


if __name__ == "__main__":
    main()
