"""Score the touch-based events (`src/touch_events.py`) against the clicks.

The clicks were made on the first detector's events, so they say, at each
of those moments, whether a pass / carry / recovery / tackle / shot
happened and who played the ball. For the new detector that gives:

- **found**: for each real clicked event, an action of the same kind
  starting from `BEFORE_S` before to `AFTER_S` after the clicked moment
  (the old events were stamped late, so the window reaches back);
- **right player**: that action's `player_from` is the clicked player's
  track, or another track of the same identity;
- **false alarms**: for each moment the person marked as no such event, an
  action of that kind starting within `FALSE_S` of it;
- **double counts**: two passes by the same player less than a second apart.

Recoveries are matched by recoveries and tackles: a ball won either way.

    python score_touch_events.py player_labels.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.track_split import player_tracks

from score_player_labels import clicked_track
from src.match_identity import names_of
from src.touch_events import for_clip

BEFORE_S, AFTER_S = 1.0, 0.5
FALSE_S = 0.5
SAME = {"pass": {"pass"}, "carry": {"carry"}, "recovery": {"recovery",
        "tackle"}, "tackle": {"tackle", "recovery"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    args = ap.parse_args()
    items = json.loads(Path(args.labels).read_text())["frames"]
    clips = sorted({i["clip"] for i in items})
    acts, info, merged, ident = {}, {}, {}, {}
    for c in clips:
        out = Path(f"output_{c}")
        acts[c] = for_clip(out)
        info[c] = json.loads((out / "clip.json").read_text())
        merged[c] = player_tracks(out)
        ident[c] = names_of(out)

    stats = defaultdict(Counter)
    for it in items:
        kind, lab, c = it["kind"], it.get("label"), it["clip"]
        if kind not in SAME or lab is None:
            continue
        fps = float(info[c]["fps"])
        f = int(it["frame"])
        if lab == "no event":
            near = [a for a in acts[c] if a.type in SAME[kind]
                    and (a.start_frame - FALSE_S * fps <= f
                         <= a.end_frame + FALSE_S * fps
                         if kind == "carry"
                         else abs(a.start_frame - f) <= FALSE_S * fps)]
            stats[kind]["rejected moments"] += 1
            stats[kind]["still detected there"] += bool(near)
            continue
        truth = clicked_track(merged[c], f, lab["x"], lab["y"],
                              info[c]["width"], info[c]["height"])
        if kind == "carry":
            # A carry covers the clicked moment: it may begin well before.
            cand = [a for a in acts[c] if a.type == "carry"
                    and a.start_frame - BEFORE_S * fps <= f
                    <= a.end_frame + AFTER_S * fps]
        else:
            cand = [a for a in acts[c] if a.type in SAME[kind]
                    and f - BEFORE_S * fps <= a.start_frame
                    <= f + AFTER_S * fps]
        stats[kind]["real"] += 1
        if not cand:
            continue
        stats[kind]["found"] += 1
        want = {f"track {truth}"}
        if truth is not None and int(truth) in ident[c]:
            want.add(ident[c][int(truth)])
        if any(a.player_from in want for a in cand):
            stats[kind]["right player"] += 1

    print("  against the clicks:")
    for kind, s in stats.items():
        print(f"    {kind:9s} real {s['real']}: found {s['found']}, right "
              f"player {s['right player']} | rejected moments "
              f"{s['rejected moments']}, still detected {s['still detected there']}")
    total = Counter()
    doubles = 0
    for c, a in acts.items():
        total.update(x.type for x in a)
        passes = sorted((x for x in a if x.type == "pass"),
                        key=lambda x: x.start_frame)
        fps = float(info[c]["fps"])
        doubles += sum(1 for x, y in zip(passes, passes[1:])
                       if x.player_from == y.player_from
                       and (y.start_frame - x.start_frame) / fps < 1.0)
    minutes = sum(m.frame.max() / float(info[c]["fps"]) / 60
                  for c, m in merged.items())
    print(f"  actions per minute over {minutes:.1f} min: " + ", ".join(
        f"{k} {v / minutes:.1f}" for k, v in sorted(total.items()))
          + f"; passes by one player under 1 s apart: {doubles}")


if __name__ == "__main__":
    main()
