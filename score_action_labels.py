"""Score event detectors against every action labelled in stretches of play.

`make_action_labeller.py` asks for every action in six stretches of live
play -- type, start and end frame, the player who did it, the receiver of
a pass, success -- independently of any detector. So both detectors can
be measured on the same footing:

- **recall**: labelled actions with a detected action of the same kind
  starting within `TOL_S` of it, one to one, nearest first;
- **precision**: detected actions inside the stretches that matched one;
- **timing**: median frames between labelled and detected start and end;
- **players**: the detected `player_from` is the clicked player (their
  track at that frame, or another track of the same identity), and for
  passes the detected `player_to` is the clicked receiver.

Tackles and recoveries are one kind here -- a ball won either way.

    python score_action_labels.py action_labels.json [action-label-v2.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from src.track_split import player_tracks

from score_player_labels import clicked_track
from src.event_schema import from_pipeline
from src.touch_events import for_clip

TOL_S = 0.4
KIND = {"tackle": "ball won", "recovery": "ball won"}


def kind(t: str) -> str:
    return KIND.get(t, t)


def names(merged, info, ident, click):
    """The names a detector may give the clicked player."""
    if click is None:
        return set()
    tid = clicked_track(merged, int(click["f"]), click["x"], click["y"],
                        info["width"], info["height"])
    if tid is None:
        return set()
    out = {f"track {tid}"}
    if int(tid) in ident:
        out.add(f"player {ident[int(tid)]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+",
                    help="one or more label files (rounds are combined)")
    args = ap.parse_args()
    blob = {"clips": [c for f in args.labels
                      for c in json.loads(Path(f).read_text())["clips"]]}
    detectors = {"touch-based (new)": {}, "nearest player (old)": {}}
    totals = {d: defaultdict(lambda: defaultdict(list)) for d in detectors}
    for c in blob["clips"]:
        clip, first, last = c["clip"], c["first"], c["last"]
        out = Path(f"output_{clip}")
        info = json.loads((out / "clip.json").read_text())
        fps = float(info["fps"])
        merged = player_tracks(out)
        idf = out / "identities.json"
        ident = ({int(k): v["identity"] for k, v in
                  json.loads(idf.read_text())["tracks"].items()}
                 if idf.exists() else {})
        found = {
            "touch-based (new)": for_clip(out),
            "nearest player (old)": [
                from_pipeline(e, fps) for e in
                json.loads((out / "events.json").read_text())],
        }
        truth = c["actions"]
        for det, acts in found.items():
            acts = [a for a in acts if first <= a.start_frame <= last]
            pairs = sorted(
                (abs(a.start_frame - t["start"]), i, j)
                for i, t in enumerate(truth) for j, a in enumerate(acts)
                if kind(a.type) == kind(t["type"])
                and abs(a.start_frame - t["start"]) <= TOL_S * fps)
            used_t, used_a = set(), set()
            for _, i, j in pairs:
                if i in used_t or j in used_a:
                    continue
                used_t.add(i)
                used_a.add(j)
                t, a = truth[i], acts[j]
                s = totals[det][kind(t["type"])]
                s["matched"].append(1)
                s["start_err"].append(abs(a.start_frame - t["start"]))
                s["end_err"].append(abs(a.end_frame - t["end"]))
                s["from_right"].append(
                    a.player_from in names(merged, info, ident, t["from"]))
                if t["type"] == "pass" and t.get("to"):
                    s["to_right"].append(
                        a.player_to in names(merged, info, ident, t["to"]))
            for i, t in enumerate(truth):
                totals[det][kind(t["type"])]["labelled"].append(1)
            for j, a in enumerate(acts):
                totals[det][kind(a.type)]["detected"].append(1)

    for det, by in totals.items():
        print(f"\n  {det}")
        print(f"    {'kind':10s} {'labelled':>8s} {'detected':>8s} "
              f"{'matched':>7s} {'recall':>6s} {'precision':>9s} "
              f"{'start':>6s} {'end':>5s} {'player':>7s} {'receiver':>8s}")
        for k, s in sorted(by.items()):
            lab, got, m = len(s["labelled"]), len(s["detected"]), len(s["matched"])
            med = lambda v: f"{np.median(v):.0f}f" if v else "-"
            frac = lambda v: f"{sum(v)}/{len(v)}" if v else "-"
            print(f"    {k:10s} {lab:8d} {got:8d} {m:7d} "
                  f"{m / max(lab, 1):6.0%} {m / max(got, 1):9.0%} "
                  f"{med(s['start_err']):>6s} {med(s['end_err']):>5s} "
                  f"{frac(s['from_right']):>7s} {frac(s['to_right']):>8s}")


if __name__ == "__main__":
    main()
