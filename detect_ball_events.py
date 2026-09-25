"""The ball leaving the pitch, and the ball entering a goal.

Both have been emitted by `src/events.py` for a long time and both have been
switched off for just as long, behind the `absolute_pitch` flag. The flag is
right to be off: the coordinates that detector works in come from the ground
plane, which this project measured as making pass F1, speed reliability and
distance reliability all worse, and which has no goal line on it in any
case.

The anchor does have a goal line on it, so these are detected here instead,
on the same per-frame pitch maps the shot detector uses.

Both are the same question asked twice. Where does the ball cross the edge
of the pitch?

  * across a touchline, or across a goal line outside the posts, and it is
    out of play;
  * across a goal line between the posts, and it is a goal.

The goal mouth is 7.32 m wide, so the two answers are 3.66 m apart at the
moment of crossing, which is well inside what the anchor can resolve -- it
agrees with itself to 0.7 m.

## What can be measured, for once

Out of play can be measured against real labels, which nothing else in this
family can. The labelled matches mark OUT, and unlike shots, three of them
fall inside the video that exists: one in the Stoke clip at 330.6 s and two
in the Reading clip at 3080.4 and 3147.1 s. Three is not many. It is three
more than the zero available for shots, and a detector that misses all three
or invents ten others is caught by it.

Goals cannot. There are none in the footage, so a synthetic control has to
stand in, exactly as it does for shots.

    python detect_ball_events.py [--check]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import detect_shots
from probe_pitch_lines import CLIPS
from src import pitch_model as pm

# Half the goal mouth: 7.32 m between the posts.
GOAL_HALF_M = 3.66

# How far past the line the ball must be before it counts as having left.
# A ball on the line is still in play, and the anchor is good to 0.7 m, so
# this keeps a ball hovering near the touchline from flickering in and out.
MARGIN_M = 1.0

# Two crossings closer together than this are one event seen twice.
MERGE_SECONDS = 3.0

# How long a gap in placement is enough to forget where the ball was.
#
# A crossing is a transition, so strictly it needs the ball seen inside and
# then outside. Placement is sparse -- half the frames have no pitch map at
# all -- and a ball can be seen outside, vanish for twenty seconds, and be
# seen outside again having been thrown in, played, and kicked out a second
# time in between. Insisting on an inside sighting scores that as one event
# rather than two.
#
# So after this long with nothing placed, the previous state is discarded and
# the next sighting outside counts as a fresh crossing. The risk runs the
# other way -- a ball that genuinely stayed out gets counted twice -- and
# which risk is worse is a matter for the false-alarm column, not taste.
RESET_SECONDS = 8.0

# Where the labelled clips sit in their matches, from the file names:
# stoke_000520 is 5:20 into the match, reading_5115 is 51:15.
# Each clip, the match it was cut from and where in it. Both parts matter:
# filtering labels by time alone pools the two matches, so the Reading
# window picked up Stoke's labels from the same minute of a different game
# and reported three crossings where there are two.
CLIP_SOURCES = {
    "output_soccernet_w3": ("34498b39-Labels-ball.json", 320.0),
    "output_soccernet_reading": ("53f06df4-Labels-ball.json", 3075.0),
}
CLIP_OFFSETS = {name: offset for name, (_, offset) in CLIP_SOURCES.items()}

# How close a detection has to be to a label to count, matching the
# convention the rest of this project scores events at.
TOLERANCE_S = 2.0


def outside(x, y):
    """Is the ball off the pitch here, and if so through where?"""
    if y < -MARGIN_M or y > pm.PITCH_WIDTH_M + MARGIN_M:
        return "touchline"
    if x < -MARGIN_M or x > pm.PITCH_LENGTH_M + MARGIN_M:
        middle = abs(y - pm.PITCH_WIDTH_M / 2.0)
        return "goal" if middle <= GOAL_HALF_M else "goal line"
    return None


def find_ball_events(ball: pd.DataFrame, maps, fps: float,
                     reset_seconds: float = RESET_SECONDS):
    """Frames where the ball crosses the edge of the pitch."""
    placed = []
    for row in ball.itertuples():
        point = detect_shots.to_pitch(maps, row.frame, row.px, row.py)
        if point is not None:
            placed.append((int(row.frame), float(row.time_s),
                           float(point[0]), float(point[1])))

    events, was_in, last_seen = [], True, None
    for frame, when, x, y in placed:
        if last_seen is not None and when - last_seen > reset_seconds:
            was_in = True            # blind for too long to claim otherwise
        last_seen = when
        where = outside(x, y)
        if where is None:
            was_in = True
            continue
        if not was_in:
            continue                 # still out, not a fresh crossing
        was_in = False
        kind = "goal" if where == "goal" else "out_of_play"
        events.append({"frame": frame, "time_s": when, "x": x, "y": y,
                       "event_type": kind, "through": where})

    merged = []
    for event in events:
        if merged and event["time_s"] - merged[-1]["time_s"] < MERGE_SECONDS:
            continue
        merged.append(event)
    return merged, len(placed)


def labelled(kind: str, source: str, offset_s: float, duration_s: float):
    """Labels of one kind inside a clip, from that clip's own match."""
    path = (Path("/root/.claude/uploads/"
                 "cd4d7e67-1dd4-5fa1-975c-2f5b3217663b") / source)
    if not path.exists():
        return []
    blob = json.loads(path.read_text())
    out = []
    for row in blob["annotations"]:
        if row["label"] != kind:
            continue
        when = int(row["position"]) / 1000.0
        if offset_s <= when <= offset_s + duration_s:
            out.append(when - offset_s)
    return sorted(out)


def synthetic_check():
    """A ball walked off the pitch, and another into the net."""
    identity = np.eye(3)
    ok = True
    for name, path, expect in (
            ("out over a touchline", [(52.5, 30.0), (52.5, 40.0),
                                      (52.5, 55.0), (52.5, 70.0)],
             "out_of_play"),
            ("goal between the posts", [(20.0, 34.0), (12.0, 34.0),
                                        (5.0, 34.0), (-2.0, 34.0)], "goal"),
            ("behind, outside the posts", [(20.0, 20.0), (12.0, 18.0),
                                           (5.0, 16.0), (-2.0, 15.0)],
             "out_of_play")):
        rows, maps = [], {}
        for k, (x, y) in enumerate(path):
            rows.append({"frame": k, "time_s": k / 25.0, "px": x, "py": y,
                         "confidence": 1.0})
            maps[k] = identity
        found, _ = find_ball_events(pd.DataFrame(rows), maps, 25.0)
        got = found[0]["event_type"] if found else "nothing"
        good = got == expect
        ok &= good
        print(f"   {name:<28s} -> {got:<12s} {'ok' if good else 'WRONG'}")
    return ok


def score(clip_name: str, out_dir: str, events, placed: int, duration: float):
    """One clip's crossings against its own match's labels."""
    outs = [e for e in events if e["event_type"] == "out_of_play"]
    goals = [e for e in events if e["event_type"] == "goal"]
    source_offset = CLIP_SOURCES.get(out_dir)
    if source_offset is None:
        return {"name": clip_name, "placed": placed, "found": len(outs),
                "truth": None, "matched": None, "false": None,
                "goals": goals}
    source, offset = source_offset
    truth = labelled("OUT", source, offset, duration)
    matched = sum(1 for t in truth
                  if any(abs(e["time_s"] - t) <= TOLERANCE_S for e in outs))
    false = sum(1 for e in outs
                if not any(abs(e["time_s"] - t) <= TOLERANCE_S for t in truth))
    return {"name": clip_name, "placed": placed, "found": len(outs),
            "truth": len(truth), "matched": matched, "false": false,
            "goals": goals}


def print_row(row):
    truth = "unknown" if row["truth"] is None else str(row["truth"])
    matched = "-" if row["matched"] is None else str(row["matched"])
    false = "-" if row["false"] is None else str(row["false"])
    print(f"  {row['name']:>24s} {row['placed']:7d} {row['found']:6d} "
          f"{truth:>9s} {matched:>8s} {false:>6s}")
    for event in row["goals"]:
        print(f"      GOAL at t={event['time_s']:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--reset", type=float, default=RESET_SECONDS,
                    help="seconds of no placement after which the ball's "
                         "last known side of the line is forgotten; pass a "
                         "very large number to require an inside sighting")
    ap.add_argument("--sweep", action="store_true",
                    help="score several reset values off one anchor pass, "
                         "which is the expensive part")
    args = ap.parse_args()

    if args.check:
        print("A ball walked over each edge of the pitch, to see which "
              "answer comes back.\n")
        ok = synthetic_check()
        print(f"\n   {'all pass' if ok else 'SOMETHING IS WRONG'}")
        return

    resets = [2.0, 4.0, 8.0, 16.0, 1e9] if args.sweep else [args.reset]
    print("The ball leaving the pitch, detected on the anchor's maps and "
          "scored\nagainst the OUT labels that fall inside these clips.\n")

    rng = np.random.default_rng(0)
    collected = {reset: [] for reset in resets}
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info = json.loads((path / "clip.json").read_text())
        ball = detect_shots.ball_track(path)
        if ball.empty:
            continue
        maps = detect_shots.anchors_for(path, info, ball.frame.tolist(), rng,
                                        args.frames)
        duration = info["n_frames"] / info["fps"]
        for reset in resets:
            events, placed = find_ball_events(ball, maps, info["fps"], reset)
            collected[reset].append(score(name, out_dir, events, placed,
                                          duration))

    for reset in resets:
        if len(resets) > 1:
            forget = ("never" if reset > 1e8 else f"{reset:.0f} s")
            print(f"\n=== forget the ball's side after {forget} ===")
        print(f"  {'clip':>24s} {'placed':>7s} {'found':>6s} "
              f"{'labelled':>9s} {'matched':>8s} {'false':>6s}")
        for row in collected[reset]:
            print_row(row)
        scored = [r for r in collected[reset] if r["truth"] is not None]
        if scored:
            print(f"  {'total':>24s} {'':>7s} {'':>6s} "
                  f"{sum(r['truth'] for r in scored):9d} "
                  f"{sum(r['matched'] for r in scored):8d} "
                  f"{sum(r['false'] for r in scored):6d}")

    print(f"\n  Scored at +/-{TOLERANCE_S:.0f} s, the convention the rest of "
          f"this project uses.\n  Three labelled crossings is a small test "
          f"and the only real one available\n  for this family -- the "
          f"footage contains no goals at all.\n")


if __name__ == "__main__":
    main()
