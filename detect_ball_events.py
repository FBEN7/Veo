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

# Where the labelled clips sit in their matches, from the file names:
# stoke_000520 is 5:20 into the match, reading_5115 is 51:15.
CLIP_OFFSETS = {"output_soccernet_w3": 320.0,
                "output_soccernet_reading": 3075.0}

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


def find_ball_events(ball: pd.DataFrame, maps, fps: float):
    """Frames where the ball crosses the edge of the pitch."""
    placed = []
    for row in ball.itertuples():
        point = detect_shots.to_pitch(maps, row.frame, row.px, row.py)
        if point is not None:
            placed.append((int(row.frame), float(row.time_s),
                           float(point[0]), float(point[1])))

    events, was_in = [], True
    for frame, when, x, y in placed:
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


def labelled_outs(offset_s: float, duration_s: float):
    """OUT labels falling inside a clip, in clip time."""
    uploads = Path("/root/.claude/uploads/"
                   "cd4d7e67-1dd4-5fa1-975c-2f5b3217663b")
    out = []
    for path in sorted(uploads.glob("*Labels-ball.json")):
        blob = json.loads(path.read_text())
        for row in blob["annotations"]:
            if row["label"] != "OUT":
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        print("A ball walked over each edge of the pitch, to see which "
              "answer comes back.\n")
        ok = synthetic_check()
        print(f"\n   {'all pass' if ok else 'SOMETHING IS WRONG'}")
        return

    print("The ball leaving the pitch, detected on the anchor's maps and "
          "scored\nagainst the OUT labels that fall inside these clips.\n")
    print(f"  {'clip':>24s} {'placed':>7s} {'found':>6s} {'labelled':>9s} "
          f"{'matched':>8s} {'false':>6s}")

    rng = np.random.default_rng(0)
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
        events, placed = find_ball_events(ball, maps, info["fps"])
        outs = [e for e in events if e["event_type"] == "out_of_play"]
        goals = [e for e in events if e["event_type"] == "goal"]

        offset = CLIP_OFFSETS.get(out_dir)
        if offset is None:
            print(f"  {name:>24s} {placed:7d} {len(outs):6d} "
                  f"{'unknown':>9s} {'-':>8s} {'-':>6s}")
            continue
        duration = info["n_frames"] / info["fps"]
        truth = labelled_outs(offset, duration)
        matched = sum(1 for t in truth
                      if any(abs(e["time_s"] - t) <= TOLERANCE_S
                             for e in outs))
        false = sum(1 for e in outs
                    if not any(abs(e["time_s"] - t) <= TOLERANCE_S
                               for t in truth))
        print(f"  {name:>24s} {placed:7d} {len(outs):6d} {len(truth):9d} "
              f"{matched:8d} {false:6d}")
        for event in goals:
            print(f"      GOAL at t={event['time_s']:.1f}s")

    print(f"\n  Scored at +/-{TOLERANCE_S:.0f} s, the convention the rest of "
          f"this project uses.\n  Three labelled crossings is a small test "
          f"and the only real one available\n  for this family -- the "
          f"footage contains no goals at all.")


if __name__ == "__main__":
    main()
