"""Throw-ins, corners, goal kicks and penalties, from where play restarts.

A set piece is not a thing to be recognised in a picture. It is a stoppage
followed by the ball being put back into play from a particular place, and
the place is what names it. With pitch coordinates the classification is
almost a lookup:

    on a touchline          throw-in
    in a corner arc         corner
    inside the six-yard box goal kick
    on the penalty spot     penalty

Nothing here needs a new detector. The ball leaving the pitch is already
found by `detect_ball_events`; this waits for it to come back and asks where
from.

## Which ones this can and cannot do

The four above all follow the ball going out, so the stoppage is visible in
the ball track and the restart position separates them. Corners and goal
kicks are the same crossing -- over a goal line, outside the posts -- and
differ only in who touched it last, which is why the restart position rather
than the crossing is what decides.

A free kick is different in kind. Nothing about the ball leaving the pitch
precedes it; it follows a foul, and a foul is not visible in a ball track at
all. What can be seen is the shape -- the ball stationary, then struck -- and
that is also what a throw-in, a corner and a goal kick look like, so on its
own it names nothing. Free kicks are therefore reported only as "a restart
that was not any of the others", which is honest about being a residue
rather than a detection.

## Measured on the three that exist

The labelled matches mark THROW IN, and three fall inside the clips that
exist: one in the Stoke clip and two in Reading. The same three crossings
that let out-of-play be measured let their restarts be measured too, which
makes this the second event family in this project with real positives
behind it rather than a synthetic stand-in.

    python detect_set_pieces.py [--check]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import detect_ball_events as ball_events
import detect_shots
from probe_pitch_lines import CLIPS
from src import pitch_model as pm

# How long after the ball goes out a restart may be, before this gives up.
# A throw-in takes twenty seconds when nobody is hurrying.
MAX_RESTART_S = 45.0

# How close to each landmark the restart has to be. Generous against the
# anchor's 0.7 m and the fact that nobody takes a throw-in from exactly on
# the line.
TOUCHLINE_M = 3.0
CORNER_M = 4.0
SPOT_M = 2.0

# The ball has to actually be put into play, not merely be lying there.
RESTART_MIN_SPEED_MS = 3.0


def classify(x, y):
    """What kind of restart happens from here?"""
    near_goal_line = min(x, pm.PITCH_LENGTH_M - x)
    near_touchline = min(y, pm.PITCH_WIDTH_M - y)

    if near_goal_line <= CORNER_M and near_touchline <= CORNER_M:
        return "corner"
    for spot in (pm.PENALTY_SPOT_DEPTH_M,
                 pm.PITCH_LENGTH_M - pm.PENALTY_SPOT_DEPTH_M):
        if abs(x - spot) <= SPOT_M and abs(y - pm.PITCH_WIDTH_M / 2) <= SPOT_M:
            return "penalty"
    if (near_goal_line <= pm.SIX_YARD_DEPTH_M + 1.0
            and abs(y - pm.PITCH_WIDTH_M / 2) <= pm.SIX_YARD_WIDTH_M / 2 + 1.0):
        return "goal kick"
    if near_touchline <= TOUCHLINE_M:
        return "throw in"
    return "free kick"


def find_restarts(ball: pd.DataFrame, maps, fps: float, stoppages):
    """Where the ball comes back into play after each stoppage."""
    placed = []
    for row in ball.itertuples():
        point = detect_shots.to_pitch(maps, row.frame, row.px, row.py)
        if point is not None:
            placed.append((float(row.time_s), float(point[0]),
                           float(point[1])))
    if len(placed) < 2:
        return []

    times = np.array([p[0] for p in placed])
    xs = np.array([p[1] for p in placed])
    ys = np.array([p[2] for p in placed])

    out = []
    for stoppage in stoppages:
        after = np.flatnonzero((times > stoppage["time_s"]) &
                               (times <= stoppage["time_s"] + MAX_RESTART_S))
        if after.size < 2:
            continue
        # The first moment the ball is back on the pitch AND moving: lying
        # still by the corner flag is not the restart, being struck is.
        found = None
        for i, j in zip(after[:-1], after[1:]):
            gap = times[j] - times[i]
            if gap <= 0:
                continue
            if ball_events.outside(xs[i], ys[i]) is not None:
                continue
            speed = float(np.hypot(xs[j] - xs[i], ys[j] - ys[i]) / gap)
            if speed >= RESTART_MIN_SPEED_MS:
                found = i
                break
        if found is None:
            continue
        kind = classify(xs[found], ys[found])
        out.append({"time_s": float(times[found]), "event_type": kind,
                    "x": float(xs[found]), "y": float(ys[found]),
                    "after_stoppage_s": float(times[found]
                                              - stoppage["time_s"])})
    return out


def synthetic_check():
    """One restart from each landmark, to see it named."""
    cases = (("on the touchline", 40.0, 0.5, "throw in"),
             ("in the corner arc", 1.0, 1.0, "corner"),
             ("in the six-yard box", 3.0, 34.0, "goal kick"),
             ("on the penalty spot", 11.0, 34.0, "penalty"),
             ("out by the D", 25.0, 34.0, "free kick"))
    ok = True
    for name, x, y, expect in cases:
        got = classify(x, y)
        good = got == expect
        ok &= good
        print(f"   {name:<22s} -> {got:<10s} {'ok' if good else 'WRONG'}")
    return ok





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int,
                    default=detect_shots.ANCHOR_FRAMES)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--clip", default=None,
                    help="substring of the output directory, to score one "
                         "clip instead of all of them")
    args = ap.parse_args()

    if args.check:
        print("A restart from each landmark on the pitch, to see it "
              "named.\n")
        ok = synthetic_check()
        print(f"\n   {'all pass' if ok else 'SOMETHING IS WRONG'}")
        return

    print("Restarts after the ball goes out, named by where it comes back "
          "from,\nand scored against the THROW IN labels inside these "
          "clips.\n")
    # Out-of-play is scored here too rather than in its own run: the
    # stoppages are already computed to find the restarts, and the anchors
    # behind them are the expensive part.
    print(f"  {'clip':>22s} {'OUT found':>10s} {'OUT true':>9s} "
          f"{'matched':>8s} {'false':>6s} {'restarts':>9s} {'kinds':>24s} "
          f"{'throws':>7s} {'ok':>4s}")

    rng = np.random.default_rng(0)
    for name, out_dir in CLIPS:
        if args.clip and args.clip not in out_dir:
            continue
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info = json.loads((path / "clip.json").read_text())
        ball = detect_shots.ball_track(path)
        if ball.empty:
            continue
        maps = detect_shots.anchors_for(path, info, ball.frame.tolist(), rng,
                                        args.frames)
        stoppages, _ = ball_events.find_ball_events(ball, maps, info["fps"])
        stoppages = [s for s in stoppages if s["event_type"] == "out_of_play"]
        restarts = find_restarts(ball, maps, info["fps"], stoppages)

        from collections import Counter
        kinds = Counter(r["event_type"] for r in restarts)
        source_offset = ball_events.CLIP_SOURCES.get(out_dir)
        if source_offset is None:
            print(f"  {name:>22s} {len(stoppages):10d} {'?':>9s} {'-':>8s} "
                  f"{'-':>6s} {len(restarts):9d} {str(dict(kinds)):>24s} "
                  f"{'-':>7s} {'-':>4s}")
            continue
        source, offset = source_offset
        duration = info["n_frames"] / info["fps"]
        outs_true = ball_events.labelled("OUT", source, offset, duration)
        out_matched = sum(1 for t in outs_true
                          if any(abs(s["time_s"] - t)
                                 <= ball_events.TOLERANCE_S
                                 for s in stoppages))
        out_false = sum(1 for s in stoppages
                        if not any(abs(s["time_s"] - t)
                                   <= ball_events.TOLERANCE_S
                                   for t in outs_true))
        truth = ball_events.labelled("THROW IN", source, offset, duration)
        throws = [r for r in restarts if r["event_type"] == "throw in"]
        matched = sum(1 for t in truth
                      if any(abs(r["time_s"] - t) <= ball_events.TOLERANCE_S
                             for r in throws))
        print(f"  {name:>22s} {len(stoppages):10d} {len(outs_true):9d} "
              f"{out_matched:8d} {out_false:6d} {len(restarts):9d} "
              f"{str(dict(kinds)):>24s} {len(truth):7d} {matched:4d}")

    print("\n  A restart is only looked for after the ball has been seen to "
          "leave, so a\n  missed crossing is a missed set piece -- the two "
          "measurements are linked.\n  Free kicks are a residue here, not a "
          "detection: nothing in a ball track\n  shows a foul.")


if __name__ == "__main__":
    main()
