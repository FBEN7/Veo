"""Play stopping, read from the players instead of the ball.

The frames settle why out of play fails: at Reading's 72.1 s crossing there
is no ball in the picture at all, and the one detection near it sits on a
player's fluorescent boot. A broadcast camera follows the ball, so at the
moment it leaves the pitch the ball is at or past the edge of frame. Every
approach tried so far -- carrying the anchor further, in steps, at four
spacings, through a multi-scale walk, tripling the anchor budget, the grass
around the ball -- is a way of saying *where the ball is*, and there is no
ball to place.

What that frame does contain is twenty-two players who have stopped running,
staff waiting on the touchline and a referee walking. A stoppage is written
all over the players, and player tracks do not need the ball to be in shot.

## The measurement, and what would make it worthless

Collective player motion falls when play stops. So: median player speed per
frame, and a stoppage is where it drops and stays down having been up.

Two things could make that look better than it is, and both are checked.

**The baseline.** Football contains plenty of slow passages that are not
stoppages, so a detector firing on every lull would score well on recall
and be useless. False alarms are reported next to matches throughout.

**The delay.** Play does not stop at the instant the ball crosses; the
players take a second or two to notice. Scoring at +/-2 s, the convention
used everywhere else here, may be tighter than the thing being detected, so
the timing offset of each match is reported rather than hidden inside a
tolerance.

This cannot say where the ball went, so it cannot name a throw-in against a
goal kick. It is a detector of stoppages, which is what a crossing looks
like from the players' side.

    python probe_stoppage.py [--check]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from detect_ball_events import CLIP_SOURCES, TOLERANCE_S, labelled
from src import pixel_scale

# Below this, in metres per second, the pitch has stopped moving. Walking
# pace is about 1.4 m/s and a jog 3; players at a stoppage stand or amble.
STILL_MS = 1.2

# How long it must stay down. A stoppage lasts many seconds; a moment of
# everyone slowing is a lull in play.
STILL_SECONDS = 2.0

# Smoothing before any of this, in seconds. Per-frame speeds are noisy
# enough that a single frame means nothing.
SMOOTH_SECONDS = 0.6

MERGE_SECONDS = 5.0


def crowd_speed(tracks: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Median player speed per frame, in metres per second."""
    players = tracks[tracks.cls == "player"].copy()
    if players.empty:
        return pd.DataFrame(columns=["frame", "time_s", "speed_ms"])
    players = players.sort_values(["track_id", "frame"])
    grouped = players.groupby("track_id")
    players["dx"] = grouped.x.diff()
    players["dy"] = grouped.y.diff()
    players["dt"] = grouped.time_s.diff()
    moving = players[(players.dt > 0) & players.dx.notna()].copy()
    if moving.empty:
        return pd.DataFrame(columns=["frame", "time_s", "speed_ms"])
    moving["speed_ms"] = np.hypot(moving.dx, moving.dy) / moving.dt
    # A track that jumps across the pitch in one frame is a swapped
    # identity, not a sprinter; it would drag the median up at exactly the
    # moments tracking is hardest.
    moving = moving[moving.speed_ms < 12.0]
    per_frame = (moving.groupby("frame")
                 .agg(time_s=("time_s", "first"),
                      speed_ms=("speed_ms", "median"))
                 .reset_index())
    window = max(int(SMOOTH_SECONDS * fps), 1)
    per_frame["speed_ms"] = (per_frame.speed_ms.rolling(window, center=True,
                                                        min_periods=1)
                             .median())
    return per_frame


def stoppages(speed: pd.DataFrame, still_ms: float, still_s: float,
              merge_s: float = MERGE_SECONDS):
    """Where the pitch stops moving, having been moving."""
    if speed.empty:
        return []
    times = speed.time_s.to_numpy(dtype=float)
    values = speed.speed_ms.to_numpy(dtype=float)
    below = values < still_ms

    found, running = [], False
    start = 0
    for k, quiet in enumerate(below):
        if quiet and not running:
            running, start = True, k
        elif not quiet and running:
            running = False
            if times[k - 1] - times[start] >= still_s:
                found.append({"time_s": float(times[start]),
                              "held_s": float(times[k - 1] - times[start])})
    if running and times[-1] - times[start] >= still_s:
        found.append({"time_s": float(times[start]),
                      "held_s": float(times[-1] - times[start])})

    merged = []
    for event in found:
        if merged and event["time_s"] - merged[-1]["time_s"] < merge_s:
            continue
        merged.append(event)
    return merged


def offsets(found, truth):
    """Signed delay from each labelled crossing to the nearest stoppage."""
    out = []
    for when in truth:
        if not found:
            continue
        nearest = min(found, key=lambda e: abs(e["time_s"] - when))
        out.append(nearest["time_s"] - when)
    return out


def synthetic_check():
    """A pitch that stops, one that never does, and one that dips."""
    ok = True
    cases = (("play stops for 5 s", [4.0] * 50 + [0.3] * 125 + [4.0] * 50, 1),
             ("play never stops", [4.0] * 225, 0),
             ("one slow second", [4.0] * 100 + [0.3] * 20 + [4.0] * 105, 0))
    for name, series, expect in cases:
        frame = pd.DataFrame({"frame": range(len(series)),
                              "time_s": [k / 25.0 for k in range(len(series))],
                              "speed_ms": series})
        got = len(stoppages(frame, STILL_MS, STILL_SECONDS))
        good = got == expect
        ok &= good
        print(f"   {name:<24s} -> {got} stoppage(s), expected {expect}  "
              f"{'ok' if good else 'WRONG'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        print("A pitch brought to a halt and let go again.\n")
        ok = synthetic_check()
        print(f"\n   {'all pass' if ok else 'SOMETHING IS WRONG'}")
        return

    print("Stoppages read from player motion, scored against the same OUT "
          "labels\nevery ball-based attempt was scored on.\n")
    print(f"  {'clip':>16s} {'still <':>8s} {'held':>6s} {'found':>6s} "
          f"{'labelled':>9s} {'matched':>8s} {'false':>6s} {'delay':>14s}")

    for out_dir, (source, offset) in CLIP_SOURCES.items():
        path = Path(out_dir)
        if not (path / "tracks_teams.parquet").exists():
            continue
        info = json.loads((path / "clip.json").read_text())
        tracks = pd.read_parquet(path / "tracks_teams.parquet")
        metric, _ = pixel_scale.prepare_tracks_for_events(tracks, None,
                                                          verbose=False)
        speed = crowd_speed(metric, info["fps"])
        truth = labelled("OUT", source, offset,
                         info["n_frames"] / info["fps"])

        for still in (0.8, 1.2, 1.6):
            for held in (1.0, 2.0, 4.0):
                found = stoppages(speed, still, held)
                matched = sum(1 for t in truth
                              if any(abs(e["time_s"] - t) <= TOLERANCE_S
                                     for e in found))
                false = sum(1 for e in found
                            if not any(abs(e["time_s"] - t) <= TOLERANCE_S
                                       for t in truth))
                delays = offsets(found, truth)
                shown = (", ".join(f"{d:+.0f}" for d in delays)
                         if delays else "-")
                print(f"  {out_dir[-16:]:>16s} {still:8.1f} {held:6.0f} "
                      f"{len(found):6d} {len(truth):9d} {matched:8d} "
                      f"{false:6d} {shown:>14s}", flush=True)

    print("\n  'delay' is seconds from each labelled crossing to the nearest "
          "stoppage,\n  signed, and it is the number to read first: play "
          "stops when the players\n  notice, not when the ball crosses, so a "
          "consistent positive offset is the\n  detector working and a "
          "scatter is it not.")


if __name__ == "__main__":
    main()
