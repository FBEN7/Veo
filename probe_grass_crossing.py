"""The ball leaving the pitch, without a pitch map.

Out of play and set pieces have failed all session for one reason, and it is
not geometry. An anchor comes from the centre circle, the centre circle is
at midfield, the ball goes out at the edges, so the frames that carry a
pitch map and the frames that contain a crossing are anti-correlated. On the
three labelled crossings in this footage the ball is detected 43, 74 and 51
times and placed on the pitch zero times. Tripling the anchor budget found
one of the three. Carrying the map further, in steps, at four spacings and
through a multi-scale walk, found none.

Every one of those attempts was trying to answer "where is the ball on the
pitch" in order to answer "is the ball off the pitch". The second question
is much easier than the first, and this project already computes something
that answers it directly.

## grass_frac

`ball_pitch_filter.annotate_grass_fraction` records, for every ball
candidate on every frame, how much of its surroundings is grass. It exists
to throw out ball detections that are really a sock or a hoarding. It is
also, unmodified, a statement about whether the ball is on the playing
surface -- and unlike the anchor it is available on **100% of ball
detections**, including at the edges, because it is computed from the pixels
around the ball rather than from a map of the pitch.

A ball over the touchline is over the crowd, the hoardings, the technical
area. Its grass fraction collapses. No homography is involved and the
symmetry of the pitch never comes into it, exactly as it never came into
which goal a shot was aimed at.

## What this cannot do, said first

The ball's image position is not its ground position. A ball lofted high
over midfield is seen against the stands and its grass fraction collapses
while it is nowhere near leaving the pitch. That is the confound, it is
structural, and it is why this is measured against labels rather than
asserted.

It also cannot tell a touchline from a goal line, so it says "out" and not
"throw-in or goal kick". The restart classifier needs a position for that
and still needs an anchor.

    python probe_grass_crossing.py [--check]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from detect_ball_events import CLIP_SOURCES, TOLERANCE_S, labelled

# Below this share of grass around it, the ball is not over the pitch.
GRASS_OUT = 0.35

# Readings in a row required, as everywhere else here: one is not evidence.
MIN_SUPPORT = 2

# Two crossings closer together than this are one event seen twice.
MERGE_SECONDS = 3.0


def crossings(ball: pd.DataFrame, threshold: float, min_support: int,
              merge_s: float = MERGE_SECONDS):
    """Frames where the ball stops being over grass, having been over it."""
    rows = ball.dropna(subset=["grass_frac"]).sort_values("frame")
    if rows.empty:
        return []
    times = rows.time_s.to_numpy(dtype=float)
    grass = rows.grass_frac.to_numpy(dtype=float)

    found, was_on = [], True
    for k in range(len(rows)):
        if grass[k] >= threshold:
            was_on = True
            continue
        if not was_on:
            continue
        support = int(np.sum(grass[k:k + min_support] < threshold))
        if support < min_support:
            continue
        was_on = False
        found.append({"time_s": float(times[k]), "grass": float(grass[k]),
                      "support": support})

    merged = []
    for event in found:
        if merged and event["time_s"] - merged[-1]["time_s"] < merge_s:
            continue
        merged.append(event)
    return merged


def score(found, truth):
    matched = sum(1 for t in truth
                  if any(abs(e["time_s"] - t) <= TOLERANCE_S for e in found))
    false = sum(1 for e in found
                if not any(abs(e["time_s"] - t) <= TOLERANCE_S for t in truth))
    return matched, false


def synthetic_check():
    """A ball carried off the grass, and one that merely dips."""
    ok = True
    cases = (
        ("walks off the grass", [1.0, 1.0, 0.9, 0.2, 0.1, 0.05], 1),
        ("stays on the grass", [1.0, 0.9, 1.0, 0.95, 1.0, 0.9], 0),
        ("one low reading", [1.0, 1.0, 0.1, 1.0, 1.0, 0.9], 0),
    )
    for name, series, expect in cases:
        frame = pd.DataFrame({"frame": range(len(series)),
                              "time_s": [k / 25.0 for k in range(len(series))],
                              "grass_frac": series})
        got = len(crossings(frame, GRASS_OUT, MIN_SUPPORT))
        good = got == expect
        ok &= good
        print(f"   {name:<24s} -> {got} crossing(s), expected {expect}  "
              f"{'ok' if good else 'WRONG'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        print("A ball taken off the grass and back, to see what is "
              "called.\n")
        ok = synthetic_check()
        print(f"\n   {'all pass' if ok else 'SOMETHING IS WRONG'}")
        return

    print("Crossings from the grass around the ball, with no pitch map, "
          "against the\nsame labels every other attempt was scored on.\n")
    print(f"  {'clip':>16s} {'grass <':>8s} {'support':>8s} {'rows':>6s} "
          f"{'found':>6s} {'labelled':>9s} {'matched':>8s} {'false':>6s}")

    for out_dir, (source, offset) in CLIP_SOURCES.items():
        path = Path(out_dir)
        grass_path = path / "tracks_grass.parquet"
        if not grass_path.exists():
            continue
        info = json.loads((path / "clip.json").read_text())
        tracks = pd.read_parquet(grass_path)
        ball = tracks[tracks.cls == "ball"]
        if ball.empty or "grass_frac" not in ball.columns:
            continue
        # One reading per frame, the most confident, matching how the rest
        # of this pipeline picks a ball.
        ball = (ball.sort_values("confidence").groupby("frame").tail(1)
                .sort_values("frame").reset_index(drop=True))
        truth = labelled("OUT", source, offset,
                         info["n_frames"] / info["fps"])

        for threshold in (0.2, 0.35, 0.5, 0.7):
            for support in (2, 4):
                found = crossings(ball, threshold, support)
                matched, false = score(found, truth)
                print(f"  {out_dir[-16:]:>16s} {threshold:8.2f} "
                      f"{support:8d} {len(ball):6d} {len(found):6d} "
                      f"{len(truth):9d} {matched:8d} {false:6d}", flush=True)

    print("\n  Every ball detection has a grass fraction, so unlike the "
          "anchor this is\n  available exactly where these events happen. "
          "What it cannot do is tell a\n  lofted ball seen against the "
          "stands from one that has left the pitch.")


if __name__ == "__main__":
    main()
