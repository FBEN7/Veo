"""Did requiring re-identification to respect the kit break the physical stats?

`track_reid.merge_fragments` now refuses to join two fragments whose team
differs, because joining across kits was relabelling a rejected non-player's
fragment with an accepted player's id and putting it back into possession.
That was measured on events, where it changed nothing and halved the
contamination.

The cost was flagged and not measured: refusing those joins leaves more, and
shorter, tracks -- 89 became 98 on one window, 128 became 143 on another --
and `stats.physical_stats` computes distance covered and top speed per track.
A player split across two ids has their distance split with them, and a
fragment under 20 frames is dropped from the table entirely.

There is no ground truth for distance covered here, so this is not accuracy.
What it can establish is the size and the direction of the change, and
whether the totals stay inside what football allows.

One reason to expect an improvement rather than a regression: a cross-kit
merge joined two *different people*. The straight-line jump between them was
being counted as distance, and the speed implied by that jump as speed.

    python check_physical_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.track_reid as reid
from analyse_pass_outcome import WINDOWS
from score_soccernet import run_pipeline
from src.stats import physical_stats, MAX_SPEED_KMH, SPRINT_KMH

CLIP_MINUTES = 1.5


def without_team_constraint():
    """Restore the pre-fix behaviour by blanking the team inside matching.

    Monkeypatched rather than exposed as an option, so the comparison costs
    the shipped code nothing.
    """
    original = reid._best_match

    def patched(frag, i, max_gap_frames, fps):
        saved = frag.team
        frag.team = np.array([""] * len(saved), dtype=object)
        try:
            return original(frag, i, max_gap_frames, fps)
        finally:
            frag.team = saved

    return original, patched


def summarise(stats: pd.DataFrame) -> dict:
    if stats.empty:
        return dict(n=0)
    return dict(
        n=len(stats),
        total_km=stats.distance_m.sum() / 1000.0,
        median_m=float(stats.distance_m.median()),
        median_top=float(stats.top_speed_kmh.median()),
        max_top=float(stats.top_speed_kmh.max()),
        p95_top=float(np.percentile(stats.top_speed_kmh, 95)),
        sprints=int(stats.n_sprints.sum()),
        median_minutes=float(stats.minutes_tracked.median()),
    )


def run(window, patched=False):
    out_dir = next(w[1] for w in WINDOWS if w[0] == window)
    clip = json.loads((Path(out_dir) / "clip.json").read_text())["path"]

    original, patch = without_team_constraint()
    if patched:
        reid._best_match = patch
    try:
        _, _, metric = run_pipeline(clip, Path(out_dir), return_tracks=True)
    finally:
        reid._best_match = original

    players = metric[metric.cls == "player"]
    stats = physical_stats(metric)
    return summarise(stats), players.track_id.nunique()


def main():
    rows = []
    for window, *_ in WINDOWS:
        before, n_before = run(window, patched=True)
        after, n_after = run(window, patched=False)
        rows.append((window, n_before, n_after, before, after))

    print(f"Per-track physical stats over {CLIP_MINUTES:.1f}-minute clips.\n"
          "'before' joins fragments across kits, 'after' refuses to.\n")

    def line(label, key, fmt="{:.2f}"):
        print(f"  {label:26s}", end="")
        for _, _, _, b, a in rows:
            if b.get("n") and a.get("n"):
                print(f"  {fmt.format(b[key]):>8s} -> {fmt.format(a[key]):<8s}",
                      end="")
        print()

    print("  " + " " * 26 + "".join(
        f"  {w.split()[0]:>18s}" for w, *_ in rows))
    print(f"  {'player tracks':26s}" + "".join(
        f"  {nb:>8d} -> {na:<8d}" for _, nb, na, _, _ in rows))
    line("tracks in the table", "n", "{:.0f}")
    line("total distance (km)", "total_km")
    line("median per track (m)", "median_m", "{:.0f}")
    line("median minutes tracked", "median_minutes")
    line("median top speed (km/h)", "median_top")
    line("95th pct top speed", "p95_top")
    line("max top speed", "max_top")
    line("sprint frames", "sprints", "{:.0f}")

    print(f"\nA top speed is clipped at {MAX_SPEED_KMH:.0f} km/h and a sprint "
          f"counted above {SPRINT_KMH:.0f}.")
    print("Football: an elite outfielder covers 10-12 km per 90 minutes and "
          "tops out at 30-36 km/h.")

    per90 = []
    for w, _, _, b, a in rows:
        if b.get("n") and a.get("n"):
            # Two teams on the pitch, so the per-team total is half the sum,
            # scaled from the clip length to 90 minutes and divided by the
            # eleven players a team fields.
            scale = (90.0 / CLIP_MINUTES) / 11.0 / 2.0
            per90.append((w, b["total_km"] * scale, a["total_km"] * scale))
    print("\nImplied km per player per 90 (crude: assumes 11 a side all "
          "tracked):")
    for w, b, a in per90:
        print(f"  {w:16s} {b:6.1f} -> {a:6.1f}")


if __name__ == "__main__":
    main()
