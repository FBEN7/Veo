"""Choosing a sprint count that counts sprints.

`stats.physical_stats` reports `n_sprints` as the number of *frames* whose
frame-to-frame speed exceeds 20 km/h. That is wrong in three separate ways:

  * it counts frames, not sprints. At 25 fps a single one-second sprint
    scores 25, and the figure printed in a report as "sprints" is really
    "frames spent above 20 km/h" -- a duration wearing a count's name;
  * it thresholds the per-frame speed, which is the same noisy signal whose
    tail made the old top-speed statistic unusable (see `VEO_FOOTAGE.md`);
  * it is not normalised by how long the track was followed, so a player
    tracked twice as long scores twice as many.

The same test that chose the top-speed statistic applies here. There is no
ground truth, but a statistic measuring a player agrees between the first
half of their track and the second. Counts are compared as a rate per minute,
since each half covers half the time.

Candidates vary two things: the speed signal the threshold is applied to, and
whether a sprint must be sustained. A sprint that must last a fifth of a
second cannot be a single mis-placed frame.

Plausibility anchor: an outfielder makes roughly 30-60 runs above 20 km/h in
a match, so about 0.3-0.7 per minute, and most nine-second fragments should
contain none at all.

    python sweep_sprints.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_pass_outcome import WINDOWS
from score_soccernet import run_pipeline
from src.stats import MAX_SPEED_KMH, SPRINT_KMH
from sweep_top_speed import _series, MIN_TRACK_S

# Spans over which speed is measured for the threshold, in seconds. 0.0 means
# frame to frame, as shipped.
SPANS_S = (0.0, 0.4)

# How long speed must stay above the threshold to count as one sprint.
MIN_DURATIONS_S = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)


def speed_series(x, y, t, span_s: float):
    """Speed at each sample, measured over a centred span.

    Returns (speed, time). A span of zero is the frame-to-frame difference the
    shipped code uses; anything larger divides the displacement across the
    span by the time it took, so one bad frame contributes its error over the
    whole span rather than over a single frame interval.
    """
    if len(t) < 3:
        return np.array([]), np.array([])
    if span_s <= 0:
        dt = np.diff(t)
        ok = dt > 0
        sp = np.hypot(np.diff(x), np.diff(y))[ok] / dt[ok] * 3.6
        return sp, t[1:][ok]

    half = span_s / 2.0
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right") - 1
    ok = hi > lo
    if not ok.any():
        return np.array([]), np.array([])
    lo, hi, tt = lo[ok], hi[ok], t[ok]
    dt = t[hi] - t[lo]
    good = dt > 0
    sp = np.hypot(x[hi] - x[lo], y[hi] - y[lo])[good] / dt[good] * 3.6
    return sp, tt[good]


def count_sprints(sp, tt, min_duration_s: float) -> int:
    """Contiguous stretches above the sprint threshold, long enough to count."""
    if sp.size == 0:
        return 0
    over = (sp > SPRINT_KMH) & (sp < MAX_SPEED_KMH)
    if not over.any():
        return 0

    # Boundaries of each run of True.
    edges = np.diff(np.concatenate(([0], over.view(np.int8), [0])))
    starts = np.nonzero(edges == 1)[0]
    ends = np.nonzero(edges == -1)[0] - 1
    if min_duration_s <= 0:
        return int(len(starts))
    return int(sum(1 for s, e in zip(starts, ends)
                   if (tt[e] - tt[s]) >= min_duration_s))


def statistics_for(x, y, t) -> dict[str, float]:
    """Sprint rate per minute under each candidate, plus the shipped figure."""
    out = {}
    minutes = max((t[-1] - t[0]) / 60.0, 1e-9)

    sp0, tt0 = speed_series(x, y, t, 0.0)
    over = (sp0 > SPRINT_KMH) & (sp0 < MAX_SPEED_KMH)
    out["frames>20 (shipped)"] = float(over.sum()) / minutes

    for span in SPANS_S:
        sp, tt = speed_series(x, y, t, span)
        for dur in MIN_DURATIONS_S:
            label = (f"span {span:.1f}s, hold {dur:.1f}s")
            out[label] = count_sprints(sp, tt, dur) / minutes
    return out


def main():
    rows = []
    for window, out_dir, *_ in WINDOWS:
        clip = json.loads((Path(out_dir) / "clip.json").read_text())["path"]
        _, _, metric = run_pipeline(clip, Path(out_dir), return_tracks=True)
        players = metric[(metric.cls == "player")
                         & (metric.team.isin(["team_A", "team_B"]))]
        for tid, g in players.groupby("track_id"):
            if len(g) < 20:
                continue
            x, y, t = _series(g)
            if (t[-1] - t[0]) < MIN_TRACK_S:
                continue
            mid = len(t) // 2
            rows.append((statistics_for(x, y, t),
                         statistics_for(x[:mid], y[:mid], t[:mid]),
                         statistics_for(x[mid:], y[mid:], t[mid:])))

    if not rows:
        print("no tracks long enough")
        return
    names = list(rows[0][0])
    print(f"{len(rows)} tracks of at least {MIN_TRACK_S:.0f}s, "
          "sprint rate per minute\n")
    print(f"  {'candidate':22s} {'split-half r':>12s} {'median':>8s} "
          f"{'mean':>7s} {'p95':>7s} {'zero':>6s}")
    for n in names:
        a = np.array([r[1][n] for r in rows], float)
        b = np.array([r[2][n] for r in rows], float)
        w = np.array([r[0][n] for r in rows], float)
        ok = np.isfinite(a) & np.isfinite(b)
        r = (float(np.corrcoef(a[ok], b[ok])[0, 1])
             if ok.sum() > 3 and a[ok].std() > 0 and b[ok].std() > 0
             else float("nan"))
        print(f"  {n:22s} {r:12.2f} {np.median(w):8.2f} {w.mean():7.2f} "
              f"{np.percentile(w, 95):7.2f} {(w == 0).mean():5.0%}")

    print(f"\nThreshold {SPRINT_KMH:.0f} km/h. Football: roughly 0.3-0.7 runs "
          "above it per minute,\nand most nine-second fragments contain none.")


if __name__ == "__main__":
    main()
