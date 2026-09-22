"""Choosing a top-speed statistic that measures the player, not the tracker.

`stats.physical_stats` reports each track's **maximum** frame-to-frame speed
after a 5-frame median smooth, discarding anything above 40 km/h. A maximum
over a noisy signal measures the noise, and it does: a track's maximum is
about double its own 95th percentile, and 7-17% of tracks reach the 40 km/h
discard threshold. See `VEO_FOOTAGE.md`.

There is no ground-truth speed for this footage, so the candidates cannot be
scored for accuracy. They can be scored for **reliability**, which is the
property actually in question. A statistic that measures a player should give
the same answer on the first half of their track as on the second; one that
measures jitter should not. Each long track is therefore split in two and the
halves compared.

Reliability alone is not enough -- a statistic that always returned 7 would be
perfectly reliable -- so three things are reported together:

    split-half r   agreement between a track's two halves
    spread         the population still has to separate quick players from
                   slow ones, so a collapsed distribution is a failure
    at the cap     share of tracks reaching 40 km/h, which nothing on a
                   football pitch should

Candidates: the maximum as shipped, high percentiles of per-frame speed, and
the fastest average over a rolling window of a given duration -- the last
being what GPS tracking in sport actually reports, because peak speed is
sustained by definition.

    python sweep_top_speed.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_pass_outcome import WINDOWS
from score_soccernet import run_pipeline
from src.stats import MAX_SPEED_KMH

# Rolling windows tried, in seconds. Below about 0.2 s a window spans three
# frames and inherits the jitter it is meant to remove; above 1 s a genuine
# sprint is averaged with the deceleration after it.
WINDOWS_S = (0.2, 0.3, 0.5, 0.8, 1.2)

PERCENTILES = (90, 95, 99)

# A track must last at least this long to be split in half and still leave
# each half able to hold the longest window tried.
MIN_TRACK_S = 3.0

SMOOTH_FRAMES = 5


def _series(g: pd.DataFrame):
    g = g.sort_values("time_s")
    x = g.x.rolling(SMOOTH_FRAMES, min_periods=1, center=True).median().to_numpy()
    y = g.y.rolling(SMOOTH_FRAMES, min_periods=1, center=True).median().to_numpy()
    return x, y, g.time_s.to_numpy(dtype=float)


def per_frame_speeds(x, y, t):
    dt = np.diff(t)
    ok = dt > 0
    sp = np.hypot(np.diff(x), np.diff(y))[ok] / dt[ok] * 3.6
    return sp[np.isfinite(sp) & (sp < MAX_SPEED_KMH)]


def window_speed(x, y, t, window_s: float, pct: float = 100.0) -> float:
    """Speed sustained over `window_s`, at the given percentile of windows.

    Displacement between two points divided by the time between them, so a
    single mis-placed frame inflates the result by its own error over the
    window rather than over one frame interval.

    ``pct`` matters more than the window did. Taking the *maximum* over
    windows is still an extreme-value statistic -- a bad frame corrupts every
    window containing it, and the maximum then selects the worst of them --
    which is why the 100th percentile scores worse for reliability than a
    plain per-frame percentile does.
    """
    if len(t) < 2 or (t[-1] - t[0]) < window_s:
        return float("nan")
    # For each start, the first sample at least window_s later.
    j = np.searchsorted(t, t + window_s, side="left")
    ok = j < len(t)
    if not ok.any():
        return float("nan")
    i = np.nonzero(ok)[0]
    j = j[ok]
    dt = t[j] - t[i]
    good = dt > 0
    if not good.any():
        return float("nan")
    sp = np.hypot(x[j] - x[i], y[j] - y[i])[good] / dt[good] * 3.6
    sp = sp[np.isfinite(sp) & (sp < MAX_SPEED_KMH)]
    if not sp.size:
        return float("nan")
    return float(np.percentile(sp, pct)) if pct < 100 else float(sp.max())


def statistics_for(x, y, t) -> dict[str, float]:
    out = {}
    sp = per_frame_speeds(x, y, t)
    out["max (shipped)"] = float(sp.max()) if sp.size else np.nan
    for p in PERCENTILES:
        out[f"p{p} per-frame"] = (float(np.percentile(sp, p))
                                  if sp.size else np.nan)
    for w in WINDOWS_S:
        out[f"win {w:.1f}s max"] = window_speed(x, y, t, w)
    for w in (0.3, 0.5, 0.8):
        for pct in (90, 95):
            out[f"win {w:.1f}s p{pct}"] = window_speed(x, y, t, w, pct)
    return out


def collect():
    """Every long track, with each statistic on its two halves and whole."""
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
            whole = statistics_for(x, y, t)
            first = statistics_for(x[:mid], y[:mid], t[:mid])
            second = statistics_for(x[mid:], y[mid:], t[mid:])
            rows.append(dict(window=window, track_id=int(tid),
                             whole=whole, first=first, second=second))
    return rows


def main():
    rows = collect()
    if not rows:
        print("no tracks long enough")
        return
    names = list(rows[0]["whole"])
    print(f"{len(rows)} tracks of at least {MIN_TRACK_S:.0f}s across "
          f"{len({r['window'] for r in rows})} windows\n")

    print(f"  {'statistic':16s} {'split-half r':>12s} {'median':>8s} "
          f"{'p95':>7s} {'max':>7s} {'spread':>7s} {'at cap':>7s}")
    for n in names:
        a = np.array([r["first"][n] for r in rows], dtype=float)
        b = np.array([r["second"][n] for r in rows], dtype=float)
        w = np.array([r["whole"][n] for r in rows], dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        r = (float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 3
             else float("nan"))
        wf = w[np.isfinite(w)]
        if not wf.size:
            continue
        # Spread as the interquartile range: a statistic that cannot tell a
        # quick player from a slow one is useless however reliable it is.
        spread = float(np.percentile(wf, 75) - np.percentile(wf, 25))
        cap = float((wf >= MAX_SPEED_KMH - 1.0).mean())
        print(f"  {n:16s} {r:12.2f} {np.median(wf):8.1f} "
              f"{np.percentile(wf, 95):7.1f} {wf.max():7.1f} "
              f"{spread:7.1f} {cap:6.0%}")

    print("\nFootball: an outfielder's peak is 30-36 km/h, and a nine-second "
          "fragment\nusually contains no sprint at all, so a sane median sits "
          "well below that.")


if __name__ == "__main__":
    main()
