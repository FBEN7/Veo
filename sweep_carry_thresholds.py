#!/usr/bin/env python3
"""Sweep what makes a possession spell count as a carry.

Carry is suggestive but unestablished: above chance at +/-1 s on two of four
windows, and over-produced everywhere -- 26 to 39 emitted against 16 to 25
real. Merging near-duplicates did not help, so they are not fragments of one
drive. Re-timestamping did not help, so it is not where in the spell the event
is placed.

What is left is the definition. SoccerNet's DRIVE is purposeful progression
with the ball; our carry is any possession spell lasting CARRY_MIN_TIME_S and
covering CARRY_MIN_DISTANCE_M. Those two constants are the whole of the
distinction, and neither has ever been swept. A player standing on the ball
for a second satisfies them if the ball drifts two metres.

Scored on all four windows, three from one match and one from another, with
mean F1 and the count of windows above chance reported together. The lesson
from the merge sweep is that a value winning on the windows it was chosen from
proves nothing; only a value that helps on average, across matches, is worth
taking.

Usage:
    python sweep_carry_thresholds.py
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

import score_soccernet as sc
import sweep_pass_thresholds as sw
from sweep_carry_timestamp import WINDOWS
from src import events as ev

DISTANCE_GRID = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
TIME_GRID = (0.4, 0.8, 1.2, 2.0)


def score_carry(tracks, absolute, truths, duration, tolerance, n_draws):
    with contextlib.redirect_stdout(io.StringIO()):
        events = ev.detect_events(tracks, absolute_pitch=absolute)

    preds = sorted(e["timestamp_s"] for e in events
                   if sc.OURS_TO_GROUP.get(e["event_type"]) == "carry")
    if not preds or not truths:
        return dict(n=len(preds), precision=0.0, recall=0.0, f1=0.0,
                    p=float("nan"))

    pairs, _, _ = sc.match_one_to_one(preds, truths, tolerance)
    prec = len(pairs) / len(preds)
    rec = len(pairs) / len(truths)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    rng = np.random.default_rng(1)
    draws = np.empty(n_draws)
    for k in range(n_draws):
        rand = sorted(rng.uniform(0.0, duration, len(preds)))
        pr, _, _ = sc.match_one_to_one(rand, truths, tolerance)
        pp = len(pr) / len(preds)
        rr = len(pr) / len(truths)
        draws[k] = 2 * pp * rr / (pp + rr) if (pp + rr) else 0.0

    return dict(n=len(preds), precision=prec, recall=rec, f1=f1,
                p=float((draws >= f1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=1.0)
    ap.add_argument("--draws", type=int, default=500)
    args = ap.parse_args()

    prepared = {}
    for name, cfg in WINDOWS.items():
        tracks, absolute, prof = sw.build_tracks(cfg["dir"])
        duration = prof.n_frames / prof.fps
        truths = sorted(l["t"] for l in sc.load_window(
            str(cfg["labels"]), cfg["offset"], duration)
            if l["group"] == "carry")
        prepared[name] = (tracks, absolute, truths, duration)
        print(f"{name}: {len(truths)} carry labels")

    original = (ev.CARRY_MIN_DISTANCE_M, ev.CARRY_MIN_TIME_S)
    results = {}

    print(f"\ncarry gates, tolerance +/-{args.tolerance:g}s, "
          f"{args.draws} draws")
    print(f"{'min_dist':>9} {'min_time':>9} {'mean n':>7} {'mean P':>7} "
          f"{'mean R':>7} {'mean F1':>8} {'n<0.05':>7}   per-window F1")

    for dist in DISTANCE_GRID:
        for tmin in TIME_GRID:
            ev.CARRY_MIN_DISTANCE_M = dist
            ev.CARRY_MIN_TIME_S = tmin
            f1s, ps, ns, precs, recs = [], [], [], [], []
            for name, (tracks, absolute, truths, duration) in prepared.items():
                r = score_carry(tracks, absolute, truths, duration,
                                args.tolerance, args.draws)
                results[(dist, tmin, name)] = r
                f1s.append(r["f1"])
                ps.append(r["p"])
                ns.append(r["n"])
                precs.append(r["precision"])
                recs.append(r["recall"])
            n_sig = sum(1 for x in ps if x < 0.05)
            print(f"{dist:>9.1f} {tmin:>9.1f} {np.mean(ns):>7.0f} "
                  f"{np.mean(precs):>7.2f} {np.mean(recs):>7.2f} "
                  f"{np.mean(f1s):>8.3f} {n_sig:>5}/4   "
                  + " ".join(f"{x:.2f}" for x in f1s))

    ev.CARRY_MIN_DISTANCE_M, ev.CARRY_MIN_TIME_S = original

    # Rank by windows above chance first, then mean F1 -- a setting that is
    # significant everywhere beats one that is excellent in two places and
    # absent in two others.
    summary = {}
    for (dist, tmin, name), r in results.items():
        key = (dist, tmin)
        summary.setdefault(key, []).append(r)
    ranked = sorted(
        summary.items(),
        key=lambda kv: (sum(1 for r in kv[1] if r["p"] < 0.05),
                        np.mean([r["f1"] for r in kv[1]])),
        reverse=True)

    print("\n" + "=" * 74)
    print("best settings, ranked by windows above chance then mean F1")
    print("=" * 74)
    current = (ev.CARRY_MIN_DISTANCE_M, ev.CARRY_MIN_TIME_S)
    for (dist, tmin), rs in ranked[:5]:
        mark = "  <- current" if (dist, tmin) == current else ""
        print(f"  dist {dist:>5.1f} m, time {tmin:>4.1f} s  "
              f"{sum(1 for r in rs if r['p'] < 0.05)}/4 above chance, "
              f"mean F1 {np.mean([r['f1'] for r in rs]):.3f}{mark}")
    cur = summary.get(current)
    if cur:
        print(f"\n  current setting: "
              f"{sum(1 for r in cur if r['p'] < 0.05)}/4 above chance, "
              f"mean F1 {np.mean([r['f1'] for r in cur]):.3f}")

    Path("carry_threshold_sweep.json").write_text(json.dumps(
        {f"{d}_{t}_{n}": v for (d, t, n), v in results.items()}, indent=2))
    print("\nwritten to carry_threshold_sweep.json")


if __name__ == "__main__":
    main()
