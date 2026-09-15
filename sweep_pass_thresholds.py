#!/usr/bin/env python3
"""Choose the pass gates by measurement, and check they transfer.

The emitted passes had a median travel of 0.6 m over 0.12 s, against 23 real
passes in the window and 56 emitted. Raising the gates will obviously improve
precision on the window used to choose them; the question is whether the
values transfer, which is the only thing that makes them a fix rather than a
fit.

So each combination is scored on both windows, and the value chosen on one is
reported against the other. A gate that looks excellent on window 1 and
mediocre on window 2 was fitted to window 1.

Usage:
    python sweep_pass_thresholds.py            # sweep and cross-check
    python sweep_pass_thresholds.py --final    # one setting, full 2000 draws
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import score_soccernet as sc
from src import (pixel_scale, ball_selection, track_reid, ball_pitch_filter,
                 player_filter, camera_motion, auto_tune)
from src import events as ev

WINDOWS = {
    "w1 (30:20)": dict(dir="output_soccernet", offset=1820.0),
    "w2 (66:50)": dict(dir="output_soccernet_w2", offset=4010.0),
}

LABELS = ("/root/.claude/uploads/cd4d7e67-1dd4-5fa1-975c-2f5b3217663b/"
          "34498b39-Labels-ball.json")

DIST_GRID = (0.3, 2.0, 3.0, 5.0, 8.0)
INTERVAL_GRID = (0.0, 0.1, 0.2, 0.4)


def build_tracks(out_dir: str):
    """Metric tracks from the cached detection, exactly as the scorer builds them."""
    tr = pd.read_parquet(f"{out_dir}/tracks_grass.parquet")
    prof = auto_tune.VideoProfile.load(f"{out_dir}/profile.json")
    filtered = ball_pitch_filter.filter_ball_by_pitch(tr)
    scale = pixel_scale.estimate_px_per_m(filtered)
    motion_path = Path(out_dir) / "camera_motion.npy"
    if motion_path.exists():
        filtered = camera_motion.compensate(filtered, np.load(motion_path))
    capped = player_filter.filter_players(filtered)
    selected = ball_selection.select_single_ball(capped, scale, fps=prof.fps)
    merged = track_reid.merge_fragments(selected, scale, fps=prof.fps)
    return pixel_scale.prepare_tracks_for_events(merged, None, verbose=False) + (prof,)


def score_pass(tracks, absolute, labels, duration, tolerance, n_draws):
    """Pass precision, recall, F1 and permutation p at one tolerance."""
    with contextlib.redirect_stdout(io.StringIO()):
        events = ev.detect_events(tracks, absolute_pitch=absolute)

    truths = sorted(l["t"] for l in labels if l["group"] == "pass")
    preds = sorted(e["timestamp_s"] for e in events
                   if sc.OURS_TO_GROUP.get(e["event_type"]) == "pass")
    if not preds:
        return dict(n=0, precision=0.0, recall=0.0, f1=0.0, chance=0.0,
                    p=float("nan"), total_events=len(events))

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
                chance=float(draws.mean()), p=float((draws >= f1).mean()),
                total_events=len(events))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=1.0)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--final", action="store_true",
                    help="score the current constants with 2000 draws")
    args = ap.parse_args()

    prepared = {}
    for name, cfg in WINDOWS.items():
        tracks, absolute, prof = build_tracks(cfg["dir"])
        duration = prof.n_frames / prof.fps
        labels = [l for l in sc.load_window(LABELS, cfg["offset"], duration)
                  if l["group"]]
        prepared[name] = (tracks, absolute, labels, duration)
        n_pass = sum(1 for l in labels if l["group"] == "pass")
        print(f"{name}: {len(labels)} scored labels, {n_pass} pass-type, "
              f"{duration:.0f}s")

    if args.final:
        print(f"\ncurrent constants: MIN_PASS_DISTANCE_M="
              f"{ev.MIN_PASS_DISTANCE_M}, MIN_PASS_INTERVAL_S="
              f"{ev.MIN_PASS_INTERVAL_S}")
        for name, (tracks, absolute, labels, duration) in prepared.items():
            r = score_pass(tracks, absolute, labels, duration,
                           args.tolerance, 2000)
            print(f"  {name}: {r['n']:>3} passes  P {r['precision']:.2f}  "
                  f"R {r['recall']:.2f}  F1 {r['f1']:.2f}  "
                  f"chance {r['chance']:.2f}  p {r['p']:.3f}")
        return

    original = (ev.MIN_PASS_DISTANCE_M, ev.MIN_PASS_INTERVAL_S)
    results = {}

    print(f"\nsweep at tolerance +/-{args.tolerance:g}s, {args.draws} draws")
    print(f"{'min_dist':>9} {'min_int':>8} | "
          + " | ".join(f"{n:^34}" for n in WINDOWS))
    print(f"{'':>9} {'':>8} | "
          + " | ".join(f"{'n':>4} {'P':>5} {'R':>5} {'F1':>5} {'p':>6} "
                       for _ in WINDOWS))

    for dist in DIST_GRID:
        for interval in INTERVAL_GRID:
            ev.MIN_PASS_DISTANCE_M = dist
            ev.MIN_PASS_INTERVAL_S = interval
            row = []
            for name, (tracks, absolute, labels, duration) in prepared.items():
                r = score_pass(tracks, absolute, labels, duration,
                               args.tolerance, args.draws)
                results[(dist, interval, name)] = r
                row.append(f"{r['n']:>4} {r['precision']:>5.2f} "
                           f"{r['recall']:>5.2f} {r['f1']:>5.2f} {r['p']:>6.3f} ")
            print(f"{dist:>9.1f} {interval:>8.2f} | " + " | ".join(row))

    ev.MIN_PASS_DISTANCE_M, ev.MIN_PASS_INTERVAL_S = original

    # Cross-check: the setting chosen on one window, judged on the other.
    print("\n" + "=" * 74)
    print("CROSS-CHECK  (a gate fitted to one window will not transfer)")
    print("=" * 74)
    names = list(WINDOWS)
    for tune_on in names:
        test_on = [n for n in names if n != tune_on][0]
        best = max(
            ((d, i) for d in DIST_GRID for i in INTERVAL_GRID),
            key=lambda k: results[(k[0], k[1], tune_on)]["f1"])
        rt = results[(best[0], best[1], tune_on)]
        rv = results[(best[0], best[1], test_on)]
        print(f"  tuned on {tune_on}: dist {best[0]:.1f} interval {best[1]:.2f}")
        print(f"    on {tune_on:<12} F1 {rt['f1']:.2f}  P {rt['precision']:.2f}  "
              f"R {rt['recall']:.2f}  p {rt['p']:.3f}")
        print(f"    on {test_on:<12} F1 {rv['f1']:.2f}  P {rv['precision']:.2f}  "
              f"R {rv['recall']:.2f}  p {rv['p']:.3f}   <- held out")

    Path("pass_threshold_sweep.json").write_text(json.dumps(
        {f"{d}_{i}_{n}": v for (d, i, n), v in results.items()}, indent=2))
    print("\nwritten to pass_threshold_sweep.json")


if __name__ == "__main__":
    main()
