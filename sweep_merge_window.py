#!/usr/bin/env python3
"""Choose the event merge window by measurement, and check it transfers.

Every false positive the detector produces sits within five seconds of a real
labelled event, most within one, so collapsing near-duplicates addresses the
fault actually present. The risk is the mirror image: football contains
genuine quick exchanges, and a window set too wide deletes them.

Precision will rise with any window at all -- fewer events can only help it --
so precision alone cannot choose one. F1 can, and the value chosen on one
window is reported against the other, because a merge window fitted to one 90
seconds of football is not a fix.

Usage:
    python sweep_merge_window.py
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

import score_soccernet as sc
import sweep_pass_thresholds as sw
from src import events as ev

LABELS = ("/root/.claude/uploads/cd4d7e67-1dd4-5fa1-975c-2f5b3217663b/"
          "34498b39-Labels-ball.json")

WINDOW_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)


def score(tracks, absolute, labels, duration, group, tolerance, n_draws):
    with contextlib.redirect_stdout(io.StringIO()):
        events = ev.detect_events(tracks, absolute_pitch=absolute)

    truths = sorted(l["t"] for l in labels if l["group"] == group)
    preds = sorted(e["timestamp_s"] for e in events
                   if sc.OURS_TO_GROUP.get(e["event_type"]) == group)
    if not preds or not truths:
        return dict(n=0, precision=0.0, recall=0.0, f1=0.0, p=float("nan"))

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
    for name, cfg in sw.WINDOWS.items():
        tracks, absolute, prof = sw.build_tracks(cfg["dir"])
        duration = prof.n_frames / prof.fps
        labels = [l for l in sc.load_window(LABELS, cfg["offset"], duration)
                  if l["group"]]
        prepared[name] = (tracks, absolute, labels, duration)

    original = (ev.MERGE_WINDOW_S, dict(ev.MERGE_WINDOW_BY_TYPE))
    results = {}

    for group in ("pass", "carry"):
        print(f"\n{'=' * 78}")
        print(f"{group}  at tolerance +/-{args.tolerance:g}s")
        print("=" * 78)
        print(f"{'merge':>7} | "
              + " | ".join(f"{n:^33}" for n in sw.WINDOWS))
        print(f"{'':>7} | "
              + " | ".join(f"{'n':>4} {'P':>5} {'R':>5} {'F1':>5} {'p':>6} "
                           for _ in sw.WINDOWS))
        for w in WINDOW_GRID:
            # Clear the per-type override as well, or the swept value never
            # reaches the type that has one -- which silently pinned carry at
            # 1.0 s through an entire sweep and produced identical rows.
            ev.MERGE_WINDOW_S = w
            ev.MERGE_WINDOW_BY_TYPE.clear()
            row = []
            for name, (tracks, absolute, labels, duration) in prepared.items():
                r = score(tracks, absolute, labels, duration, group,
                          args.tolerance, args.draws)
                results[(group, w, name)] = r
                row.append(f"{r['n']:>4} {r['precision']:>5.2f} "
                           f"{r['recall']:>5.2f} {r['f1']:>5.2f} {r['p']:>6.3f} ")
            print(f"{w:>7.1f} | " + " | ".join(row))

    ev.MERGE_WINDOW_S = original[0]
    ev.MERGE_WINDOW_BY_TYPE.clear()
    ev.MERGE_WINDOW_BY_TYPE.update(original[1])

    print("\n" + "=" * 78)
    print("CROSS-CHECK  (a window fitted to one 90 seconds is not a fix)")
    print("=" * 78)
    names = list(sw.WINDOWS)
    for group in ("pass", "carry"):
        # Mean F1 across every window: a value is only worth having if it
        # helps on average, not if it wins on the two it was chosen from.
        print(f"\n  {group}: mean F1 across all {len(names)} windows")
        for w in WINDOW_GRID:
            f1s = [results[(group, w, n)]["f1"] for n in names]
            ps = [results[(group, w, n)]["p"] for n in names]
            n_sig = sum(1 for x in ps if x < 0.05)
            print(f"    merge {w:>4.1f}s  mean F1 {np.mean(f1s):.3f}  "
                  f"({', '.join(f'{x:.2f}' for x in f1s)})  "
                  f"{n_sig}/{len(names)} above chance")
        for tune_on in names:
            test_on = [n for n in names if n != tune_on][0]
            best = max(WINDOW_GRID,
                       key=lambda w: results[(group, w, tune_on)]["f1"])
            rt = results[(group, best, tune_on)]
            rv = results[(group, best, test_on)]
            base_t = results[(group, 0.0, tune_on)]
            base_v = results[(group, 0.0, test_on)]
            print(f"  {group:<6} tuned on {tune_on}: merge {best:.1f}s")
            print(f"    {tune_on:<12} F1 {base_t['f1']:.2f} -> {rt['f1']:.2f}  "
                  f"P {base_t['precision']:.2f} -> {rt['precision']:.2f}  "
                  f"R {base_t['recall']:.2f} -> {rt['recall']:.2f}")
            print(f"    {test_on:<12} F1 {base_v['f1']:.2f} -> {rv['f1']:.2f}  "
                  f"P {base_v['precision']:.2f} -> {rv['precision']:.2f}  "
                  f"R {base_v['recall']:.2f} -> {rv['recall']:.2f}   <- held out")

    Path("merge_window_sweep.json").write_text(json.dumps(
        {f"{g}_{w}_{n}": v for (g, w, n), v in results.items()}, indent=2))
    print("\nwritten to merge_window_sweep.json")


if __name__ == "__main__":
    main()
