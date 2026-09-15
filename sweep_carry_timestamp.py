#!/usr/bin/env python3
"""Where in a carry should its timestamp sit?

Carry detection clears chance at +/-2 s on every window but fails at +/-1 s on
two of four, and its absolute timing error is 0.48 s against 0.32 s for pass.
So the events are being found and mistimed, not missed.

The error has no consistent sign -- -0.36, -0.12, -0.36, +0.20 s across the
four windows -- so it is not an offset to subtract. The likelier fault is
definitional. We timestamp a carry when possession begins; SoccerNet's DRIVE
marks when the player starts driving with the ball, and a player receives,
controls, and only then moves. Those are different instants, and which one we
report is a choice rather than a measurement.

This scores the alternatives against the labels on all four windows:

  start       possession onset, what is emitted today
  midpoint    halfway through the carry
  end         where the carry finishes
  +offset     start shifted by a constant, as a control -- if a constant
              shift helps as much as a reference point, the reference point
              is not the explanation

Usage:
    python sweep_carry_timestamp.py
"""

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np

import score_soccernet as sc
import sweep_pass_thresholds as sw
from src import events as ev

UPLOADS = Path("/root/.claude/uploads/cd4d7e67-1dd4-5fa1-975c-2f5b3217663b")
STOKE = UPLOADS / "34498b39-Labels-ball.json"
READING = UPLOADS / "53f06df4-Labels-ball.json"

WINDOWS = {
    "w1 (30:20)": dict(dir="output_soccernet", offset=1820.0, labels=STOKE),
    "w2 (66:50)": dict(dir="output_soccernet_w2", offset=4010.0, labels=STOKE),
    "w3 (05:20)": dict(dir="output_soccernet_w3", offset=320.0, labels=STOKE),
    "reading": dict(dir="output_soccernet_reading", offset=5115.0,
                    labels=READING),
}

SHIFTS = (-0.4, -0.2, 0.2, 0.4)


def carry_times(events, mode: str, shift: float = 0.0):
    """Timestamps for carries under one reference-point choice."""
    out = []
    for e in events:
        if e.get("event_type") != "carry":
            continue
        t = float(e["timestamp_s"])
        dur = float(e.get("carry_duration_s") or 0.0)
        if mode == "start":
            out.append(t + shift)
        elif mode == "midpoint":
            out.append(t + dur / 2.0)
        elif mode == "end":
            out.append(t + dur)
    return sorted(out)


def evaluate(preds, truths, duration, tolerance, n_draws):
    if not preds or not truths:
        return dict(f1=0.0, p=float("nan"), err=float("nan"))
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

    err = (np.median([abs(preds[pi] - truths[ti]) for pi, ti, _ in pairs])
           if pairs else float("nan"))
    return dict(f1=f1, p=float((draws >= f1).mean()), err=float(err),
                precision=prec, recall=rec, n=len(preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=1.0)
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()

    prepared = {}
    for name, cfg in WINDOWS.items():
        tracks, absolute, prof = sw.build_tracks(cfg["dir"])
        duration = prof.n_frames / prof.fps
        labels = [l for l in sc.load_window(str(cfg["labels"]), cfg["offset"],
                                            duration) if l["group"] == "carry"]
        with contextlib.redirect_stdout(io.StringIO()):
            events = ev.detect_events(tracks, absolute_pitch=absolute)
        prepared[name] = (events, sorted(l["t"] for l in labels), duration)
        print(f"{name}: {len(labels)} carry labels")

    variants = [("start", 0.0), ("midpoint", 0.0), ("end", 0.0)]
    variants += [(f"start{s:+.1f}s", s) for s in SHIFTS]

    print(f"\ncarry timestamp, tolerance +/-{args.tolerance:g}s, "
          f"{args.draws} draws")
    print(f"{'variant':<14} "
          + " ".join(f"{n:>17}" for n in WINDOWS)
          + f" {'mean F1':>9} {'n<0.05':>7}")

    rows = {}
    for label, shift in variants:
        mode = "start" if label.startswith("start") else label
        cells, f1s, ps = [], [], []
        for name, (events, truths, duration) in prepared.items():
            preds = carry_times(events, mode, shift)
            r = evaluate(preds, truths, duration, args.tolerance, args.draws)
            cells.append(f"F1 {r['f1']:.2f} p {r['p']:.3f}")
            f1s.append(r["f1"])
            ps.append(r["p"])
        rows[label] = (np.mean(f1s), sum(1 for x in ps if x < 0.05))
        print(f"{label:<14} " + " ".join(f"{c:>17}" for c in cells)
              + f" {np.mean(f1s):>9.3f} {sum(1 for x in ps if x < 0.05):>5}/4")

    best = max(rows.items(), key=lambda kv: (kv[1][1], kv[1][0]))
    print(f"\nbest by windows-above-chance then mean F1: {best[0]} "
          f"({best[1][1]}/4, mean F1 {best[1][0]:.3f})")
    print("A constant shift that matches a reference point means the "
          "reference point is not the explanation.")


if __name__ == "__main__":
    main()
