#!/usr/bin/env python3
"""Look at what the detector gets wrong, not just how often.

Precision and recall say how much is wrong. They do not say whether a false
positive is a hallucinated event in quiet play or a real event reported a
second late, and those need different fixes -- the first is a logic problem,
the second a timing one. Nor do they say whether misses are spread evenly or
concentrated where the ball was never detected, which would make event
detection a downstream symptom rather than a fault of its own.

Everything here is measured against the same SoccerNet labels used for
scoring, on both windows.
"""

import argparse
import contextlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import score_soccernet as sc
import sweep_pass_thresholds as sw
from src import events as ev

LABELS = ("/root/.claude/uploads/cd4d7e67-1dd4-5fa1-975c-2f5b3217663b/"
          "34498b39-Labels-ball.json")
TOL = 1.0


def analyse(name, cfg):
    tracks, absolute, prof = sw.build_tracks(cfg["dir"])
    duration = prof.n_frames / prof.fps
    all_labels = sc.load_window(LABELS, cfg["offset"], duration)
    labels = [l for l in all_labels if l["group"]]

    with contextlib.redirect_stdout(io.StringIO()):
        events = ev.detect_events(tracks, absolute_pitch=absolute)

    print("\n" + "=" * 74)
    print(f"{name}   {len(labels)} scorable labels, {len(events)} events")
    print("=" * 74)

    # ---- recall by the label's own type, not by our coarse grouping -------
    # "pass" lumps PASS, HIGH PASS, CROSS and HEADER together. If recall
    # differs sharply between them the grouping is hiding a real weakness.
    print(f"\nrecall by label type (tolerance +/-{TOL:g}s)")
    print(f"  {'label':<26} {'n':>4} {'found':>6} {'recall':>7}")
    preds_by_group = {
        g: sorted(e["timestamp_s"] for e in events
                  if sc.OURS_TO_GROUP.get(e["event_type"]) == g)
        for g in ("pass", "carry", "recovery")
    }
    for label_type in sorted({l["label"] for l in labels}):
        subset = [l for l in labels if l["label"] == label_type]
        group = subset[0]["group"]
        truths = sorted(l["t"] for l in subset)
        pairs, _, _ = sc.match_one_to_one(preds_by_group[group], truths, TOL)
        n_found = len(pairs)
        print(f"  {label_type:<26} {len(subset):>4} {n_found:>6} "
              f"{n_found / len(subset):>7.2f}")

    # ---- are false positives hallucinations, or mistimed real events? -----
    # Distance from each unmatched detection to the nearest label of any kind.
    # Clustered close means the event is real and the timing is off; far means
    # the detector invented it.
    print("\nfalse positives: distance to the nearest label of any type")
    label_times = np.array(sorted(l["t"] for l in labels))
    for group in ("pass", "carry"):
        truths = sorted(l["t"] for l in labels if l["group"] == group)
        preds = preds_by_group[group]
        pairs, fp_idx, fn_idx = sc.match_one_to_one(preds, truths, TOL)
        if not fp_idx:
            continue
        gaps = np.array([np.min(np.abs(label_times - preds[i])) for i in fp_idx])
        buckets = [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 1e9)]
        counts = [int(((gaps >= a) & (gaps < b)).sum()) for a, b in buckets]
        print(f"  {group:<8} {len(fp_idx):>3} false positives   "
              + "  ".join(f"{a:.0f}-{b:.0f}s: {c}" if b < 1e9
                          else f">{a:.0f}s: {c}"
                          for (a, b), c in zip(buckets, counts)))
        print(f"  {'':8} median gap to nearest label {np.median(gaps):.2f}s")

    # ---- timing error on the events we do get right ----------------------
    print("\ntiming error on matched events")
    for group in ("pass", "carry"):
        truths = sorted(l["t"] for l in labels if l["group"] == group)
        preds = preds_by_group[group]
        pairs, _, _ = sc.match_one_to_one(preds, truths, TOL)
        if not pairs:
            continue
        signed = np.array([preds[pi] - truths[ti] for pi, ti, _ in pairs])
        print(f"  {group:<8} n={len(signed):>3}  median {np.median(signed):+.2f}s  "
              f"p10 {np.percentile(signed, 10):+.2f}  "
              f"p90 {np.percentile(signed, 90):+.2f}  "
              f"|error| median {np.median(np.abs(signed)):.2f}s")

    # ---- do misses coincide with the ball being untracked? ---------------
    # If they do, event detection is limited by ball coverage rather than by
    # its own logic, and improving the event rules cannot help.
    ball = tracks[tracks.cls == "ball"]
    covered = set(ball.time_s.round(2))
    ball_times = np.array(sorted(ball.time_s.to_numpy()))

    def ball_gap(t):
        if not len(ball_times):
            return np.inf
        return float(np.min(np.abs(ball_times - t)))

    print("\nball coverage at the moment of each label")
    for group in ("pass", "carry"):
        truths = sorted(l["t"] for l in labels if l["group"] == group)
        preds = preds_by_group[group]
        pairs, _, fn_idx = sc.match_one_to_one(preds, truths, TOL)
        hit_t = [truths[ti] for _, ti, _ in pairs]
        miss_t = [truths[i] for i in fn_idx]
        if not miss_t:
            continue
        hit_gap = np.median([ball_gap(t) for t in hit_t]) if hit_t else np.nan
        miss_gap = np.median([ball_gap(t) for t in miss_t])
        print(f"  {group:<8} nearest ball detection: "
              f"found events {hit_gap:.2f}s, missed events {miss_gap:.2f}s "
              f"({len(miss_t)} missed)")

    # ---- team attribution, which nothing has checked yet -----------------
    # SoccerNet records which side performed each action. Our team labels are
    # arbitrary (team_A/team_B), so the test is whether the *mapping* is
    # consistent: if it is, one assignment of A/B to left/right explains most
    # matched events. A coin-flip detector scores 0.5 whichever way round.
    print("\nteam attribution on matched events")
    raw = json.loads(Path(LABELS).read_text())
    side_by_time = {round(float(a["position"]) / 1000.0 - cfg["offset"], 3):
                    a.get("team") for a in raw["annotations"]}
    agree = Counter()
    for group in ("pass", "carry"):
        truths_full = [l for l in labels if l["group"] == group]
        truths = sorted(l["t"] for l in truths_full)
        ev_by_time = {e["timestamp_s"]: e.get("team")
                      for e in events
                      if sc.OURS_TO_GROUP.get(e["event_type"]) == group}
        preds = preds_by_group[group]
        pairs, _, _ = sc.match_one_to_one(preds, truths, TOL)
        for pi, ti, _ in pairs:
            ours = ev_by_time.get(preds[pi])
            theirs = side_by_time.get(round(truths[ti], 3))
            if ours and theirs and ours != "unknown":
                agree[(ours, theirs)] += 1
    if agree:
        total = sum(agree.values())
        # Best consistent mapping of our two labels onto left/right.
        a_left = agree[("team_A", "left")] + agree[("team_B", "right")]
        a_right = agree[("team_A", "right")] + agree[("team_B", "left")]
        best = max(a_left, a_right)
        print(f"  {total} matched events carry a team on both sides")
        print(f"  best consistent mapping explains {best}/{total} "
              f"= {best / total:.2f}  (0.50 = no information)")
        print(f"  breakdown: {dict(agree)}")
    else:
        print("  no matched events carry a team on both sides")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    for name, cfg in sw.WINDOWS.items():
        analyse(name, cfg)


if __name__ == "__main__":
    main()
