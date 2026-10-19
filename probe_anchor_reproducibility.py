"""Asking an anchor to be found twice, and what it costs.

Shot recall was measured at 0 of 6, and one of the three windows failed for
a specific, traceable reason: a single spurious RANSAC fit on a frame with
no centre circle in it became the nearest anchor to the shot, and placed the
ball forty metres from where it was struck. Re-running the circle detector
on that frame with twenty independent draws finds a circle zero times. The
anchoring run got one fit, and it does not reproduce.

RANSAC is a sampler, not a function. A rare spurious fit is not an anomaly
to be surprised by; it is what a sampler does occasionally. Nothing
downstream asks an anchor to prove itself.

## The gate

Fit twice, with independent draws, and keep the anchor only if the two maps
place the ground within `REPRODUCE_TOLERANCE_M` of each other. No ground
truth is needed and no other frame is involved, which is what makes it cheap
enough to run on every candidate.

## What has to be measured before it ships

**Yield**, because it can only take anchors away. On the window that failed,
6 frames of 240 anchored at all; a gate that halves that may cost more in
coverage than it saves in accuracy. Both numbers are here.

**Whether it rejects the right ones.** A gate that removes anchors at random
would also look like it was working, since the bad anchor is rare. So the
frames it rejects are listed, and the known-bad frame is checked by name.

    python probe_anchor_reproducibility.py [--frames 240] [--clip w3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fit_pitch_anchor import REPRODUCE_TOLERANCE_M, anchored_frames
from probe_pitch_lines import CLIPS

# Windows cut around labelled shots, which is where this was found.
SHOT_CLIPS = (("Stoke 13:02", "output_stoke_1302"),
              ("Stoke 42:07", "output_stoke_4207"),
              ("Stoke 70:01", "output_stoke_7001"))


def measure(name: str, out_dir: str, n_frames: int):
    """Anchors with the gate and without, from the same random stream."""
    path = Path(out_dir)
    if not (path / "clip.json").exists():
        return None
    plain = anchored_frames(path, n_frames, np.random.default_rng(0),
                            use_penalty_arc=False)[1]
    gated = anchored_frames(path, n_frames, np.random.default_rng(0),
                            use_penalty_arc=False, reproduce=True)[1]
    kept = {int(i) for i, _ in gated}
    all_idx = {int(i) for i, _ in plain}
    return {"name": name, "out_dir": out_dir, "tried": n_frames,
            "plain": len(all_idx), "gated": len(kept),
            "dropped": sorted(all_idx - kept),
            "added": sorted(kept - all_idx)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--clip", default=None)
    args = ap.parse_args()

    print(f"Anchors kept when a frame has to produce the same map twice, "
          f"within\n{REPRODUCE_TOLERANCE_M:.0f} m. Same random stream both "
          f"ways, so the difference is the gate.\n")
    print(f"  {'clip':>14s} {'tried':>6s} {'plain':>6s} {'gated':>6s} "
          f"{'kept':>6s}   dropped")

    clips = list(SHOT_CLIPS) + [(n, d) for n, d in CLIPS]
    seen = set()
    for name, out_dir in clips:
        if out_dir in seen:
            continue
        seen.add(out_dir)
        if args.clip and args.clip not in out_dir:
            continue
        result = measure(name, out_dir, args.frames)
        if result is None:
            continue
        share = (result["gated"] / result["plain"]
                 if result["plain"] else float("nan"))
        dropped = (", ".join(str(i) for i in result["dropped"][:8])
                   or "none")
        print(f"  {result['name'][-14:]:>14s} {result['tried']:6d} "
              f"{result['plain']:6d} {result['gated']:6d} {share:6.0%}   "
              f"{dropped}", flush=True)
        if result["added"]:
            print(f"      (gate ADDED {result['added']} -- it should only "
                  f"ever take away; investigate)")

    print("\n  The gate can only remove anchors, so any gain has to come "
          "from the ones\n  it removes being wrong. Frame 667 of the 13:02 "
          "window is the one known\n  to be spurious -- the circle detector "
          "finds nothing there on 20 of 20\n  independent draws -- so it "
          "should appear in that row's dropped list.")


if __name__ == "__main__":
    main()
