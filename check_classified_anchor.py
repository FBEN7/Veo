"""Does the classifier make the penalty-arc anchor safe to switch on?

Three earlier versions of the penalty-arc anchor were measured and all three
failed the same way: they put anchors tens of metres from where neighbouring
frames put them, while looking perfectly healthy against the markings,
because the pitch is symmetric and the marking score cannot see the
difference. The geometry could not tell a half-hidden centre circle from a
penalty D, so the anchor kept shifting arcs that should have stayed put.

A trained classifier can tell them apart, at least on some clips. This asks
whether that is enough to make the anchor usable.

## Judged by a model that has never seen the clip

Each clip is evaluated with the model trained *without* it. Using a model
that saw the clip would be marking its own homework, and the question is
precisely whether the classifier generalises -- on two clips of four it
does not, and an evaluation that hid that would be worse than no evaluation.

## Scored by the check that is not blind

Frame-to-frame disagreement, never the marking distance. Two anchors on a
pitch that did not move must describe the same ground, and that comparison
has no symmetry to be fooled by -- it is what caught all three previous
attempts while the marking score was calling them fine.

The bar is the anchor as it ships:

    circle-circle pairs   median 0.7 m   worst 7.5 m   0% past 20 m

Penalty-arc anchors are worth having only if they add coverage without
moving those numbers. Anything appearing tens of metres out is the same
failure again, wearing a network.

    python check_classified_anchor.py [--frames 40]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from check_anchor_consistency import MAX_PAIR_GAP, disagreement
from fit_pitch_anchor import anchored_frames
from probe_pitch_lines import CLIPS
from propagate_anchor import frame_to_frame
from src import marking_model

BLUNDER_M = 20.0


def pairs_for(info, rows, images):
    """Every comparable pair, split by whether a classified arc is in it."""
    maps = {index: homography for index, homography, _ in rows}
    kinds = {index: kind for index, _, kind in rows}
    plain, mixed = [], []
    keys = sorted(images)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if b - a > MAX_PAIR_GAP:
                continue
            warp, _ = frame_to_frame(images[a], images[b])
            if warp is None:
                continue
            value = disagreement(maps[a], maps[b], warp,
                                 info["width"], info["height"])
            if not np.isfinite(value):
                continue
            if kinds[a] == "circle" and kinds[b] == "circle":
                plain.append(value)
            else:
                mixed.append(value)
    return plain, mixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()

    print("Penalty-arc anchors gated by a classifier that never saw the "
          "clip it is\njudging, scored by frame-to-frame disagreement.\n")
    print(f"  {'clip':>14s} {'model':>7s} {'circle':>7s} {'arc':>5s} "
          f"{'circle-circle':>22s} {'pairs with an arc':>22s}")

    pooled = {"plain": [], "mixed": []}
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        classifier = marking_model.load(exclude=name)
        rng = np.random.default_rng(0)
        info, rows = anchored_frames(path, args.frames, rng,
                                     use_penalty_arc=classifier is not None,
                                     with_kind=True, classifier=classifier)
        if len(rows) < 2:
            continue

        cap = cv2.VideoCapture(info["path"])
        images = {}
        for index, _, _ in rows:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                images[index] = frame
        cap.release()

        plain, mixed = pairs_for(info, rows, images)
        circles = sum(1 for _, _, kind in rows if kind == "circle")
        arcs = len(rows) - circles

        def describe(values):
            if not values:
                return f"{'-':>22s}"
            array = np.array(values)
            return (f"{array.size:3d} med {np.median(array):5.1f} "
                    f"worst {array.max():6.1f}")

        print(f"  {name:>14s} {'yes' if classifier else 'none':>7s} "
              f"{circles:7d} {arcs:5d} {describe(plain):>22s} "
              f"{describe(mixed):>22s}")
        pooled["plain"] += plain
        pooled["mixed"] += mixed

    for label, values in pooled.items():
        if not values:
            print(f"\n  {label}: none")
            continue
        array = np.array(values)
        print(f"\n  {label}: {array.size} pairs, median "
              f"{np.median(array):.1f} m, worst {array.max():.1f} m, "
              f"{np.mean(array > BLUNDER_M):.0%} past {BLUNDER_M:.0f} m")

    print("\n  The bar is the shipping anchor: 0.7 m median, 7.5 m worst, "
          "nothing past\n  20 m. Arcs earn their place by adding frames "
          "without touching that.")


if __name__ == "__main__":
    main()
