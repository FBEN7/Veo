"""Tell the centre circle from the penalty D by looking at more than one frame.

Every attempt to separate them inside a single frame has failed, and the
reason is now clear. A centre circle with most of itself hidden -- behind
players, or running out of shot -- presents the same evidence as a penalty
arc. Same 9.15 m radius, because both are struck from the same measurement
in the laws of the game. Same residual, measured at 0.7 px either way. A
straight line near where a chord should be, because a partial arc gives a
biased centre and a true diameter measured against a centre that is a few
metres out reads as a chord. Even enough of the visible arc on one side of
that line.

Three filters were added on that basis and each one refused more frames
while the survivors stayed wrong, misplaced by the 41.5 m that separates
the centre spot from a penalty spot.

What no single frame can supply, the clip can. **An arc painted on the grass
does not move.** Warp a neighbouring frame onto this one -- the same warp
propagation already uses, measured from image features -- and its arc pixels
land on the same circle. Where the two frames hide different parts of it,
the union shows more than either.

And the quantity that separates the two arcs is exactly the one that grows:

  * the penalty D is the part of its circle outside the penalty area, which
    is 2*acos(5.5/9.15) = 106 degrees of it, and no number of frames can make
    that bigger, because the rest of the circle is not painted on the pitch;
  * a centre circle is a whole one, so every frame that hides a different
    part of it adds to what the union covers.

So pool the arc pixels across a window of frames and refit. An arc that
stays near 106 degrees is a D. One that opens out past 200 was a centre
circle all along.

## Where the labels come from

There is no ground truth here, so the check has to make its own. For an
ambiguous frame near an unambiguous one, build both candidate maps -- the
arc as a centre circle, meaning no shift, and the arc as a D, meaning a
41.5 m shift -- and carry the neighbour's anchor onto this frame through the
measured warp. The candidate that agrees with the neighbour is the right
one. That is the same frame-to-frame comparison that caught every previous
version of this, and it is independent of the markings, so it cannot be
fooled by the pitch's symmetry.

Those labels cannot themselves be the decision rule: they need a confident
anchor within warp range, which is the case where the frame did not need
help. They are here to say whether the accumulated span is a rule worth
trusting where no such neighbour exists.

    python probe_arc_accumulation.py [--frames 60] [--window 4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from check_anchor_consistency import disagreement
from fit_pitch_anchor import (anchored_frames, penalty_arc_candidate,
                              plausible_anchor)
from probe_centre_circle import arc_pixels, find_circle, fit_arc
from probe_pitch_lines import CLIPS
from propagate_anchor import frame_to_frame
from src import pitch_model as pm

# How far away the neighbours are that get pooled in. Far enough that the
# view has moved and hides something different; near enough that the warp
# between the two frames is well determined.
NEIGHBOUR_GAPS = (8, 25, 50, 100)

# Pooled pixels are thinned to keep the RANSAC sampling spread out rather
# than concentrated in whichever frame contributed most.
MAX_POOLED_PIXELS = 6000

# A neighbour anchor has to be this close to be trusted as a label.
LABEL_MAX_GAP = 200

# An arc centre this far from one of the three candidate positions is not
# called either way. The candidates are 41.5 m apart, so this is generous.
LABEL_TOLERANCE_M = 14.0

# The widest arc treated as ambiguous. Anything below this is too little to
# fit; anything at or above MIN_ARC_SPAN_DEG was never in doubt.
AMBIGUOUS_SPAN_MIN = 60.0


def pooled_arc(images, target, rng, gaps=NEIGHBOUR_GAPS):
    """Refit the target frame's arc using neighbouring frames' pixels too."""
    own, _ = arc_pixels(images[target])
    ys, xs = np.nonzero(own > 0)
    if len(xs) == 0:
        return None, 0
    points = [np.column_stack([xs, ys]).astype(np.float32)]

    used = 0
    for gap in gaps:
        for index in (target - gap, target + gap):
            if index not in images:
                continue
            warp, _ = frame_to_frame(images[index], images[target])
            if warp is None:
                continue
            mask, _ = arc_pixels(images[index])
            ys, xs = np.nonzero(mask > 0)
            if len(xs) == 0:
                continue
            here = np.column_stack([xs, ys]).astype(np.float32)
            moved = cv2.perspectiveTransform(here.reshape(-1, 1, 2), warp)
            points.append(moved.reshape(-1, 2))
            used += 1

    pooled = np.vstack(points)
    grew = len(pooled) / max(len(points[0]), 1)
    if len(pooled) > MAX_POOLED_PIXELS:
        pooled = pooled[rng.choice(len(pooled), MAX_POOLED_PIXELS,
                                   replace=False)]
    return fit_arc(pooled, rng, min_span_deg=1.0), used, grew


def label_from_neighbour(target, centre_px, images, anchors, info):
    """What a confidently anchored neighbour says this arc's centre is.

    Simpler than asking which *reading* of the arc the neighbour supports,
    and it needs nothing from the arc but where its middle is. Carry the
    neighbour's map onto this frame through the measured warp, put the arc's
    centre through it, and see where on the pitch it lands. The centre spot
    and the two penalty spots are 41.5 m apart, so there is no difficulty
    telling which it is near.

    The first version of this compared two candidate maps instead, which
    needed the arc's own rotation -- and on an ambiguous arc that rotation
    is exactly what is not available. It labelled two frames out of twelve.
    """
    best = None
    for index, homography in anchors.items():
        if abs(index - target) > LABEL_MAX_GAP or index not in images:
            continue
        if best is not None and abs(index - target) >= best[0]:
            continue
        warp, _ = frame_to_frame(images[index], images[target])
        if warp is None:
            continue
        try:
            carried = homography @ np.linalg.inv(warp)
        except np.linalg.LinAlgError:
            continue
        mapped = carried @ np.array([centre_px[0], centre_px[1], 1.0])
        if abs(mapped[2]) < 1e-9:
            continue
        where = mapped[:2] / mapped[2]
        if not np.all(np.isfinite(where)):
            continue
        best = (abs(index - target), where)

    if best is None:
        return None
    where = best[1]
    named = {"circle": np.array(pm.PITCH_CENTRE_M),
             "penalty_arc": None}
    to_centre = float(np.linalg.norm(where - named["circle"]))
    to_spot = min(float(np.hypot(where[0] - spot, where[1] - 34.0))
                  for spot in pm.PENALTY_SPOTS_M)
    if min(to_centre, to_spot) > LABEL_TOLERANCE_M:
        return None                       # near neither; say nothing
    if abs(to_centre - to_spot) < 5.0:
        return None                       # too close to call
    return "circle" if to_centre < to_spot else "penalty_arc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("Arc pixels pooled across neighbouring frames, then refitted.\n"
          "A penalty D cannot open past 106 degrees; a centre circle can.\n")
    print(f"  {'clip':>14s} {'ambiguous':>10s} {'labelled':>9s} "
          f"{'span alone':>21s} {'span pooled':>21s}")

    rows = []
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info = json.loads((path / "clip.json").read_text())
        _, confident = anchored_frames(path, args.frames, rng,
                                       use_penalty_arc=False)
        anchors = dict(confident)

        cap = cv2.VideoCapture(info["path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = sorted({int(i) for i in
                         np.linspace(0, total - 1, args.frames)})
        # Neighbours are read too, so the pool has something to draw on.
        extra = {index + sign * gap for index in wanted
                 for gap in NEIGHBOUR_GAPS for sign in (-1, 1)}
        images = {}
        for index in sorted(set(wanted) | {i for i in extra
                                           if 0 <= i < total}):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                images[index] = frame
        cap.release()

        here = []
        for target in wanted:
            if target not in images:
                continue
            if find_circle(images[target], rng) is not None:
                continue                  # unambiguous already
            # Any arc big enough to fit at all. Narrowing this to the frames
            # that already pass the chord and one-sided tests was the first
            # version's mistake: those filters are what is under test, so
            # using them to choose the sample left twelve frames in total.
            alone = find_circle(images[target], rng,
                                min_span_deg=AMBIGUOUS_SPAN_MIN)
            if alone is None:
                continue
            pooled, used, grew = pooled_arc(images, target, rng)
            if pooled is None or used == 0:
                continue
            truth = label_from_neighbour(target, alone["ellipse"][0],
                                         images, anchors, info)
            here.append({
                "alone": alone["span_deg"] if alone else float("nan"),
                "pooled": pooled["span_deg"],
                "alone_residual": alone["residual_px"] if alone else np.nan,
                "pooled_residual": pooled["residual_px"],
                "neighbours": used,
                "grew": grew,
                "label": truth,
            })

        if not here:
            print(f"  {name:>14s} {0:10d}")
            continue

        def spread(key, label):
            values = [row[key] for row in here if row["label"] == label
                      and np.isfinite(row[key])]
            return (f"{np.median(values):5.0f} (n={len(values)})"
                    if values else "    -      ")

        labelled = sum(1 for row in here if row["label"])
        print(f"  {name:>14s} {len(here):10d} {labelled:9d} "
              f"{spread('alone', 'circle'):>10s} "
              f"{spread('alone', 'penalty_arc'):>10s} "
              f"{spread('pooled', 'circle'):>10s} "
              f"{spread('pooled', 'penalty_arc'):>10s}")
        rows += here

    labelled = [row for row in rows if row["label"]]
    if not labelled:
        print("\n  nothing could be labelled")
        return

    print(f"\n  {len(labelled)} frames labelled by a neighbouring anchor, of "
          f"{len(rows)} ambiguous.\n")
    print(f"  {'':>14s} {'circle':>16s} {'penalty D':>16s}")
    for key in ("alone", "pooled"):
        parts = []
        for label in ("circle", "penalty_arc"):
            values = [row[key] for row in labelled if row["label"] == label
                      and np.isfinite(row[key])]
            parts.append(f"{np.median(values):6.0f} deg (n={len(values):2d})"
                         if values else f"{'-':>16s}")
        print(f"  {key:>14s} {parts[0]:>16s} {parts[1]:>16s}")

    # The threshold that would separate them best, and how well it does.
    best = None
    for threshold in range(110, 300, 10):
        correct = sum(
            1 for row in labelled
            if np.isfinite(row["pooled"])
            and ((row["pooled"] >= threshold) == (row["label"] == "circle")))
        usable = sum(1 for row in labelled if np.isfinite(row["pooled"]))
        if usable and (best is None or correct > best[1]):
            best = (threshold, correct, usable)
    if best:
        threshold, correct, usable = best
        print(f"\n  Pooled span at {threshold} degrees separates them on "
              f"{correct}/{usable} labelled frames ({correct / usable:.0%}).")

    # Did pooling fail because the warps smear the arc, or because the
    # neighbours were showing the same part of it? The residual says which.
    for key, title in (("alone_residual", "residual alone"),
                       ("pooled_residual", "residual pooled")):
        values = [row[key] for row in rows if np.isfinite(row[key])]
        if values:
            print(f"  {title:>16s} {np.median(values):6.2f} px")
    grew = [row["grew"] for row in rows if np.isfinite(row["grew"])]
    if grew:
        used = [row["neighbours"] for row in rows]
        print(f"  {'pixels pooled':>16s} {np.median(grew):6.1f}x, from "
              f"{np.median(used):.0f} neighbours")
    print("\n  A pooled residual much worse than the one alone means the "
          "warps are smearing\n  the arc rather than completing it. One that "
          "holds means the neighbours had\n  nothing new to show.\n")

    print("\n  The pooled span has to beat the span alone, and by enough to "
          "be worth\n  the warps. A D that stays near 106 degrees however "
          "many frames are added\n  is a D; a circle that opens out was "
          "never anything else.")


if __name__ == "__main__":
    main()
