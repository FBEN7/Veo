"""Training data for a marking classifier, labelled by the anchors we trust.

Every attempt in this project to say *which* marking a detected line or arc
is has been geometric, and the hardest case has beaten all of them: a centre
circle half hidden behind players looks, to every measurement available,
exactly like a penalty arc. Same 9.15 m radius, same 0.7 px residual, same
span, a line where a chord should be. Three single-frame filters and two
multi-frame ones were built, measured and abandoned.

None of them looked at the picture. They measured the shape of the thing and
never what is around it -- and a centre circle sits in open grass with a line
through its middle, while a penalty arc sits against the edge of a box, with
a goal behind it and a six-yard box inside that. Those are not subtle
differences to the eye. They are simply not expressible as a property of the
arc's own geometry, which is all the code has ever been allowed to see.

## Where the labels come from, for nothing

The anchored frames already know. Where a frame carries a confident anchor
-- and those agree with each other to 0.7 m -- the pitch model says where
every marking should appear in that image. So each detected marking pixel
can be handed the identity of the model marking it lands on.

The anchor supplies identity, the detector supplies position, and neither is
asked for what it is bad at. That matters, because 0.7 m of anchor error
would badly misplace a rendered line: a marking is 12 cm wide and 0.7 m is
about 25 px, so labels drawn by projecting the model would sit well off the
real thing. Assigning identity to pixels that were *detected* is immune to
that, as long as the model's markings are further apart than the anchor's
error. They are: the closest pair on the pitch is 5.5 m, against 0.7 m of
error, which is seven times the margin.

## The classes are symmetric on purpose

"Goal line", never "left goal line". The pitch is exactly symmetric end for
end, and this project has repeatedly been caught asking for the one thing an
image of it cannot supply. Asked only what KIND of marking it is looking at,
the symmetry stops being the problem and becomes why the classes are clean.

## A licence note that shapes what this can be used for

Four of the five clips are SoccerNet, whose terms are non-commercial and
no-redistribution. Anything learned from them cannot ship in the product.
Every sample therefore records which clip it came from, so a model intended
for the product can be trained on the Veo footage alone, and the SoccerNet
clips used only to measure whether the method works at all. The crops
themselves are written outside the repository and gitignored.

    python build_marking_crops.py [--frames 120] [--per-frame 60]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from fit_pitch_anchor import anchored_frames, plausible_anchor
from probe_centre_circle import marking_pixels
from probe_pitch_lines import CLIPS
from propagate_anchor import frame_to_frame
from src import pitch_model as pm

# How much of the picture a crop covers, and what it is squashed to. Big
# enough to contain the context that distinguishes the classes -- a penalty
# arc with the box beside it, a touchline with grass on one side only -- and
# small enough to train on four CPU threads.
# How much PITCH a crop covers, in metres, rather than how many pixels.
#
# A fixed pixel size is a different physical footprint on every clip, and
# that is a real defect rather than a detail. At roughly 20 px/m on 720p
# broadcast the old 160 px crop covered about 8 m; on Veo at 35 px/m it
# covered under 5. So the classifier was shown the same marking at wildly
# different scales and, held out on Veo, recognised not one centre circle.
#
# It also has to be large enough to contain what separates the classes. A
# penalty arc is identifiable because the penalty-area line cuts it 5.5 m
# from its centre and the box, six-yard box and goal lie beyond that; a
# centre circle has open grass and a halfway line. Seeing only 8 m of pitch
# around an arc pixel may not reach any of it. Twenty-four metres does, with
# room to spare, and the anchor already knows how many pixels that is.
FOOTPRINT_M = 24.0
CROP_OUT = 64

# Guard rails, in pixels, for when the anchor's local scale is nonsense.
MIN_CROP_PX = 48
MAX_CROP_PX = 900

# A detected pixel is labelled only if it lands this close to a marking in
# the model. Comfortably above the anchors' 0.7 m agreement and comfortably
# below the 5.5 m that separates the closest pair of markings.
LABEL_TOLERANCE_M = 1.5

# Crops are taken from the lower part of the frame, which is where the
# ground is; above that is stands, and a ground-plane map means nothing.
SKY_FRACTION = 0.45

# How far an anchor is carried to label a frame it does not own. The
# propagation measurement puts a borrow at 0.7 m at a third of a second and
# 1.1 m at eight seconds, against a 5.5 m gap between the closest markings,
# so the identity it hands over is safe well past this.
BORROW_REACH = 150

OUT_DIR = Path("marking_crops")


def pixels_per_metre(homography, x, y):
    """How many pixels a metre of pitch covers, here in this frame.

    The homography maps image to pitch, so its Jacobian at a point gives
    metres per pixel; the square root of its determinant is the area scale,
    and one over that is pixels per metre. Taken locally rather than once
    per frame because perspective makes a metre near the camera many more
    pixels than a metre at the far touchline.
    """
    step = 1.0
    here = np.array([[x, x + step, x], [y, y, y + step], [1.0, 1.0, 1.0]])
    mapped = homography @ here
    if np.any(np.abs(mapped[2]) < 1e-9):
        return None
    mapped = mapped[:2] / mapped[2]
    if not np.all(np.isfinite(mapped)):
        return None
    jacobian = np.column_stack([mapped[:, 1] - mapped[:, 0],
                                mapped[:, 2] - mapped[:, 0]]) / step
    area = abs(float(np.linalg.det(jacobian)))
    if area < 1e-12:
        return None
    return 1.0 / np.sqrt(area)


def crop_at(frame, x, y, scale, footprint_m=FOOTPRINT_M):
    """A fixed number of METRES of pitch about a point, squashed to a square."""
    size = int(round(footprint_m * scale))
    if not (MIN_CROP_PX <= size <= MAX_CROP_PX):
        return None
    half = size // 2
    patch = cv2.copyMakeBorder(frame, half, half, half, half,
                               cv2.BORDER_REPLICATE)
    x0, y0 = int(x), int(y)
    patch = patch[y0:y0 + 2 * half, x0:x0 + 2 * half]
    if patch.shape[0] < 4 or patch.shape[1] < 4:
        return None
    return cv2.resize(patch, (CROP_OUT, CROP_OUT),
                      interpolation=cv2.INTER_AREA)


def label_pixels(homography, xs, ys, tree, labels):
    """The kind of marking each detected pixel belongs to, or nothing."""
    points = np.vstack([xs, ys, np.ones(len(xs))])
    mapped = homography @ points
    good = np.abs(mapped[2]) > 1e-9
    out = np.zeros(len(xs), dtype=np.int64)
    if not np.any(good):
        return out
    pitch = (mapped[:2, good] / mapped[2, good]).T
    finite = np.all(np.isfinite(pitch), axis=1)
    distances, index = tree.query(pitch[finite])
    assigned = np.where(distances <= LABEL_TOLERANCE_M, labels[index], 0)
    slot = np.zeros(good.sum(), dtype=np.int64)
    slot[finite] = assigned
    out[good] = slot
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--per-frame", type=int, default=60)
    ap.add_argument("--footprint-m", type=float, default=FOOTPRINT_M,
                    help="metres of pitch a crop covers")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    model_points, model_labels = pm.labelled_model_points()
    tree = cKDTree(model_points)

    OUT_DIR.mkdir(exist_ok=True)
    print("Marking crops, labelled by the frames that already carry a "
          "trusted anchor.\n")
    print(f"  {'clip':>14s} {'own+lent':>9s} {'crops':>7s} {'labelled':>9s} "
          f"{'classes seen':>13s}")

    totals = np.zeros(len(pm.MARKING_CLASSES), dtype=np.int64)
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info, anchors = anchored_frames(path, args.frames, rng,
                                        use_penalty_arc=False)
        if not anchors:
            print(f"  {name:>14s} {0:9d}")
            continue

        # Labelling only the anchored frames teaches only midfield. A
        # confident anchor needs a centre circle, a centre circle is at
        # midfield, so those frames contain the halfway line, the circle and
        # occasionally a touchline -- and never a penalty area. The first
        # run of this produced 3439 centre-circle crops and not one penalty
        # arc, which is the very class the whole exercise is for.
        #
        # Borrowed anchors do not have that problem. They are good to 0.7 m
        # at a third of a second and 1.1 m at eight, and they reach the
        # frames the camera has moved on to -- which is where the boxes are.
        cap = cv2.VideoCapture(info["path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        owned = {index: homography for index, homography in anchors}
        sources = {index: None for index in owned}
        wanted = sorted({int(i) for i in
                         np.linspace(0, total - 1, args.frames)}
                        | set(owned))

        images = {}
        for index in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                images[index] = frame

        labelled_maps = dict(owned)
        for index in wanted:
            if index in labelled_maps or index not in images:
                continue
            near = min((i for i in owned if abs(i - index) <= BORROW_REACH
                        and i in images),
                       key=lambda i: abs(i - index), default=None)
            if near is None:
                continue
            warp, _ = frame_to_frame(images[near], images[index])
            if warp is None:
                continue
            try:
                carried = owned[near] @ np.linalg.inv(warp)
            except np.linalg.LinAlgError:
                continue
            if plausible_anchor(carried, info):
                labelled_maps[index] = carried
                sources[index] = near

        crops, targets, frames_used = [], [], []
        offered = 0
        for index in sorted(labelled_maps):
            homography = labelled_maps[index]
            frame = images.get(index)
            if frame is None:
                continue
            mask = marking_pixels(frame)
            mask[:int(frame.shape[0] * SKY_FRACTION)] = 0
            ys, xs = np.nonzero(mask > 0)
            if len(xs) < 20:
                continue
            take = min(args.per_frame, len(xs))
            pick = rng.choice(len(xs), take, replace=False)
            xs, ys = xs[pick], ys[pick]
            offered += take

            kinds = label_pixels(homography, xs, ys, tree, model_labels)
            for x, y, kind in zip(xs, ys, kinds):
                if kind == 0:
                    continue          # detected, but not a modelled marking
                scale = pixels_per_metre(homography, x, y)
                if scale is None:
                    continue
                patch = crop_at(frame, x, y, scale, args.footprint_m)
                if patch is None:
                    continue
                crops.append(patch)
                targets.append(kind)
                frames_used.append(index)
        cap.release()

        if not crops:
            print(f"  {name:>14s} {len(anchors):9d} {0:7d}")
            continue
        crops = np.stack(crops)
        targets = np.asarray(targets, dtype=np.int64)
        np.savez_compressed(
            OUT_DIR / f"{out_dir}.npz", crops=crops, labels=targets,
            frames=np.asarray(frames_used), clip=name)
        counts = np.bincount(targets, minlength=len(pm.MARKING_CLASSES))
        totals += counts
        print(f"  {name:>14s} {len(anchors):4d}+{len(labelled_maps) - len(anchors):<4d} "
              f"{len(crops):7d} {len(crops) / max(offered, 1):8.0%} "
              f"{int((counts > 0).sum()):13d}")

    print(f"\n  {'class':>20s} {'crops':>8s}")
    for i, label in enumerate(pm.MARKING_CLASSES):
        if totals[i]:
            print(f"  {label:>20s} {totals[i]:8d}")
    print(f"\n  A low 'labelled' share means detected markings are landing "
          f"nowhere near\n  the model -- the anchor would be wrong, not the "
          f"detector. The classes\n  that matter are the last two, and they "
          f"have to appear on different frames.")


if __name__ == "__main__":
    main()
