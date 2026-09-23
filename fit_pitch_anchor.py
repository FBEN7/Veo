"""Assemble the anchor: horizon from the lines, scale and origin from the circle.

Neither half is available everywhere. Straight lines give the horizon, but
only on penalty-area frames, where two families are populated. The centre
circle gives scale and origin, but only at midfield. Measured, they share a
frame once in thirty -- so the assembly is not "use both", it is "use each
where it exists and carry it to where it does not".

The horizon is the one that carries. It belongs to the camera rather than to
the frame, and a broadcast camera is bolted to a gantry, so its image moves
only as the camera turns -- which the motion estimator already tracks. A
horizon measured on a penalty-area frame therefore transfers to a midfield
frame by the displacement between them.

With a horizon, the plane is affinely rectified. The circle then supplies
what remains: an affine image of a circle is an ellipse, and the map taking
that ellipse back to a 9.15 m circle is fixed up to a rotation, which the
halfway line resolves.

The check is every OTHER marking. The halfway line will not do, and the first
version of this used it: the halfway line runs through the circle's centre,
that centre maps to (52.5, 34) by construction, and the line's direction is
what fixes the rotation -- so its endpoints can only miss x = 52.5 if the
rotation is wrong, and the rotation was fitted from them. It reported 3 to 4
metres and most of what it was measuring was itself.

What is free is the rest of the markings. Nothing about a touchline, a
penalty-area edge or a six-yard box enters the fit at any point, so mapping
those and asking how far they land from the nearest real pitch line is a test
the anchor cannot have been steered toward.

That test still needs a floor, because "distance to the nearest pitch line"
is not a demanding target: the lines across the pitch sit at 0, 5.5, 16.5,
52.5, 88.5, 99.5 and 105 m, so a position picked at random is already a few
metres from one of them. The floor is measured rather than reasoned about --
the same anchor is displaced by a random offset and scored again, which is
what an anchor carrying no information would score.

    python fit_pitch_anchor.py [--frames 40]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from probe_centre_circle import find_circle
from probe_pitch_lines import CLIPS, line_segments
from src import pitch_model as pm
import probe_vanishing_points as vp

# A segment is taken for the halfway line when its line passes this close to
# the centre of the circle, in pixels. The halfway line is a diameter, so it
# runs through the centre exactly; the tolerance is for detection noise.
HALFWAY_MAX_MISS_PX = 22.0

# The halfway line must be long enough to trust its direction.
HALFWAY_MIN_LENGTH_PX = 90.0

# Random displacements used to measure what an uninformative anchor scores.
CHANCE_TRIALS = 40
CHANCE_OFFSET_M = 20.0


def horizons_from_lines(out_dir: Path, n_frames: int, rng):
    """Horizon lines on the frames whose straight markings determine one."""
    info = json.loads((out_dir / "clip.json").read_text())
    cap = cv2.VideoCapture(info["path"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    found = []
    for idx in np.linspace(0, total - 1, n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        segments, _ = line_segments(frame)
        family_a, family_b = vp._split_families(segments)
        if not family_a:
            continue
        vp_a, support_a = vp.vanishing_point(family_a, rng)
        vp_b, support_b = vp.vanishing_point(family_b, rng)
        if vp_a is None or vp_b is None:
            continue
        if min(support_a, support_b) < vp.GATE_MIN_SUPPORT_LINES:
            continue
        if (support_a / len(family_a) < vp.GATE_MIN_SUPPORT_FRACTION
                or support_b / len(family_b) < vp.GATE_MIN_SUPPORT_FRACTION):
            continue
        found.append((int(idx), np.cross(vp_a, vp_b)))
    cap.release()
    return info, found


def transfer_horizon(horizon, motion, from_frame: int, to_frame: int):
    """Carry a horizon between frames using the camera's own displacement.

    A line a*x + b*y + c = 0 under an image shift (dx, dy) becomes
    a*x + b*y + (c - a*dx - b*dy) = 0. The shift is what the camera-motion
    estimator measures, so the horizon moves with the camera rather than
    being re-measured on a frame that cannot determine it.
    """
    if motion is None or len(motion) <= max(from_frame, to_frame):
        return horizon
    shift = motion[to_frame] - motion[from_frame]
    a, b, c = horizon
    return np.array([a, b, c - a * float(shift[0]) - b * float(shift[1])])


def halfway_line(segments, centre):
    """The detected segment that runs through the circle's centre."""
    best = None
    for x1, y1, x2, y2 in segments:
        if np.hypot(x2 - x1, y2 - y1) < HALFWAY_MIN_LENGTH_PX:
            continue
        line = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
        norm = np.hypot(line[0], line[1])
        if norm < 1e-9:
            continue
        miss = abs(line @ np.array([centre[0], centre[1], 1.0])) / norm
        if miss > HALFWAY_MAX_MISS_PX:
            continue
        if best is None or miss < best[0]:
            best = (miss, (x1, y1, x2, y2))
    return None if best is None else best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("Horizon from the straight lines, scale and origin from the "
          "circle.\nThe halfway line fixes only the rotation, so where it "
          "lands is a free check:\nit must come out at x = 52.5 m.\n")

    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info, horizons = horizons_from_lines(path, args.frames, rng)
        if not horizons:
            print(f"{name}: no frame determines a horizon\n")
            continue

        motion_path = path / "camera_motion.npy"
        motion = np.load(motion_path) if motion_path.exists() else None

        cap = cv2.VideoCapture(info["path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rows = []
        for idx in np.linspace(0, total - 1, args.frames).astype(int):
            idx = int(idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            circle = find_circle(frame, rng)
            if circle is None:
                continue

            source, horizon = min(horizons, key=lambda h: abs(h[0] - idx))
            horizon = transfer_horizon(horizon, motion, source, idx)

            (cx, cy), _, _ = circle["ellipse"]
            halfway = halfway_line(circle["segments"], (cx, cy))
            direction = None
            if halfway is not None:
                direction = np.array([halfway[2] - halfway[0],
                                      halfway[3] - halfway[1]], dtype=float)

            homography = pm.metric_from_circle(horizon, circle["ellipse"],
                                               direction)
            if homography is None:
                continue

            # The free check: every marking the fit never saw.
            model_x = np.array(pm.LINES_ACROSS_M)
            model_y = np.array(pm.LINES_ALONG_M)
            errors = []
            for segment in circle["segments"]:
                if halfway is not None and segment == halfway:
                    continue
                pts = np.array([[segment[0], segment[2]],
                                [segment[1], segment[3]], [1.0, 1.0]])
                mapped = homography @ pts
                if np.any(np.abs(mapped[2]) < 1e-9):
                    continue
                mapped = mapped[:2] / mapped[2]
                # A pitch line is at constant x or constant y; score it
                # against whichever it is closer to being.
                mid = mapped.mean(axis=1)
                if abs(mapped[0, 0] - mapped[0, 1]) < abs(mapped[1, 0]
                                                          - mapped[1, 1]):
                    errors.append(float(np.abs(model_x - mid[0]).min()))
                else:
                    errors.append(float(np.abs(model_y - mid[1]).min()))
            miss = float(np.median(errors)) if errors else float("nan")

            # The floor: the same markings under a displaced anchor.
            chance = []
            for _ in range(CHANCE_TRIALS):
                offset = rng.uniform(-CHANCE_OFFSET_M, CHANCE_OFFSET_M, 2)
                shifted = []
                for segment in circle["segments"]:
                    if halfway is not None and segment == halfway:
                        continue
                    pts = np.array([[segment[0], segment[2]],
                                    [segment[1], segment[3]], [1.0, 1.0]])
                    mapped = homography @ pts
                    if np.any(np.abs(mapped[2]) < 1e-9):
                        continue
                    mapped = mapped[:2] / mapped[2]
                    mid = mapped.mean(axis=1) + offset
                    if abs(mapped[0, 0] - mapped[0, 1]) < abs(mapped[1, 0]
                                                              - mapped[1, 1]):
                        shifted.append(float(np.abs(model_x - mid[0]).min()))
                    else:
                        shifted.append(float(np.abs(model_y - mid[1]).min()))
                if shifted:
                    chance.append(float(np.median(shifted)))
            floor = float(np.median(chance)) if chance else float("nan")

            rows.append((idx, source, miss, circle["residual_px"], floor))

        cap.release()
        if not rows:
            print(f"{name}: no frame has both a circle and a horizon to "
                  f"carry\n")
            continue

        print(f"{name}: {len(horizons)} horizon frames, {len(rows)} anchored")
        for idx, source, miss, residual, floor in rows:
            print(f"   f{idx:5d}  horizon from f{source:<5d}  "
                  f"other markings land {miss:6.1f} m from a real line  "
                  f"(chance {floor:5.1f} m, circle residual {residual:.2f} px)")
        # `miss` is already a distance; subtracting 52.5 from it again was
        # the first version of this line, and it reported 3.6 m as 48.9.
        finite = np.array([r[2] for r in rows if np.isfinite(r[2])])
        floors = np.array([r[4] for r in rows if np.isfinite(r[4])])
        if finite.size:
            beat = (f", chance floor {np.median(floors):.1f} m"
                    if floors.size else "")
            print(f"   -> median error {np.median(finite):.1f} m, "
                  f"worst {finite.max():.1f} m, over {finite.size} "
                  f"frames{beat}\n")
        else:
            print("   -> no frame had a halfway line to check against\n")

    print("Markings landing metres from any real pitch line mean the anchor "
          "is wrong,\nwhatever the circle residual says. A pitch line is "
          "12 cm wide.")


if __name__ == "__main__":
    main()
