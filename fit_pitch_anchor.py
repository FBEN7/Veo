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

## Two fixes, one of which was not one

Rejecting implausible anchors works and is kept. A fit that centres the
picture off the pitch, or claims to show ninety metres of it, is not a near
miss but a failure that should never have been reported; one such frame was
carrying a median error of 189 m. Refusing those, and dropping individual
markings that map more than 25 m outside the pitch, takes the worst case from
189 m to 4.9 m on one clip and 8.0 to 2.8 on another.

Carrying the horizon as a rotation rather than a translation is correct in
principle and does not help. Adding the measured displacement to a line
treats a turn as a slide, which is wrong away from the image centre by
sec^2 of the angle -- and the horizon sits hundreds of rows above the frame,
so it is wrong exactly where the horizon lives. With the outlier rejection
held fixed, the two compare:

    clip        rotation            translation
    w1          2.1 m (worst 4.9)   1.7 m (worst 3.6)
    w2          2.6 m (worst 14.4)  2.6 m (worst 6.5)
    w3          2.6 m (worst 4.3)   1.8 m (worst 5.0)
    reading     1.6 m (worst 2.8)   1.9 m (worst 5.6)

Three to nine frames a clip is too few to separate those, and the simpler
model is no worse, so translation stays the default and `--rotate-horizon`
keeps the other available. The theoretically better correction is not worth
claiming on this evidence.

## The focal length, which is the real find here

The rotation needs a focal length, and two orthogonal vanishing points give
one directly: (v_a - c) . (v_b - c) = -f^2. It comes out at 2331, 2983, 2648
and 3295 px across the four clips, with an interquartile spread of 0.02 to
0.14 -- against 1663, 1614, 1790 and 1799 from the camera's pan, at a spread
of 0.16 to 0.27.

The geometry-based estimate is both larger, by 1.6 to 1.8 times, and steadier
than the motion-based one that `ground_plane.py` ships. That plane is the
component measured to make pass F1, speed reliability and distance
reliability all worse, and its focal length was already the suspect. This is a
third independent line of evidence against it and, unlike the others, it
hands over a replacement number.

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

# An anchor is refused outright when it implies something a football camera
# cannot be seeing. Without this one frame's median error was 189 metres,
# which is not an error in the anchor so much as an anchor that should never
# have been reported: a fit that puts the middle of the picture a hundred
# metres off the pitch is not a near miss.
MAX_CENTRE_OFFSET_M = 40.0     # how far off the pitch the view may be centred
MAX_VISIBLE_SPAN_M = 90.0      # and how much pitch it may claim to show
MIN_VISIBLE_SPAN_M = 8.0

# Mapped markings beyond the pitch by this much are dropped before scoring.
# A segment near the horizon maps to hundreds of metres, and one of those in
# the sample moves a median.
MAX_MARKING_OFFSET_M = 25.0


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
        focal = pm.focal_from_vanishing_points(
            vp_a, vp_b, (info["width"] / 2.0, info["height"] / 2.0))
        found.append((int(idx), np.cross(vp_a, vp_b), focal))
    cap.release()
    return info, found


def transfer_horizon(horizon, motion, from_frame, to_frame, focal, principal):
    """Carry a horizon between frames by the turn the camera actually made.

    The first version added the measured displacement to the line, which
    treats a rotation as a translation. Away from the image centre that is
    wrong by sec^2 of the angle, and the horizon sits hundreds of rows above
    the frame -- so it was wrong precisely where the horizon lives. With a
    focal length from the vanishing points the turn can be reconstructed
    instead, and a line transforms under it as l' = H^-T l.
    """
    if motion is None or len(motion) <= max(from_frame, to_frame):
        return horizon
    shift = motion[to_frame] - motion[from_frame]
    turn = pm.rotation_homography(focal, principal,
                                  float(shift[0]), float(shift[1]))
    if turn is None:
        a, b, c = horizon
        return np.array([a, b, c - a * float(shift[0]) - b * float(shift[1])])
    return np.linalg.inv(turn).T @ np.asarray(horizon, dtype=float)


def plausible_anchor(homography, info) -> bool:
    """Refuse an anchor that implies an impossible view of a football pitch."""
    width, height = info["width"], info["height"]
    corners = np.array([[0, width, width, 0],
                        [height * 0.45, height * 0.45, height, height],
                        [1, 1, 1, 1]], dtype=float)
    mapped = homography @ corners
    if np.any(np.abs(mapped[2]) < 1e-9):
        return False
    mapped = mapped[:2] / mapped[2]
    if not np.all(np.isfinite(mapped)):
        return False

    span_x, span_y = float(np.ptp(mapped[0])), float(np.ptp(mapped[1]))
    if not (MIN_VISIBLE_SPAN_M <= span_x <= MAX_VISIBLE_SPAN_M):
        return False
    if not (MIN_VISIBLE_SPAN_M <= span_y <= MAX_VISIBLE_SPAN_M):
        return False

    centre = mapped.mean(axis=1)
    if not (-MAX_CENTRE_OFFSET_M <= centre[0]
            <= pm.PITCH_LENGTH_M + MAX_CENTRE_OFFSET_M):
        return False
    if not (-MAX_CENTRE_OFFSET_M <= centre[1]
            <= pm.PITCH_WIDTH_M + MAX_CENTRE_OFFSET_M):
        return False
    return True


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


# The vanishing line, when the perspective is weak enough to ignore.
#
# This is the default and it took a head-to-head to believe. The horizon
# apparatus -- two vanishing points, a gate, a transfer between frames -- is
# what most of this file was built for, and on identical frames the anchor
# that skips all of it is better:
#
#     clip        horizon anchor   circle alone
#     w1                  3.3 m          1.4 m
#     w2                  2.5 m          2.1 m
#     w3                  2.2 m          2.3 m
#     reading             2.1 m          0.8 m
#
# The reason is not that perspective does not exist. It is that these are
# narrow views -- a broadcast camera following play, and a Veo virtual camera
# more so -- and across a narrow field of view the projective part of the
# mapping is small, while the horizon estimated to remove it carries real
# error and the transfer between frames adds more. Removing a small
# distortion with a noisy correction is worse than leaving it.
#
# It also unlocks the Veo clip, which has no horizon at all: 4 frames of 60
# yield two vanishing points and their spread is 522 to 875 px at every gate.
# Veo has good circles and no straight-line geometry, so an anchor that needs
# only the circle is the only kind it can have.
AT_INFINITY = pm.AT_INFINITY_LINE


def penalty_arc_candidate(frame, rng, info):
    """A penalty-D fit, with the end of the pitch left undecided.

    Everything here is determined except one bit: the arc's centre is a
    penalty spot, but which of the two? The chord fixes the rotation, the
    radius fixes the scale, and the bulge fixes the map up to that bit --
    which is exactly the bit the pitch's symmetry hides, so nothing in this
    frame can settle it. `resolve_end` does, from the camera's motion.

    Returns the map with the arc's centre still placed at the middle of the
    pitch, plus the arc's supporting pixels.
    """
    circle = find_circle(frame, rng, min_span_deg=pm.D_SPAN_MIN)
    if circle is None:
        return None
    if not (pm.D_SPAN_MIN <= circle["span_deg"] <= pm.D_SPAN_MAX):
        return None

    # Unrotated only to measure how far lines sit from the arc's centre,
    # which a rotation cannot change.
    unrotated = pm.metric_from_circle(AT_INFINITY, circle["ellipse"], None)
    if unrotated is None:
        return None
    chord = pm.penalty_arc_chord(unrotated, circle["segments"])
    if chord is None:
        return None
    # ... and the arc must lie on one side of it. A diameter bisects its
    # circle; a chord does not. Without this, a half-hidden centre circle
    # whose biased centre puts the halfway line 5.5 m away passes as a D and
    # gets shifted 41.5 m.
    if not pm.arc_is_one_sided(unrotated, chord, circle["support"]):
        return None

    # The chord runs parallel to the goal line, the same family as the
    # halfway line, so it fixes the rotation the same way -- and it is the
    # only line that can, since nothing crosses a D's centre.
    direction = np.array([chord[2] - chord[0], chord[3] - chord[1]],
                         dtype=float)
    provisional = pm.metric_from_circle(AT_INFINITY, circle["ellipse"],
                                        direction)
    if provisional is None:
        return None
    return provisional, circle["support"]


def view_centre_on_pitch(reference, reference_index, index, motion, info):
    """Where the middle of frame `index` sits along the pitch, in metres.

    Carried from an anchored frame by the camera's own movement. A scene
    point seen at p in the reference frame is seen at p + (motion[index] -
    motion[reference]) here, so the middle of this frame was at c minus that
    displacement in the reference, and the reference's map says where that
    is on the pitch.

    This is the same translation carry that was measured to drift -- 141 px
    over 600 frames on the Veo clip -- and the drift does not matter, because
    of what it is being asked. The two candidate penalty spots are 41.5 m
    from the middle of the pitch in opposite directions, and 141 px is about
    4 m at these scales. Ten times the margin, for one bit.
    """
    if motion is None or len(motion) <= max(index, reference_index):
        return None
    shift = motion[index] - motion[reference_index]
    centre = np.array([info["width"] / 2.0 - shift[0],
                       info["height"] * 0.75 - shift[1], 1.0])
    mapped = reference @ centre
    if abs(mapped[2]) < 1e-9:
        return None
    value = float(mapped[0] / mapped[2])
    return value if np.isfinite(value) else None


def resolve_end(provisional, support, index, references, motion, info):
    """Settle which penalty spot the arc is, from two independent signs.

    The pitch cannot answer this on its own. Its markings are symmetric
    about the halfway line, so a map placing the arc at x = 11 and one
    placing it at x = 94 put every visible marking onto a real line and
    score identically -- which is how an earlier version passed a 94% check
    while being wrong often enough to put two anchors 114 m apart.

    Two things here are not symmetric, and they are independent of each
    other:

    **Where the camera is looking.** It follows play, so its accumulated pan
    says which half of the pitch is in view, measured against the frames
    whose centre circle pins them to the middle. This is evidence from
    outside the geometry, which is the only kind that can settle it.

    **Which way the arc bulges.** The D is the part of its circle outside
    the penalty area, so it always swells away from its goal. Once the
    rotation is pinned -- and it is, by the camera sitting on one side of
    the pitch -- that direction names the end too.

    Either alone is a guess. Both agreeing is a decision, and where they
    disagree the frame is refused rather than guessed at: a D anchor is
    worth having only if it is worth trusting, since the failure mode is a
    shot at the wrong end of the pitch that looks perfectly reasonable.
    """
    if not references:
        return None
    nearest, reference = min(references.items(),
                             key=lambda item: abs(item[0] - index))
    where = view_centre_on_pitch(reference, nearest, index, motion, info)
    if where is None:
        return None
    # Midfield is a candidate too, and leaving it out was a mistake. The
    # question was put as "which penalty spot", so the answer was always one
    # of them -- even on frames where the camera was plainly looking at the
    # centre of the pitch and the arc was a half-hidden centre circle. Those
    # got shifted 41.5 m, which is exactly the gap between the two answers
    # the question allowed. Asking "which of these three" lets the pan say
    # that the arc should not be moved at all.
    candidates = (*pm.PENALTY_SPOTS_M, pm.PITCH_CENTRE_M[0])
    from_camera = min(candidates, key=lambda spot: abs(spot - where))
    if from_camera == pm.PITCH_CENTRE_M[0]:
        return None

    homography, from_bulge = pm.anchor_from_penalty_arc(provisional, support)
    if homography is None:
        return None
    if from_bulge != from_camera:
        return None
    return homography


def anchored_frames(out_dir: Path, n_frames: int, rng, use_rotation=False,
                    use_horizon: bool = False,
                    use_penalty_arc: bool = True,
                    with_kind: bool = False):
    """Frames carrying a full image-to-pitch map, with that map.

    `with_kind` adds where each anchor came from -- "circle" or
    "penalty_arc". Asking the question by running this twice and diffing the
    two answers does not work: the RANSAC in the circle detector draws from
    the `rng` passed in, so the second run finds a different set of frames
    from the first, and the difference is mostly that rather than the kind.
    A diagnostic built that way reported a clip with no penalty-arc anchors
    and two pairs involving one.
    """
    info = json.loads((out_dir / "clip.json").read_text())
    horizons = []
    motion = None
    if use_horizon:
        info, horizons = horizons_from_lines(out_dir, n_frames, rng)
        if not horizons:
            return info, []
    motion_path = out_dir / "camera_motion.npy"
    if motion_path.exists():
        motion = np.load(motion_path)
    principal = (info["width"] / 2.0, info["height"] / 2.0)

    cap = cv2.VideoCapture(info["path"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out, pending = [], []
    for idx in np.linspace(0, total - 1, n_frames).astype(int):
        idx = int(idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        circle = find_circle(frame, rng)
        if circle is None:
            # No centre circle. Keep the penalty D, if this is one, and
            # decide which end of the pitch it is once every centre-circle
            # anchor on the clip is known -- they are what the camera's pan
            # is measured against.
            if use_penalty_arc and not use_horizon:
                candidate = penalty_arc_candidate(frame, rng, info)
                if candidate is not None:
                    pending.append((idx, candidate[0], candidate[1]))
            continue
        if use_horizon:
            source, horizon, focal = min(horizons,
                                         key=lambda h: abs(h[0] - idx))
            horizon = transfer_horizon(
                horizon, motion, source, idx,
                focal if use_rotation else None, principal)
        else:
            horizon = AT_INFINITY
        (cx, cy), _, _ = circle["ellipse"]
        halfway = halfway_line(circle["segments"], (cx, cy))
        direction = (None if halfway is None
                     else np.array([halfway[2] - halfway[0],
                                    halfway[3] - halfway[1]], dtype=float))
        homography = pm.metric_from_circle(horizon, circle["ellipse"],
                                           direction)
        if homography is None or not plausible_anchor(homography, info):
            continue
        out.append((idx, homography))
    cap.release()
    kinds = {idx: "circle" for idx, _ in out}

    # Second pass over the penalty-D frames, now that the centre-circle
    # anchors exist to measure the camera's pan against.
    if pending:
        references = dict(out)
        for idx, provisional, support in pending:
            settled = resolve_end(provisional, support, idx, references,
                                  motion, info)
            if settled is not None and plausible_anchor(settled, info):
                out.append((idx, settled))
                kinds[idx] = "penalty_arc"
        out.sort(key=lambda row: row[0])
    if with_kind:
        return info, [(idx, homography, kinds[idx]) for idx, homography in out]
    return info, out


def marking_error(homography, segments, info):
    """Median distance from mapped markings to a real pitch line, in metres."""
    model_x = np.array(pm.LINES_ACROSS_M)
    model_y = np.array(pm.LINES_ALONG_M)
    errors = []
    for segment in segments:
        pts = np.array([[segment[0], segment[2]],
                        [segment[1], segment[3]], [1.0, 1.0]], dtype=float)
        mapped = homography @ pts
        if np.any(np.abs(mapped[2]) < 1e-9):
            continue
        mapped = mapped[:2] / mapped[2]
        mid = mapped.mean(axis=1)
        if (mid[0] < -MAX_MARKING_OFFSET_M
                or mid[0] > pm.PITCH_LENGTH_M + MAX_MARKING_OFFSET_M
                or mid[1] < -MAX_MARKING_OFFSET_M
                or mid[1] > pm.PITCH_WIDTH_M + MAX_MARKING_OFFSET_M):
            continue
        if abs(mapped[0, 0] - mapped[0, 1]) < abs(mapped[1, 0] - mapped[1, 1]):
            errors.append(float(np.abs(model_x - mid[0]).min()))
        else:
            errors.append(float(np.abs(model_y - mid[1]).min()))
    return float(np.median(errors)) if errors else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--horizon", action="store_true",
                    help="use the vanishing-point horizon instead of taking "
                         "it at infinity; kept because it was most of the "
                         "work here, but it is measurably worse -- see the "
                         "module docstring")
    ap.add_argument("--rotate-horizon", action="store_true",
                    help="with --horizon, carry it as a rotation rather than "
                         "a translation; correct in principle and not "
                         "measurably better")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    # This reports the anchor that SHIPS. An earlier version of this function
    # always took the horizon route, long after `anchored_frames` had stopped
    # doing so by default -- so the headline script was describing a method
    # that had already been measured and rejected, and the Veo figure quoted
    # from it (3.7 m against a 4.9 m floor) was not what the pipeline would
    # actually produce. Both paths are still available; the default is the
    # one that runs.
    print("The circle fixes scale and origin; the halfway line fixes only "
          "the rotation.\nEvery other marking is then a free check -- nothing "
          "about a touchline or a\npenalty area enters the fit at any "
          "point.\n")

    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info, anchors = anchored_frames(path, args.frames, rng,
                                        use_rotation=args.rotate_horizon,
                                        use_horizon=args.horizon)
        if not anchors:
            print(f"{name}: no frame yields an anchor\n")
            continue

        cap = cv2.VideoCapture(info["path"])
        rows = []
        for idx, homography in anchors:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            segments, _ = line_segments(frame)
            if len(segments) < 3:
                continue
            miss = marking_error(homography, segments, info)
            if not np.isfinite(miss):
                continue

            chance = []
            for _ in range(CHANCE_TRIALS):
                shifted = np.eye(3)
                shifted[:2, 2] = rng.uniform(-CHANCE_OFFSET_M,
                                             CHANCE_OFFSET_M, 2)
                value = marking_error(shifted @ homography, segments, info)
                if np.isfinite(value):
                    chance.append(value)
            rows.append((idx, miss, len(segments),
                         float(np.median(chance)) if chance else float("nan")))
        cap.release()

        if not rows:
            print(f"{name}: anchors found, none with markings to check\n")
            continue
        print(f"{name}: {len(rows)} anchored frames")
        for idx, miss, count, floor in rows:
            print(f"   f{idx:5d}  {count:3d} markings land {miss:6.1f} m from "
                  f"a real line  (chance {floor:5.1f} m)")
        finite = np.array([r[1] for r in rows])
        floors = np.array([r[3] for r in rows if np.isfinite(r[3])])
        beat = f", chance floor {np.median(floors):.1f} m" if floors.size else ""
        print(f"   -> median error {np.median(finite):.1f} m, "
              f"worst {finite.max():.1f} m, over {finite.size} frames{beat}\n")

    print("Markings landing metres from any real pitch line mean the anchor "
          "is wrong,\nwhatever the circle residual says. A pitch line is "
          "12 cm wide.")


if __name__ == "__main__":
    main()
