"""What is actually happening on the frames where the anchor is 50 m wrong?

The median anchor error is about a metre and the tail is tens of metres. A
gate built on anchors disagreeing with their neighbours did nothing about it,
which is a sign that the question was wrong: "is this anchor an outlier" is
a statistical question, and a fifty-metre error is not a statistical event.
It is a specific mistake, and there are only so many available.

Three candidates, each of which predicts a particular displacement:

**The arc is not the centre circle.** The penalty arc -- the D outside the
box -- has a radius of 9.15 m, exactly like the centre circle, because both
are struck from the same measurement. Its centre is the penalty spot, 11 m
from the goal line. Mistaking one for the other therefore puts the whole map
out by 52.5 - 11 = 41.5 m along the pitch, in one direction or the other,
with the scale and the rotation perfectly fine. That is the right size for
the errors being seen.

**The pitch is the wrong way round.** A circle is unchanged by turning it
through 180 degrees and so is the halfway line, so the two of them together
fix the rotation only up to a flip. A flipped anchor maps (x, y) to
(105 - x, 68 - y): harmless at the centre spot, tens of metres wrong
anywhere else.

**The rotation was never fixed at all.** `metric_from_circle` takes the
rotation from a line through the circle's centre and falls back to the
identity when it cannot find one -- silently. On a frame where the camera is
not level that is a free error of however many degrees the camera is rolled.

The test applies each correction to each anchored frame and asks whether the
error collapses. A frame that goes from 47 m to 1 m under a 41.5 m shift was
not a noisy anchor; it was the D.

## What it found: the tail is the penalty D, and the flip is separate

    129 anchored frames, 24 of them over 5 m (19%)

    best correction   frames   median before   median after
    as is                  1            5.6m           5.6m
    D left                 9            7.0m           2.0m
    D right                5            7.9m           2.5m
    flipped                2           20.1m          20.1m
    flip + D left          4            7.3m           2.1m
    flip + D right         3            7.7m           2.6m

**Twenty-one of the twenty-four tail frames collapse under a shift of
exactly 41.5 m**, from 7 m to 2 m. They had anchored on the penalty D
instead of the centre circle. It is an easy mistake and a hard one to
notice: both arcs are struck at 9.15 m radius from the same measurement in
the laws of the game, so the scale is right, the residual is right (0.7 px
in the tail and 0.7 px outside it, identical) and the ellipse is a perfectly
good ellipse. Everything about the fit is sound except which arc it is.

The two frames the flip "fixes" score the same before and after, which is
the symmetry doing what it does -- `marking_error` cannot see a flip, so it
cannot be used to find one either. That is `check_orientation.py`'s job and
it is a separate bug with a separate fix.

## And the signal to refuse on

                          in the tail   not in the tail
    has a diameter line           67%               69%
    arc span (degrees)            150               240
    circle residual (px)          0.7               0.7

The arc span separates them and nothing else does. That is exactly what the
geometry predicts: the D is the part of its circle outside the penalty area,
the chord sits 5.5 m from a 9.15 m centre, and the arc spans
2*acos(5.5/9.15) = 106 degrees and never more. `MIN_ARC_SPAN_DEG` was 120,
so the D went straight through; `tune_circle_gate.py` sweeps it and it is
now 200.

Worth noting what did NOT separate them. A diameter line through the
circle's centre should distinguish the centre circle, which the halfway line
bisects, from the D, whose chord misses the centre by 5.5 m -- and it does
not, 67% against 69%. The tolerance in `halfway_line` is 22 px, which at
these scales is wide enough for the D's chord to pass as a diameter. Worth
knowing before building anything else on that test.

    python diagnose_tail.py [--frames 60]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fit_pitch_anchor import (halfway_line, marking_error, plausible_anchor)
from probe_centre_circle import find_circle
from probe_pitch_lines import CLIPS, line_segments
from src import pitch_model as pm

# What mistaking the penalty arc for the centre circle costs, in metres along
# the pitch: the centre spot at 52.5 against the penalty spot at 11.
PENALTY_ARC_OFFSET_M = 52.5 - 11.0

# Above this an anchor is in the tail rather than merely imprecise.
TAIL_M = 5.0


def shift(dx, dy=0.0):
    out = np.eye(3)
    out[0, 2], out[1, 2] = dx, dy
    return out


def flip():
    """Turn the pitch end for end, about its centre."""
    out = np.eye(3)
    out[:2, :2] = -np.eye(2)
    out[0, 2] = pm.PITCH_LENGTH_M
    out[1, 2] = pm.PITCH_WIDTH_M
    return out


CORRECTIONS = {
    "as is": np.eye(3),
    "D left": shift(-PENALTY_ARC_OFFSET_M),
    "D right": shift(+PENALTY_ARC_OFFSET_M),
    "flipped": flip(),
    "flip+Dl": flip() @ shift(-PENALTY_ARC_OFFSET_M),
    "flip+Dr": flip() @ shift(+PENALTY_ARC_OFFSET_M),
}


def anchors_with_detail(out_dir: Path, n_frames: int, rng):
    """Every anchored frame, with what the circle and the halfway line did."""
    info = json.loads((out_dir / "clip.json").read_text())
    cap = cv2.VideoCapture(info["path"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for index in np.linspace(0, total - 1, n_frames).astype(int):
        index = int(index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        circle = find_circle(frame, rng)
        if circle is None:
            continue
        (cx, cy), axes, _ = circle["ellipse"]
        line = halfway_line(circle["segments"], (cx, cy))
        direction = (None if line is None
                     else np.array([line[2] - line[0], line[3] - line[1]],
                                   dtype=float))
        homography = pm.metric_from_circle(
            np.array([0.0, 0.0, 1.0]), circle["ellipse"], direction)
        if homography is None or not plausible_anchor(homography, info):
            continue
        segments, _ = line_segments(frame)
        if len(segments) < 3:
            continue
        out.append({
            "index": index, "homography": homography, "segments": segments,
            "has_line": line is not None,
            "span": circle.get("span_deg", float("nan")),
            "residual": circle["residual_px"],
        })
    cap.release()
    return info, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("Each anchored frame, scored as it stands and under each specific "
          "mistake\nit might be making.\n")

    rows = []
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info, anchors = anchors_with_detail(path, args.frames, rng)
        for entry in anchors:
            scores = {}
            for label, correction in CORRECTIONS.items():
                fixed = correction @ entry["homography"]
                scores[label] = marking_error(fixed, entry["segments"], info)
            entry["scores"] = scores
            entry["clip"] = name
            rows.append(entry)

    if not rows:
        print("no anchored frames")
        return

    plain = np.array([r["scores"]["as is"] for r in rows])
    tail = [r for r in rows if np.isfinite(r["scores"]["as is"])
            and r["scores"]["as is"] > TAIL_M]
    good = [r for r in rows if np.isfinite(r["scores"]["as is"])
            and r["scores"]["as is"] <= TAIL_M]

    print(f"  {len(rows)} anchored frames, {len(tail)} of them over "
          f"{TAIL_M:.0f} m "
          f"({len(tail) / max(1, len(rows)):.0%})\n")

    # Does a specific correction rescue the tail?
    print(f"  Of the {len(tail)} frames in the tail, the best correction is:")
    print(f"  {'correction':>10s} {'frames':>7s} {'median before':>14s} "
          f"{'median after':>13s}")
    chosen = {}
    for entry in tail:
        best = min(CORRECTIONS,
                   key=lambda k: (entry["scores"][k]
                                  if np.isfinite(entry["scores"][k])
                                  else np.inf))
        chosen.setdefault(best, []).append(entry)
    for label in CORRECTIONS:
        group = chosen.get(label, [])
        if not group:
            continue
        before = [e["scores"]["as is"] for e in group]
        after = [e["scores"][label] for e in group]
        print(f"  {label:>10s} {len(group):7d} {np.median(before):13.1f}m "
              f"{np.median(after):12.1f}m")

    # And the two structural signals, against the same split.
    print(f"\n  {'':>22s} {'in the tail':>12s} {'not in the tail':>16s}")
    line_tail = np.mean([e["has_line"] for e in tail]) if tail else np.nan
    line_good = np.mean([e["has_line"] for e in good]) if good else np.nan
    print(f"  {'has a diameter line':>22s} {line_tail:11.0%} "
          f"{line_good:15.0%}")
    for key, label in (("span", "arc span (deg)"),
                       ("residual", "circle residual (px)")):
        a = [e[key] for e in tail if np.isfinite(e[key])]
        b = [e[key] for e in good if np.isfinite(e[key])]
        if a and b:
            print(f"  {label:>22s} {np.median(a):11.1f} {np.median(b):15.1f}")

    print("\n  A correction that collapses a group of tail frames names the "
          "mistake they\n  were making. A signal that separates the tail from "
          "the rest is one the\n  anchor could refuse on, without needing any "
          "markings to notice.")


if __name__ == "__main__":
    main()
