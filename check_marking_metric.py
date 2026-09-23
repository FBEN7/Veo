"""What does the marking metric read when the anchor is exactly right?

Every accuracy figure for the anchor is `marking_error`: map the detected
segments to the pitch and measure how far each lands from the nearest real
pitch line. One to two metres, against a chance floor of five or six.

The question this asks is the one that was never asked. A metre of that is
the anchor being wrong. How much of it is the *metric* being imprecise?

There is reason to suspect a good deal. The metric takes a segment's mapped
midpoint, decides from its orientation whether it belongs to a line of
constant x or constant y, and takes the distance to the nearest line of that
family. Every step of that is approximate. A segment detected slightly off
the line it came from, or spanning a corner, or belonging to a marking the
model does not list, contributes a residual with no anchor error behind it
at all.

So: a synthetic control. Take a real anchor and *declare it correct*.
Generate the pitch's actual markings, project them into the image through
that anchor, cut them into segments the length the detector really produces,
add the endpoint noise it really has -- and score them with the same
function. Whatever it reads is what a perfect anchor scores. Anything below
that number is unmeasurable by this metric, and improving an anchor already
there would not show up.

The second reason to run this is the scale pathology. Twice now, a fit free
to change the scale has driven it to the edge of whatever bound it was
given -- to 10% of a 12% bound per frame, and to 20% of a 20% bound per
clip, on three clips out of five and in the same direction. Something
rewards shrinking, and the obvious suspect is the metric itself, which this
tests by scoring the same perfect anchor scaled up and down.

## The metric is sound, and the suspicion about it was wrong

    clip            perfect anchor   real anchor   segments
    SoccerNet w1             0.06m         0.90m         17
    SoccerNet w2             0.07m         0.95m         23
    SoccerNet w3             0.05m         0.88m         18
    reading                  0.02m         0.82m          7
    Veo                      0.07m         2.90m         11
    pooled                   0.06m         1.10m

Six centimetres. A perfect anchor scores essentially zero, so the metric has
ample resolving power and the 1.1 m the real anchors score is all of it real
error. The hypothesis behind writing this -- that a good part of the residual
was the measurement rather than the anchor -- is wrong, and the anchor is
genuinely about a metre off on broadcast and worse on Veo.

## And the scale pathology is real but not where it was blamed

    scale   metric reads
     0.85          1.42m
     0.90          1.09m
     0.95          0.66m
     1.00          0.05m
     1.05          0.29m
     1.10          0.40m

The metric is asymmetric in scale, but in the OPPOSITE direction to the one
that would explain the fits: shrinking a correct anchor by 10% costs 1.09 m
and growing it by 10% costs 0.40 m, so `marking_error` punishes shrinking
nearly three times as hard. It does not reward it, and the earlier note
saying it did was wrong.

What the fits actually minimise is not this. It is `refine_anchor._loss`, a
weighted Huber over the *detected* segments, and the difference is the
detections. Real segments include ones that are not pitch markings at all --
a shadow, a boot, the edge of a technical area -- and shrinking the map drags
those toward the middle of the pitch, where the model lines are closer
together and any stray point is near one. It also pulls over-long segments
back under the length filter that would otherwise drop them. So the fit was
not exploiting the metric; it was exploiting the outliers, which the Huber
and the length bound between them do not fully suppress.

That keeps the scoring trustworthy and condemns the fitting, which is the
right way round: every accuracy figure in this project rests on the first
and nothing now rests on the second.

    python check_marking_metric.py [--frames 20] [--noise 1.5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from fit_pitch_anchor import anchored_frames, marking_error
from probe_pitch_lines import CLIPS, line_segments
from src import pitch_model as pm

# How long a detected segment is, in pixels. The line detector's own minimum
# is what makes a large arc look straight, and it sets the scale here too.
SEGMENT_PX = 55.0

# Scales to score the perfect anchor at, to see which way the metric leans.
SCALES = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1)


def synthetic_segments(homography, width, height, noise_px, rng):
    """The real pitch markings, seen through this anchor, as the detector
    would report them: short straight pieces with noisy endpoints."""
    try:
        to_image = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return []
    out = []
    for x1, y1, x2, y2 in pm.model_lines():
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 1e-6:
            continue
        steps = max(2, int(length / 0.5))
        pitch = np.array([np.linspace(x1, x2, steps),
                          np.linspace(y1, y2, steps),
                          np.ones(steps)])
        mapped = to_image @ pitch
        good = np.abs(mapped[2]) > 1e-9
        mapped = mapped[:, good]
        if mapped.shape[1] < 2:
            continue
        pixels = mapped[:2] / mapped[2]
        inside = ((pixels[0] >= 0) & (pixels[0] < width)
                  & (pixels[1] >= height * 0.45) & (pixels[1] < height))
        # Walk the visible part, cutting it into detector-length pieces.
        start = None
        for i, here in enumerate(inside):
            if here and start is None:
                start = i
            if start is not None and (not here or i == len(inside) - 1):
                run = pixels[:, start:i + 1]
                start = None
                if run.shape[1] < 2:
                    continue
                steps_along = np.linalg.norm(np.diff(run, axis=1), axis=0)
                travelled = np.concatenate([[0.0], np.cumsum(steps_along)])
                cut = 0.0
                while cut + SEGMENT_PX <= travelled[-1]:
                    a = np.interp(cut, travelled, run[0]), \
                        np.interp(cut, travelled, run[1])
                    b = np.interp(cut + SEGMENT_PX, travelled, run[0]), \
                        np.interp(cut + SEGMENT_PX, travelled, run[1])
                    jitter = rng.normal(0.0, noise_px, 4)
                    out.append((a[0] + jitter[0], a[1] + jitter[1],
                                b[0] + jitter[2], b[1] + jitter[3]))
                    cut += SEGMENT_PX
    return out


def scaled(homography, factor):
    centre = np.array(pm.PITCH_CENTRE_M)
    out = np.eye(3)
    out[:2, :2] = factor * np.eye(2)
    out[:2, 2] = centre - factor * centre
    return out @ homography


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--noise", type=float, default=1.5,
                    help="endpoint noise in pixels")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("A real anchor, declared correct, scored against the markings it "
          "itself\nimplies. Whatever this reads is the metric's own floor.\n")
    print(f"  {'clip':>14s} {'frames':>7s} {'perfect anchor':>15s} "
          f"{'real anchor':>12s} {'segments':>9s}")

    floors, reals, cached = [], [], []
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info, anchors = anchored_frames(path, args.frames, rng)
        if not anchors:
            continue
        cap = cv2.VideoCapture(info["path"])
        clean, actual, counts = [], [], []
        for index, homography in anchors:
            fake = synthetic_segments(homography, info["width"],
                                      info["height"], args.noise, rng)
            if len(fake) < 3:
                continue
            value = marking_error(homography, fake, info)
            if np.isfinite(value):
                clean.append(value)
                counts.append(len(fake))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                segments, _ = line_segments(frame)
                if len(segments) >= 3:
                    real = marking_error(homography, segments, info)
                    if np.isfinite(real):
                        actual.append(real)
        cap.release()
        cached.append((info, anchors))
        if not clean:
            continue
        print(f"  {name:>14s} {len(clean):7d} {np.median(clean):14.2f}m "
              f"{(np.median(actual) if actual else np.nan):11.2f}m "
              f"{np.mean(counts):9.0f}")
        floors += clean
        reals += actual

    if floors:
        print(f"\n  {'pooled':>14s} {len(floors):7d} {np.median(floors):14.2f}m "
              f"{(np.median(reals) if reals else np.nan):11.2f}m")

    # And which way the metric leans when the scale is wrong.
    print(f"\n  A perfect anchor, deliberately rescaled:")
    print(f"  {'scale':>8s} {'metric reads':>13s}")
    # The same anchors as above, kept rather than re-detected: finding the
    # circle on every frame of every clip is the expensive part here, and
    # re-running it once per scale was thirty passes to answer one question.
    rng = np.random.default_rng(1)
    for factor in SCALES:
        values = []
        for info, anchors in cached:
            for index, homography in anchors:
                fake = synthetic_segments(homography, info["width"],
                                          info["height"], args.noise, rng)
                if len(fake) < 3:
                    continue
                value = marking_error(scaled(homography, factor), fake, info)
                if np.isfinite(value):
                    values.append(value)
        if values:
            print(f"  {factor:8.2f} {np.median(values):12.2f}m")

    print("\n  If shrinking a correct anchor makes the metric read BETTER, "
          "the metric\n  rewards shrinking, and no fit that is free to "
          "change scale can be trusted.")


if __name__ == "__main__":
    main()
