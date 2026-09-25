"""Find the goal frame, the one landmark in the stadium that is unique.

Fitting the pitch to its markings failed, and failed in a way that said why:
a pitch is a repetitive grid, so many poses put most points near some line,
and the fit cannot tell which line is which. Every touchline looks like every
other touchline.

A goal does not. There are two in the ground, they are 7.32 by 2.44 metres,
and those dimensions are in the laws rather than estimated. Four corners of a
rectangle of known size determine the camera pose *and* its focal length on
their own -- which matters here because the focal length measured from the
pan is the weakest link in everything downstream, spreading 0.16 to 0.27
across baselines, and a goal-based anchor does not use it at all.

The goal is also in frame exactly when it is needed. The camera follows play,
so the goal is visible when play is near it, which is when shots happen.

What this probe does is the first half: find the thing. It does not work, and
the record of how it fails is the point of keeping it.

A goal is two bright near-vertical posts, three times further apart than they
are tall, joined across the top by a crossbar, with net between them. Three
increasingly specific versions of that test were built and all three found
stadium furniture instead:

    posts + crossbar + aspect ratio         fired on 88-92% of frames
    ...rejecting anything on the grass      54-74%
    ...requiring a post to stand on it      24-64%

The first fired nearly everywhere, which was itself the giveaway: the camera
is on midfield for most of a match and there is no goal to see there.
Inspected, it had paired the *halfway line* -- bright, thin and near-vertical
from a camera on the halfway line -- with an advertising hoarding edge, at an
aspect ratio inside tolerance by coincidence.

Rejecting candidates lying on the grass fixed that and moved the problem into
the stands, which are full of bright vertical structures. Requiring a post to
stand on the pitch -- foot in the grass, head above it -- was the right idea
and still fails, because that is exactly the geometry of a hoarding edge at
the touchline, and of the marquee posts beside the Veo pitch.

The net is the one thing in a stadium that is only ever a goal, and at this
resolution it is not separable. Measured on a frame where the goal is plainly
visible, the net's interior reads saturation 167, value 169, Laplacian 85
against the crowd's 163, 190, 122 -- and the crowd is what sits directly
behind the goal. A mesh photographed at 720p in front of a stand looks like
the stand.

So this is a job for a trained detector rather than hand-written
morphology. The pipeline already runs YOLO; a goal-frame class would learn
the appearance that these rules keep failing to describe. That needs
annotated goals, which is a data dependency rather than a research problem.

    python probe_goal_frame.py [--save]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.detect_track_hybrid import _grass_mask

# The goal, per the laws of the game.
GOAL_WIDTH_M = 7.32
GOAL_HEIGHT_M = 2.44
GOAL_ASPECT = GOAL_WIDTH_M / GOAL_HEIGHT_M       # 3.0

# A post is a thin bright vertical thing. The top-hat kernel is wide and
# short, which keeps structures narrower than the kernel horizontally --
# posts, and little else that is also vertical.
POST_TOPHAT_W = 17
POST_MIN_BRIGHTNESS = 18

# Posts are near-vertical in the image: the camera has essentially no roll,
# and a goalpost is plumb.
POST_MAX_TILT_DEG = 18.0
POST_MIN_LENGTH_PX = 22

# The crossbar is foreshortened and can slope a long way in the image.
BAR_MAX_TILT_DEG = 35.0
BAR_MIN_LENGTH_PX = 45

# How far the aspect ratio of a candidate may sit from the real 3.0 before it
# is rejected. Generous, because perspective stretches it: a goal seen from
# an angle is narrower in the image than one seen square on.
ASPECT_TOLERANCE = (1.1, 7.0)

# The crossbar has to sit near the tops of both posts, within this fraction
# of post height.
BAR_TOP_TOLERANCE = 0.45

# The share of a candidate's length that may lie on the grass.
#
# This is the test that makes the rest work, and without it the detector is
# worthless. A goalpost stands *above* the pitch, against the crowd; a pitch
# marking lies *on* it. Everything else about them is alike -- both are thin,
# bright and, for the halfway line seen from a camera on the halfway line,
# both are near-vertical in the image.
#
# Without this the detector fired on 88-92% of frames, which was itself the
# giveaway, since the camera is on midfield for most of a match and there is
# no goal to see there. Inspected, it had paired the halfway line with an
# advertising hoarding edge and found an aspect ratio inside tolerance by
# coincidence.
MAX_ON_GRASS = 0.35

# ...and a post has to *stand on* the pitch: its foot in the grass, its head
# out of it. Requiring only "mostly off the grass" was not enough, because
# everything in the stands satisfies that trivially -- stanchions, seat rows,
# signage, and on the Veo clip the burned-in scoreboard. Those are all off
# the grass and none of them is standing on it.
#
# This is the constraint that separates a goalpost from every other bright
# vertical thing in a stadium.
BASE_GRASS_RADIUS_PX = 7
BASE_MIN_GRASS = 0.25
TOP_MAX_GRASS = 0.10


def _segments(mask, min_length):
    found = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=30,
                            minLineLength=min_length, maxLineGap=8)
    if found is None:
        return []
    return [tuple(int(v) for v in s) for s in np.asarray(found).reshape(-1, 4)]


def _pitch_region(frame: np.ndarray) -> np.ndarray:
    """The playing surface, as the largest connected patch of grass."""
    grass = _grass_mask(frame)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (grass > 0).astype(np.uint8), 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        grass = ((labels == biggest) * 255).astype(np.uint8)
    return grass


def _fraction_on_grass(seg, grass, samples: int = 16) -> float:
    """How much of a segment lies on the playing surface."""
    x1, y1, x2, y2 = seg
    t = np.linspace(0.0, 1.0, samples)
    xs = np.clip((x1 + (x2 - x1) * t).astype(int), 0, grass.shape[1] - 1)
    ys = np.clip((y1 + (y2 - y1) * t).astype(int), 0, grass.shape[0] - 1)
    return float((grass[ys, xs] > 0).mean())


def _grass_around(point, grass, radius: int = BASE_GRASS_RADIUS_PX) -> float:
    """Share of grass in a small disc around a point."""
    x, y = int(round(point[0])), int(round(point[1]))
    h, w = grass.shape
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((grass[y0:y1, x0:x1] > 0).mean())


def _stands_on_pitch(seg, grass) -> bool:
    """Foot on the grass, head above it -- which only a post does."""
    x1, y1, x2, y2 = seg
    top, base = ((x1, y1), (x2, y2)) if y1 < y2 else ((x2, y2), (x1, y1))
    return (_grass_around(base, grass) >= BASE_MIN_GRASS
            and _grass_around(top, grass) <= TOP_MAX_GRASS)


def _tilt(x1, y1, x2, y2, reference="vertical"):
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    if reference == "vertical":
        return abs(abs(angle) - 90.0)
    return min(abs(angle), abs(abs(angle) - 180.0))


def find_goal(frame: np.ndarray):
    """The best goal-shaped configuration in the frame, or None.

    Returns the two post segments, the crossbar, and the four corners in the
    order the calibration wants them: left base, left top, right top, right
    base.
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grass = _pitch_region(frame)

    vertical = cv2.morphologyEx(
        grey, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (POST_TOPHAT_W, 1)))
    horizontal = cv2.morphologyEx(
        grey, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, POST_TOPHAT_W)))

    posts = [s for s in _segments((vertical > POST_MIN_BRIGHTNESS).astype(
        np.uint8) * 255, POST_MIN_LENGTH_PX)
        if _tilt(*s, "vertical") <= POST_MAX_TILT_DEG
        and _fraction_on_grass(s, grass) <= MAX_ON_GRASS
        and _stands_on_pitch(s, grass)]
    bars = [s for s in _segments((horizontal > POST_MIN_BRIGHTNESS).astype(
        np.uint8) * 255, BAR_MIN_LENGTH_PX)
        if _tilt(*s, "horizontal") <= BAR_MAX_TILT_DEG
        and _fraction_on_grass(s, grass) <= MAX_ON_GRASS]

    if len(posts) < 2 or not bars:
        return None

    def ends(seg):
        x1, y1, x2, y2 = seg
        top, base = ((x1, y1), (x2, y2)) if y1 < y2 else ((x2, y2), (x1, y1))
        return np.array(top, float), np.array(base, float)

    best = None
    for i in range(len(posts)):
        for j in range(i + 1, len(posts)):
            top_a, base_a = ends(posts[i])
            top_b, base_b = ends(posts[j])
            if top_a[0] > top_b[0]:
                top_a, base_a, top_b, base_b = top_b, base_b, top_a, base_a

            height = (np.linalg.norm(top_a - base_a)
                      + np.linalg.norm(top_b - base_b)) / 2.0
            width = abs(top_b[0] - top_a[0])
            if height < POST_MIN_LENGTH_PX or width < BAR_MIN_LENGTH_PX:
                continue
            aspect = width / height
            if not (ASPECT_TOLERANCE[0] <= aspect <= ASPECT_TOLERANCE[1]):
                continue

            # A crossbar spanning the tops, near both of them.
            for bar in bars:
                bx1, by1, bx2, by2 = bar
                left, right = ((bx1, by1), (bx2, by2)) if bx1 < bx2 else \
                    ((bx2, by2), (bx1, by1))
                if left[0] > top_a[0] + width * 0.35:
                    continue
                if right[0] < top_b[0] - width * 0.35:
                    continue
                drop = (abs(left[1] - top_a[1]) + abs(right[1] - top_b[1])) / 2
                if drop > height * BAR_TOP_TOLERANCE:
                    continue

                # Prefer the biggest plausible structure: a goal is large,
                # and the confusers -- flag posts, hoarding corners -- are
                # small.
                score = height * width / (1.0 + drop)
                if best is None or score > best[0]:
                    best = (score, posts[i], posts[j], bar,
                            np.array([base_a, top_a, top_b, base_b]),
                            aspect, height, width)

    if best is None:
        return None
    return dict(score=float(best[0]), post_a=best[1], post_b=best[2],
                bar=best[3], corners=best[4], aspect=float(best[5]),
                height_px=float(best[6]), width_px=float(best[7]))


CLIPS = (("SoccerNet w1", "output_soccernet"),
         ("SoccerNet w2", "output_soccernet_w2"),
         ("SoccerNet w3", "output_soccernet_w3"),
         ("reading", "output_soccernet_reading"),
         ("Veo", "output_veo"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--out", default="/tmp/goal_frames")
    args = ap.parse_args()

    print("Looking for a goal-shaped configuration: two near-vertical posts, "
          "joined\nacross the top, three or so times wider than tall.\n")
    print(f"  {'clip':>14s} {'frames':>7s} {'found':>7s} {'rate':>6s} "
          f"{'aspect':>8s} {'height px':>10s}")

    for name, out_dir in CLIPS:
        info = Path(out_dir) / "clip.json"
        if not info.exists():
            continue
        path = json.loads(info.read_text())["path"]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        hits, aspects, heights, saved = 0, [], [], 0
        for idx in np.linspace(0, total - 1, args.frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            goal = find_goal(frame)
            if goal is None:
                continue
            hits += 1
            aspects.append(goal["aspect"])
            heights.append(goal["height_px"])

            if args.save and saved < 3:
                canvas = frame.copy()
                for seg in (goal["post_a"], goal["post_b"], goal["bar"]):
                    cv2.line(canvas, seg[:2], seg[2:], (0, 0, 255), 2)
                for corner in goal["corners"]:
                    cv2.circle(canvas, tuple(corner.astype(int)), 5,
                               (0, 255, 255), -1)
                Path(args.out).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(args.out) /
                                f"{out_dir}_f{int(idx)}.png"), canvas)
                saved += 1

        cap.release()
        n = args.frames
        print(f"  {name:>14s} {n:7d} {hits:7d} {hits / n:6.0%} "
              f"{(np.median(aspects) if aspects else float('nan')):8.2f} "
              f"{(np.median(heights) if heights else float('nan')):10.1f}")

    print(f"\n  The real aspect ratio is {GOAL_ASPECT:.1f}. Anything far "
          "from it, or found in\n  nearly every frame, is finding something "
          "else -- the camera is on midfield\n  most of a match and there "
          "is no goal to see there.\n")
    print("  Every rate above is still too high and every one was inspected: "
          "these are\n  hoarding edges at the touchline, which have a "
          "goalpost's exact geometry.\n  See the module docstring. This "
          "approach is recorded, not recommended.")


if __name__ == "__main__":
    main()
