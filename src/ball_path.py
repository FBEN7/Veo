"""Choose the ball's path, instead of the best-looking blob on each frame.

`detect_shots.ball_track` took the most confident detection per frame,
independently, with no memory between frames. Measured on the shot windows
that produces a track whose median jump between *consecutive* frames is 30
to 180 px, with 465 to 781 jumps beyond 150 px. A ball travelling 33 m/s at
these zooms moves about 60 px per frame, so a 150 px step is not a ball --
it is the detector having found a sock, a line marking or a distant blob and
the selection having no reason to prefer yesterday's ball to today's.

The detections are not the problem. On stoke_1302 there are 3476 candidates
across 1612 frames, and 968 of those frames offer two or more. The ball is
usually in the list. Picking it needs the one thing a per-frame maximum
cannot use: the frames either side.

## What this does

A shortest path over candidates. Each candidate is a node scored by how
confident the detector was; each pair of candidates on nearby frames is an
edge scored by the speed it implies. The cheapest path through the clip is
the track. Frames whose candidates are all implausible are simply not on the
path, which is the right answer -- a gap is honest where a teleport is not.

This is the standard formulation for exactly this problem and is worth
stating plainly rather than dressing up: it is Viterbi over detections, with
a constant-velocity prior expressed as a speed penalty.

## The speed scale is not a magic number

A pixel is not a distance, and the same ball is 30 px/frame at one end of
the pitch and 90 at the other. So the limit is derived per clip from the
players: a footballer is about 1.75 m and `crop_h` says how many pixels that
is, which gives pixels per metre without any calibration, homography or
anchor. `MAX_BALL_SPEED_MS` then sets the edge cost in metres per second,
where it means something.

## What it is checked against

A synthetic clip: one smooth trajectory, plus distractors that are *more
confident* than the ball on a third of frames. A per-frame maximum follows
the distractors by construction. The path has to recover the trajectory, and
`selftest` fails if it does not.

That control is the honest half. The half that matters is downstream --
whether shots and goals are found -- because a tracker tuned for smoothness
and then praised for smoothness has proved nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A footballer, the same assumption `ground_plane` makes, used here only to
# turn pixels into metres so the speed limit means something.
PLAYER_HEIGHT_M = 1.75

# Above this the edge is not offered at all. A struck ball reaches about 35
# m/s; this leaves room for a mis-measured scale without admitting teleports.
MAX_BALL_SPEED_MS = 45.0

# Where the speed penalty starts biting, in m/s. Below it a step is free;
# above it the cost grows quadratically.
#
# This was 12, which was a mistake with a specific consequence: a struck
# ball travels far faster than that, so the tracker was penalising exactly
# the motion the shot detector exists to find. Measured, the chosen track's
# 95th-percentile speed came out at 9.5 to 12.6 m/s against a
# `SHOT_MIN_SPEED_MS` of 13.0 -- the tracker was quietly capping the ball
# below the shot threshold.
#
# 25 m/s is a firmly struck ball, so ordinary football motion costs nothing
# and the hard limit does the rejecting. Raising it to 25 roughly doubles
# the frames above 13 m/s with no increase in impossible steps.
SOFT_BALL_SPEED_MS = 25.0

# How far ahead an edge may reach, in frames. The ball is occluded, leaves
# frame and is missed; a path that cannot skip would have to take a
# distractor instead.
MAX_GAP_FRAMES = 12

# What a skipped frame costs. Without it the path would rather jump a gap
# than pay for any motion at all, and would return a sparse track of
# stationary-looking points.
GAP_COST = 0.35

# How much the detector's own confidence counts. Low, deliberately: the
# per-frame maximum is exactly the failure being replaced, so confidence
# breaks ties rather than deciding.
CONFIDENCE_WEIGHT = 0.6

# What including one more point is worth. This is what makes the path long
# rather than merely cheap, and leaving it out was a real bug: scoring a
# path by its average cost per node, a single high-confidence blob scores
# better than any honest track, and `selftest` duly returned a one-point
# "trajectory". With a reward per node the problem is the right one --
# maximise points explained, minus the implausibility of explaining them.
NODE_REWARD = 1.0

# Tried and reverted: preferring candidates with grass around them. It kept
# the path off the advertising hoardings, which on three unplaced labelled
# shots is where the path ran, but measured on six windows it lost more
# than it gained -- shots 3 of 10 to 2, goals 1 of 2 to 0, out of play 2 of
# 4 to 1 -- because the frames those events need are the ones without
# grass behind the ball: the net, the stands over the bar, the hoardings
# beyond the line. See EVENT_ACCURACY.md.

# What a candidate the ball classifier is sure is not a ball loses, as a
# share of `NODE_REWARD` -- equal to it, as with the grass weighting that
# failed, but judged on the object rather than its background. Uses the
# `p_ball` column where tracks carry one (`src/ball_classifier.py`).
CLASSIFIER_WEIGHT = NODE_REWARD

# A segment shorter than this is not a track, it is a coincidence.
MIN_SEGMENT = 4

# A segment that goes nowhere for this long is not the ball.
#
# The speed penalty is `max(0, speed - SOFT)^2`, which is *zero* for a
# candidate that does not move. So a static false positive -- a blob in the
# crowd, a fleck on an advertising board -- offers a long chain at no cost,
# while the real ball pays speed and gap penalties for actually moving. The
# optimum was biased towards things that stay still, which is the opposite
# of a ball in play.
#
# Seen directly on reading_2519: through the goal at frame 500 the path sat
# on a point in the stands for thirty frames. Drawing the candidates showed
# why -- at that moment every ball candidate is in the crowd and the real
# ball is not detected at all.
#
# A ball at a free kick genuinely is still, and this will drop it. That is
# an accepted cost: a stationary ball carries no event, and a stationary
# false positive corrupts every velocity fitted near it.
STATIC_DISPLACEMENT_M = 2.0
STATIC_SECONDS = 1.0


def pixels_per_metre(tracks: pd.DataFrame) -> float | None:
    """Scale from player heights, with no calibration of any kind."""
    if "crop_h" not in tracks.columns:
        return None
    players = tracks[tracks.cls == "player"] if "cls" in tracks else tracks
    heights = players.crop_h.to_numpy(float)
    heights = heights[np.isfinite(heights) & (heights > 5)]
    if heights.size < 50:
        return None
    return float(np.median(heights)) / PLAYER_HEIGHT_M


def choose(candidates: pd.DataFrame, fps: float, px_per_m: float,
           max_gap: int = MAX_GAP_FRAMES) -> pd.DataFrame:
    """The cheapest plausible path through per-frame ball candidates.

    `candidates` needs frame, px, py and confidence. Returns the chosen rows
    in frame order, at most one per frame.
    """
    if candidates.empty:
        return candidates
    rows = candidates.sort_values("frame").reset_index(drop=True)
    frame = rows.frame.to_numpy(int)
    x = rows.px.to_numpy(float)
    y = rows.py.to_numpy(float)
    conf = rows.confidence.to_numpy(float)
    conf = np.where(np.isfinite(conf), conf, 0.5)

    # Metres per pixel-step, so the speed limit is physical.
    to_ms = fps / max(px_per_m, 1e-6)

    n = len(rows)
    cost = CONFIDENCE_WEIGHT * (1.0 - conf) - NODE_REWARD
    if "p_ball" in rows.columns:
        p_ball = rows.p_ball.to_numpy(float)
        p_ball = np.where(np.isfinite(p_ball), np.clip(p_ball, 0.0, 1.0), 1.0)
        cost = cost + CLASSIFIER_WEIGHT * (1.0 - p_ball)
    best = cost.copy()
    came = np.full(n, -1, dtype=int)

    # Candidates are frame-ordered, so the predecessors of node i are a
    # contiguous block ending just before its frame. `start` walks forward
    # with i rather than being searched for each time.
    start = 0
    for i in range(n):
        while frame[i] - frame[start] > max_gap:
            start += 1
        for j in range(start, i):
            step = frame[i] - frame[j]
            if step <= 0:
                continue
            speed = float(np.hypot(x[i] - x[j], y[i] - y[j])) * to_ms / step
            if speed > MAX_BALL_SPEED_MS:
                continue
            penalty = (max(0.0, speed - SOFT_BALL_SPEED_MS)
                       / SOFT_BALL_SPEED_MS) ** 2
            total = best[j] + penalty + GAP_COST * (step - 1) + cost[i]
            if total < best[i]:
                best[i] = total
                came[i] = j

    end = int(np.argmin(best))

    path, node = [], end
    while node >= 0:
        path.append(node)
        node = came[node]
    return rows.iloc[path[::-1]].reset_index(drop=True)


def _is_static(path: pd.DataFrame, fps: float, px_per_m: float) -> bool:
    """A long run that covers no ground: crowd, hoarding, not a ball."""
    frames = path.frame.to_numpy(float)
    seconds = (frames.max() - frames.min()) / max(fps, 1e-6)
    if seconds < STATIC_SECONDS:
        return False
    x = path.px.to_numpy(float) / px_per_m
    y = path.py.to_numpy(float) / px_per_m
    travelled = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    return travelled < STATIC_DISPLACEMENT_M


def segments(candidates: pd.DataFrame, fps: float, px_per_m: float,
             max_gap: int = MAX_GAP_FRAMES,
             min_segment: int = MIN_SEGMENT) -> pd.DataFrame:
    """Every plausible run of ball, not just the single best one.

    One path over a whole clip was the first version and it was wrong. The
    ball goes out, is occluded by a player, leaves frame and is missed;
    across ninety seconds that happens constantly, and a single chain cannot
    span it. It kept 19-29% of frames where the per-frame maximum kept
    57-74% -- trading away most of the track to buy plausibility, which for
    a shot detector needing six points in a twelve-frame window is a poor
    bargain.

    So paths are taken repeatedly: best one, remove its frames, again, until
    nothing of length survives. Each run is internally plausible; between
    runs the ball really was lost, and a `segment` column says where those
    breaks are so nothing downstream fits a velocity across one.
    """
    remaining = candidates
    found, index = [], 0
    while not remaining.empty:
        path = choose(remaining, fps, px_per_m, max_gap)
        if len(path) < min_segment:
            break
        path = path.copy()
        keep = not _is_static(path, fps, px_per_m)
        if keep:
            path["segment"] = index
            found.append(path)
            index += 1
        remaining = remaining[~remaining.frame.isin(set(path.frame))]
    if not found:
        return candidates.iloc[0:0]
    return (pd.concat(found, ignore_index=True)
            .sort_values("frame").reset_index(drop=True))


def track(tracks: pd.DataFrame, fps: float,
          px_per_m: float | None = None) -> pd.DataFrame:
    """The ball's path through a clip's detections."""
    balls = tracks[tracks.cls == "ball"] if "cls" in tracks.columns else tracks
    if balls.empty:
        return balls
    if px_per_m is None:
        px_per_m = pixels_per_metre(tracks)
    if not px_per_m:
        # No scale, no physical limit; fall back to the old behaviour rather
        # than inventing one.
        return (balls.sort_values("confidence").groupby("frame").tail(1)
                .sort_values("frame").reset_index(drop=True))
    return segments(balls, fps, px_per_m)


def selftest(verbose: bool = True) -> bool:
    """A smooth ball among distractors that are more confident than it is."""
    rng = np.random.default_rng(11)
    fps, px_per_m, n = 25.0, 45.0, 240

    # The ball: a steady run across the frame at a footballer's pace.
    t = np.arange(n)
    true_x = 200.0 + 6.0 * t
    true_y = 400.0 + 40.0 * np.sin(t / 30.0)

    rows = []
    for i in range(n):
        rows.append({"frame": i, "cls": "ball", "px": true_x[i],
                     "py": true_y[i], "confidence": 0.45,
                     "crop_h": 10.0})
        # Distractors: brighter to the detector, scattered anywhere.
        if i % 3 == 0:
            rows.append({"frame": i, "cls": "ball",
                         "px": float(rng.uniform(0, 1280)),
                         "py": float(rng.uniform(0, 720)),
                         "confidence": 0.9, "crop_h": 10.0})
    # Players, only so the scale can be measured the way it is in practice.
    for i in range(0, n, 4):
        rows.append({"frame": i, "cls": "player", "px": 500.0, "py": 400.0,
                     "confidence": 0.9,
                     "crop_h": PLAYER_HEIGHT_M * px_per_m})

    frame = pd.DataFrame(rows)
    got = track(frame, fps)
    naive = (frame[frame.cls == "ball"].sort_values("confidence")
             .groupby("frame").tail(1).sort_values("frame"))

    def strays(picked):
        merged = picked.merge(pd.DataFrame({"frame": t, "tx": true_x,
                                            "ty": true_y}), on="frame")
        gap = np.hypot(merged.px - merged.tx, merged.py - merged.ty)
        return float((gap > 20).mean()), len(picked)

    off_path, n_path = strays(got)
    off_naive, n_naive = strays(naive)
    ok = off_path < 0.02 and n_path > 0.9 * n
    if verbose:
        print(f"  {'per-frame maximum':>20s}  {n_naive:3d} points, "
              f"{off_naive:.0%} off the ball")
        print(f"  {'cheapest path':>20s}  {n_path:3d} points, "
              f"{off_path:.0%} off the ball   "
              f"{'ok' if ok else 'WRONG'}")
    return ok


if __name__ == "__main__":
    print("A ball among distractors the detector likes better.\n")
    raise SystemExit(0 if selftest() else 1)
