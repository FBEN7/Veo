"""Find shots in the ball track, and measure the half of that which can be measured.

The xG model has been finished and verified for some time -- `check_xg.py`
recovers known coefficients to within a few percent and its closing line is
"the pipeline is ready for real shots". Nothing has ever fed it one, because
nothing in this pipeline detects a shot. This does.

## What a shot is, in terms the pipeline has

A struck ball, moving fast, on a line that would take it into a goal, from
somewhere a player would shoot from. Each of those is available now in a way
it was not before:

  * the ball track exists, at about 73% frame coverage;
  * the anchor puts it on the pitch in metres, to 0.7 m where a frame is
    anchored and 0.7-1.1 m where one is borrowed;
  * `pitch_model` knows where the goals are.

So: ball speed in metres per second over the ground, sustained; a velocity
pointing into the mouth of a goal rather than merely towards that end; and
an origin close enough to shoot from. A pass across the box is fast and in
the right place but does not point at the goal. A clearance points at the
goal but starts eighty metres away.

## Which goal, which is not the problem it was

The pitch is symmetric and this project has established at length that
geometry cannot say which physical goal is which. It does not need to here.
A shot is aimed at the goal it is aimed at: take whichever goal the ball is
travelling towards, and the distance and angle that come out are the same
numbers either way, because they are measured to that goal. The symmetry
that has cost this project so much is, for xG, genuinely irrelevant -- it
would matter only for saying which TEAM shot, which needs team identity
anyway.

## What can and cannot be measured here

Recall cannot be. The labels hold 50 shots and goals across two full
matches, and not one of them falls inside the video that exists: the Stoke
clip covers 320-410 s with labelled shots at 261 s and 803 s on either side,
the Reading clip covers 3075-3165 s with shots at 2878 s and 3377 s. Six
minutes of football, verified shot-free.

Verified shot-free is worth something, though, and it is the half that can
be measured. Every detection on those clips is a false alarm, and a detector
that cries shot twice a minute on football where there are none is caught
without a single positive example.

The other half is a synthetic control, which this project has used before to
separate "the code is broken" from "the signal is absent". A ballistic shot
is written into the ball track at a known place, speed and direction, and
the detector has to find it. That does not prove it finds real shots; it
proves that when a shot is present and the ball is tracked, this fires.

    python detect_shots.py [--clip output_veo] [--self-check]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from fit_pitch_anchor import anchored_frames, plausible_anchor
from probe_pitch_lines import CLIPS
from propagate_anchor import frame_to_frame
from src import shot_geometry as sg
from src import pitch_model as pm
from src import xg as xg_module

# A struck ball. Passes and dribbles live below this; measured ball speeds in
# football put shots at 20-30 m/s and passes mostly under 15.
SHOT_MIN_SPEED_MS = 13.0

# Over how many frames the speed is measured, and for how many it must hold.
SPEED_WINDOW = 3
SUSTAIN_FRAMES = 3

# The ball's line, extended, has to cross the goal line inside the mouth --
# with this much slack for the anchor's own error, since 0.7 m at 20 m out
# is about two degrees.
MOUTH_SLACK_M = 2.0

# How far out a shot may be taken from. Beyond this it is a clearance or a
# hopeful punt, and xG would be meaningless anyway.
MAX_SHOT_DISTANCE_M = 35.0

# Two detections closer together than this are the same shot seen twice.
MERGE_SECONDS = 2.0

# How far an anchor is carried to put the ball on the pitch.
BORROW_REACH = 150

# Anchors are computed every few frames rather than on every one. Matching
# each of several thousand ball frames against an anchor would mean as many
# ORB fits and holding every frame in memory at once -- gigabytes, for a map
# that barely changes between neighbours. At 2 px/frame of camera movement,
# two frames is about 0.2 m on the ground, well inside the anchor's own
# 0.7 m, so a ball may use the anchor of a frame this close to it.
ANCHOR_GRID = 4


def ball_track(out_dir: Path) -> pd.DataFrame:
    """One ball position per frame, the most confident where there are several."""
    tracks = pd.read_parquet(out_dir / "tracks.parquet")
    ball = tracks[tracks.cls == "ball"]
    if ball.empty:
        return ball
    best = ball.sort_values("confidence").groupby("frame").tail(1)
    return best.sort_values("frame").reset_index(drop=True)


def anchors_for(out_dir: Path, info, frames_wanted, rng, n_frames=120):
    """A pitch map on a grid of frames, carried from the anchored ones.

    Only the anchored frames are held in memory; every other frame is read,
    matched and discarded, so this costs one ORB fit per grid step rather
    than one per ball position, and a few frames of memory rather than all
    of them.
    """
    _, owned = anchored_frames(out_dir, n_frames, rng, use_penalty_arc=False)
    if not owned:
        return {}
    maps = dict(owned)

    cap = cv2.VideoCapture(info["path"])
    sources = {}
    for index, _ in owned:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            sources[int(index)] = frame

    wanted = sorted({int(f) - int(f) % ANCHOR_GRID for f in frames_wanted})
    for index in wanted:
        if index in maps:
            continue
        near = min((i for i in sources if abs(i - index) <= BORROW_REACH),
                   key=lambda i: abs(i - index), default=None)
        if near is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        warp, _ = frame_to_frame(sources[near], frame)
        if warp is None:
            continue
        try:
            carried = maps[near] @ np.linalg.inv(warp)
        except np.linalg.LinAlgError:
            continue
        if plausible_anchor(carried, info):
            maps[index] = carried
    cap.release()
    return maps


def to_pitch(maps, frame_index, px, py):
    """Place a ball, using the nearest grid anchor within half a step."""
    index = int(frame_index)
    homography = maps.get(index)
    if homography is None:
        # Explicit None checks: `a or b` on numpy arrays asks for their
        # truth value, which an array of more than one element does not have.
        for offset in range(1, ANCHOR_GRID // 2 + 1):
            for candidate in (maps.get(index - offset),
                              maps.get(index + offset)):
                if candidate is not None:
                    homography = candidate
                    break
            if homography is not None:
                break
    if homography is None:
        return None
    mapped = homography @ np.array([px, py, 1.0])
    if abs(mapped[2]) < 1e-9:
        return None
    out = mapped[:2] / mapped[2]
    return out if np.all(np.isfinite(out)) else None


def crosses_mouth(x, y, vx, vy, goal):
    """Would a ball leaving here on this heading go in?"""
    goal_x = 0.0 if goal == "left" else pm.PITCH_LENGTH_M
    if abs(vx) < 1e-6:
        return False
    steps = (goal_x - x) / vx
    if steps <= 0:                       # travelling away from this goal
        return False
    crossing = y + vy * steps
    half = sg.GOAL_HALF_M + MOUTH_SLACK_M
    return abs(crossing - pm.PITCH_WIDTH_M / 2.0) <= half


def find_shots(ball: pd.DataFrame, maps, fps: float):
    """Frames where the ball is struck towards a goal."""
    placed = []
    for row in ball.itertuples():
        point = to_pitch(maps, row.frame, row.px, row.py)
        if point is not None:
            placed.append((int(row.frame), float(row.time_s),
                           float(point[0]), float(point[1])))
    if len(placed) < SPEED_WINDOW + 1:
        return [], len(placed)

    frames = np.array([p[0] for p in placed])
    times = np.array([p[1] for p in placed])
    xs = np.array([p[2] for p in placed])
    ys = np.array([p[3] for p in placed])

    hits = []
    for i in range(SPEED_WINDOW, len(placed)):
        j = i - SPEED_WINDOW
        gap = times[i] - times[j]
        if gap <= 0 or frames[i] - frames[j] > 2 * SPEED_WINDOW:
            continue                     # a jump across missing detections
        vx, vy = (xs[i] - xs[j]) / gap, (ys[i] - ys[j]) / gap
        speed = float(np.hypot(vx, vy))
        if speed < SHOT_MIN_SPEED_MS or speed > 45.0:
            continue
        for goal in sg.GOALS:
            if not crosses_mouth(xs[j], ys[j], vx, vy, goal):
                continue
            distance = sg.distance_to_goal(xs[j], ys[j], goal)
            if distance > MAX_SHOT_DISTANCE_M:
                continue
            hits.append({"frame": int(frames[j]), "time_s": float(times[j]),
                         "x": float(xs[j]), "y": float(ys[j]),
                         "speed_ms": speed, "goal": goal,
                         "distance_m": float(distance)})
            break

    # Sustained, then merged: a struck ball is fast for several frames
    # running, and one shot should be reported once.
    kept, run = [], []
    for hit in hits:
        if run and hit["frame"] - run[-1]["frame"] <= 2 * SPEED_WINDOW:
            run.append(hit)
        else:
            if len(run) >= SUSTAIN_FRAMES:
                kept.append(run[0])
            run = [hit]
    if len(run) >= SUSTAIN_FRAMES:
        kept.append(run[0])

    merged = []
    for hit in kept:
        if merged and hit["time_s"] - merged[-1]["time_s"] < MERGE_SECONDS:
            continue
        merged.append(hit)
    return merged, len(placed)


def score(shots):
    """Attach xG, where a model is available."""
    path = Path("models/xg_xghub.json")
    if not path.exists() or not shots:
        return shots
    model = xg_module.XGModel.load(path)
    for shot in shots:
        angle = sg.goal_mouth_angle(shot["x"], shot["y"], shot["goal"])
        shot["angle_rad"] = float(angle)
        shot["xg"] = float(model.predict(distance_m=np.array([shot["distance_m"]]),
                                         angle_rad=np.array([angle]))[0])
    return shots


def synthetic_check(fps=25.0):
    """A shot written into a track, to prove the detector fires on one."""
    # Struck from 16 m out, central, at 22 m/s towards the left goal.
    frames, rows = 40, []
    x, y, vx, vy = 16.0, 34.0, -22.0, 0.0
    identity = np.eye(3)
    maps = {}
    for k in range(frames):
        t = k / fps
        rows.append({"frame": k, "time_s": t, "px": x + vx * t,
                     "py": y + vy * t, "confidence": 1.0})
        maps[k] = identity
    ball = pd.DataFrame(rows)
    shots, placed = find_shots(ball, maps, fps)
    print(f"  synthetic shot: {len(shots)} detected from {placed} placed "
          f"ball positions")
    if shots:
        shot = score(shots)[0]
        print(f"    at {shot['distance_m']:.1f} m, {shot['speed_ms']:.0f} m/s"
              + (f", xG {shot['xg']:.3f}" if "xg" in shot else ""))
    return len(shots) == 1


def inject_shot(ball, maps, info, fps, goal="left"):
    """Write a real shot into a real clip, through the real pitch maps.

    The synthetic control tests `find_shots` on a perfect track with an
    identity map, which is most of the chain missing. This puts a shot into
    an actual clip: a trajectory is laid out in metres, pushed back through
    the frame's own anchor into pixels, and dropped into the ball track. It
    then has to survive everything a real shot would -- the anchor's error,
    the grid the anchors sit on, the coverage gaps.

    Returns the modified track and the frame the shot starts on, or None if
    the clip has no run of anchored frames to put one in.
    """
    anchored = sorted(maps)
    if not anchored:
        return None, None
    span = int(SUSTAIN_FRAMES * SPEED_WINDOW + 2 * ANCHOR_GRID + 8)
    start = None
    for candidate in anchored:
        covered = sum(1 for f in range(candidate, candidate + span)
                      if f in maps or (f - f % ANCHOR_GRID) in maps)
        if covered >= span * 0.8:
            start = candidate
            break
    if start is None:
        return None, None

    goal_x = 0.0 if goal == "left" else pm.PITCH_LENGTH_M
    sign = 1.0 if goal == "left" else -1.0
    x0, y0 = goal_x + sign * 16.0, pm.PITCH_WIDTH_M / 2.0
    speed = 22.0

    rows = []
    for k in range(span):
        frame = start + k
        homography = maps.get(frame)
        if homography is None:
            homography = maps.get(frame - frame % ANCHOR_GRID)
        if homography is None:
            continue
        t = k / fps
        x = x0 - sign * speed * t
        y = y0
        try:
            to_image = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            continue
        pixel = to_image @ np.array([x, y, 1.0])
        if abs(pixel[2]) < 1e-9:
            continue
        pixel = pixel[:2] / pixel[2]
        if not np.all(np.isfinite(pixel)):
            continue
        rows.append({"frame": frame, "time_s": frame / fps,
                     "px": float(pixel[0]), "py": float(pixel[1]),
                     "cls": "ball", "confidence": 1.0,
                     "track_id": -1, "crop_h": 0.0,
                     "detection_method": "injected"})
    if len(rows) < SUSTAIN_FRAMES + SPEED_WINDOW:
        return None, None

    injected = pd.DataFrame(rows)
    kept = ball[~ball.frame.isin(injected.frame)]
    out = pd.concat([kept, injected], ignore_index=True)
    return out.sort_values("frame").reset_index(drop=True), start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--inject", action="store_true",
                    help="put a known shot into each real clip, through that "
                         "clip's own anchors, and see whether it is found")
    args = ap.parse_args()

    if args.self_check:
        print("A shot that is definitely there, to separate a silent "
              "detector from\nfootage with no shots in it.\n")
        ok = synthetic_check()
        print(f"\n  {'passes' if ok else 'FAILS'}: the detector "
              f"{'fires' if ok else 'does not fire'} on a shot it is given.")
        return

    print("Shots in the ball track. The SoccerNet clips are verified "
          "shot-free, so\nevery detection on them is a false alarm.\n")
    print(f"  {'clip':>14s} {'ball frames':>12s} {'placed':>7s} "
          f"{'shots':>6s} {'per 90 min':>11s}")

    rng = np.random.default_rng(0)
    for name, out_dir in CLIPS:
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info = json.loads((path / "clip.json").read_text())
        ball = ball_track(path)
        if ball.empty:
            print(f"  {name:>14s} {'no ball track':>12s}")
            continue
        maps = anchors_for(path, info, ball.frame.tolist(), rng, args.frames)
        if args.inject:
            planted, start = inject_shot(ball, maps, info, info["fps"])
            if planted is None:
                print(f"  {name:>14s} {len(ball):12d} {'-':>7s} "
                      f"{'no anchored run long enough':>30s}")
                continue
            shots, placed = find_shots(planted, maps, info["fps"])
            shots = score(shots)
            found = [s for s in shots if abs(s["frame"] - start) <= 12]
            print(f"  {name:>14s} {len(ball):12d} {placed:7d} "
                  f"{len(shots):6d} {'FOUND' if found else 'missed':>11s}")
            for shot in found[:1]:
                print(f"      planted at f{start}, found f{shot['frame']}, "
                      f"{shot['distance_m']:.1f} m, {shot['speed_ms']:.0f} m/s"
                      + (f", xG {shot['xg']:.3f}" if "xg" in shot else ""))
            continue
        shots, placed = find_shots(ball, maps, info["fps"])
        shots = score(shots)
        minutes = info["n_frames"] / info["fps"] / 60.0
        print(f"  {name:>14s} {len(ball):12d} {placed:7d} {len(shots):6d} "
              f"{len(shots) / minutes * 90:11.1f}")
        for shot in shots[:4]:
            extra = f", xG {shot['xg']:.3f}" if "xg" in shot else ""
            print(f"      t={shot['time_s']:6.1f}s  {shot['distance_m']:5.1f} m"
                  f"  {shot['speed_ms']:4.0f} m/s  -> {shot['goal']}{extra}")

    print("\n  'placed' is how many ball positions had a pitch map to sit on; "
          "the rest\n  are invisible to this whatever they were doing. On "
          "shot-free football the\n  shot count should be zero, and anything "
          "else is the false-alarm rate.")


if __name__ == "__main__":
    main()
