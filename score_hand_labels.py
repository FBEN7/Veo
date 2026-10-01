"""Score the pipeline against hand-written labels for a clip.

Every accuracy figure in this project comes from two SoccerNet broadcast
matches. Nothing measures whether any of it transfers, and the plausibility
checks in the run output explicitly cannot: "roughly 22 players, one ball,
human speeds" catches a pipeline that is badly wrong and not one that is
subtly so, which is the failure this project has actually had.

Hand labels fix that for one clip. This reads them and scores what they
can reach.

## The format, which is meant to survive being typed by a person

One event per line, a time and a name, times relative to the start of the
clip:

    00:04  out  touchline
    00:11  throw in
    0:52   shot
    1:03   out goal line
    69     corner
    02:13  goal

Blank lines and anything after `#` are ignored. Times may be `m:ss`,
`h:mm:ss` or plain seconds. Names are matched loosely -- "throw in",
"throw-in" and "throwin" are one thing -- because a label file is worth more
than the discipline needed to type it consistently.

Unknown names are reported rather than dropped silently. A line nobody
scores is usually a typo, and a typo in the truth is worse than one in the
code: it makes a working detector look broken.

## What can and cannot be scored here

**Shots are the prize.** The detector has never been shown to find a shot it
was not handed: no labelled footage available contains one, so its recall is
unmeasured while it ships numbers. One labelled shot changes that.

**Goals, corners and throw-ins** are scored where labelled.

**Out of play cannot be rescued by labelling.** On a camera that follows the
ball, the ball is at or past the edge of frame when it crosses -- at one
labelled crossing in the SoccerNet footage there is no ball in the picture
at all. The blind spot belongs to the camera. It is scored anyway, because
a measured zero is worth having written down.

    python score_hand_labels.py labels.txt --out output_veo
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# The tolerance every other event in this project is scored at.
TOLERANCE_S = 2.0

# Spellings that mean the same event. Matched after stripping anything that
# is not a letter, so "throw-in", "throw in" and "ThrowIn" all arrive here
# as "throwin".
ALIASES = {
    "shot": "shot", "shots": "shot", "attempt": "shot", "tir": "shot",
    "goal": "goal", "but": "goal", "scored": "goal",
    "out": "out", "outofplay": "out", "touchline": "out", "sortie": "out",
    "goalline": "out", "behind": "out",
    "throwin": "throw in", "throw": "throw in", "touche": "throw in",
    "corner": "corner", "cornerkick": "corner",
    "goalkick": "goal kick", "gk": "goal kick",
    "freekick": "free kick", "fk": "free kick", "foul": "free kick",
    "penalty": "penalty", "pen": "penalty",
    "save": "save", "offside": "offside", "kickoff": "kick off",
}

# What this can score, and with what.
SCOREABLE = ("shot", "goal", "out", "throw in", "corner", "goal kick")


def read_time(text: str) -> float | None:
    parts = text.split(":")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return values[0] * 60 + values[1]
    if len(values) == 3:
        return values[0] * 3600 + values[1] * 60 + values[2]
    return None


def read_labels(path: Path):
    """Parse a hand-written label file into events, and report what failed."""
    events, unknown, unparsed = [], [], []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        # Split on any run of whitespace: a person aligning columns types
        # two spaces, and partitioning on one leaves the rest starting with
        # a blank, which then reads as an empty event name.
        pieces = line.split(None, 1)
        head = pieces[0]
        rest = pieces[1].strip() if len(pieces) > 1 else ""
        when = read_time(head)
        if when is None:
            unparsed.append((number, raw.strip()))
            continue
        key = re.sub(r"[^a-z]", "", rest.lower())
        name = ALIASES.get(key)
        if name is None:
            # "out touchline" and "out goal line" both start with a word
            # that names the event; try the first word before giving up.
            words = rest.lower().split()
            first = re.sub(r"[^a-z]", "", words[0]) if words else ""
            name = ALIASES.get(first)
        if name is None:
            unknown.append((number, rest or "(nothing)"))
            continue
        events.append({"time_s": when, "event_type": name,
                       "detail": rest})
    return sorted(events, key=lambda e: e["time_s"]), unknown, unparsed


def match(found_times, truth_times, tolerance=TOLERANCE_S):
    """Greedy one-to-one matching, nearest first.

    One detection cannot satisfy two labels and one label cannot excuse two
    detections, which a per-event `any()` test quietly allows.
    """
    pairs = sorted(((abs(f - t), k, j)
                    for k, f in enumerate(found_times)
                    for j, t in enumerate(truth_times)
                    if abs(f - t) <= tolerance))
    used_found, used_truth, matched = set(), set(), []
    for gap, k, j in pairs:
        if k in used_found or j in used_truth:
            continue
        used_found.add(k)
        used_truth.add(j)
        matched.append((k, j, gap))
    return matched, used_found, used_truth


def report(name, found_times, truth_times):
    matched, _, _ = match(found_times, truth_times)
    hits = len(matched)
    recall = hits / len(truth_times) if truth_times else float("nan")
    precision = hits / len(found_times) if found_times else float("nan")
    if matched:
        delay = float(np.median([found_times[k] - truth_times[j]
                                 for k, j, _ in matched]))
        shown = f"{delay:+.1f}s"
    else:
        shown = "-"
    print(f"  {name:>12s} {len(truth_times):9d} {len(found_times):7d} "
          f"{hits:8d} {recall:8.2f} {precision:10.2f} {shown:>10s}")


def match_name(labels: Path):
    """The match a label file names in its header, or None.

    "# Stoke City - Huddersfield Town, from 13:02 of the match; ..." gives
    "Stoke City - Huddersfield Town".
    """
    try:
        first = labels.read_text().splitlines()[0]
    except (OSError, IndexError):
        return None
    if not first.startswith("#"):
        return None
    return first.lstrip("# ").split(",")[0].strip() or None


def goal_maps(out_dir, info, ball, args):
    """A placer built from the goal detector, calibrated by the corners.

    Two stages, and the division of labour is the point. The corners are
    clicked **once per clip** and, with the centre circle, fix the camera
    position, which a gantry camera keeps for the whole window. The
    detector then supplies a box on every sampled frame, and a box is
    enough to recover the rest of the pose once position is known.
    """
    import numpy as np
    from ultralytics import YOLO

    from check_goal_corners import load, poses_for_clip
    from src.goal_placer import build
    from src.ground_plane import GroundPlane

    rows = [r for r in load(Path(args.corners))
            if Path(r["out_dir"]).resolve() == Path(out_dir).resolve()]
    if not rows:
        print(f"  no corners in {args.corners} for {out_dir}")
        return None

    plane_path = Path(out_dir) / "ground_plane.json"
    blob = json.loads(plane_path.read_text()) if plane_path.exists() else {}
    plane = None if not blob or blob.get("refused") else GroundPlane(**blob)

    poses = poses_for_clip(rows, plane)
    if not poses:
        print("  the corners did not yield a pose for this clip")
        return None
    eye = np.mean([p.camera_position() for p in poses.values()], axis=0)
    seed = float(np.median([p.focal_px for p in poses.values()]))

    # The corners fix the camera only to a line. Where on it comes from the
    # centre circle, when the clip shows enough of it (`camera_position`).
    circles = None
    if not getattr(args, "old_camera", False):
        from check_goal_corners import located_poses, pixels, usable_rows
        from fit_pitch_anchor import halfway_line
        from probe_centre_circle import find_circle
        from src.camera_position import corner_ray, locate, scan_circles
        from src.goal_placer import POSE_GRID

        cx, cy = info["width"] / 2.0, info["height"] / 2.0
        ray = corner_ray([pixels(r) for r in usable_rows(rows)], cx, cy)
        grid = sorted({int(f) - int(f) % POSE_GRID
                       for f in ball.frame.tolist()})
        circles = scan_circles(info["path"], grid, find_circle, halfway_line)
        with_line = [c for c in circles.values() if c[1] is not None]
        found, report = (locate(with_line, ray, eye, seed, cx, cy)
                         if ray is not None else
                         (None, {"refused": "no corner line"}))
        store = getattr(args, "camera_store", None)
        match = match_name(Path(args.labels))
        clip = Path(out_dir).name.replace("output_", "")
        if found is None and store and ray is not None and match:
            # Located clips of the same match lend their camera's height,
            # which does not depend on which goal a clip calibrated from.
            from src.camera_position import borrow_height

            known = (json.loads(Path(store).read_text())
                     if Path(store).exists() else {})
            heights = [v["camera"][1] for k, v in known.items()
                       if k != clip and v.get("match") == match
                       and v.get("how") == "located"]
            if heights:
                borrowed, why = borrow_height(ray, heights)
                if borrowed is None:
                    print(f"  [camera] circle refused ({report.get('refused')})"
                          f"; borrowing {why['height_m']:.1f} m from "
                          f"{why['from']} clips of this match refused too: "
                          f"{why['refused']}")
                else:
                    print(f"  [camera] circle refused ({report.get('refused')})"
                          f"; placed at {why['height_m']:.1f} m, the height "
                          f"of {why['from']} located clips of this match: "
                          f"({borrowed[0]:.1f}, {borrowed[1]:.1f}, "
                          f"{borrowed[2]:.1f}), {why['off_halfway_m']:.1f} m "
                          f"from the halfway line, was ({eye[0]:.1f}, "
                          f"{eye[1]:.1f}, {eye[2]:.1f})")
                    found, report = borrowed, {"how": "borrowed height"}
        if found is None:
            print(f"  [camera] kept the corners' own position "
                  f"({eye[0]:.1f}, {eye[1]:.1f}, {eye[2]:.1f}): "
                  f"{report.get('refused')}")
        elif report.get("how") == "borrowed height":
            located = located_poses(rows, found)
            eye = found
            if located:
                seed = float(np.median([p.focal_px
                                        for p in located.values()]))
        else:
            if store and match:
                known = (json.loads(Path(store).read_text())
                         if Path(store).exists() else {})
                known[clip] = {"camera": [round(float(v), 2) for v in found],
                               "match": match, "how": "located"}
                Path(store).write_text(json.dumps(known, indent=1))
            located = located_poses(rows, found)
            print(f"  [camera] located at ({found[0]:.1f}, {found[1]:.1f}, "
                  f"{found[2]:.1f}), was ({eye[0]:.1f}, {eye[1]:.1f}, "
                  f"{eye[2]:.1f}); {report['agreeing']} of "
                  f"{report['tried']} circle frames cross the corners' line "
                  f"(median miss {report['miss_m']:.1f} m, spread "
                  f"{report['spread_m']:.1f} m)")
            eye = found
            if located:
                seed = float(np.median([p.focal_px
                                        for p in located.values()]))

    goal = build(info["path"], ball.frame.tolist(), eye, seed,
                 info["width"], info["height"], YOLO(args.goal_weights),
                 verbose=True)
    if getattr(args, "midfield", False):
        # The centre circle, refitted with the camera held where the goal
        # corners put it. Same frame as the goal, so no orientation check.
        from fit_pitch_anchor import halfway_line
        from probe_centre_circle import find_circle
        from src.goal_placer import LandmarkPlacer, build_midfield

        midfield = build_midfield(info["path"], ball.frame.tolist(), eye,
                                  seed, info["width"], info["height"],
                                  find_circle, halfway_line,
                                  skip=goal.poses, circles=circles,
                                  verbose=True)
        return LandmarkPlacer(goal, midfield)

    if not args.anchor_too:
        return goal

    # The anchor covers midfield, where the goal is out of frame. Combining
    # is refused unless the two can be shown to point the same way.
    import detect_shots
    from src.goal_placer import combine

    rng = np.random.default_rng(0)
    maps = detect_shots.anchors_for(Path(out_dir), info, ball.frame.tolist(),
                                    rng, detect_shots.ANCHOR_FRAMES)
    if not maps:
        print("  [both] no anchors on this clip; using the goal alone")
        return goal
    tracks = pd.read_parquet(Path(out_dir) / "tracks.parquet")
    return combine(goal, detect_shots.AnchorPlacer(maps),
                   sorted(goal.poses), info["width"], info["height"],
                   players=tracks[tracks.cls == "player"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", help="the hand-written label file")
    ap.add_argument("--out", default="output_veo",
                    help="the pipeline's output directory for that clip")
    ap.add_argument("--no-reproduce", dest="reproduce", action="store_false",
                    help="accept an anchor from a single fit, as this did "
                         "before the reproducibility gate shipped")
    ap.add_argument("--frames", type=int, default=None,
                    help="anchor budget; defaults to the shipped value")
    ap.add_argument("--goal-weights",
                    help="a trained goal detector. With --corners, events "
                         "are placed from the goal instead of the centre "
                         "circle. The weights are trained on footage under "
                         "a non-commercial agreement and are therefore not "
                         "in this repository; supply your own.")
    ap.add_argument("--midfield", action="store_true",
                    help="also place the ball at midfield, from the centre "
                         "circle refitted with the camera held")
    ap.add_argument("--old-camera", action="store_true",
                    help="hold the camera where the goal corners alone put "
                         "it, instead of locating it with the centre circle "
                         "(for comparison)")
    ap.add_argument("--camera-store",
                    help="JSON shared across runs: each located camera is "
                         "recorded with its match, and a clip the centre "
                         "circle cannot locate borrows the camera height of "
                         "located clips from the same match")
    ap.add_argument("--anchor-too", action="store_true",
                    help="also use the centre-circle anchor where the goal "
                         "is out of frame, if the two can be shown to point "
                         "the same way")
    ap.add_argument("--corners",
                    help="hand-clicked goal corners, which calibrate the "
                         "clip's camera position once so every frame's pose "
                         "can come from a detected box")
    args = ap.parse_args()

    path = Path(args.labels)
    if not path.exists():
        raise SystemExit(f"no label file at {path}")
    truth, unknown, unparsed = read_labels(path)

    print(f"Read {len(truth)} labelled events from {path.name}.\n")
    for number, text in unparsed:
        print(f"  line {number}: no time at the start -- {text!r}")
    for number, text in unknown:
        print(f"  line {number}: don't know the event {text!r}")
    if unknown or unparsed:
        print("  (those lines are not scored; fix or ignore as you like)\n")

    counts = pd.Series([e["event_type"] for e in truth]).value_counts()
    print("  labelled:", ", ".join(f"{n} {k}" for k, n in counts.items()))

    out_dir = Path(args.out)
    if not (out_dir / "clip.json").exists():
        raise SystemExit(f"no pipeline output in {out_dir} -- run the "
                         f"pipeline on that clip first")
    info = json.loads((out_dir / "clip.json").read_text())

    import detect_ball_events as ball_events
    import detect_set_pieces as set_pieces
    import detect_shots

    ball = detect_shots.ball_track(out_dir)
    if ball.empty:
        raise SystemExit("no ball track in that output directory")
    if args.goal_weights and args.corners:
        maps = goal_maps(out_dir, info, ball, args)
        if maps is None:
            raise SystemExit("could not calibrate this clip from the corners")
        print(f"  placing from the goal: poses on {len(maps)} sampled "
              f"frames\n")
    else:
        rng = np.random.default_rng(0)
        frames = args.frames or detect_shots.ANCHOR_FRAMES
        maps = detect_shots.anchors_for(out_dir, info, ball.frame.tolist(),
                                        rng, frames,
                                        reproduce=args.reproduce)
    shots, placed = detect_shots.find_shots(ball, maps, info["fps"])
    if hasattr(maps, "pose_at"):
        # With a camera pose per frame, shots are read where they cross the
        # goal plane (`src/goal_plane.py`): a ball in the air is placed far
        # beyond itself on the grass, and five of the twelve labelled shot
        # and goal moments were in the air.
        from src import goal_plane

        shots = goal_plane.find_shots(ball, maps, info["fps"])
    shots = detect_shots.score(detect_shots.attribute(shots, out_dir, ball))
    stoppages, _ = ball_events.find_ball_events(ball, maps, info["fps"])
    outs = [e for e in stoppages if e["event_type"] == "out_of_play"]
    goals = [e for e in stoppages if e["event_type"] == "goal"]
    if hasattr(maps, "pose_at"):
        # A goal is an on-target crossing followed by the ball in the net,
        # read at the goal plane; the grass-crossing rule above misses any
        # goal scored in the air. Timed at the line, not the strike.
        goals = [{"time_s": s["crossing_s"], "event_type": "goal"}
                 for s in shots if s.get("outcome") == "goal"]
    restarts = set_pieces.find_restarts(ball, maps, info["fps"], outs)

    print(f"\n  ball placed on the pitch: {placed} of {len(ball)} "
          f"({placed / max(len(ball), 1):.0%})")
    if hasattr(maps, "used"):
        # Counted across every detector that asked, so these are placement
        # requests rather than distinct ball positions; the split is what
        # matters.
        total = max(sum(maps.used.values()), 1)
        print("  placed by: " + ", ".join(
            f"{k} {v / total:.0%}" for k, v in maps.used.items()))
    print()
    print(f"  {'event':>12s} {'labelled':>9s} {'found':>7s} {'matched':>8s} "
          f"{'recall':>8s} {'precision':>10s} {'delay':>10s}")

    def times(kind):
        return [e["time_s"] for e in truth if e["event_type"] == kind]

    report("shot", [s["time_s"] for s in shots], times("shot"))
    report("goal", [g["time_s"] for g in goals], times("goal"))
    report("out", [o["time_s"] for o in outs], times("out"))
    for kind in ("throw in", "corner", "goal kick"):
        report(kind, [r["time_s"] for r in restarts
                      if r["event_type"] == kind], times(kind))

    if times("shot") and shots:
        print("\n  Shots found, with what the model makes of them:")
        for shot in shots:
            extra = f", xG {shot['xg']:.3f}" if "xg" in shot else ""
            print(f"    t={shot['time_s']:6.1f}s  "
                  f"{shot['distance_m']:5.1f} m, "
                  f"{shot['speed_ms']:4.0f} m/s{extra}")

    print("\n  Shots are read where they cross the goal plane; across six "
          "labelled\n  windows that finds 3 of 10 shots and 1 of 2 goals "
          "(EVENT_ACCURACY.md).\n  The 'goal' row is a separate detector, "
          "the ball crossing the line on\n  the grass, and misses goals "
          "scored in the air. Out of play is expected\n  to be thin on a "
          "camera that follows the ball; a measured zero is still\n  worth "
          "writing down.")


if __name__ == "__main__":
    main()
