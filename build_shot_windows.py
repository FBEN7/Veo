"""Cut a shot test set out of a full match, so shots become measurable at all.

Every accuracy figure in this project rests on four 90-second windows, and
those windows contain **zero shots and zero goals**. Not few -- none. So a
shot detector cannot currently be shown to work or caught failing, and
nothing downstream of one (goals, assists, chances created, xG) can be
validated either.

The labels already hold what is needed. Across the two SoccerNet matches:

    Stoke - Huddersfield    21 SHOT + 1 GOAL
    Reading - Fulham        23 SHOT + 5 GOAL

all 50 marked `visible`, every one carrying the team that took it. What is
missing is video at those moments: the label files describe whole matches and
only 90-second clips were ever cut from them.

This builds the missing test set. Windows are cut around each labelled shot,
plus control windows containing none, and a manifest records each clip's
match time so the existing scorer reads it with no changes.

Two modes, because the full matches are large and this container is not:

    --plan      needs no video at all. Writes the exact ffmpeg commands to
                cut the clips, to run wherever the match files already are.

    default     cuts here with OpenCV, when the match file is on this machine.

Both cutting paths are frame-accurate, which took two attempts. Stream-copy
(`-c copy`) is faster, smaller and lossless, and it snaps the cut to the
nearest keyframe -- one to two seconds on broadcast H.264. Event scoring
matches at a tolerance of one second, and every label in the clip is placed
relative to its first frame, so a keyframe-snapped clip silently shifts the
entire ground truth. The scorer already warns that getting this offset wrong
"misaligns every label against every detection". So the plan re-encodes,
which makes `-ss` exact, and the OpenCV path reads the match through once in
order rather than seeking, because seeking compressed video is not
frame-accurate either -- a fact this codebase learned in `validate_motion`.

Controls are not optional. A detector that fires constantly scores perfectly
on clips chosen because they contain a shot; precision is only meaningful
against stretches where the answer is no.

    python build_shot_windows.py --labels Labels-ball.json --match stoke --plan
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
from pathlib import Path

SHOT_LABELS = ("SHOT", "GOAL")

# Seconds of build-up kept before each shot, and of aftermath after it. The
# build-up is the longer side on purpose: a shot is recognised from the ball
# leaving a player at speed, and the pass or carry that set it up is part of
# what distinguishes it from a clearance.
DEFAULT_BEFORE_S = 40.0
DEFAULT_AFTER_S = 20.0

# Shots closer together than this share one window rather than producing two
# overlapping clips of nearly the same footage.
MERGE_WITHIN_S = 30.0

# A control window must be at least this far from any labelled shot, so that
# "no shot here" is true with margin rather than by a second.
CONTROL_CLEARANCE_S = 25.0


def load_shots(labels_path: Path):
    """Visible SHOT and GOAL labels, in seconds of match video."""
    data = json.loads(labels_path.read_text())
    out = []
    for ann in data.get("annotations", []):
        if ann.get("label") not in SHOT_LABELS:
            continue
        # A label the broadcast never showed cannot be detected from the
        # broadcast, and counting it against a detector would be measuring
        # the director. All 50 in this corpus are visible; the filter is here
        # so that stays true rather than being assumed.
        if ann.get("visibility") not in (None, "visible"):
            continue
        out.append(dict(t=float(ann["position"]) / 1000.0,
                        label=ann["label"],
                        team=ann.get("team")))
    return sorted(out, key=lambda a: a["t"])


def cluster(shots, merge_within: float = MERGE_WITHIN_S):
    """Group shots close enough to share a window."""
    groups = []
    for shot in shots:
        if groups and shot["t"] - groups[-1][-1]["t"] <= merge_within:
            groups[-1].append(shot)
        else:
            groups.append([shot])
    return groups


def plan_windows(shots, duration_s, before, after, n_controls, seed=0):
    """Shot windows and control windows, as (kind, start, length, shots)."""
    groups = cluster(shots)
    windows = []
    for group in groups:
        start = max(0.0, group[0]["t"] - before)
        end = min(duration_s, group[-1]["t"] + after)
        windows.append(dict(kind="shot", start=start, length=end - start,
                            shots=group))

    # Controls: sampled where no shot is anywhere near, and not overlapping a
    # shot window already chosen.
    rng = random.Random(seed)
    length = before + after
    taken = [(w["start"], w["start"] + w["length"]) for w in windows]
    shot_times = [s["t"] for s in shots]
    controls, attempts = [], 0
    while len(controls) < n_controls and attempts < n_controls * 200:
        attempts += 1
        start = rng.uniform(0.0, max(0.0, duration_s - length))
        end = start + length
        if any(start - CONTROL_CLEARANCE_S < t < end + CONTROL_CLEARANCE_S
               for t in shot_times):
            continue
        if any(start < b and a < end for a, b in taken):
            continue
        taken.append((start, end))
        controls.append(dict(kind="control", start=start, length=length,
                             shots=[]))
    windows.extend(controls)
    return sorted(windows, key=lambda w: w["start"])


def name_for(match: str, window) -> str:
    mm, ss = divmod(int(window["start"]), 60)
    return f"{match}_{window['kind']}_{mm:03d}m{ss:02d}s"


def write_manifest(out_dir: Path, match: str, labels_path: Path, windows,
                   video_name: str):
    entries = []
    for w in windows:
        entries.append(dict(
            name=name_for(match, w),
            clip=f"{name_for(match, w)}.mp4",
            offset_s=round(w["start"], 3),
            duration_s=round(w["length"], 3),
            kind=w["kind"],
            labels=str(labels_path),
            source_video=video_name,
            shots=[dict(t=round(s["t"], 3), label=s["label"], team=s["team"])
                   for s in w["shots"]],
        ))
    path = out_dir / f"manifest_{match}.json"
    path.write_text(json.dumps(entries, indent=2))
    return path, entries


def cut_with_opencv(video: Path, out_dir: Path, match: str, windows):
    """Cut every window in one ordered pass over the match.

    Deliberately not `cap.set(CAP_PROP_POS_FRAMES, ...)` per window. Seeking
    compressed video lands on a keyframe rather than the frame asked for, and
    a clip that starts a second early carries every one of its labels a
    second out of place. Reading straight through costs one decode of the
    match and is exact.
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not fps or fps <= 0:
        raise SystemExit(f"could not read frame rate from {video}")

    plan = []
    for w in windows:
        plan.append(dict(name=name_for(match, w),
                         first=int(round(w["start"] * fps)),
                         last=int(round((w["start"] + w["length"]) * fps)),
                         writer=None, written=0))
    last_frame = max(p["last"] for p in plan)

    index, open_writers = 0, 0
    while index <= last_frame:
        ok, frame = cap.read()
        if not ok:
            break
        for p in plan:
            if p["first"] <= index < p["last"]:
                if p["writer"] is None:
                    dest = out_dir / f"{p['name']}.mp4"
                    p["writer"] = cv2.VideoWriter(
                        str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                        (width, height))
                    if not p["writer"].isOpened():
                        raise SystemExit(
                            "OpenCV could not open a video writer")
                    open_writers += 1
                p["writer"].write(frame)
                p["written"] += 1
            elif index >= p["last"] and p["writer"] is not None:
                p["writer"].release()
                p["writer"] = None
                open_writers -= 1
                print(f"  {p['name']}.mp4: {p['written']} frames")
        index += 1

    for p in plan:
        if p["writer"] is not None:
            p["writer"].release()
            print(f"  {p['name']}.mp4: {p['written']} frames")
    cap.release()


def write_plan(out_dir: Path, match: str, windows, video_name: str):
    """ffmpeg commands to cut the same windows wherever the match lives."""
    lines = [
        "#!/bin/sh",
        "# Cut the shot test set. Run this where the full match file is.",
        "#",
        "# This re-encodes rather than stream-copying, on purpose. With -c",
        "# copy the cut snaps to the nearest keyframe -- one to two seconds on",
        "# broadcast H.264 -- and every label in the clip is positioned",
        "# relative to its first frame, so the whole ground truth would shift",
        "# by more than the tolerance events are scored at. Putting -ss before",
        "# -i still seeks fast; re-encoding is what makes it land on the exact",
        "# frame. CRF 20 is visually lossless for this purpose and audio is",
        "# dropped, so the clips stay small enough to move around.",
        "set -e",
        f'VIDEO="{video_name}"',
        'OUT="."',
        "",
    ]
    for w in windows:
        name = name_for(match, w)
        dest = shlex.quote(f"{name}.mp4")
        lines.append(
            f'ffmpeg -nostdin -y -ss {w["start"]:.3f} -i "$VIDEO" '
            f'-t {w["length"]:.3f} -c:v libx264 -crf 20 -preset veryfast '
            f'-an -sn "$OUT"/{dest}')
    path = out_dir / f"cut_{match}.sh"
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)
    return path


def self_check(video: Path, out_dir: Path):
    """Cut two windows from a known clip and prove the frames line up.

    A cut that starts a second early is invisible: the clip plays, the
    pipeline runs, every number comes out, and every label is a second out of
    place. Comparing a cut frame against its source is not enough on its own
    because the writer is lossy, so this compares against a neighbourhood and
    checks the *best* match is the frame that was asked for.
    """
    import cv2
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    windows = [dict(kind="shot", start=12.0, length=4.0, shots=[]),
               dict(kind="control", start=40.0, length=4.0, shots=[])]
    cut_with_opencv(video, out_dir, "selfcheck", windows)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    failures = 0
    for w in windows:
        name = f"{name_for('selfcheck', w)}.mp4"
        want = int(round(w["start"] * fps))
        clip = cv2.VideoCapture(str(out_dir / name))
        ok, got = clip.read()
        clip.release()
        if not ok:
            print(f"  {name}: unreadable")
            failures += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        diffs = {}
        for i in range(want + 5):
            ok2, frame = cap.read()
            if not ok2:
                break
            if want - 4 <= i <= want + 4:
                diffs[i] = float(np.abs(frame.astype(int)
                                        - got.astype(int)).mean())
        best = min(diffs, key=diffs.get)
        verdict = "exact" if best == want else "MISALIGNED"
        print(f"  {name}: best match {best - want:+d} frames from the one "
              f"asked for -- {verdict}")
        if best != want:
            failures += 1
    cap.release()

    if failures:
        raise SystemExit(f"{failures} window(s) did not land on the right "
                         "frame; the labels would be shifted")
    print("  Cutting is frame-accurate.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true",
                    help="prove the cutter is frame-accurate and exit")
    ap.add_argument("--labels")
    ap.add_argument("--match",
                    help="short name used to prefix the clips, e.g. stoke")
    ap.add_argument("--video", help="the full match file, if it is here")
    ap.add_argument("--out", default="shot_windows")
    ap.add_argument("--before", type=float, default=DEFAULT_BEFORE_S)
    ap.add_argument("--after", type=float, default=DEFAULT_AFTER_S)
    ap.add_argument("--controls", type=int, default=15,
                    help="windows containing no shot, for precision")
    ap.add_argument("--plan", action="store_true",
                    help="write ffmpeg commands instead of cutting here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.self_check:
        video = args.video
        if not video:
            probe = Path("output_soccernet/clip.json")
            if not probe.exists():
                raise SystemExit("--self-check needs --video, or a cached "
                                 "output_soccernet/clip.json to borrow")
            video = json.loads(probe.read_text())["path"]
        print(f"Cutting two windows from {Path(video).name} and checking "
              "where they land.\n")
        self_check(Path(video), Path(args.out) / "selfcheck")
        return

    if not args.labels or not args.match:
        raise SystemExit("--labels and --match are required")
    labels_path = Path(args.labels)
    shots = load_shots(labels_path)
    if not shots:
        raise SystemExit(f"no visible SHOT or GOAL labels in {labels_path}")

    data = json.loads(labels_path.read_text())
    duration_s = max(float(a["position"]) / 1000.0
                     for a in data["annotations"]) + 30.0

    windows = plan_windows(shots, duration_s, args.before, args.after,
                           args.controls, args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(args.video).name if args.video else "FULL_MATCH.mp4"
    manifest, entries = write_manifest(out_dir, args.match, labels_path,
                                       windows, video_name)

    n_shot = sum(1 for w in windows if w["kind"] == "shot")
    n_ctrl = sum(1 for w in windows if w["kind"] == "control")
    total_s = sum(w["length"] for w in windows)
    goals = sum(1 for s in shots if s["label"] == "GOAL")

    print(f"{data.get('UrlLocal', labels_path.name)}")
    print(f"  {len(shots)} visible shot events ({goals} goals) over "
          f"{duration_s / 60:.0f} min")
    print(f"  -> {n_shot} shot windows + {n_ctrl} control windows, "
          f"{total_s / 60:.0f} min of video")
    print(f"  manifest: {manifest}")

    if args.plan or not args.video:
        script = write_plan(out_dir, args.match, windows, video_name)
        print(f"  cut script: {script}")
        if not args.video and not args.plan:
            print("\n  No --video given, so only the plan was written. Run the "
                  "script\n  where the match file is, then bring the clips "
                  "back next to the manifest.")
        return

    print(f"  cutting from {args.video} ...")
    cut_with_opencv(Path(args.video), out_dir, args.match, windows)


if __name__ == "__main__":
    main()
