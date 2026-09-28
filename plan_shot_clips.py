"""Find the windows of a labelled match that contain shots, and cut them.

Shot recall is the largest hole in this project. The detector ships numbers
and has never been shown to find a shot it was not handed: an injected
control proves the geometry works when a shot is present and tracked, and
says nothing about whether real shots are found. The reason is mundane --
the clips that exist bracket the labelled shots without containing any.

The labels know where the shots are. This reads them, packs the shots into
windows small enough to upload, and cuts those windows out of the match.

## What it optimises

Shots per window, not windows per shot. A window holding three labelled
shots measures three times as much recall per megabyte as one holding one,
and shots cluster -- a spell of pressure produces several inside a minute.
So windows are packed greedily from the densest cluster down, and ranked.

Each window also reports the other labelled events inside it, because they
come free. A window with OUT and THROW IN labels measures the crossing
detector at the same time, on footage chosen for a different reason, which
is a fairer test of it than footage chosen for crossings.

## One timeline, not two halves

SoccerNet normally ships a file per half with positions measured inside it,
and this script was written expecting that. These label files are not like
that: every annotation is tagged half "1" and the positions run continuously
to 101 minutes, with `halftime` marking where the break falls on that same
timeline. The clips already measured here confirm it -- the Reading window
sits at 51:15, past that match's 47:10 half-time, and its labels line up on
a continuous offset.

So the script reads which layout a file uses instead of assuming, says
which it found, and only asks for a half when the labels are actually split
by one. Getting this wrong does not fail loudly: it cuts a different minute
of the match and the detector looks broken.

## Licence

This reads your own copy of your own labels and cuts your own copy of the
video. It prints timings and writes clips locally; nothing derived from the
footage is committed here, which is the rule this repository has followed
throughout. The SoccerNet terms are non-commercial, and the data is not to
be passed to anyone who has not signed for it themselves.

    python plan_shot_clips.py --labels Labels-ball.json
    python plan_shot_clips.py --labels Labels-ball.json --cut 1_720p.mkv --half 1

Cutting needs ffmpeg on the PATH. The planning half is tested here; the
encode is not, because this container has no ffmpeg. If the first clip comes
out wrong, say so rather than working around it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

SHOT_LABELS = ("SHOT", "GOAL")

# 90 seconds at 720p lands at 21-27 MB, which is the shape of every clip
# already measured here and fits a 30 MB upload.
WINDOW_S = 90.0

# Seconds of run-up kept before the first shot in a window. A shot needs its
# build-up: the detector reads the ball's flight from before it is struck,
# and the pass that made the chance has to be in shot to be linked to it.
LEAD_S = 20.0


def read_events(path: Path):
    """Every labelled event, and whether the file is one timeline or two.

    Returns (events, split_by_half). When it is one timeline, every event is
    filed under half 1 and its time is measured from the start of the match.
    """
    blob = json.loads(path.read_text())
    out = []
    for row in blob.get("annotations", []):
        game_time = str(row.get("gameTime", ""))
        half = game_time.split("-")[0].strip()
        if not half.isdigit():
            continue
        out.append({"half": int(half),
                    "time_s": int(row["position"]) / 1000.0,
                    "label": row["label"],
                    "team": row.get("team"),
                    "visibility": row.get("visibility", "visible")})
    halves = {e["half"] for e in out}
    longest = max((e["time_s"] for e in out), default=0.0)
    break_at = halftime_seconds(blob.get("halftime"))
    # Two markers of a continuous file: only one half is ever named, and the
    # timeline runs well past where half-time was marked.
    split = len(halves) > 1 or (break_at is not None
                                and longest <= break_at + 60.0)
    return sorted(out, key=lambda e: (e["half"], e["time_s"])), split


def halftime_seconds(marker) -> float | None:
    """The `halftime` field, "1 - 49:25", as seconds."""
    if not marker or "-" not in str(marker):
        return None
    clock_part = str(marker).split("-", 1)[1].strip()
    try:
        minutes, seconds = clock_part.split(":")
        return int(minutes) * 60 + int(seconds)
    except ValueError:
        return None


def pack(events, window_s: float, lead_s: float):
    """Greedy windows, each starting `lead_s` before a shot it contains."""
    shots = [e for e in events
             if e["label"] in SHOT_LABELS and e["visibility"] == "visible"]
    windows, taken = [], set()
    for k, shot in enumerate(shots):
        if k in taken:
            continue
        start = max(0.0, shot["time_s"] - lead_s)
        end = start + window_s
        inside = [j for j, other in enumerate(shots)
                  if j not in taken and other["half"] == shot["half"]
                  and start <= other["time_s"] <= end]
        taken.update(inside)
        held = [shots[j] for j in inside]
        others = [e for e in events
                  if e["half"] == shot["half"] and start <= e["time_s"] <= end
                  and e["label"] not in SHOT_LABELS]
        windows.append({"half": shot["half"], "start_s": start,
                        "seconds": window_s, "shots": held,
                        "others": others})
    # Densest first: a window with three shots is worth three times as much
    # per megabyte as one with a single shot.
    windows.sort(key=lambda w: (-len(w["shots"]), w["half"], w["start_s"]))
    return windows


def clock(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def describe(windows, top: int, split: bool):
    half_head = f"{'half':>5s}" if split else ""
    print(f"  {'#':>3s}{half_head} {'from':>7s} {'to':>7s} {'shots':>6s} "
          f"  what else is labelled inside")
    for index, window in enumerate(windows[:top], start=1):
        kinds = {}
        for event in window["others"]:
            kinds[event["label"]] = kinds.get(event["label"], 0) + 1
        interesting = {k: v for k, v in kinds.items()
                       if k in ("OUT", "THROW IN", "CROSS", "FREE KICK",
                                "PLAYER SUCCESSFUL TACKLE")}
        shots = ", ".join(f"{e['label'].lower()} at {clock(e['time_s'])}"
                          for e in window["shots"])
        half_cell = f"{window['half']:5d}" if split else ""
        print(f"  {index:3d}{half_cell} "
              f"{clock(window['start_s']):>7s} "
              f"{clock(window['start_s'] + window['seconds']):>7s} "
              f"{len(window['shots']):6d}   "
              + ", ".join(f"{v} {k.lower()}" for k, v in
                          sorted(interesting.items())))
        print(f"      {shots}")


def cut(video: Path, window, out_dir: Path, target_mb: float,
        split: bool) -> Path:
    bitrate = int(target_mb * 8 * 1024 * 1024 / window["seconds"])
    target = out_dir / (window_name(window, split) + ".mp4")
    command = [
        "ffmpeg", "-y", "-ss", f"{window['start_s']:.3f}", "-i", str(video),
        "-t", f"{window['seconds']:.3f}",
        "-c:v", "libx264", "-preset", "slow", "-b:v", str(bitrate),
        "-maxrate", str(int(bitrate * 1.3)), "-bufsize", str(int(bitrate * 2)),
        "-pix_fmt", "yuv420p", "-an", str(target),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return target


def window_name(window, split: bool) -> str:
    """A name that says where the clip came from."""
    where = (f"half{window['half']}_" if split else "m")
    return (f"{where}{clock(window['start_s']).replace(':', '')}"
            f"_{len(window['shots'])}shots")


def write_truth(windows, out_dir: Path, split: bool = False):
    """A label file per window, in the format score_hand_labels.py reads."""
    written = []
    for window in windows:
        name = window_name(window, split) + ".txt"
        origin = (f"half {window['half']}" if split else "the match")
        lines = [f"# from {clock(window['start_s'])} of {origin}, "
                 f"times relative to the clip"]
        for event in sorted(window["shots"] + window["others"],
                            key=lambda e: e["time_s"]):
            if event["label"] not in SHOT_LABELS and event["label"] not in (
                    "OUT", "THROW IN", "CORNER", "FREE KICK"):
                continue
            offset = event["time_s"] - window["start_s"]
            lines.append(f"{clock(offset)}  {event['label'].lower()}")
        path = out_dir / name
        path.write_text("\n".join(lines) + "\n")
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True,
                    help="a SoccerNet Labels-ball.json for the match")
    ap.add_argument("--seconds", type=float, default=WINDOW_S)
    ap.add_argument("--lead", type=float, default=LEAD_S)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--cut", default=None,
                    help="the video for one half; cuts the windows for it")
    ap.add_argument("--half", type=int, default=None,
                    help="which half --cut is; only needed when the labels "
                         "are actually split by half, which this reports")
    ap.add_argument("--take", type=int, default=3,
                    help="how many of the ranked windows to cut")
    ap.add_argument("--out", default="shot_clips")
    ap.add_argument("--target-mb", type=float, default=28.0)
    args = ap.parse_args()

    labels = Path(args.labels)
    if not labels.exists():
        sys.exit(f"no label file at {labels}")
    events, split = read_events(labels)
    windows = pack(events, args.seconds, args.lead)

    total_shots = sum(len(w["shots"]) for w in windows)
    layout = ("split into halves, so each window names the half it belongs to"
              if split else
              "one continuous timeline for the whole match, so times run "
              "from kick-off")
    print(f"{labels.name}: {total_shots} visible shots and goals, "
          f"packed into {len(windows)} windows of {args.seconds:.0f} s.\n"
          f"  These labels are {layout}.\n")
    describe(windows, args.top, split)

    if not args.cut:
        print(f"\n  Nothing cut. Add --cut <the match video> to cut these "
              f"windows out of it\n  (and --half, if the line above says "
              f"the labels are split by half).")
        return

    if split and args.half is None:
        sys.exit("these labels are split by half, so --cut needs --half")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg needs to be on the PATH to cut")
    video = Path(args.cut)
    if not video.exists():
        sys.exit(f"no video at {video}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted_half = args.half if split else None
    chosen = [w for w in windows
              if wanted_half is None or w["half"] == wanted_half][:args.take]
    if not chosen:
        sys.exit(f"no windows in half {args.half}")

    print(f"\n  cutting {len(chosen)} window(s) from {video.name}:")
    for window in chosen:
        target = cut(video, window, out_dir, args.target_mb, split)
        size_mb = target.stat().st_size / 1e6
        flag = "" if size_mb <= 30 else "   TOO BIG -- lower --seconds"
        print(f"    {target.name}  {size_mb:.1f} MB{flag}")

    truth = write_truth(chosen, out_dir, split)
    print(f"\n  and a label file beside each one:")
    for path in truth:
        print(f"    {path.name}")
    print(f"\n  Upload the clips and their label files together. The labels "
          f"are already\n  in the format score_hand_labels.py reads, so "
          f"nothing needs typing.")


if __name__ == "__main__":
    main()
