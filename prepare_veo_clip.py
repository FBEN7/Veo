"""Cut an upload-sized clip out of a Veo panorama, on your machine.

Everything measured in this repository comes from two broadcast matches, and
the clip called "Veo" here is a 640x360 auto-follow crop with a scoreboard
burnt into it. That crop is the reason out-of-play cannot be detected: a
camera that chases the ball has not caught up when the ball leaves the
pitch, so there is nothing in frame at the moment that matters.

The panorama does not have that problem, and it removes more besides -- a
camera that never moves has one homography for the whole match, so most of
what this project has fought with stops existing.

This cuts clips small enough to upload while keeping the resolution, which
is the part that matters. Run it where the panorama is.

    python prepare_veo_clip.py match.mp4 --at 12:30 --seconds 90
    python prepare_veo_clip.py match.mp4 --at 12:30 --at 47:05 --at 68:20

Needs ffmpeg on the PATH. Nothing else, and no part of this repository.

Not run end to end before being committed: the container this was written in
has no ffmpeg, so the argument handling and the time parsing are tested and
the encode is not. If the first clip comes out wrong, say so rather than
working around it.

## What to choose

Pick passages containing what cannot currently be measured:

  * the ball going out and coming back -- a throw-in, a corner, a goal kick;
  * a shot, ideally one that scores;
  * two minutes of ordinary play, for possession and team assignment.

Three clips of ninety seconds is far more useful than one of five minutes,
because they can cover three different things.

## Resolution, not length

If a clip will not fit, shorten it rather than shrinking it. Half the
picture's width is half the pixels on the ball, and the ball in the current
footage is already only about eight pixels across.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# The upload limit for a session, with room for container overhead.
TARGET_MB = 28.0

# Audio is not used anywhere in this pipeline and costs bitrate the picture
# could have had.
DROP_AUDIO = True


def seconds(stamp: str) -> float:
    """Accept 90, 1:30 or 01:30:00."""
    parts = [float(p) for p in str(stamp).split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"cannot read a time from {stamp!r}")


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    width, height, rate = out.split(",")[:3]
    num, _, den = rate.partition("/")
    fps = float(num) / float(den or 1)
    return {"width": int(width), "height": int(height), "fps": fps}


def cut(source: Path, start_s: float, length_s: float, out_path: Path,
        target_mb: float) -> None:
    """One clip, encoded to land just under the limit at full resolution."""
    # Bits available for the picture, over the clip's length.
    bitrate = int(target_mb * 8 * 1024 * 1024 / length_s)
    command = [
        "ffmpeg", "-y",
        # Seeking before -i is fast; re-encoding below makes it frame-exact.
        "-ss", f"{start_s:.3f}", "-i", str(source), "-t", f"{length_s:.3f}",
        "-c:v", "libx264", "-preset", "slow",
        "-b:v", str(bitrate), "-maxrate", str(int(bitrate * 1.3)),
        "-bufsize", str(int(bitrate * 2)),
        "-pix_fmt", "yuv420p",
    ]
    command += ["-an"] if DROP_AUDIO else []
    command += [str(out_path)]
    subprocess.run(command, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="the panoramic recording")
    ap.add_argument("--at", action="append", required=True,
                    help="start time, as 12:30 or 750; repeatable")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--out", default="veo_clips")
    ap.add_argument("--target-mb", type=float, default=TARGET_MB)
    args = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg and ffprobe need to be on the PATH.")
    source = Path(args.source)
    if not source.exists():
        sys.exit(f"no such file: {source}")

    info = probe(source)
    print(f"{source.name}: {info['width']}x{info['height']} @ "
          f"{info['fps']:.0f} fps")
    if info["width"] < 1920:
        print("  NOTE: that is narrower than a Veo panorama usually is. If "
              "this is the\n        auto-follow edit rather than the full "
              "recording, it has the same\n        blind spot as broadcast "
              "and will not answer the question.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stamp in args.at:
        start = seconds(stamp)
        name = f"veo_{str(stamp).replace(':', '')}_{int(args.seconds)}s.mp4"
        target = out_dir / name
        print(f"  cutting {args.seconds:.0f}s from {stamp} -> {name}",
              flush=True)
        cut(source, start, args.seconds, target, args.target_mb)
        size_mb = target.stat().st_size / 1e6
        flag = "" if size_mb <= 30 else "   TOO BIG -- use fewer seconds"
        print(f"    {size_mb:.1f} MB{flag}")

    print(f"\nClips are in {out_dir}/. Upload them, and a text file of what "
          f"happens in\nthem, one event per line, times relative to the "
          f"start of each clip:\n\n"
          f"    00:04  out  touchline\n"
          f"    00:11  throw in\n"
          f"    00:52  shot\n"
          f"    01:03  out  goal line\n"
          f"    01:09  corner\n\n"
          f"Rough seconds are fine -- everything here is scored at a "
          f"two-second\ntolerance. Twenty lines is enough to measure what "
          f"two broadcast matches\ncurrently stand in for.")


if __name__ == "__main__":
    main()
