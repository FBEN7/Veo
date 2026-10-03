"""Run `main.py` end to end on a synthetic match, and check what comes out.

Nothing in this repository exercised `main.py`. Every measurement runs
through `score_soccernet.py` or `run_veo_analysis.py`, which share the
detectors but not the script that actually produces the report. That gap let
a real bug ship: the possession speed gate asked for the ball's speed from
columns that hold **pixels** in this path and **metres** in the measured
one, so it compared an inflated speed against a threshold in km/h and threw
out 95% of the frames. The report then dropped the possession figure
silently, because it reads "no frames" as "no data".

That bug was found by reading the code. This is the test that would have
found it.

## A synthetic match, not a real one

Twenty-two players on known paths, a ball passed between them, rendered as
coloured rectangles on green so that team assignment has colours to cluster
and re-identification has crops to compare. It is not football and it is not
meant to be: the assertions below are about **contracts and orders of
magnitude**, which is what an end-to-end test can hold without becoming a
second implementation of the pipeline.

## What is asserted, and why each one

  * **the report exists and is not empty.** The weakest possible check, and
    it catches an exception in any of the eight stages between tracks and
    PDF;
  * **players are placed on the pitch in metres.** The coordinate contract
    the bug broke: `to_pitch_coords` leaves `px`/`py` in pixels and puts
    metres in `x`/`y`, and anything reading the wrong pair is out by the
    pixels-per-metre scale;
  * **possession is reported at all, over a decent share of frames.** The
    specific regression. A gate that throws out almost everything leaves a
    share that is arithmetically fine and means nothing;
  * **speeds and distances are human.** Catches a scale error anywhere in
    the chain, which is the failure this project has had most often.

Slow, because it runs the real thing. Not part of any fast loop.

    python test_main_pipeline.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent

WIDTH, HEIGHT, FPS = 960, 540, 25
SECONDS = 12
PITCH_LENGTH_M, PITCH_WIDTH_M = 105.0, 68.0

# Pixels per metre, so the synthetic match has a known scale to recover.
PX_PER_M = 8.0

TEAM_COLOURS = ((40, 40, 220), (220, 80, 40))      # BGR: red kit, blue kit


def pitch_to_pixels(x_m, y_m):
    """Where a point on the pitch lands in the picture."""
    return (WIDTH / 2 + (x_m - PITCH_LENGTH_M / 2) * PX_PER_M,
            HEIGHT / 2 + (y_m - PITCH_WIDTH_M / 2) * PX_PER_M)


def synthetic_paths():
    """Twenty-two players walking, and a ball moving between two of them."""
    rng = np.random.default_rng(7)
    players = []
    for team in (0, 1):
        for k in range(11):
            start_x = 20.0 + team * 55.0 + (k % 4) * 6.0
            start_y = 8.0 + (k // 4) * 14.0 + rng.uniform(-2, 2)
            # A steady walk, about 1.5 m/s, turned around mid-clip.
            players.append({"team": team, "x0": start_x, "y0": start_y,
                            "vx": rng.uniform(-1.5, 1.5),
                            "vy": rng.uniform(-1.0, 1.0)})
    # Tracks 1 and 12 are the carriers, one from each team; see `carrier`.
    assert players[0]["team"] == 0 and players[11]["team"] == 1
    return players


def ball_at(t: float):
    """Where the ball is, in metres, at time `t`."""
    return 30.0 + 20.0 * (t / SECONDS), 34.0


def carrier(t: float) -> int:
    """Which track is dribbling: one from each team, a half each.

    The handover matters. With a single carrier the expected possession is
    100/0, which any rule returning a constant would also produce; splitting
    it gives the test a share to be wrong about.
    """
    return 1 if t < SECONDS / 2 else 12


def write_clip(path: Path, players):
    """Render the match, and return the tracks a detector would have made."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    rows = []
    n_frames = SECONDS * FPS
    for frame_index in range(n_frames):
        t = frame_index / FPS
        canvas = np.full((HEIGHT, WIDTH, 3), (60, 140, 60), dtype=np.uint8)
        # A halfway line, so the picture is not featureless for tracking.
        cv2.line(canvas, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT),
                 (230, 230, 230), 2)

        ball_x, ball_y = ball_at(t)
        holding = carrier(t)

        for track_id, player in enumerate(players, start=1):
            if track_id == holding:
                # The dribbler stays a metre off the ball, which is what
                # possession means and what the 3 m rule has to see.
                x_m, y_m = ball_x - 1.0, ball_y + 0.5
            else:
                turn = 1.0 if t < SECONDS / 2 else -1.0
                x_m = player["x0"] + player["vx"] * t * turn
                y_m = player["y0"] + player["vy"] * t * turn
            x_m = float(np.clip(x_m, 1.0, PITCH_LENGTH_M - 1.0))
            y_m = float(np.clip(y_m, 1.0, PITCH_WIDTH_M - 1.0))
            px, py = pitch_to_pixels(x_m, y_m)
            height_px = 1.8 * PX_PER_M
            colour = TEAM_COLOURS[player["team"]]
            cv2.rectangle(canvas,
                          (int(px - 3), int(py - height_px)),
                          (int(px + 3), int(py)), colour, -1)
            rows.append({"frame": frame_index, "time_s": t,
                         "track_id": track_id, "cls": "player",
                         "px": px, "py": py, "crop_h": height_px,
                         "detection_method": "yolo", "confidence": 0.9})

        # The ball moves at about 1.7 m/s: slow enough to be held, which is
        # what the speed gate is there to decide.
        bx, by = pitch_to_pixels(ball_x, ball_y)
        cv2.circle(canvas, (int(bx), int(by)), 3, (250, 250, 250), -1)
        rows.append({"frame": frame_index, "time_s": t, "track_id": 999,
                     "cls": "ball", "px": bx, "py": by, "crop_h": 6.0,
                     "detection_method": "yolo", "confidence": 0.8})

        writer.write(canvas)
    writer.release()
    return pd.DataFrame(rows)


def homography() -> np.ndarray:
    """Pixels to metres, the inverse of `pitch_to_pixels`."""
    forward = np.array([[PX_PER_M, 0.0,
                         WIDTH / 2 - PITCH_LENGTH_M / 2 * PX_PER_M],
                        [0.0, PX_PER_M,
                         HEIGHT / 2 - PITCH_WIDTH_M / 2 * PX_PER_M],
                        [0.0, 0.0, 1.0]])
    return np.linalg.inv(forward)


def build_fixture(work: Path):
    """A video, an homography and a tracks file, as main.py expects them."""
    (work / "output").mkdir(parents=True, exist_ok=True)
    video = work / "match.mp4"
    tracks = write_clip(video, synthetic_paths())
    tracks.to_parquet(work / "output" / "tracks.parquet")
    np.save(work / "output" / "homography.npy", homography())
    return video


def run_main(work: Path, video: Path):
    """The real script, in its own directory."""
    command = [sys.executable, str(REPO / "main.py"), str(video),
               "--skip-detect", "--label", "Synthetic",
               "--homography", "output/homography.npy",
               "--out", "output/report.pdf", "--db", "output/match.db"]
    # Inherit the environment rather than building one: a stripped env lost
    # the paths pandas needs for its own dependencies, which fails the test
    # for a reason that has nothing to do with the pipeline.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(command, cwd=work, capture_output=True, text=True,
                          env=env)


def check(work: Path, result) -> bool:
    ok = True

    def say(name, passed, detail=""):
        nonlocal ok
        ok &= passed
        print(f"   {name:<42s} {'ok' if passed else 'FAILED'}  {detail}")

    say("main.py exits cleanly", result.returncode == 0,
        "" if result.returncode == 0 else f"exit {result.returncode}")
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-12:]
        print("\n".join(f"      {line}" for line in tail))
        return False

    report = work / "output" / "report.pdf"
    say("a report is written", report.exists() and report.stat().st_size > 5000,
        f"{report.stat().st_size / 1024:.0f} kB" if report.exists() else "missing")

    pitch_path = work / "output" / "tracks_teams.parquet"
    say("projected tracks are written", pitch_path.exists())
    if not pitch_path.exists():
        return False

    tracks = pd.read_parquet(pitch_path)
    players = tracks[tracks.cls == "player"]

    # The contract the bug broke: metres in x/y, pixels still in px/py.
    on_pitch = (players.x.between(-5, PITCH_LENGTH_M + 5)
                & players.y.between(-5, PITCH_WIDTH_M + 5))
    say("players are placed on the pitch in metres", bool(on_pitch.all()),
        f"x {players.x.min():.0f}-{players.x.max():.0f} m")

    pixel_scale = players.px.max() > PITCH_LENGTH_M * 2
    say("px/py are still pixels, not metres", bool(pixel_scale),
        f"px max {players.px.max():.0f}")

    # Possession: reported at all, over a decent share of the ball's frames.
    sys.path.insert(0, str(REPO))
    from src import stats as stats_module
    held = stats_module.possession_timeline(tracks)
    ball_frames = tracks[tracks.cls == "ball"].frame.nunique()
    share = len(held) / max(ball_frames, 1)
    say("possession is measured on most ball frames", share > 0.5,
        f"{len(held)} of {ball_frames} frames ({share:.0%})")

    poss = stats_module.possession_proxy(tracks)
    say("both teams are always reported",
        poss.get("team_A") is not None and poss.get("team_B") is not None,
        json.dumps({k: v for k, v in poss.items() if k != "frames_used"}))

    # The fixture hands the ball over at half time, so the answer is known:
    # about half each. A rule that names one team throughout passes every
    # check above and fails this one.
    if poss.get("team_A") is not None:
        biggest = max(poss["team_A"], poss["team_B"])
        say("possession splits about evenly, as staged", biggest < 0.75,
            f"{poss['team_A']:.2f} / {poss['team_B']:.2f}")

    # Orders of magnitude, which catch a scale error anywhere in the chain.
    phys = stats_module.physical_stats(tracks)
    if not phys.empty and "top_speed_kmh" in phys.columns:
        fastest = float(phys.top_speed_kmh.max())
        say("top speed is humanly possible", 1.0 < fastest < 45.0,
            f"{fastest:.1f} km/h")
    if not phys.empty and "distance_m" in phys.columns:
        walked = float(phys.distance_m.median())
        say("players cover a sane distance", 1.0 < walked < 500.0,
            f"{walked:.0f} m in {SECONDS} s")
    return ok


def main():
    work = Path(tempfile.mkdtemp(prefix="veo_main_test_"))
    print(f"Running main.py end to end on a synthetic match.\n"
          f"  working directory: {work}\n")
    try:
        video = build_fixture(work)
        result = run_main(work, video)
        ok = check(work, result)
        print(f"\n   {'all pass' if ok else 'SOMETHING IS WRONG'}")
        return 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
