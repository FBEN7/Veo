"""Every check in this repository that can run without footage.

Two tiers, because they need different things.

**Controls** are the synthetic checks each detector carries: a ball walked
over each edge of the pitch, a restart from each landmark, a shot planted in
a known place, a possession timeline that is right, backwards, or a coin
flip. They need no video and no model weights, and they run in seconds. They
are what a change should be checked against before anything else.

**Pipeline** runs `main.py` end to end on a synthetic match. It imports the
detector stack -- ultralytics and torch -- because `main.py` does, even
with `--skip-detect`, so it is slower and needs the full requirements. It is
the only thing here that would catch a change breaking the report.

What is deliberately *not* here: everything scored against SoccerNet. Those
measurements need footage under an NDA which is not in the repository and
will not be in CI. `EVENT_ACCURACY.md` records what they said and how to
reproduce them; nothing automated can check them.

    python run_tests.py              # controls, then the pipeline
    python run_tests.py --controls   # just the fast ones
    python run_tests.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent

# Each entry: a name, the command, and what it is checking.
CONTROLS = [
    ("pitch orientation",
     # Runs standalone and exits non-zero on failure, so no pytest is
     # needed: the fast tier stays installable from requirements.txt alone.
     [sys.executable, "test_pitch_orientation.py"],
     "the anchor's direction of play is pinned to the camera, not to the "
     "order a line detector happens to return its endpoints in"),
    ("ball leaving the pitch",
     [sys.executable, "detect_ball_events.py", "--check"],
     "a ball walked over a touchline, into a goal, and behind the posts"),
    ("set-piece naming",
     [sys.executable, "detect_set_pieces.py", "--check"],
     "a restart from each of the five landmarks is named correctly"),
    ("shot geometry",
     [sys.executable, "detect_shots.py", "--self-check"],
     "a shot laid out in metres comes back with the distance it was given"),
    ("possession scoring",
     [sys.executable, "check_possession.py", "--check"],
     "a timeline that is right, one backwards, and one no better than a "
     "coin flip that still reports the share exactly"),
    ("crossings from grass",
     [sys.executable, "probe_grass_crossing.py", "--check"],
     "a ball taken off the grass, and a single stray reading that is not a "
     "crossing"),
    ("stoppages from players",
     [sys.executable, "probe_stoppage.py", "--check"],
     "a pitch brought to a halt, one that never stops, and one that dips"),
    ("per-player ball metrics",
     [sys.executable, "test_player_metrics.py"],
     "an intercepted pass credits the interceptor and not the passer"),
    ("the ball's path",
     [sys.executable, "src/ball_path.py"],
     "a ball among distractors the detector scores higher, where taking the "
     "best blob per frame lands on the wrong object a third of the time"),
    ("pose from goal corners",
     [sys.executable, "src/goal_pose.py"],
     "a known camera recovered from a projected goal, three views and a "
     "refusal when only two corners are given"),
    ("poses between grid frames",
     [sys.executable, "test_pose_interpolation.py"],
     "a panning, zooming camera placed between solved poses to 7 mm, where "
     "reusing the nearest pose is 0.8 m out"),
    ("camera from goal and circle",
     [sys.executable, "-m", "src.camera_position"],
     "a camera 38 m from its starting guess located to 0.07 m, circle "
     "frames from a lower or nearer camera refused, and the one "
     "displacement the geometry cannot see pinned as accepted"),
    ("ball height from its flight",
     [sys.executable, "-m", "src.ball_height"],
     "a cross and a long ball recognised as in the air and a pass as on "
     "the grass, from a located camera, with half the frames missing"),
]

PIPELINE = [
    ("main.py end to end",
     [sys.executable, "test_main_pipeline.py"],
     "the real report script on a synthetic match: coordinates in metres, "
     "possession over most frames and split as staged, human speeds"),
]


def run(name: str, command: list[str], why: str, verbose: bool) -> bool:
    started = time.time()
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    took = time.time() - started
    # Several of these report a failure and still exit 0, because they were
    # written to be read by a person. Both conditions have to hold.
    output = (result.stdout or "") + (result.stderr or "")
    passed = result.returncode == 0 and "SOMETHING IS WRONG" not in output
    print(f"  {'PASS' if passed else 'FAIL'}  {name:<26s} {took:5.1f}s")
    if not passed or verbose:
        for line in output.strip().splitlines()[-20:]:
            print(f"        {line}")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", action="store_true",
                    help="only the fast checks, which need no model weights")
    ap.add_argument("--list", action="store_true",
                    help="say what each check covers, and run nothing")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    tiers = [("controls", CONTROLS)]
    if not args.controls:
        tiers.append(("pipeline", PIPELINE))

    if args.list:
        for tier, checks in tiers:
            print(f"\n{tier}:")
            for name, _, why in checks:
                print(f"  {name}\n      {why}")
        return 0

    failures = []
    for tier, checks in tiers:
        print(f"\n{tier}")
        for name, command, why in checks:
            if not run(name, command, why, args.verbose):
                failures.append(name)

    print()
    if failures:
        print(f"  {len(failures)} failed: {', '.join(failures)}")
        return 1
    total = sum(len(c) for _, c in tiers)
    print(f"  all {total} passed")
    print("\n  Note that none of this measures accuracy. Every number in "
          "EVENT_ACCURACY.md\n  comes from footage under an NDA that is not "
          "in this repository, so the\n  controls check that the code does "
          "what it claims on cases with known\n  answers -- not that it "
          "works on football.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
