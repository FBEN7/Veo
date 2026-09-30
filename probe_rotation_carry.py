"""Can a goal pose be carried into midfield by the camera's own rotation?

The centre-circle anchor puts its camera 62-72 m up against 15.6 and 15.9 m
from two independent methods (`probe_anchor_height.py`), so midfield has no
trustworthy geometry. The goal does, but it is only in frame in the
attacking third.

What connects the two is the camera. The goal corners fix its *position*
per clip -- a broadcast camera is bolted to a gantry, and the bundle
reprojects every corner frame from one position to under a pixel. Between
frames it only rotates and zooms, which makes the frame-to-frame warp

    H = K_b R K_a^-1

a rotation between two intrinsic matrices. Given the focal at frame a,
the rotation and the focal at frame b both fall out of H. Chaining them
carries a pose from a frame where the goal is visible into frames where it
is not -- with three unknowns per step rather than the eight of a general
homography, which is what drifted 21 m in 125 steps when this project
chained full homographies before.

## It checks itself

The camera leaves the goal and comes back to it. So a pose can be carried
from the end of one goal sighting to the start of the next, and compared
with the pose measured *directly* there from the detected box. Nothing is
fitted to make those agree; the far-end pose was never an input.

    carrying works      if the carried and measured poses place a pitch
                        point within a few metres, near the box-pose floor
                        of 1.8 m median
    carrying fails      if the disagreement grows with the gap and passes
                        what a pitch point can tolerate

If it works, midfield gets geometry without anyone clicking anything. If it
fails, manual midfield landmarks become worth asking for, and not before.

## What it found: carrying does not work on this footage

    gaps attempted               9, across three clips
    broke (frames not matched)   6
    carried                      3, all on stoke_4207
      536 -> 592    2.2 s    6.52 m    focal ratio 1.124
      608 -> 792    7.4 s   20.64 m    focal ratio 0.558
     1160 -> 1480  12.8 s    7.11 m    focal ratio 0.809

Against a floor of 1.8 m -- the box-derived pose's own error -- even the
shortest carry is 3.6 times over.

Two separate failures.

**Most gaps cannot be crossed at all.** Six of nine broke on a pair of
frames with too few feature matches. The longest, reading_2519's 64 s from
frame 580 to 2176, is long enough to contain a cut to a replay or another
camera, and across a cut frame N and N+1 come from different cameras. No
constraint on the warp carries a pose across that, and none should.

**Where it crosses, the zoom drifts.** Error does not track gap length -- a
12.8 s carry came out at 7.1 m and a 7.4 s one at 20.6 -- it tracks the
focal ratio at the far end, 0.558 on the worst. Rotation is well
constrained from each warp; the per-step focal is a ratio of row norms, and
its noise compounds multiplicatively along the chain. Three unknowns per
step rather than eight did not save it, because the one that fails is the
one that multiplies.

So midfield geometry needs a per-frame observation, as the goal did, rather
than a pose carried from somewhere else.

    python probe_rotation_carry.py --corners goal_corners.json \
        --goal-weights best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.goal_pose import GoalPose, ground_point  # noqa: E402

# Frames per carrying step. The goal placer's grid is 4, so carried poses
# land on the same frames as measured ones.
STEP = 4

# A gap shorter than this is the detector blinking, not the camera leaving.
MIN_GAP_FRAMES = 20

# Probe pixels, low and central, where the pitch is.
PROBES = ((0.35, 0.72), (0.50, 0.78), (0.65, 0.72))


def rotation_from_homography(h, focal_a: float, cx: float, cy: float):
    """Rotation and new focal from a pure-rotation, changing-zoom warp.

    With pixel coordinates centred on the principal point, H becomes
    diag(f_b, f_b, 1) R diag(1/f_a, 1/f_a, 1) up to scale. Multiplying on
    the right by diag(f_a, f_a, 1) leaves M = s diag(f_b, f_b, 1) R, whose
    rows are s f_b r1, s f_b r2 and s r3. Rows of a rotation are unit
    length, so f_b is the ratio of the first two row norms to the third.
    """
    centre = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    hc = centre @ np.asarray(h, dtype=float) @ np.linalg.inv(centre)
    m = hc @ np.diag([focal_a, focal_a, 1.0])

    n1, n2, n3 = (float(np.linalg.norm(m[i])) for i in range(3))
    if min(n1, n2, n3) < 1e-12:
        return None, None
    focal_b = (n1 + n2) / (2.0 * n3)
    if not np.isfinite(focal_b) or focal_b <= 0:
        return None, None

    raw = np.vstack([m[0] / focal_b, m[1] / focal_b, m[2]]) / n3
    u, _, vt = np.linalg.svd(raw)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        rot = -rot
    return rot, float(focal_b)


def carry(pose: GoalPose, rot_ab, focal_b: float) -> GoalPose:
    """Apply one camera rotation to a pose; the position does not move."""
    rot_a, _ = cv2.Rodrigues(pose.rvec)
    eye = pose.camera_position()
    rot_b = rot_ab @ rot_a
    rvec_b, _ = cv2.Rodrigues(rot_b)
    return GoalPose(rvec=rvec_b, tvec=(-rot_b @ eye).reshape(3, 1),
                    focal_px=focal_b, cx=pose.cx, cy=pose.cy,
                    reprojection_px=0.0, n_corners=0)


def disagreement(a: GoalPose, b: GoalPose, width: int, height: int):
    """Median distance, in metres, between two poses' placement of probes."""
    gaps = []
    for u, v in PROBES:
        p = ground_point(a, u * width, v * height)
        q = ground_point(b, u * width, v * height)
        if p is None or q is None:
            continue
        gaps.append(float(np.hypot(p[0] - q[0], p[1] - q[1])))
    return float(np.median(gaps)) if gaps else None


def selftest(verbose: bool = True) -> bool:
    """Recover a known rotation and zoom from the warp they produce."""
    cx, cy = 640.0, 360.0
    ok = True
    for yaw, pitch, f_a, f_b in ((4.0, 1.0, 1800.0, 1800.0),
                                 (-9.0, -2.0, 1800.0, 2400.0),
                                 (15.0, 3.0, 2600.0, 1500.0)):
        rot, _ = cv2.Rodrigues(np.radians([pitch, yaw, 0.0]))
        k_a = np.array([[f_a, 0, cx], [0, f_a, cy], [0, 0, 1.0]])
        k_b = np.array([[f_b, 0, cx], [0, f_b, cy], [0, 0, 1.0]])
        h = k_b @ rot @ np.linalg.inv(k_a)
        got_rot, got_f = rotation_from_homography(h * 3.7, f_a, cx, cy)
        if got_rot is None:
            print("  FAILED to recover")
            ok = False
            continue
        angle = float(np.degrees(np.arccos(np.clip(
            (np.trace(got_rot @ rot.T) - 1) / 2, -1, 1))))
        good = angle < 0.01 and abs(got_f - f_b) < 0.5
        ok &= good
        if verbose:
            print(f"  pan {yaw:+5.1f}  zoom {f_a:.0f}->{f_b:.0f}   "
                  f"rotation err {angle:.4f} deg   focal {got_f:7.1f}   "
                  f"{'ok' if good else 'WRONG'}")
    return ok


def gaps_of(frames):
    """Where the goal leaves frame and comes back: (last seen, next seen)."""
    frames = sorted(frames)
    return [(a, b) for a, b in zip(frames, frames[1:])
            if b - a >= MIN_GAP_FRAMES]


def measure(clip: str, corners_path: Path, weights: Path):
    from ultralytics import YOLO

    import detect_shots
    from check_goal_corners import load, poses_for_clip
    from src.ground_plane import GroundPlane
    from src import goal_placer
    from propagate_anchor import frame_to_frame

    out = Path(f"output_{clip}")
    info = json.loads((out / "clip.json").read_text())
    width, height = info["width"], info["height"]
    rows = [r for r in load(corners_path)
            if Path(r["out_dir"]).resolve() == out.resolve()]
    blob = json.loads((out / "ground_plane.json").read_text()) \
        if (out / "ground_plane.json").exists() else {}
    plane = None if not blob or blob.get("refused") else GroundPlane(**blob)
    poses = poses_for_clip(rows, plane)
    if not poses:
        return []
    eye = np.mean([p.camera_position() for p in poses.values()], axis=0)
    seed = float(np.median([p.focal_px for p in poses.values()]))

    ball = detect_shots.ball_track(out)
    everything = list(range(0, int(info["n_frames"]), STEP))
    placer = goal_placer.build(info["path"], everything, eye, seed,
                               width, height, YOLO(str(weights)))
    measured = placer.poses

    cap = cv2.VideoCapture(info["path"])
    results = []
    for a, b in gaps_of(measured):
        pose = measured[a]
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        ok, prev = cap.read()
        if not ok:
            continue
        failed = False
        for f in range(a + STEP, b + 1, STEP):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                failed = True
                break
            warp, _ = frame_to_frame(prev, frame)
            if warp is None:
                failed = True
                break
            rot, focal = rotation_from_homography(warp, pose.focal_px,
                                                  width / 2.0, height / 2.0)
            if rot is None:
                failed = True
                break
            pose = carry(pose, rot, focal)
            prev = frame
        if failed:
            results.append((clip, a, b, None, None, None))
            continue
        end = measured.get(b - b % STEP) or measured.get(b)
        if end is None:
            continue
        gap_m = disagreement(pose, end, width, height)
        focal_ratio = pose.focal_px / end.focal_px
        results.append((clip, a, b, gap_m, focal_ratio,
                        (b - a) // STEP))
    cap.release()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*",
                    default=["reading_2519", "reading_0737", "stoke_4207"])
    ap.add_argument("--corners", required=True)
    ap.add_argument("--goal-weights", required=True)
    args = ap.parse_args()

    print("Recovering a known rotation and zoom from the warp they "
          "produce.\n")
    if not selftest():
        raise SystemExit("rotation recovery is wrong; carrying would be "
                         "meaningless")

    print("\n\nCarried from one goal sighting to the next, against the pose "
          "measured there.\n")
    print(f"  {'clip':>13s} {'from':>5s} {'to':>5s} {'steps':>6s} "
          f"{'seconds':>8s} {'disagree m':>11s} {'focal ratio':>12s}")
    everything = []
    for clip in args.clips:
        for c, a, b, gap_m, ratio, steps in measure(
                clip, Path(args.corners), Path(args.goal_weights)):
            if gap_m is None:
                print(f"  {c:>13s} {a:5d} {b:5d}   carrying broke "
                      f"(no match between frames)")
                continue
            everything.append((steps, gap_m))
            print(f"  {c:>13s} {a:5d} {b:5d} {steps:6d} {(b - a) / 25:8.1f} "
                  f"{gap_m:11.2f} {ratio:12.3f}")

    if everything:
        steps = np.array([s for s, _ in everything])
        gaps = np.array([g for _, g in everything])
        print(f"\n  {len(gaps)} gaps carried.  median disagreement "
              f"{np.median(gaps):.2f} m, 90th pct "
              f"{np.percentile(gaps, 90):.2f} m")
        short = gaps[steps <= 15]
        long_ = gaps[steps > 15]
        if short.size:
            print(f"  up to 15 steps (~2.4 s): median {np.median(short):.2f} m"
                  f" over {short.size}")
        if long_.size:
            print(f"  over 15 steps:           median {np.median(long_):.2f} m"
                  f" over {long_.size}")
        print("\n  The floor is the box-derived pose itself: 1.8 m median "
              "against the\n  corner pose. Carried disagreement near that "
              "floor means the carry adds\n  little error of its own.")


if __name__ == "__main__":
    main()
