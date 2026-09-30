"""Check hand-clicked goal corners against things whose answers are known.

`src/goal_pose.py` recovers a camera from four corners of the goal mouth and
was verified on synthetic cameras, where it is exact. Synthetic exactness
proves the algebra and nothing about football, so this runs it on the real
clicks and asks questions that have answers independent of the clicks.

## The three checks, and why each can fail

**Focal length, two ways.** A rectangle's projection determines the focal
length on its own: the two pairs of parallel edges give two vanishing
points, and for perpendicular directions f^2 = -(v1 - c).(v2 - c). That uses
only the corners. `ground_plane.estimate_focal` gets the same number from
how the camera *rotates* between frames, using no corners at all. Two
unrelated routes to one quantity, so agreement is evidence and disagreement
localises the fault.

**A camera that stays put.** A broadcast camera sits on a gantry and pans;
it does not fly around the ground. Frames from one clip showing one goal
must therefore put the camera in nearly the same place. Each frame is solved
alone, so agreement across them is not built in -- it is a property the
clicks either have or do not.

**A camera above the pitch, at a plausible height.** Recovered heights are
compared with `ground_plane`'s independent estimate from player sizes.

None of these needs a ground-truth distance, which is fortunate, because
nothing in this footage carries one.

    python check_goal_corners.py --corners goal_corners.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from goal_pose import (GOAL_HEIGHT_M, GOAL_WIDTH_M,  # noqa: E402
                       bundle, shot_geometry, solve)
from ground_plane import GroundPlane  # noqa: E402
from propagate_goal_labels import DIRS  # noqa: E402

# The page that produced the first corner file divided every click by one
# page-wide frame size instead of the clip's own. Every clip was 1280x720
# except the Veo one at 640x360, whose corners therefore came back at
# exactly half. The factor is exact, so such a file is repaired rather than
# re-clicked; files carrying a per-frame size need nothing.
LEGACY_ASSUMED = (1280, 720)


def load(path: Path):
    """Corner rows, with legacy half-scale frames repaired."""
    blob = json.loads(Path(path).read_text())
    assumed = (blob.get("width"), blob.get("height"))
    rows, fixed = [], []
    for row in blob["frames"]:
        if not row.get("corners"):
            continue
        out_dir = DIRS.get(row["clip"])
        if not out_dir or not (Path(out_dir) / "clip.json").exists():
            continue
        info = json.loads((Path(out_dir) / "clip.json").read_text())
        width, height = info["width"], info["height"]
        corners = row["corners"]
        if assumed == LEGACY_ASSUMED and (width, height) != LEGACY_ASSUMED:
            sx, sy = assumed[0] / width, assumed[1] / height
            corners = [None if c is None else [c[0] * sx, c[1] * sy]
                       for c in corners]
            fixed.append(row["id"])
        rows.append({**row, "corners": corners, "width": width,
                     "height": height, "path": info["path"],
                     "out_dir": out_dir})
    if fixed:
        print(f"  rescaled from a page-wide frame size: {', '.join(fixed)}\n")
    return rows


def pixels(row):
    """Corners in pixels, or None where a corner was not clicked."""
    return [None if c is None else
            np.array([c[0] * row["width"], c[1] * row["height"]])
            for c in row["corners"]]


def _meet(p1, p2, p3, p4):
    """Where the line p1p2 meets p3p4, in homogeneous coordinates."""
    line_a = np.cross(np.append(p1, 1.0), np.append(p2, 1.0))
    line_b = np.cross(np.append(p3, 1.0), np.append(p4, 1.0))
    return np.cross(line_a, line_b)


def focal_from_rectangle(corners, cx: float, cy: float):
    """Focal length from the projection of a rectangle, corners alone.

    Edges 1-4 and 2-3 run along the goal line; 1-2 and 4-3 run up the posts.
    The two vanishing points of perpendicular world directions satisfy
    (v1 - c).(v2 - c) + f^2 = 0. Returns None when the rectangle is too
    near-parallel in the image to place a vanishing point.
    """
    if any(c is None for c in corners):
        return None
    p1, p2, p3, p4 = corners
    out = []
    for a, b, c_, d in ((p1, p4, p2, p3), (p1, p2, p4, p3)):
        v = _meet(a, b, c_, d)
        if abs(v[2]) < 1e-9:
            return None              # parallel in the image: at infinity
        out.append(v[:2] / v[2])
    centre = np.array([cx, cy])
    dot = float(np.dot(out[0] - centre, out[1] - centre))
    return float(np.sqrt(-dot)) if dot < 0 else None


# Frames shot from a different camera than the one following play: two goal
# celebrations with broadcast graphics over them, and one from behind the
# goal. Identified by drawing the corners on the frames and looking. They
# are excluded because the bundle's whole premise is one camera in one
# place, and a replay camera is neither.
#
# The behind-goal case is also detectable without a list -- seen from
# behind, the left and right posts swap, so corner 1 lands to the *right* of
# corner 4 -- and `looks_mirrored` checks for it. The two celebration views
# are not mirrored, only differently placed, so the list stays.
REPLAY_CAMERAS = {"reading_0737:963", "stoke_1302:1445", "stoke_7001:1927"}


def looks_mirrored(corners) -> bool:
    """Seen from behind the goal, the posts swap sides."""
    if corners[0] is None or corners[3] is None:
        return False
    return bool(corners[0][0] > corners[3][0])


def usable_rows(rows):
    """Corner frames from the main camera with all four corners clicked."""
    return [r for r in rows
            if r["id"] not in REPLAY_CAMERAS
            and not looks_mirrored(r["corners"])
            and all(c is not None for c in r["corners"])]


def located_poses(rows, eye):
    """Corner poses with the camera held at a located position."""
    from src.goal_pose import pose_at

    out = {}
    for row in usable_rows(rows):
        cx, cy = row["width"] / 2.0, row["height"] / 2.0
        guess = focal_from_rectangle(pixels(row), cx, cy) or 2500.0
        pose = pose_at(pixels(row), eye, guess, cx, cy)
        if pose is not None:
            out[row["id"]] = pose
    return out


def poses_for_clip(rows, plane, verbose: bool = False):
    """Best pose available for each frame of one clip, keyed by frame id.

    Focal length is the thing that limits this, so it is taken from the
    strongest source available, in order:

      1. a bundle over the clip's frames, which shares one camera position
         and fits a focal per frame -- the most constrained, and the only
         one that uses the fact that a gantry camera does not move;
      2. failing that (one usable frame), the frame's own solve, seeded from
         the ground plane's focal where there is a ground plane and from the
         rectangle's vanishing points where there is not.

    That second fallback is what unblocks reading_1155 and reading_2519,
    whose clips yield no ground plane at all. Their focal was never
    unobtainable -- it was simply being asked for from the one source that
    could not supply it.
    """
    usable = usable_rows(rows)
    if not usable:
        return {}

    cx = usable[0]["width"] / 2.0
    cy = usable[0]["height"] / 2.0
    rect = [focal_from_rectangle(pixels(r), cx, cy) for r in usable]
    seeds = [f for f in rect if f]
    seed = (plane.focal_px if plane else
            (float(np.median(seeds)) if seeds else None))
    if seed is None:
        return {}

    if len(usable) >= 2:
        fitted = bundle([pixels(r) for r in usable], seed, cx, cy)
        if fitted is not None:
            _, poses, _ = fitted
            return {r["id"]: p for r, p in zip(usable, poses)}

    out = {}
    for row, f_rect in zip(usable, rect):
        pose = solve(pixels(row), plane.focal_px if plane else f_rect, cx, cy)
        if pose is None and f_rect:
            pose = solve(pixels(row), f_rect, cx, cy)
        if pose is not None:
            out[row["id"]] = pose
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corners", required=True)
    ap.add_argument("--cameras",
                    help="JSON of located camera positions by clip, as "
                         "score_hand_labels prints them")
    args = ap.parse_args()
    cameras = ({k: np.asarray(v, dtype=float) for k, v in
                json.loads(Path(args.cameras).read_text()).items()}
               if args.cameras else None)

    rows = load(Path(args.corners))
    planes: dict[str, GroundPlane] = {}
    for row in rows:
        clip = row["clip"]
        if clip not in planes:
            path = Path(row["out_dir"]) / "ground_plane.json"
            blob = json.loads(path.read_text()) if path.exists() else {}
            planes[clip] = (None if not blob or blob.get("refused")
                            else GroundPlane(**blob))

    print("Focal length two ways: from the corners, and from camera "
          "rotation.\n")
    print(f"  {'frame':>18s} {'corners':>9s} {'rotation':>9s} {'ratio':>7s}")
    solved = []
    for row in rows:
        plane = planes[row["clip"]]
        cx = row["width"] / 2.0
        cy = row["height"] / 2.0
        f_rect = focal_from_rectangle(pixels(row), cx, cy)
        f_rot = plane.focal_px if plane else None
        if f_rect and f_rot:
            print(f"  {row['id']:>18s} {f_rect:9.0f} {f_rot:9.0f} "
                  f"{f_rect/f_rot:7.2f}")
        else:
            print(f"  {row['id']:>18s} {'-' if not f_rect else f'{f_rect:.0f}':>9s} "
                  f"{'-' if not f_rot else f'{f_rot:.0f}':>9s} {'':>7s}")
        row["f_rect"], row["f_rot"] = f_rect, f_rot

    ratios = np.array([r["f_rect"] / r["f_rot"] for r in rows
                       if r["f_rect"] and r["f_rot"]])
    if ratios.size:
        print(f"\n  {ratios.size} comparable.  median ratio "
              f"{np.median(ratios):.2f}   "
              f"within 20%: {int(np.sum(np.abs(ratios-1) <= 0.2))} "
              f"of {ratios.size}")

    print("\n\nCamera position from each frame alone, in metres from the "
          "left post.\n")
    print(f"  {'frame':>18s} {'across':>7s} {'up':>6s} {'out':>7s} "
          f"{'reproj':>7s}")
    for row in rows:
        focal = row["f_rect"] or row["f_rot"]
        if not focal:
            continue
        pose = solve(pixels(row), focal, row["width"] / 2.0,
                     row["height"] / 2.0)
        row["pose"] = pose
        if pose is None:
            print(f"  {row['id']:>18s} {'refused':>22s}")
            continue
        eye = pose.camera_position()
        solved.append(row)
        print(f"  {row['id']:>18s} {eye[0]:7.1f} {eye[1]:6.1f} {eye[2]:7.1f} "
              f"{pose.reprojection_px:6.2f}px")

    print("\n\nA camera on a gantry does not move. Spread within one clip:\n")
    print(f"  {'clip':>16s} {'n':>3s} {'height m':>9s} {'spread m':>9s} "
          f"{'plane m':>8s}")
    for clip in sorted({r["clip"] for r in solved}):
        group = [r for r in solved if r["clip"] == clip]
        if len(group) < 2:
            continue
        eyes = np.array([r["pose"].camera_position() for r in group])
        spread = float(np.max(np.linalg.norm(eyes - eyes.mean(axis=0),
                                             axis=1)))
        plane = planes[clip]
        print(f"  {clip:>16s} {len(group):3d} {eyes[:,1].mean():9.1f} "
              f"{spread:9.1f} "
              f"{(f'{plane.h_cam_m:.1f}' if plane else '-'):>8s}")

    print("\n  Camera heights from the corners and from player sizes are "
          "independent.\n  The spread is the test the clicks could fail "
          "outright: one gantry,\n  several frames, solved separately.")

    shot_table(rows, planes, cameras)


# The moment of the labelled shot in every window cut around one.
SHOT_FRAME = 481


def shot_table(rows, planes, cameras=None):
    """Distance and angle at each labelled shot moment.

    `cameras` maps a clip to a located camera position
    (`src.camera_position`). Where given, the shot frame's pose is refitted
    with the camera held there instead of wherever the corners' focal
    guess put it.

    The position measured is the **shooter's feet**, not the ball. A player
    is unambiguously standing on the pitch, which is the assumption the
    whole ground intersection rests on; a ball is often in flight, above the
    plane, and its ray then meets the ground far beyond where it really is.
    Taking the ball gave 171.7 m and 50.9 m on two of these windows. Taking
    the nearest player to the ball gives 25.9 and 18.0.
    """
    import pandas as pd

    by_clip = {}
    for row in rows:
        by_clip.setdefault(row["clip"], []).append(row)

    model = None
    model_path = Path("models/xg_xghub.json")
    if model_path.exists():
        from src.xg import XGModel
        model = XGModel.load(model_path)

    print(f"\n\nAt the shot moment, frame {SHOT_FRAME}, from the shooter's "
          f"feet.\n")
    print(f"  {'window':>14s} {'plane':>6s} {'distance':>9s} {'angle':>6s} "
          f"{'reproj':>8s} {'xG':>6s}")
    got = total = 0
    scored = []
    for clip, group in sorted(by_clip.items()):
        shot = [r for r in group if r["frame"] == SHOT_FRAME]
        if not shot:
            continue
        total += 1
        plane = planes.get(clip)
        if cameras and clip in cameras:
            pose = located_poses(group, cameras[clip]).get(shot[0]["id"])
            label = "loc"
        else:
            pose = poses_for_clip(group, plane).get(shot[0]["id"])
            label = "yes" if plane else "no"
        if pose is None:
            print(f"  {clip:>14s} {label:>6s}   no pose")
            continue
        tracks = pd.read_parquet(Path(shot[0]["out_dir"]) / "tracks.parquet")
        ball = tracks[(tracks.cls == "ball")
                      & (abs(tracks.frame - SHOT_FRAME) <= 3)]
        players = tracks[(tracks.cls == "player")
                         & (tracks.frame == SHOT_FRAME)]
        if ball.empty or players.empty:
            print(f"  {clip:>14s} {label:>6s}   no ball or player here")
            continue
        bx, by = float(ball.px.iloc[0]), float(ball.py.iloc[0])
        near = ((players.px - bx) ** 2 + (players.py - by) ** 2) ** 0.5
        shooter = players.loc[near.idxmin()]
        geom = shot_geometry(pose, float(shooter.px), float(shooter.py))
        if geom is None:
            print(f"  {clip:>14s} {label:>6s}   refused, beyond the pitch")
            continue
        got += 1
        shown = "-"
        if model is not None:
            # Same conventions the model was fitted under: distance to the
            # middle of the goal, and the angle the two posts subtend. Both
            # are what `shot_geometry` returns, so nothing is converted
            # except degrees to radians.
            value = float(model.predict(
                distance_m=[geom[0]],
                angle_rad=[np.radians(geom[1])])[0])
            scored.append((clip, geom[0], geom[1], value))
            shown = f"{value:.3f}"
        print(f"  {clip:>14s} {label:>6s} {geom[0]:8.1f} m {geom[1]:5.0f}d "
              f"{pose.reprojection_px:6.2f}px {shown:>6s}")

    print(f"\n  {got} of {total} shot moments carry a measured distance and "
          f"angle.")
    print("\n  Reprojection cannot arbitrate the focal length here: a focal "
          "fitted to\n  these very corners will always reproject them best, "
          "while focal and\n  distance trade off against each other. The "
          "single-frame rows are the\n  weaker ones for that reason; the "
          "bundled rows are constrained by a\n  camera that has to be in one "
          "place across several frames.")

    if scored:
        total_xg = sum(v for _, _, _, v in scored)
        print(f"\n  Total xG over {len(scored)} labelled shots: "
              f"{total_xg:.2f}, so about one goal in "
              f"{1.0 / max(total_xg, 1e-9):.0f} of these sets.")
        print("\n  The model is the reduced xGHub fit in "
              "models/xg_xghub.json: classical\n  geometry only, held out by "
              "match, AUC 0.766 and Brier 0.0859 against a\n  0.0976 base "
              "rate. The orientation features that are xGHub's actual\n  "
              "contribution need pose estimation this pipeline does not have.")
        print("\n  These are six shots. The number is what the geometry "
              "implies, not a\n  measurement of anything about these "
              "matches.")


if __name__ == "__main__":
    main()
