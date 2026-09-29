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
                       shot_geometry, solve)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corners", required=True)
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
