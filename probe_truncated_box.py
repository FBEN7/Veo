"""Poses from detected goal boxes, against the corner poses: before and after.

Two faults, found together.

**What a box is.** `goal_pose.pose_from_box` read a box as the rectangle
enclosing the goal. The hand labels the detector learned from run from the
top of the left post to the base of the right post -- one diagonal of the
goal's slanted outline, to a median 3.5 px on 24 of 25 frames. Where the
crossbar slopes, the enclosing rectangle is 45-62 px from that, and box
poses landed a median 18 m from the corner poses.

**Truncation.** When the goal runs off the picture, the box's edge on that
side is the frame's. It was fitted as the goal's, dragging the goal back
inside. Now scored one-sidedly, with roll held near zero -- a gantry camera
does not roll, measured -1.7 to +0.2 degrees on the corner frames -- and a
pose is refused when more than half the goal would lie outside.

Measured on real frames, with located cameras, as the median metres between
where a box pose and the corner pose put a grid of ground points across the
penalty area:

    whole boxes, 15 corner frames      old 18.44 m (90th 22.12)   new 0.53 m (90th 1.05)
    same boxes cut by a pretend edge   old 31.66 m (90th 38.35)   new 1.09 m (90th 3.54)
    stoke_1302 frame 638, a 34 px sliver of goal at the edge:
                                       old 23-34 m off painted lines, new refused

    python probe_truncated_box.py --corners goal_corners.json \\
        --cameras cameras.json --goal-weights best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.goal_pose import MODEL, GoalPose, ground_point, pose_from_box

# Ground points in the goal frame (X across from the left post, Z out),
# covering the penalty area where shots are taken.
GRID = [(x, z) for x in (-8.0, 0.0, 3.66, 7.32, 15.0)
        for z in (5.5, 11.0, 16.5, 22.0)]


def old_pose_from_box(box, eye, focal_guess, cx, cy, max_nfev=600):
    """The version before the fix, kept here to compare against."""
    from scipy.optimize import least_squares

    eye = np.asarray(eye, dtype=np.float64)
    observed = np.asarray(box, dtype=np.float64)

    def bbox(rvec, focal):
        rot, _ = cv2.Rodrigues(rvec)
        camera = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]])
        pts, _ = cv2.projectPoints(MODEL, rvec, (-rot @ eye).reshape(3, 1),
                                   camera, None)
        pts = pts.reshape(-1, 2)
        return np.array([pts[:, 0].min(), pts[:, 1].min(),
                         pts[:, 0].max(), pts[:, 1].max()])

    target = np.array([3.66, 1.22, 0.0])
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    start, _ = cv2.Rodrigues(np.vstack([right, np.cross(forward, right),
                                        forward]))
    fit = least_squares(lambda q: bbox(q[:3], np.exp(q[3])) - observed,
                        np.concatenate([start.ravel(), [np.log(focal_guess)]]),
                        method="lm", max_nfev=max_nfev)
    rvec = fit.x[:3].reshape(3, 1)
    rot, _ = cv2.Rodrigues(rvec)
    return GoalPose(rvec=rvec, tvec=(-rot @ eye).reshape(3, 1),
                    focal_px=float(np.exp(fit.x[3])), cx=cx, cy=cy,
                    reprojection_px=float(np.sqrt(np.mean(fit.fun ** 2))),
                    n_corners=0)


def disagreement(truth: GoalPose, pose, width, height):
    """Median metres between where two poses put the same ground points."""
    if pose is None:
        return None
    rot, _ = cv2.Rodrigues(truth.rvec)
    errs = []
    for x, z in GRID:
        px, _ = cv2.projectPoints(np.array([[x, 0.0, z]]), truth.rvec,
                                  truth.tvec, truth.camera_matrix, None)
        u, v = px.reshape(2)
        if not (0 <= u < width and 0 <= v < height):
            continue
        got = ground_point(pose, u, v)
        errs.append(np.inf if got is None else np.hypot(got[0] - x,
                                                        got[1] - z))
    return float(np.median(errs)) if errs else None


def main():
    from ultralytics import YOLO

    from check_goal_corners import load, located_poses, usable_rows

    ap = argparse.ArgumentParser()
    ap.add_argument("--corners", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--goal-weights", required=True)
    args = ap.parse_args()

    cameras = json.loads(Path(args.cameras).read_text())
    rows = usable_rows(load(Path(args.corners)))
    detector = YOLO(args.goal_weights)

    whole, cut = {"old": [], "new": []}, {"old": [], "new": []}
    print("Box poses against the corner pose, median metres over a grid "
          "across the penalty area.\n")
    print(f"  {'frame':>18s} {'whole old':>10s} {'new':>6s}   "
          f"{'cut old':>8s} {'new':>6s}")
    for row in rows:
        clip = row["clip"]
        if clip not in cameras:
            continue
        eye = np.asarray(cameras[clip], dtype=float)
        truth = located_poses([row], eye).get(row["id"])
        info = json.loads((Path(row["out_dir"]) / "clip.json").read_text())
        width, height = info["width"], info["height"]
        cx, cy = width / 2.0, height / 2.0
        cap = cv2.VideoCapture(info["path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
        ok, frame = cap.read()
        cap.release()
        if truth is None or not ok:
            continue
        result = detector.predict(frame, conf=0.15, verbose=False)[0]
        if not len(result.boxes):
            continue
        box = result.boxes.xyxy.cpu().numpy()[
            int(np.argmax(result.boxes.conf.cpu().numpy()))].astype(float)

        # Measurement 1: the whole box.
        w_old = disagreement(truth, old_pose_from_box(box, eye, 2500.0, cx, cy),
                             width, height)
        w_new = disagreement(truth, pose_from_box(box, eye, 2500.0, cx, cy,
                                                  frame_size=(width, height)),
                             width, height)
        # Measurement 2: a pretend right edge through the goal's middle.
        # Only the edge test sees the pretend width; the camera's principal
        # point is unchanged, as it would be on a real truncated frame.
        edge = float(box[0] + 0.6 * (box[2] - box[0]))
        clipped = np.array([box[0], box[1], edge, box[3]])
        c_old = disagreement(truth, old_pose_from_box(clipped, eye, 2500.0,
                                                      cx, cy), width, height)
        c_new = disagreement(truth, pose_from_box(clipped, eye, 2500.0, cx, cy,
                                                  frame_size=(edge, height)),
                             width, height)
        for d, key, val in ((whole, "old", w_old), (whole, "new", w_new),
                            (cut, "old", c_old), (cut, "new", c_new)):
            if val is not None:
                d[key].append(val)
        fmt = lambda v: "   -  " if v is None else f"{v:6.2f}"
        print(f"  {row['id']:>18s} {fmt(w_old):>10s} {fmt(w_new)}   "
              f"{fmt(c_old):>8s} {fmt(c_new)}")

    def summary(values):
        v = np.asarray(values)
        return (f"median {np.median(v):5.2f} m, 90th pct "
                f"{np.percentile(v, 90):5.2f} m, n={len(v)}")

    print("\n  whole boxes   old:", summary(whole["old"]))
    print("                new:", summary(whole["new"]))
    print("  cut boxes     old:", summary(cut["old"]))
    print("                new:", summary(cut["new"]))

    # Measurement 3: the real truncated stretch on stoke_1302.
    if "stoke_1302" in cameras:
        eye = np.asarray(cameras["stoke_1302"], dtype=float)
        info = json.loads(Path("output_stoke_1302/clip.json").read_text())
        width, height = info["width"], info["height"]
        cap = cv2.VideoCapture(info["path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, 638)
        ok, frame = cap.read()
        cap.release()
        result = detector.predict(frame, conf=0.15, verbose=False)[0]
        if ok and len(result.boxes):
            box = result.boxes.xyxy.cpu().numpy()[
                int(np.argmax(result.boxes.conf.cpu().numpy()))].astype(float)
            print(f"\n  stoke_1302 frame 638, goal box "
                  f"{np.round(box).astype(int).tolist()} in a {width} px "
                  f"frame")
            # Painted, clicked by eye: the six-yard box's near corner and
            # the penalty area's far corner, in pitch metres.
            marks = {"six-yard corner (5.5, 43.2)": ((885, 325), (5.5, 43.16)),
                     "penalty-area corner (16.5, 54.2)":
                         ((235, 290), (16.5, 54.16))}
            for name, pose in (
                    ("old fit", old_pose_from_box(box, eye, 2500.0,
                                                  width / 2, height / 2)),
                    ("new fit", pose_from_box(box, eye, 2500.0, width / 2,
                                              height / 2,
                                              frame_size=(width, height)))):
                out = []
                for label, ((u, v), (tx, ty)) in marks.items():
                    got = None if pose is None else ground_point(pose, u, v)
                    if got is None:
                        out.append(f"{label}: refused")
                        continue
                    px_, py_ = got[1], got[0] + (68.0 - 7.32) / 2.0
                    out.append(f"{label}: ({px_:.1f}, {py_:.1f}), "
                               f"{np.hypot(px_ - tx, py_ - ty):.1f} m off")
                print(f"    {name}: " + "; ".join(out))


if __name__ == "__main__":
    main()
