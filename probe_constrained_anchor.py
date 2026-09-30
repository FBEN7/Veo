"""Refit the centre-circle anchor with the camera's position held fixed.

The anchor fails on tilt. Its implied camera sits 62-72 m up against 15.6
and 15.9 m from two independent methods (`probe_anchor_height.py`), while
the players it places still land on a believable patch of pitch -- a
homography with the wrong tilt maps the visible trapezoid onto plausible
ground and distorts depth inside it.

Where the tilt comes from is written in `fit_pitch_anchor.py` itself. The
horizon is measured on penalty-area frames, where straight lines give
vanishing points, and then *carried* to midfield by adding the camera's
measured displacement -- which, its own docstring says, "treats a turn as a
slide, which is wrong away from the image centre by sec^2 of the angle --
and the horizon sits hundreds of rows above the frame, so it is wrong
exactly where the horizon lives." Carrying is also what failed outright in
`probe_rotation_carry.py`.

So this drops the carried horizon entirely. The goal corners fix the
camera's *position* per clip, and a broadcast camera does not travel.
With position known, a midfield frame has four unknowns -- rotation and
focal -- and the centre circle and halfway line over-determine them. The
tilt stops being something a small ellipse has to pin down on its own.

## The check never sees the fit

The fit uses the circle's observed arc pixels and the halfway line, and
nothing else. **Player sizes** never enter it. A correct pose predicts how
many pixels tall a 1.75 m player should be at every spot on the pitch, so
comparing that with the boxes the detector drew is a test the fit cannot
have been steered towards. The same test is run on the old anchor, as a
comparison rather than an absolute: detector boxes are not exactly a
player's height, so what matters is which pose predicts them better and
how consistently.

Stated before running: the refit is right if its predicted-to-observed
ratio sits near a constant close to 1 with a tight spread, and the old
anchor's does not.

    python probe_constrained_anchor.py --corners goal_corners.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import pitch_model as pm  # noqa: E402
from src.goal_pose import GOAL_WIDTH_M  # noqa: E402

CIRCLE_RADIUS_M = 9.15
PLAYER_HEIGHT_M = 1.75

# The centre spot in the goal frame: X across the goal line from the left
# post, Y up, Z out onto the pitch. Half a standard pitch from the goal line.
# EFL pitches run 100-105 m, so this is uncertain by about 2.5 m -- stated
# rather than hidden, and small against the 40-odd metres being corrected.
CENTRE = np.array([GOAL_WIDTH_M / 2.0, 0.0, pm.PITCH_LENGTH_M / 2.0])

# How heavily a halfway-line point counts against an arc pixel. There are
# hundreds of arc pixels and two line endpoints, so without this the line
# would be ignored.
HALFWAY_WEIGHT = 25.0


def camera(focal, cx, cy):
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])


def to_ground(pixels, rot, eye, focal, cx, cy):
    """Image pixels to the plane Y = 0 of the goal frame, vectorised."""
    pts = np.column_stack([pixels, np.ones(len(pixels))]).T
    rays = rot.T @ (np.linalg.inv(camera(focal, cx, cy)) @ pts)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -eye[1] / rays[1]
    hits = eye[:, None] + t * rays
    hits[:, ~(t > 0)] = np.nan
    return hits.T


def fit(arc_px, halfway_px, eye, focal0, cx, cy):
    """Rotation and focal, with the camera's position held fixed."""
    from scipy.optimize import least_squares

    forward = CENTRE - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    start, _ = cv2.Rodrigues(np.vstack([right, np.cross(forward, right),
                                        forward]))

    def residuals(q):
        rot, _ = cv2.Rodrigues(q[:3])
        focal = float(np.exp(q[3]))
        out = []
        hits = to_ground(arc_px, rot, eye, focal, cx, cy)
        radial = np.hypot(hits[:, 0] - CENTRE[0], hits[:, 2] - CENTRE[2])
        out.append(np.nan_to_num(radial - CIRCLE_RADIUS_M, nan=50.0))
        if halfway_px is not None:
            line = to_ground(halfway_px, rot, eye, focal, cx, cy)
            out.append(HALFWAY_WEIGHT * np.nan_to_num(
                line[:, 2] - CENTRE[2], nan=50.0))
        return np.concatenate(out)

    got = least_squares(residuals,
                        np.concatenate([start.ravel(), [np.log(focal0)]]),
                        method="trf", max_nfev=400)
    rot, _ = cv2.Rodrigues(got.x[:3])
    radial_rms = float(np.sqrt(np.mean(
        got.fun[:len(arc_px)] ** 2)))
    return rot, float(np.exp(got.x[3])), radial_rms


def predicted_heights(feet_px, rot, eye, focal, cx, cy, up):
    """Pixel height a 1.75 m player should have at each feet position."""
    ground = to_ground(feet_px, rot, eye, focal, cx, cy)
    heads = ground + PLAYER_HEIGHT_M * up
    k = camera(focal, cx, cy)
    out = []
    for g, h in zip(ground, heads):
        if not np.all(np.isfinite(g)):
            out.append(np.nan)
            continue
        pg = k @ (rot @ (g - eye))
        ph = k @ (rot @ (h - eye))
        if pg[2] <= 0 or ph[2] <= 0:
            out.append(np.nan)
            continue
        out.append(abs(pg[1] / pg[2] - ph[1] / ph[2]))
    return np.array(out)


def old_anchor_pose(h_img_to_plane, focal, cx, cy):
    """Full pose from an anchor homography, in the anchor's pitch frame.

    The same decomposition `probe_anchor_height` checks against a known
    camera, carried one step further to the rotation, so the old anchor can
    be put through the identical player-size test.
    """
    h = np.linalg.inv(np.asarray(h_img_to_plane, dtype=float))
    m = np.linalg.inv(camera(focal, cx, cy)) @ h
    scale = 2.0 / (np.linalg.norm(m[:, 0]) + np.linalg.norm(m[:, 1]))
    r1, r2, t = m[:, 0] * scale, m[:, 1] * scale, m[:, 2] * scale
    rot = np.column_stack([r1, r2, np.cross(r1, r2)])
    u, _, vt = np.linalg.svd(rot)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        rot = u @ np.diag([1.0, 1.0, -1.0]) @ vt
        t = -t
    eye = -rot.T @ t
    if eye[2] < 0:
        # H and -H are the same homography. Negating it negates r1, r2 and
        # t while r3 = r1 x r2 survives, which works out to exactly this:
        # flip the first two rotation columns, and the camera's height.
        rot = rot @ np.diag([-1.0, -1.0, 1.0])
        eye = np.array([eye[0], eye[1], -eye[2]])
    return rot, eye


def to_ground_plane_z(pixels, rot, eye, focal, cx, cy):
    """Image pixels to the plane z = 0 of the anchor's pitch frame."""
    pts = np.column_stack([pixels, np.ones(len(pixels))]).T
    rays = rot.T @ (np.linalg.inv(camera(focal, cx, cy)) @ pts)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -eye[2] / rays[2]
    hits = eye[:, None] + t * rays
    hits[:, ~(t > 0)] = np.nan
    return hits.T


def old_predicted_heights(feet_px, rot, eye, focal, cx, cy):
    ground = to_ground_plane_z(feet_px, rot, eye, focal, cx, cy)
    k = camera(focal, cx, cy)
    out = []
    for g in ground:
        if not np.all(np.isfinite(g)):
            out.append(np.nan)
            continue
        head = g + np.array([0.0, 0.0, PLAYER_HEIGHT_M])
        pg = k @ (rot @ (g - eye))
        ph = k @ (rot @ (head - eye))
        if pg[2] <= 0 or ph[2] <= 0:
            out.append(np.nan)
            continue
        out.append(abs(pg[1] / pg[2] - ph[1] / ph[2]))
    return np.array(out)


def measure(clip: str, corners: Path, frames: int):
    from fit_pitch_anchor import anchored_frames, halfway_line
    from probe_centre_circle import find_circle
    from check_goal_corners import load, poses_for_clip
    from src.ground_plane import GroundPlane

    out = Path(f"output_{clip}")
    info = json.loads((out / "clip.json").read_text())
    cx, cy = info["width"] / 2.0, info["height"] / 2.0
    rows = [r for r in load(corners)
            if Path(r["out_dir"]).resolve() == out.resolve()]
    blob = json.loads((out / "ground_plane.json").read_text()) \
        if (out / "ground_plane.json").exists() else {}
    plane = None if not blob or blob.get("refused") else GroundPlane(**blob)
    poses = poses_for_clip(rows, plane)
    if not poses:
        return None
    eye = np.mean([p.camera_position() for p in poses.values()], axis=0)
    focal0 = float(np.median([p.focal_px for p in poses.values()]))
    old_focal = plane.focal_px if plane else focal0

    tracks = pd.read_parquet(out / "tracks.parquet")
    players = tracks[(tracks.cls == "player") & (tracks.crop_h > 5)]

    rng = np.random.default_rng(0)
    _, owned = anchored_frames(out, frames, rng, use_penalty_arc=False,
                               reproduce=True)
    cap = cv2.VideoCapture(info["path"])
    new_ratio, old_ratio, fits = [], [], []
    for index, homography in owned:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        circle = find_circle(frame, np.random.default_rng(int(index)))
        if circle is None:
            continue
        arc = np.asarray(circle["support"], dtype=float)
        (ex, ey), _, _ = circle["ellipse"]
        half = halfway_line(circle["segments"], (ex, ey))
        half_px = (None if half is None else
                   np.array([[half[0], half[1]], [half[2], half[3]]],
                            dtype=float))
        rot, focal, rms = fit(arc, half_px, eye, focal0, cx, cy)
        fits.append((int(index), focal, rms))

        here = players[players.frame == int(index)]
        if len(here) < 4:
            continue
        feet = here[["px", "py"]].to_numpy(float)
        seen = here.crop_h.to_numpy(float)

        new_pred = predicted_heights(feet, rot, eye, focal, cx, cy,
                                     np.array([0.0, 1.0, 0.0]))
        o_rot, o_eye = old_anchor_pose(homography, old_focal, cx, cy)
        old_pred = old_predicted_heights(feet, o_rot, o_eye, old_focal,
                                         cx, cy)
        for p, s in zip(new_pred, seen):
            if np.isfinite(p) and p > 1:
                new_ratio.append(s / p)
        for p, s in zip(old_pred, seen):
            if np.isfinite(p) and p > 1:
                old_ratio.append(s / p)
    cap.release()
    return dict(eye=eye, focal0=focal0, fits=fits,
                new=np.array(new_ratio), old=np.array(old_ratio))


def _look_at(eye, target, up):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return np.vstack([right, np.cross(forward, right), forward])


def selftest(verbose: bool = True) -> bool:
    """A known camera, a projected circle and player, recovered exactly."""
    cx, cy, true_focal = 640.0, 360.0, 2600.0
    eye = np.array([-20.0, 15.6, 18.0])       # goal frame, 15.6 m up
    rot = _look_at(eye, CENTRE, np.array([0.0, 1.0, 0.0]))
    k = camera(true_focal, cx, cy)

    def project(points):
        cam = (rot @ (points - eye).T)
        img = k @ cam
        return (img[:2] / img[2]).T

    theta = np.linspace(0.3, 2.6, 120)           # a partial arc, as seen
    arc_world = np.column_stack([CENTRE[0] + CIRCLE_RADIUS_M * np.cos(theta),
                                 np.zeros_like(theta),
                                 CENTRE[2] + CIRCLE_RADIUS_M * np.sin(theta)])
    half_world = np.array([[CENTRE[0] - 20, 0, CENTRE[2]],
                           [CENTRE[0] + 20, 0, CENTRE[2]]])
    arc_px, half_px = project(arc_world), project(half_world)

    ok = True
    got_rot, got_f, rms = fit(arc_px, half_px, eye, 1900.0, cx, cy)
    angle = float(np.degrees(np.arccos(np.clip(
        (np.trace(got_rot @ rot.T) - 1) / 2, -1, 1))))
    good = angle < 0.05 and abs(got_f - true_focal) < 5 and rms < 0.05
    ok &= good
    if verbose:
        print(f"  refit from a partial arc, focal seeded at 1900: rotation "
              f"err {angle:.3f} deg, focal {got_f:.0f} (true 2600), "
              f"rms {rms:.3f} m   {'ok' if good else 'WRONG'}")

    # A player standing 6 m from the centre spot: predicted box height must
    # equal the projected one.
    foot = CENTRE + np.array([6.0, 0.0, -4.0])
    head = foot + np.array([0.0, PLAYER_HEIGHT_M, 0.0])
    true_h = abs(project(head[None])[0, 1] - project(foot[None])[0, 1])
    pred = predicted_heights(project(foot[None]), rot, eye, true_focal,
                             cx, cy, np.array([0.0, 1.0, 0.0]))[0]
    good = abs(pred - true_h) < 0.01
    ok &= good
    if verbose:
        print(f"  player height, goal frame: predicted {pred:.2f} px, "
              f"projected {true_h:.2f}   {'ok' if good else 'WRONG'}")

    # The old anchor's path: a camera in the pitch frame (z up), its
    # image-to-plane homography, decomposed and asked the same question.
    p_eye = np.array([40.0, -20.0, 15.6])
    p_rot = _look_at(p_eye, np.array([52.5, 34.0, 0.0]),
                     np.array([0.0, 0.0, 1.0]))
    p_t = -p_rot @ p_eye
    plane_to_img = k @ np.column_stack([p_rot[:, 0], p_rot[:, 1], p_t])
    for sign in (1.0, -1.0):                 # H and -H must agree
        d_rot, d_eye = old_anchor_pose(sign * np.linalg.inv(plane_to_img),
                                       true_focal, cx, cy)
        g = np.array([50.0, 30.0, 0.0])
        top = g + np.array([0.0, 0.0, PLAYER_HEIGHT_M])
        def proj(x):
            v = k @ (p_rot @ (x - p_eye))
            return v[:2] / v[2]
        true_h = abs(proj(top)[1] - proj(g)[1])
        pred = old_predicted_heights(proj(g)[None], d_rot, d_eye,
                                     true_focal, cx, cy)[0]
        good = (abs(pred - true_h) < 0.01
                and abs(d_eye[2] - 15.6) < 0.01)
        ok &= good
        if verbose:
            print(f"  old-anchor path, H sign {sign:+.0f}: camera "
                  f"{d_eye[2]:.2f} m up, player {pred:.2f} px vs "
                  f"{true_h:.2f}   {'ok' if good else 'WRONG'}")
    return ok


def describe(ratios):
    if ratios.size == 0:
        return "nothing measurable"
    q1, med, q3 = np.percentile(ratios, [25, 50, 75])
    return (f"median {med:5.2f}  IQR {q1:4.2f}-{q3:4.2f}  "
            f"spread {(q3 - q1) / med:4.2f}  n={ratios.size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*",
                    default=["reading_0737", "stoke_7001"])
    ap.add_argument("--corners", required=True)
    ap.add_argument("--frames", type=int, default=240)
    args = ap.parse_args()

    print("Checking the fit and both height predictions on a known "
          "camera.\n")
    if not selftest():
        raise SystemExit("the refit or the decomposition is wrong; the "
                         "comparison would be meaningless")
    print()
    print("Observed player box height over the height each pose predicts "
          "for a\n1.75 m player at that spot. Right pose: a constant near 1, "
          "tightly.\n")
    for clip in args.clips:
        got = measure(clip, Path(args.corners), args.frames)
        if got is None:
            print(f"  {clip}: no corner calibration")
            continue
        focals = [f for _, f, _ in got["fits"]]
        rms = [r for _, _, r in got["fits"]]
        print(f"  {clip}  camera held at {got['eye'][1]:.1f} m up, "
              f"{len(got['fits'])} circle frames refitted")
        if focals:
            print(f"    refit focal   median {np.median(focals):.0f} px "
                  f"(corner bundle {got['focal0']:.0f}), circle fit rms "
                  f"{np.median(rms):.2f} m")
        print(f"    old anchor    {describe(got['old'])}")
        print(f"    refit         {describe(got['new'])}\n")
    print("  'spread' is the interquartile range over the median: how much "
          "the ratio\n  wanders from player to player. A pose with the wrong "
          "tilt gets near\n  players and far players wrong by different "
          "amounts, so it shows up here\n  even where the median happens to "
          "land near 1.")


if __name__ == "__main__":
    main()
