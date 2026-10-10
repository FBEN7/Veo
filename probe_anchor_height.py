"""Check the centre-circle anchor against two independent camera heights.

The anchor underpins possession, out-of-play and set pieces, and it has
never been checked against anything outside itself. Every figure resting on
it rests on an assumption.

Checking it directly failed: the anchor and the goal pose never share a
frame -- not one, across 103 goal poses and 117 anchor maps on the same clip
-- so they cannot be compared on the same ball, and a player bridge spans
three to six seconds, over which a player moves far enough to swamp the
signal.

This needs no shared frame. An anchor is a homography from the image to the
pitch plane. Inverted it maps the plane to the image, which is
`H = K [r1 r2 t]` up to scale, and that decomposes into a camera pose.
The **camera height** falls out, and two independent estimates of it
already exist for the same clips:

  * from the hand-clicked goal corners, through `goal_pose.bundle`;
  * from player sizes, through `ground_plane.fit_height_model`, which uses
    no corners, no anchor and no homography.

Three routes, no shared frames needed, nothing fitted to anything else.

## Stated before running

    the anchor is corroborated   if its implied height sits near the other
                                 two, which agree at 15.6 and 15.9 m on
                                 reading_0737
    the anchor is wrong          if it lands somewhere else entirely, and
                                 the numbers resting on it need revisiting

A decomposition that is quietly wrong would produce a confident and
meaningless number, so `selftest` builds a camera of known height, projects
the pitch through it, forms the homography and recovers the height. It runs
first and this refuses to report clip numbers if it fails.

## What it found: the anchor does not corroborate

    clip            anchors   anchor m   players m
    reading_0737         31       62.2        15.9
    stoke_1302            1       62.7        40.1
    stoke_4207            1       72.2        24.8
    stoke_7001           31       70.9        21.9

reading_0737 is the row that carries the conclusion: 31 consistent anchors,
a height fit worth having (R^2 0.16), and the goal corners agreeing with the
players at 15.6 against 15.9. The anchor says 62.2.

The focal is not the explanation. Implied height rises monotonically with
it -- 40.3 m at 1200 px, 62.2 at the ground plane's 1851, 100.8 at the
corner bundle's 3000 -- and the focal that would bring the anchor down to
15.9 m is 473 px, a 107-degree lens on a camera filming from a gantry.

Nor is it a scale error. Placing players through the same anchors puts 16 of
them across 20 m and along 37 m, from x = 35 to 56 -- straddling the halfway
line, which is exactly where an anchor fitted to the centre circle should
be. Nothing sprawls off the pitch.

Both hold at once because a homography with the wrong **tilt** still maps
the visible trapezoid onto a believable patch of pitch while distorting
depth inside it. That is the diagnosis: perspective, not scale.

## What follows

Anchor-derived absolute positions should not be trusted along the viewing
direction. Possession is a nearest-player comparison and is relative, so it
is largely unaffected; out-of-play and set pieces rest on where the ball is
against a line, and are.

The anchor returning 62 to 72 m across two stadiums, rather than scattering,
says this is systematic in how it is built rather than noise in any clip.

    python probe_anchor_height.py [--clips stoke_1302 reading_0737]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

# A camera outside this is not filming football, and a decomposition that
# lands here has failed rather than measured.
MIN_PLAUSIBLE_M, MAX_PLAUSIBLE_M = 3.0, 80.0


def camera_from_plane_homography(h_img_to_plane, focal: float,
                                 cx: float, cy: float):
    """Camera position in plane coordinates, from an image-to-plane map.

    The anchor maps image to pitch; inverted it maps pitch to image, which
    for a plane at z = 0 is

        H ~ K [r1 r2 t]

    so K^-1 H has columns proportional to r1, r2 and t. The scale comes from
    r1 and r2 being unit vectors, the third axis from their cross product,
    and the camera centre is -R^T t.
    """
    try:
        h = np.linalg.inv(np.asarray(h_img_to_plane, dtype=float))
    except np.linalg.LinAlgError:
        return None
    camera = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    m = np.linalg.inv(camera) @ h

    n1 = float(np.linalg.norm(m[:, 0]))
    n2 = float(np.linalg.norm(m[:, 1]))
    if not (np.isfinite(n1) and np.isfinite(n2)) or min(n1, n2) < 1e-12:
        return None
    # Averaging the two norms is the usual conditioning fix: noise leaves
    # the columns unequal, and taking either alone inherits its error.
    scale = 1.0 / ((n1 + n2) / 2.0)
    r1, r2, t = m[:, 0] * scale, m[:, 1] * scale, m[:, 2] * scale

    # Re-orthonormalise: the columns are only approximately a rotation.
    r3 = np.cross(r1, r2)
    rot = np.column_stack([r1, r2, r3])
    u, _, vt = np.linalg.svd(rot)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        rot = u @ np.diag([1.0, 1.0, -1.0]) @ vt
        t = -t

    position = -rot.T @ t
    # A camera below the plane means the sign convention came out inverted,
    # which a homography cannot distinguish; the pitch is only ever viewed
    # from above.
    return -position if position[2] < 0 else position


def selftest(verbose: bool = True) -> bool:
    """Recover a known camera height from a homography built to have it."""
    focal, cx, cy = 1800.0, 640.0, 360.0
    camera = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    ok = True
    for height, across, out in ((15.6, -30.0, 40.0), (24.8, 10.0, 55.0),
                                (40.1, 0.0, 70.0)):
        eye = np.array([across, out, height])
        # Look at a point on the pitch ahead of the camera.
        target = np.array([0.0, 0.0, 0.0])
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        rot = np.vstack([right, np.cross(forward, right), forward])
        tvec = -rot @ eye

        # Plane z = 0 to image, then inverted to match an anchor's direction.
        plane_to_image = camera @ np.column_stack([rot[:, 0], rot[:, 1],
                                                   tvec])
        got = camera_from_plane_homography(np.linalg.inv(plane_to_image),
                                           focal, cx, cy)
        if got is None:
            print(f"  {height:5.1f} m  FAILED to decompose")
            ok = False
            continue
        err = abs(float(got[2]) - height)
        good = err < 0.05
        ok &= good
        if verbose:
            print(f"  camera at {height:5.1f} m -> recovered "
                  f"{float(got[2]):5.1f} m   error {err:.3f} m   "
                  f"{'ok' if good else 'WRONG'}")
    return ok


def heights_for(out_dir: Path, frames: int, verbose: bool = False):
    """Implied camera height from each of a clip's anchors."""
    from fit_pitch_anchor import anchored_frames
    from ground_plane import GroundPlane

    info = json.loads((out_dir / "clip.json").read_text())
    plane_path = out_dir / "ground_plane.json"
    blob = json.loads(plane_path.read_text()) if plane_path.exists() else {}
    plane = None if not blob or blob.get("refused") else GroundPlane(**blob)
    if plane is None:
        return None, None, []

    rng = np.random.default_rng(0)
    _, owned = anchored_frames(out_dir, frames, rng, use_penalty_arc=False,
                               reproduce=True)
    heights = []
    for _, homography in owned:
        got = camera_from_plane_homography(homography, plane.focal_px,
                                           info["width"] / 2.0,
                                           info["height"] / 2.0)
        if got is None:
            continue
        height = float(got[2])
        if MIN_PLAUSIBLE_M <= height <= MAX_PLAUSIBLE_M:
            heights.append(height)
    return plane, len(owned), heights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=[
        "reading_0737", "stoke_1302", "stoke_4207", "stoke_7001",
        "reading_1155", "reading_2519"])
    ap.add_argument("--frames", type=int, default=240)
    args = ap.parse_args()

    print("Recovering a known camera height from a homography built to "
          "have it.\n")
    if not selftest():
        raise SystemExit("the decomposition is wrong; clip numbers would "
                         "be meaningless")

    print("\n\nThe anchor's implied camera height, against player sizes.\n")
    print(f"  {'clip':>14s} {'anchors':>8s} {'usable':>7s} "
          f"{'anchor m':>9s} {'spread':>14s} {'players m':>10s} "
          f"{'ratio':>6s}")
    for name in args.clips:
        out_dir = Path(f"output_{name}")
        if not (out_dir / "clip.json").exists():
            continue
        plane, total, heights = heights_for(out_dir, args.frames)
        if plane is None:
            print(f"  {name:>14s} {'no ground plane, so no focal':>40s}")
            continue
        if not heights:
            print(f"  {name:>14s} {total:8d} {0:7d}   nothing plausible")
            continue
        got = np.array(heights)
        print(f"  {name:>14s} {total:8d} {len(got):7d} "
              f"{np.median(got):9.1f} "
              f"{f'{got.min():.0f}-{got.max():.0f}':>14s} "
              f"{plane.h_cam_m:10.1f} "
              f"{np.median(got) / plane.h_cam_m:6.2f}")

    print("\n  The two columns are independent: the anchor's height comes "
          "from pitch\n  markings through a homography, the players' from "
          "how tall people look\n  where. Neither was fitted to the other.")
    print("\n  On reading_0737 a third estimate exists, from the "
          "hand-clicked goal\n  corners: 15.6 m, against 15.9 from players. "
          "An anchor near those\n  corroborates all three.")


if __name__ == "__main__":
    main()
