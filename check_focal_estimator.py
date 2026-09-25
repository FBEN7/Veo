"""Hand the focal-length estimator a rotation whose focal length we chose.

This exists because the first version of `ground_plane.focal_from_homography`
returned zero on every clip, and zero is exactly what "there is no projective
signal in this footage" looks like. It would have been recorded as a property
of the video. It was a property of the code: both orthogonality constraints
had numerator and denominator inverted.

A synthetic control separates the two, and nothing else does. A real frame is
warped by `K R K^-1` for a focal length we pick, matched against the original
exactly as a real frame pair is, and the estimate compared with the truth.
Anything that cannot recover a focal length it was handed cannot be trusted
to measure one it was not.

    python check_focal_estimator.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.camera_motion import _background_mask
from src.ground_plane import focal_from_homography

# Tolerance for calling a recovery good, as a fraction of the true value.
TOLERANCE = 0.15

# Pans to try. Below about a degree the projective term is under the noise,
# which is a real limit of the method and is reported rather than hidden.
PANS_DEG = (0.5, 2.0, 5.0)
FOCALS = (700.0, 1400.0, 2800.0)

SOURCE = Path("output_soccernet/clip.json")


def warped_pair(frame, focal_px, pan_deg):
    h, w = frame.shape[:2]
    K = np.array([[focal_px, 0, w / 2.0],
                  [0, focal_px, h / 2.0],
                  [0, 0, 1.0]])
    t = np.radians(pan_deg)
    R = np.array([[np.cos(t), 0, np.sin(t)],
                  [0, 1, 0],
                  [-np.sin(t), 0, np.cos(t)]])
    H = K @ R @ np.linalg.inv(K)
    return cv2.warpPerspective(frame, H, (w, h))


def main():
    info = json.loads(SOURCE.read_text())
    cap = cv2.VideoCapture(info["path"])
    cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read a frame from {info['path']}")

    h, w = frame.shape[:2]
    orb = cv2.ORB_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_a, desc_a = orb.detectAndCompute(gray, _background_mask(frame))

    print("A frame warped by a rotation of known focal length, then handed "
          "back.\n")
    print(f"  {'true f':>8s} {'pan':>6s} {'shift px':>9s} {'recovered':>10s} "
          f"{'error':>7s}")
    failures = []
    for focal in FOCALS:
        for pan in PANS_DEG:
            warped = warped_pair(frame, focal, pan)
            kp_b, desc_b = orb.detectAndCompute(
                cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), None)
            pairs = matcher.knnMatch(desc_a, desc_b, k=2)
            good = [p[0] for p in pairs
                    if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
            src = np.float32([kp_a[m.queryIdx].pt for m in good])
            dst = np.float32([kp_b[m.trainIdx].pt for m in good])
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
            got = focal_from_homography(H, w, h)

            shift = focal * np.tan(np.radians(pan))
            if got is None:
                print(f"  {focal:8.0f} {pan:5.1f}d {shift:9.0f} "
                      f"{'none':>10s} {'':>7s}")
                if pan >= 2.0:
                    failures.append((focal, pan, None))
                continue
            err = (got - focal) / focal
            flag = "" if abs(err) <= TOLERANCE else "  <-- off"
            print(f"  {focal:8.0f} {pan:5.1f}d {shift:9.0f} {got:10.0f} "
                  f"{err:+6.1%}{flag}")
            # Small pans are expected to be unreliable; they are printed for
            # the record but only the usable range is a failure.
            if pan >= 2.0 and abs(err) > TOLERANCE:
                failures.append((focal, pan, got))

    print(f"\n  Pans under a degree are not expected to work: the projective "
          f"term\n  scales with the rotation and the noise does not. Only "
          f"{PANS_DEG[1]:.0f} degrees and\n  above are treated as a "
          f"requirement.")
    if failures:
        raise SystemExit(f"\n{len(failures)} recoveries outside "
                         f"{TOLERANCE:.0%} -- the estimator is wrong, not the "
                         "footage")
    print("\n  All recoveries within tolerance over the usable range.")


if __name__ == "__main__":
    main()
