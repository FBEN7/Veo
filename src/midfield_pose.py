"""Midfield camera pose from the centre circle, with the camera's position fixed.

The centre-circle anchor puts its camera 62-72 m up against 15.6 and 15.9 m
from two independent methods, because its tilt comes from a horizon carried
to midfield by the camera's displacement -- a step its own docstring says
"treats a turn as a slide". This replaces that.

The goal corners fix the camera's *position* per clip, and a broadcast
camera does not travel. With position known, a midfield frame has four
unknowns, rotation and focal, which the centre circle's observed arc and the
halfway line over-determine.

Validated in `probe_constrained_anchor.py` against a quantity the fit never
sees -- how tall players appear:

    clip          pose         observed / predicted    spread
    reading_0737  old anchor   19.97                   0.90
                  this          1.04  (IQR 0.96-1.09)  0.13
    stoke_7001    old anchor   28.30                   0.82
                  this          1.11  (IQR 1.02-1.20)  0.16

Poses are expressed in the calibrated goal's frame -- X across the goal line
from the left post, Y up, Z out onto the pitch -- with the circle placed
half a pitch out, so they share a frame with the goal placer by
construction.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import pitch_model as pm
from .goal_pose import GOAL_WIDTH_M

CIRCLE_RADIUS_M = 9.15

# The centre spot, half a standard pitch out from the calibrated goal line.
# EFL pitches run 100-105 m, so this carries up to about 2.5 m of
# uncertainty along the pitch.
CENTRE = np.array([GOAL_WIDTH_M / 2.0, 0.0, pm.PITCH_LENGTH_M / 2.0])

# How heavily a halfway-line endpoint counts against an arc pixel: two
# endpoints against a few hundred arc pixels would otherwise be ignored.
HALFWAY_WEIGHT = 5.0

# What a broadcast lens can be at 1280 wide, about 90 degrees across at the
# short end and 6 at the long. Also what stops the fit collapsing: the first
# version, scored on the ground, drove the focal to 1.4e10 px and a perfect
# 0.00 m by making every ray parallel.
MIN_FOCAL_PX, MAX_FOCAL_PX = 600.0, 12000.0


def camera(focal, cx, cy):
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])


def look_at(eye, target, up=np.array([0.0, 1.0, 0.0])):
    forward = np.asarray(target, dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return np.vstack([right, np.cross(forward, right), forward])


def _scorer(arc_px, halfway_px, cx, cy):
    """Residuals of a candidate camera against an observed arc and line.

    Shared by `fit`, where the camera's position is fixed, and `fit_free`,
    where it is not; only what is free differs.

    Residuals are taken in the image: the whole circle is projected through
    the candidate camera and each observed arc pixel is scored by its
    distance to that curve. On the ground the problem had a degenerate
    solution; in the image a camera that squashes the circle to a point is
    far from most of the arc. Only observed pixels are scored, never the
    other way round, since players hide part of the circle.
    """
    theta = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
    circle = np.column_stack([CENTRE[0] + CIRCLE_RADIUS_M * np.cos(theta),
                              np.zeros_like(theta),
                              CENTRE[2] + CIRCLE_RADIUS_M * np.sin(theta)])
    half = np.array([[CENTRE[0] - 40.0, 0.0, CENTRE[2]],
                     [CENTRE[0] + 40.0, 0.0, CENTRE[2]]])
    far = 400.0

    def project(points, rot, focal, eye):
        cam = rot @ (points - eye).T
        front = cam[2] > 1e-6
        img = camera(focal, cx, cy) @ cam
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (img[:2] / img[2]).T
        return uv, front

    def residuals(rot, focal, eye):
        uv, front = project(circle, rot, focal, eye)
        # Distance to the projected curve itself, not to its nearest sample:
        # 720 samples leave about a pixel between neighbours, which read as
        # 0.99 px of error on a fit that was otherwise exact. Segments join
        # samples only where both are in front of the lens.
        nxt = np.roll(np.arange(len(uv)), -1)
        keep = front & front[nxt]
        if keep.sum() < 20:
            arc = np.full(len(arc_px), far)
        else:
            a, ab = uv[keep], uv[nxt][keep] - uv[keep]
            ap = arc_px[:, None, :] - a[None, :, :]
            along = np.clip((ap * ab[None]).sum(-1)
                            / np.maximum((ab ** 2).sum(-1), 1e-12)[None],
                            0.0, 1.0)
            nearest = a[None] + along[..., None] * ab[None]
            arc = np.sqrt(((arc_px[:, None, :] - nearest) ** 2)
                          .sum(-1)).min(axis=1)
        out = [arc]
        if halfway_px is not None:
            huv, hfront = project(half, rot, focal, eye)
            if hfront.all():
                a, b = huv
                normal = np.array([b[1] - a[1], a[0] - b[0]])
                normal /= max(np.linalg.norm(normal), 1e-9)
                out.append(HALFWAY_WEIGHT * ((halfway_px - a) @ normal))
            else:
                out.append(np.full(len(halfway_px), far))
        return np.concatenate(out)

    return residuals


def fit(arc_px, halfway_px, eye, focal0, cx, cy, seed_rot=None):
    """Rotation, focal and arc residual in pixels, camera position fixed.

    `seed_rot` starts from a neighbouring frame's answer. Consecutive frames
    differ by a fraction of a degree, so this is both faster and less likely
    to settle somewhere odd than starting from a camera aimed at the spot.
    """
    from scipy.optimize import least_squares

    eye = np.asarray(eye, dtype=float)
    score = _scorer(arc_px, halfway_px, cx, cy)
    start = seed_rot if seed_rot is not None else look_at(eye, CENTRE)
    rvec0, _ = cv2.Rodrigues(start)

    def residuals(q):
        rot, _ = cv2.Rodrigues(q[:3])
        return score(rot, float(np.exp(q[3])), eye)

    seed_f = float(np.clip(focal0, MIN_FOCAL_PX * 1.001, MAX_FOCAL_PX * 0.999))
    got = least_squares(
        residuals, np.concatenate([rvec0.ravel(), [np.log(seed_f)]]),
        bounds=([-np.inf] * 3 + [np.log(MIN_FOCAL_PX)],
                [np.inf] * 3 + [np.log(MAX_FOCAL_PX)]),
        method="trf", max_nfev=600)
    rot, _ = cv2.Rodrigues(got.x[:3])
    arc_rms = float(np.sqrt(np.mean(got.fun[:len(arc_px)] ** 2)))
    return rot, float(np.exp(got.x[3])), arc_rms


def fit_free(arc_px, halfway_px, eye0, focal0, cx, cy, reach_m: float = 150.0,
             max_height_m: float = 80.0):
    """As `fit`, with the camera's position free as well.

    The circle and halfway line do not fix a position: sliding the camera
    along its line of sight to the centre spot while zooming keeps the
    picture the same. What they do fix is that line -- a ray from the
    centre spot out to the camera -- and that is what this is for
    (`camera_position.locate`). The position returned is one point on it.
    Returns (position, rotation, focal, arc rms px).
    """
    from scipy.optimize import least_squares

    eye0 = np.asarray(eye0, dtype=float)
    score = _scorer(arc_px, halfway_px, cx, cy)
    rvec0, _ = cv2.Rodrigues(look_at(eye0, CENTRE))

    def residuals(q):
        rot, _ = cv2.Rodrigues(q[:3])
        return score(rot, float(np.exp(q[3])), q[4:7])

    lo = np.array([-np.inf] * 3 + [np.log(MIN_FOCAL_PX),
                                   eye0[0] - reach_m, 1.0, eye0[2] - reach_m])
    hi = np.array([np.inf] * 3 + [np.log(MAX_FOCAL_PX),
                                  eye0[0] + reach_m, max_height_m,
                                  eye0[2] + reach_m])
    seed_f = float(np.clip(focal0, MIN_FOCAL_PX * 1.01, MAX_FOCAL_PX * 0.99))
    x0 = np.concatenate([rvec0.ravel(), [np.log(seed_f)], eye0])
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    got = least_squares(residuals, x0, bounds=(lo, hi), method="trf",
                        max_nfev=800)
    rot, _ = cv2.Rodrigues(got.x[:3])
    arc_rms = float(np.sqrt(np.mean(got.fun[:len(arc_px)] ** 2)))
    return got.x[4:7].copy(), rot, float(np.exp(got.x[3])), arc_rms
