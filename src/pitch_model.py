"""Fit a football pitch to the markings, and so find where the goal is.

Everything downstream of a shot -- the shot itself, goals, assists, chances
created, xG -- needs a goal to measure from, and this is what puts one on the
map. The markings are the only things in view whose real dimensions are
known, so fitting the known pitch to the detected lines is what turns
free-floating metres into pitch coordinates.

## Three unknowns, not eight

A homography has eight degrees of freedom and fitting one from a handful of
ambiguous line segments is the research project `probe_homography.py`
declined to start. That estimate was made before `ground_plane.py` existed.
The ground plane already pins the camera's height, its horizon and its focal
length, which fixes everything about the mapping except where the camera
stands on the pitch and which way it faces. Three unknowns remain:

    theta     the direction the pitch runs, relative to the camera
    tx, ty    where the camera sits in pitch coordinates

Three is a grid search. Eight is not.

## Theta comes from the lines themselves

Pitch markings run in two directions and only two: along the pitch and across
it. Projected onto the ground plane, the detected segments therefore cluster
around theta and theta + 90 degrees, whatever the camera is doing. Taking the
dominant direction modulo 90 pins theta to four candidates -- the four ways a
rectangle can face -- and the grid search resolves which.

## The camera does not move, and that is the test

This fits every frame independently, and a broadcast camera is bolted to a
gantry. So the implied camera position has to come out the same on every
frame of a clip, and the spread across frames is a measurement of whether the
fit is finding the pitch or finding noise. Nothing about it is circular: no
frame's fit knows about any other's, and there is no labelled answer being
fitted toward.
"""

from __future__ import annotations

import numpy as np

try:                                    # SciPy is present; the fallback keeps
    from scipy.ndimage import distance_transform_edt as _edt
except ImportError:                     # this module importable without it.
    _edt = None

import cv2

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_WIDTH_M = 7.32

PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32
SIX_YARD_DEPTH_M = 5.5
SIX_YARD_WIDTH_M = 18.32
CENTRE_CIRCLE_R_M = 9.15

# Points sampled along each detected segment before projection. The segments
# are straight in the image and curved on the ground, so projecting only the
# endpoints would throw away the curvature that identifies them.
SAMPLES_PER_SEGMENT = 14

# Ground points further away than this are dropped. Depth goes as
# 1/(row - horizon), so a segment a few pixels below the horizon lands
# kilometres away and would dominate any residual.
MAX_GROUND_DEPTH_M = 130.0

# Resolution of the rasterised pitch, in metres per cell, and how far outside
# the pitch the field extends -- the camera sees beyond the touchline.
FIELD_RESOLUTION_M = 0.25
FIELD_MARGIN_M = 12.0

# Distance beyond which a point stops counting against the fit, in metres. A
# truncated cost is what makes this robust: a third of the detected segments
# are shadows, scuffs and the edges of players, and a plain sum of squares
# would let them drag the pose.
COST_TRUNCATION_M = 2.5

# A pose is only believed when this share of points land within the
# truncation distance.
MIN_INLIER_FRACTION = 0.45


def model_lines() -> list[tuple[float, float, float, float]]:
    """The pitch, as segments in metres. Origin at a corner."""
    L, W = PITCH_LENGTH_M, PITCH_WIDTH_M
    mid_y = W / 2.0
    pa_half = PENALTY_AREA_WIDTH_M / 2.0
    sy_half = SIX_YARD_WIDTH_M / 2.0

    lines = [
        (0.0, 0.0, L, 0.0),              # touchlines
        (0.0, W, L, W),
        (0.0, 0.0, 0.0, W),              # goal lines
        (L, 0.0, L, W),
        (L / 2, 0.0, L / 2, W),          # halfway
    ]
    for near, sign in ((0.0, 1.0), (L, -1.0)):
        d = PENALTY_AREA_DEPTH_M * sign
        lines += [
            (near, mid_y - pa_half, near + d, mid_y - pa_half),
            (near, mid_y + pa_half, near + d, mid_y + pa_half),
            (near + d, mid_y - pa_half, near + d, mid_y + pa_half),
        ]
        s = SIX_YARD_DEPTH_M * sign
        lines += [
            (near, mid_y - sy_half, near + s, mid_y - sy_half),
            (near, mid_y + sy_half, near + s, mid_y + sy_half),
            (near + s, mid_y - sy_half, near + s, mid_y + sy_half),
        ]

    # Centre circle as a polyline.
    angles = np.linspace(0, 2 * np.pi, 49)
    cx, cy = L / 2, mid_y
    xs = cx + CENTRE_CIRCLE_R_M * np.cos(angles)
    ys = cy + CENTRE_CIRCLE_R_M * np.sin(angles)
    lines += [(xs[i], ys[i], xs[i + 1], ys[i + 1])
              for i in range(len(angles) - 1)]
    return lines


def distance_field(resolution_m: float = FIELD_RESOLUTION_M,
                   margin_m: float = FIELD_MARGIN_M):
    """Metres to the nearest marking, on a grid. Returns (field, origin, res).

    Precomputed once so scoring a candidate pose is a lookup per point rather
    than a distance to every line.
    """
    if _edt is None:
        raise SystemExit("scipy is required for the pitch distance field")
    width = int((PITCH_LENGTH_M + 2 * margin_m) / resolution_m)
    height = int((PITCH_WIDTH_M + 2 * margin_m) / resolution_m)
    canvas = np.ones((height, width), dtype=np.uint8)
    origin = np.array([-margin_m, -margin_m])

    for x1, y1, x2, y2 in model_lines():
        p1 = ((np.array([x1, y1]) - origin) / resolution_m).astype(int)
        p2 = ((np.array([x2, y2]) - origin) / resolution_m).astype(int)
        cv2.line(canvas, tuple(p1), tuple(p2), 0, 1)

    return _edt(canvas) * resolution_m, origin, resolution_m


def segments_to_ground(segments, plane, horizon_row: float,
                       samples: int = SAMPLES_PER_SEGMENT) -> np.ndarray:
    """Project image segments onto the ground plane, in metres.

    The points are on the pitch surface, so unlike a player's bounding box
    there is no centre-to-feet correction: the row *is* the ground row.
    """
    if not segments:
        return np.empty((0, 2))
    out = []
    for x1, y1, x2, y2 in segments:
        t = np.linspace(0.0, 1.0, samples)
        xs = x1 + (x2 - x1) * t
        ys = y1 + (y2 - y1) * t
        v = ys - horizon_row
        ok = v > 1.0
        if not ok.any():
            continue
        depth = plane.focal_px * plane.h_cam_m / v[ok]
        lateral = (xs[ok] - plane.cx) * depth / plane.focal_px
        keep = depth < MAX_GROUND_DEPTH_M
        if keep.any():
            out.append(np.column_stack([lateral[keep], depth[keep]]))
    return np.vstack(out) if out else np.empty((0, 2))


def dominant_orientation(points_by_segment) -> float:
    """The direction the markings run, in radians, modulo a right angle.

    Pitch lines run two ways and only two, so on the ground their directions
    pile up at theta and theta + 90. Doubling the angle folds those onto each
    other, which lets a single circular mean find the pair.
    """
    angles = []
    for chunk in points_by_segment:
        if len(chunk) < 2:
            continue
        delta = chunk[-1] - chunk[0]
        angles.append(np.arctan2(delta[1], delta[0]))
    if not angles:
        return 0.0
    doubled = 4.0 * np.asarray(angles)   # fold mod 90 degrees
    mean = np.arctan2(np.sin(doubled).mean(), np.cos(doubled).mean())
    return float(mean / 4.0)


def _score(points, field, origin, resolution, theta, tx, ty):
    c, s = np.cos(theta), np.sin(theta)
    x = c * points[:, 0] - s * points[:, 1] + tx
    y = s * points[:, 0] + c * points[:, 1] + ty
    col = ((x - origin[0]) / resolution).astype(np.int32)
    row = ((y - origin[1]) / resolution).astype(np.int32)
    inside = ((col >= 0) & (col < field.shape[1])
              & (row >= 0) & (row < field.shape[0]))
    if not inside.any():
        return np.inf, 0.0
    d = np.full(len(points), COST_TRUNCATION_M)
    d[inside] = np.minimum(field[row[inside], col[inside]], COST_TRUNCATION_M)
    return float(d.mean()), float((d < COST_TRUNCATION_M).mean())


def fit_pose(ground_points, points_by_segment, field, origin, resolution,
             coarse_m: float = 2.0, fine_m: float = 0.25):
    """Where the camera stands and which way the pitch runs.

    Returns a dict with theta, tx, ty, the mean truncated residual and the
    inlier fraction, or None when nothing fits.
    """
    if len(ground_points) < 40:
        return None

    base = dominant_orientation(points_by_segment)
    best = None
    # Four candidates: a rectangle looks the same every quarter turn, and
    # which way is "along" cannot be told from directions alone.
    for quarter in range(4):
        theta = base + quarter * np.pi / 2.0
        for tx in np.arange(-FIELD_MARGIN_M, PITCH_LENGTH_M + FIELD_MARGIN_M,
                            coarse_m):
            for ty in np.arange(-FIELD_MARGIN_M,
                                PITCH_WIDTH_M + FIELD_MARGIN_M, coarse_m):
                cost, inliers = _score(ground_points, field, origin,
                                       resolution, theta, tx, ty)
                if best is None or cost < best[0]:
                    best = (cost, inliers, theta, tx, ty)

    if best is None:
        return None

    # Refine locally, including a little play in theta.
    cost, inliers, theta, tx, ty = best
    for _ in range(3):
        improved = False
        for dtheta in (-0.02, 0.0, 0.02):
            for dx in (-fine_m, 0.0, fine_m):
                for dy in (-fine_m, 0.0, fine_m):
                    c, i = _score(ground_points, field, origin, resolution,
                                  theta + dtheta, tx + dx, ty + dy)
                    if c < cost:
                        cost, inliers = c, i
                        theta, tx, ty = theta + dtheta, tx + dx, ty + dy
                        improved = True
        if not improved:
            break

    return dict(theta=float(theta), tx=float(tx), ty=float(ty),
                residual_m=float(cost), inlier_fraction=float(inliers),
                n_points=int(len(ground_points)),
                usable=bool(inliers >= MIN_INLIER_FRACTION))


def fit_pose_multiframe(per_frame_points, pans_rad, field, origin, resolution,
                        **kwargs):
    """One pose for the whole clip, using every frame at once.

    Fitting each frame on its own does not work, and the reason is ambiguity
    rather than noise: a pitch is a repetitive grid of parallel lines, so
    many poses put most points near *some* line. Measured, that produces
    residuals of about a metre with three quarters of points as inliers while
    the implied camera position wanders 30 to 40 metres between frames of a
    camera bolted to a gantry.

    The camera being fixed is the constraint that removes the ambiguity.
    Ground coordinates are computed in the camera's current orientation, so
    panning rotates them and nothing else: frame i differs from frame 0 by
    the pan, which the motion estimator already measured. De-rotating each
    frame by its own pan therefore brings every observation into one common
    frame, where a single rigid pose has to explain all of them together.

    This costs no more than fitting one frame -- the de-rotation happens once
    per frame, before the search -- and it replaces N weakly constrained
    three-parameter fits with one strongly constrained one.
    """
    stacked, chunks = [], []
    for points, pan in zip(per_frame_points, pans_rad):
        if len(points) == 0 or not np.isfinite(pan):
            continue
        c, s = np.cos(-pan), np.sin(-pan)
        rotated = np.column_stack([
            c * points[:, 0] - s * points[:, 1],
            s * points[:, 0] + c * points[:, 1],
        ])
        stacked.append(rotated)
        chunks.append(rotated)
    if not stacked:
        return None
    return fit_pose(np.vstack(stacked), chunks, field, origin, resolution,
                    **kwargs)


# --- rectification from vanishing points -------------------------------------
# This route uses no ground plane at all, which is the point of it: the plane
# is the component both earlier attempts blamed. Two vanishing points give the
# horizon, the horizon gives a rectification, and in the rectified frame the
# two families of markings become axis-aligned -- so the pitch fit drops to two
# independent one-dimensional alignments against known spacings.

# Where a rectified line position stops being constant along its own segment.
# A segment that fails this was never a straight world line.
RECTIFIED_CONSTANCY = 0.08

# Rectified positions closer than this fraction of the spread are the same
# world line seen as several segments.
CLUSTER_RELATIVE_TOL = 0.02

# How close a mapped line must land to a real pitch line, in metres.
MATCH_TOLERANCE_M = 0.6

# Two lines always fit a two-point solve exactly, so two is not evidence.
MATCH_MIN_INLIERS = 3

# A broadcast frame does not show more pitch than this. Physical knowledge,
# used to reject impossible fits rather than to choose among possible ones.
MAX_VISIBLE_X_M = 85.0
MAX_VISIBLE_Y_M = 75.0

# Pitch lines at constant x (parallel to the goals) and constant y (parallel
# to the touchlines).
LINES_ACROSS_M = (0.0, 5.5, 16.5, 52.5, 88.5, 99.5, 105.0)
LINES_ALONG_M = (0.0, 13.84, 24.84, 43.16, 54.16, 68.0)


def rectify_from_vanishing_points(vp_a, vp_b, origin) -> np.ndarray:
    """Send one vanishing point to each axis at infinity.

    Rows are the three lines that must map to the axes: a line through vp_b
    and the origin becomes x = 0, a line through vp_a and the origin becomes
    y = 0, and the horizon itself becomes the line at infinity. After this
    the two families of markings are axis-aligned.
    """
    o = np.array([origin[0], origin[1], 1.0], dtype=float)
    return np.vstack([np.cross(vp_b, o), np.cross(vp_a, o),
                      np.cross(vp_a, vp_b)])


def rectified_positions(segments, rectification, axis: int):
    """Each segment's constant coordinate after rectification."""
    out = []
    for x1, y1, x2, y2 in segments:
        mapped = rectification @ np.array([[x1, x2], [y1, y2], [1.0, 1.0]])
        if np.any(np.abs(mapped[2]) < 1e-9):
            continue
        q = mapped[:2] / mapped[2]
        spread = abs(q[axis, 0] - q[axis, 1])
        if spread > RECTIFIED_CONSTANCY * (abs(q[axis, 0]) + 1e-9):
            continue
        out.append(float(q[axis].mean()))
    return out


def cluster_positions(values, relative_tol: float = CLUSTER_RELATIVE_TOL):
    """Collapse several segments of one world line into one position."""
    if not values:
        return []
    ordered = np.sort(np.asarray(values, dtype=float))
    span = max(float(np.ptp(ordered)), 1e-9)
    groups, current = [], [ordered[0]]
    for value in ordered[1:]:
        if value - current[-1] <= relative_tol * span:
            current.append(value)
        else:
            groups.append(float(np.mean(current)))
            current = [value]
    groups.append(float(np.mean(current)))
    return groups


def match_line_positions(positions, model_positions):
    """Scale and offset putting rectified positions onto real pitch lines.

    Solved exactly from two lines rather than searched: two positions and two
    model values determine the pair. Every such hypothesis is scored on how
    many of the *other* lines land on a real marking, which is why at least
    three inliers are required -- the two that generated the hypothesis are
    on it by construction and carry no information.
    """
    import itertools

    model = np.asarray(model_positions, dtype=float)
    best = None
    for i, j in itertools.combinations(range(len(positions)), 2):
        delta = positions[j] - positions[i]
        if abs(delta) < 1e-9:
            continue
        for mi, mj in itertools.permutations(range(len(model)), 2):
            scale = (model[mj] - model[mi]) / delta
            offset = model[mi] - scale * positions[i]
            mapped = scale * np.asarray(positions) + offset
            distance = np.abs(mapped[:, None] - model[None, :]).min(axis=1)
            inliers = int((distance <= MATCH_TOLERANCE_M).sum())
            if inliers < MATCH_MIN_INLIERS:
                continue
            cost = float(np.minimum(distance, MATCH_TOLERANCE_M).mean())
            if best is None or (inliers, -cost) > (best[0], -best[1]):
                best = (inliers, cost, float(scale), float(offset))
    return best


def image_to_pitch(plane, horizon_row: float, pose) -> np.ndarray:
    """The homography this whole module exists to produce.

    Image pixels to pitch metres, composed from the ground plane and the
    fitted pose:

        ground:  X = h*(x - cx)/(y - horizon),  Z = f*h/(y - horizon)
        pose:    pitch = R(theta) . (X, Z) + t
    """
    h, f, cx = plane.h_cam_m, plane.focal_px, plane.cx
    ground = np.array([
        [h, 0.0, -h * cx],
        [0.0, 0.0, f * h],
        [0.0, 1.0, -horizon_row],
    ], dtype=float)
    c, s = np.cos(pose["theta"]), np.sin(pose["theta"])
    rigid = np.array([
        [c, -s, pose["tx"]],
        [s, c, pose["ty"]],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    return rigid @ ground


def goal_centres(pose=None) -> dict[str, tuple[float, float]]:
    """Where the goals are, in pitch coordinates."""
    return {"left": (0.0, PITCH_WIDTH_M / 2.0),
            "right": (PITCH_LENGTH_M, PITCH_WIDTH_M / 2.0)}
