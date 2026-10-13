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
PENALTY_SPOT_DEPTH_M = 11.0
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


# The markings, as kinds rather than as places.
#
# Deliberately symmetric: a "goal line", never a "left goal line". A football
# pitch is exactly symmetric end for end, which has defeated several attempts
# in this project to tell one end from the other, so a classifier asked to
# name ends would be asked for something the image cannot contain. Asked only
# what KIND of marking it is looking at, the symmetry stops being a problem
# and becomes the reason the classes are clean: both touchlines look alike
# because they are alike.
#
# The distinction this exists for is the last two. A centre circle and a
# penalty arc have the same 9.15 m radius, the same residual and, when the
# circle is half hidden, the same span -- no geometric test in this project
# has separated them. They do not look the same, though: one sits in open
# grass with a line through its middle, the other sits against a penalty box.
MARKING_CLASSES = (
    "background",
    "touchline",
    "goal line",
    "halfway line",
    "penalty area edge",
    "penalty area side",
    "goal area edge",
    "goal area side",
    "centre circle",
    "penalty arc",
)
MARKING_CLASS_INDEX = {name: i for i, name in enumerate(MARKING_CLASSES)}

# How finely each marking is sampled when building the label lookup.
MODEL_SAMPLE_M = 0.25


def _sample(x1, y1, x2, y2, step=MODEL_SAMPLE_M):
    length = float(np.hypot(x2 - x1, y2 - y1))
    count = max(2, int(length / step) + 1)
    return np.column_stack([np.linspace(x1, x2, count),
                            np.linspace(y1, y2, count)])


def labelled_model_points():
    """Every marking on the pitch as points, each carrying its kind.

    Includes the penalty arcs, which `model_lines` leaves out -- they are the
    reason this exists.
    """
    L, W = PITCH_LENGTH_M, PITCH_WIDTH_M
    mid_y = W / 2.0
    pa_half, sy_half = PENALTY_AREA_WIDTH_M / 2.0, SIX_YARD_WIDTH_M / 2.0

    pieces = [
        (_sample(0.0, 0.0, L, 0.0), "touchline"),
        (_sample(0.0, W, L, W), "touchline"),
        (_sample(0.0, 0.0, 0.0, W), "goal line"),
        (_sample(L, 0.0, L, W), "goal line"),
        (_sample(L / 2, 0.0, L / 2, W), "halfway line"),
    ]
    for near, sign in ((0.0, 1.0), (L, -1.0)):
        d = PENALTY_AREA_DEPTH_M * sign
        s = SIX_YARD_DEPTH_M * sign
        pieces += [
            (_sample(near, mid_y - pa_half, near + d, mid_y - pa_half),
             "penalty area side"),
            (_sample(near, mid_y + pa_half, near + d, mid_y + pa_half),
             "penalty area side"),
            (_sample(near + d, mid_y - pa_half, near + d, mid_y + pa_half),
             "penalty area edge"),
            (_sample(near, mid_y - sy_half, near + s, mid_y - sy_half),
             "goal area side"),
            (_sample(near, mid_y + sy_half, near + s, mid_y + sy_half),
             "goal area side"),
            (_sample(near + s, mid_y - sy_half, near + s, mid_y + sy_half),
             "goal area edge"),
        ]

        # The D: the part of a 9.15 m circle about the penalty spot that
        # lies outside the penalty area, which is 2*acos(5.5/9.15) of it.
        spot_x = near + PENALTY_SPOT_DEPTH_M * sign
        # The arc runs from where the penalty-area line cuts the circle,
        # which is (16.5 - 11) = 5.5 m from the spot, round the outside.
        half = np.arccos((PENALTY_AREA_DEPTH_M - PENALTY_SPOT_DEPTH_M)
                         / CENTRE_CIRCLE_R_M)
        angles = np.linspace(-half, half, 40)
        pieces.append((np.column_stack([
            spot_x + sign * CENTRE_CIRCLE_R_M * np.cos(angles),
            mid_y + CENTRE_CIRCLE_R_M * np.sin(angles)]), "penalty arc"))

    angles = np.linspace(0.0, 2 * np.pi, 160)
    pieces.append((np.column_stack([
        L / 2 + CENTRE_CIRCLE_R_M * np.cos(angles),
        mid_y + CENTRE_CIRCLE_R_M * np.sin(angles)]), "centre circle"))

    points = np.vstack([piece for piece, _ in pieces])
    labels = np.concatenate([
        np.full(len(piece), MARKING_CLASS_INDEX[name], dtype=np.int64)
        for piece, name in pieces])
    return points, labels


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


# --- horizon plus circle -----------------------------------------------------
# The two halves that do exist, joined. Straight lines give the horizon but
# only on penalty-area frames; the circle gives scale and origin but only at
# midfield; measured, they share a frame once in thirty. So neither is used
# where it is absent -- the horizon is a property of the camera and carries
# between frames, and the circle supplies everything the horizon cannot.
#
# The horizon alone affinely rectifies the plane: send it to infinity and
# parallel world lines become parallel again, leaving an unknown 2x2 linear
# map. The circle removes exactly that. An affine image of a circle is an
# ellipse, and the linear map taking that ellipse back to a circle of radius
# 9.15 m is determined up to a rotation -- which is the pitch's orientation,
# and which the halfway line fixes.

CENTRE_CIRCLE_RADIUS_M = 9.15
PITCH_CENTRE_M = (52.5, 34.0)


def focal_from_vanishing_points(vp_a, vp_b, principal):
    """Focal length from two vanishing points of perpendicular directions.

    The classic constraint: for directions that are orthogonal in the world,
    the rays through their vanishing points are orthogonal too, so with a
    centred principal point and square pixels

        (v_a - c) . (v_b - c) = -f^2

    Pitch markings supply exactly such a pair, which makes this a focal
    length measured from the geometry in the frame rather than from the
    camera's motion. Worth having on its own: the motion-derived estimate in
    `ground_plane.py` is the weakest link in everything downstream.
    """
    a = np.asarray(vp_a, dtype=float)[:2] - np.asarray(principal, dtype=float)
    b = np.asarray(vp_b, dtype=float)[:2] - np.asarray(principal, dtype=float)
    squared = -float(a @ b)
    if not np.isfinite(squared) or squared <= 0:
        return None
    return float(np.sqrt(squared))


def rotation_homography(focal, principal, pan_px, tilt_px) -> np.ndarray:
    """The image transform of a camera that turned, not slid.

    Carrying a horizon between frames by adding the measured displacement
    treats a rotation as a translation. That is wrong away from the centre,
    and the horizon is as far from the centre as anything gets -- hundreds of
    rows above the frame -- so it is wrong exactly where it matters. A row at
    angle t from the axis moves by sec^2(t) more than the centre does, which
    on these clips is tens of percent.

    So the turn is reconstructed properly: the displacement at the image
    centre gives the angles, the angles give a rotation, and the rotation
    gives H = K R K^-1, which moves every row by the right amount.
    """
    if focal is None or focal <= 0:
        return None
    pan = float(pan_px) / focal
    tilt = float(tilt_px) / focal

    about_y = np.array([[np.cos(pan), 0.0, np.sin(pan)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(pan), 0.0, np.cos(pan)]])
    about_x = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(tilt), -np.sin(tilt)],
                        [0.0, np.sin(tilt), np.cos(tilt)]])

    intrinsics = np.array([[focal, 0.0, principal[0]],
                           [0.0, focal, principal[1]],
                           [0.0, 0.0, 1.0]])
    homography = intrinsics @ (about_y @ about_x) @ np.linalg.inv(intrinsics)

    # The sign conventions above are a choice, so the result is checked
    # rather than assumed: the image centre must end up displaced by the
    # amount that was measured. If it does not, the rotations are inverted.
    centre = np.array([principal[0], principal[1], 1.0])
    moved = homography @ centre
    moved = moved[:2] / moved[2]
    wanted = np.array([principal[0] + pan_px, principal[1] + tilt_px])
    if np.linalg.norm(moved - wanted) > 0.5 * max(
            1.0, np.hypot(pan_px, tilt_px)):
        homography = intrinsics @ np.linalg.inv(about_y @ about_x) \
            @ np.linalg.inv(intrinsics)
    return homography


def affine_rectify_from_horizon(horizon) -> np.ndarray:
    """Send the horizon to infinity, making parallel world lines parallel."""
    line = np.asarray(horizon, dtype=float)
    scale = np.hypot(line[0], line[1])
    if scale < 1e-12:
        return np.eye(3)
    line = line / scale
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [line[0], line[1], line[2]]])


def ellipse_to_conic(ellipse) -> np.ndarray:
    """cv2's ((cx, cy), (major, minor), angle) as a conic matrix."""
    (cx, cy), (major, minor), angle = ellipse
    a, b = major / 2.0, minor / 2.0
    theta = np.radians(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    shape = rotation @ np.diag([1.0 / (a * a), 1.0 / (b * b)]) @ rotation.T
    centre = np.array([cx, cy])
    conic = np.eye(3)
    conic[:2, :2] = shape
    conic[:2, 2] = -shape @ centre
    conic[2, :2] = -(shape @ centre)
    conic[2, 2] = float(centre @ shape @ centre) - 1.0
    return conic


def transform_conic(conic, homography) -> np.ndarray:
    """A conic seen through a homography: C' = H^-T C H^-1."""
    inverse = np.linalg.inv(homography)
    return inverse.T @ conic @ inverse


def conic_centre_and_shape(conic):
    """Centre, and the matrix M with (p - c)^T M (p - c) = 1."""
    upper = conic[:2, :2]
    if abs(np.linalg.det(upper)) < 1e-15:
        return None, None
    centre = -np.linalg.solve(upper, conic[:2, 2])
    constant = float(centre @ upper @ centre) - conic[2, 2]
    if abs(constant) < 1e-15:
        return None, None
    return centre, upper / constant


def _matrix_sqrt(matrix):
    values, vectors = np.linalg.eigh(matrix)
    if np.any(values <= 0):
        return None
    return vectors @ np.diag(np.sqrt(values)) @ vectors.T


# The D subtends 2*acos(5.5/9.15) = 106 degrees of its circle. Arcs are
# looked for from a little below that, to allow for a detector that clips
# the ends, and accepted as a D only well short of what a centre circle
# shows -- measured at a median of 240 degrees.
D_SPAN_MIN = 80.0
D_SPAN_MAX = 150.0

# Where the two penalty spots sit along the pitch.
PENALTY_SPOTS_M = (11.0, PITCH_LENGTH_M - 11.0)

# The D's chord is the penalty-area line, and it lies exactly this far from
# the penalty spot: 16.5 m out from the goal line against 11 m.
D_CHORD_OFFSET_M = 5.5
D_CHORD_TOLERANCE_M = 1.5

# A chord has to be a real line, not a scrap.
MIN_CHORD_PX = 60.0


# A chord cuts its circle; a diameter bisects it. For the D, every point of
# the arc lies on one side of the penalty-area line. For a centre circle,
# the halfway line runs through the middle and the arc straddles it.
ONE_SIDED_FRACTION = 0.92


def arc_is_one_sided(homography, chord, support) -> bool:
    """Does the whole arc lie on one side of this line?

    This is what tells a real D from a centre circle that happens to be
    half-hidden, and leaving it out is what made the penalty-arc anchors
    unusable. Distance from the centre was supposed to do the job -- the
    halfway line passes through the circle's centre, the penalty-area line
    misses the D's centre by 5.5 m -- but a partial arc gives a biased
    centre, so a true diameter can measure 5.5 m away from an estimate of
    where the centre is. Measured, the anchors that resulted sat 30 to 43 m
    from where neighbouring frames put them, which is the 41.5 m of a centre
    circle shifted as though it were a D.

    Which side of a line the arc falls on does not depend on the centre at
    all, so it survives that error.
    """
    points = np.column_stack([support, np.ones(len(support))]).T
    mapped = homography @ points
    if np.any(np.abs(mapped[2]) < 1e-9):
        return False
    mapped = mapped[:2] / mapped[2]
    if not np.all(np.isfinite(mapped)):
        return False

    ends = np.array([[chord[0], chord[2]], [chord[1], chord[3]],
                     [1.0, 1.0]], dtype=float)
    ends = homography @ ends
    if np.any(np.abs(ends[2]) < 1e-9):
        return False
    ends = ends[:2] / ends[2]
    line = np.cross([ends[0, 0], ends[1, 0], 1.0],
                    [ends[0, 1], ends[1, 1], 1.0])
    norm = np.hypot(line[0], line[1])
    if norm < 1e-9:
        return False
    signed = (line[0] * mapped[0] + line[1] * mapped[1] + line[2]) / norm
    share = max(float(np.mean(signed > 0.0)), float(np.mean(signed < 0.0)))
    return share >= ONE_SIDED_FRACTION


def penalty_arc_chord(homography, segments):
    """The straight line sitting where the D's chord must sit.

    This is the positive test for a D, and leaving it out is what sank the
    first version of this file. Arc span alone does not identify a D --
    plenty of short arcs are centre circles with most of themselves hidden,
    and "correcting" one of those by 41.5 m creates the very error this is
    meant to remove. Measured that way it improved 54% of the frames it
    touched, which is a coin toss.

    The D has something a partial circle does not: a line lying 5.5 m from
    its centre, which is the edge of the penalty area that cut it. Distance
    from the centre is unaffected by the rotation, so this can be measured
    before the rotation is known -- which is just as well, because on these
    frames it is not.
    """
    centre = np.array(PITCH_CENTRE_M)
    best = None
    for x1, y1, x2, y2 in segments:
        if np.hypot(x2 - x1, y2 - y1) < MIN_CHORD_PX:
            continue
        pts = np.array([[x1, x2], [y1, y2], [1.0, 1.0]], dtype=float)
        mapped = homography @ pts
        if np.any(np.abs(mapped[2]) < 1e-9):
            continue
        mapped = mapped[:2] / mapped[2]
        if not np.all(np.isfinite(mapped)):
            continue
        line = np.cross([mapped[0, 0], mapped[1, 0], 1.0],
                        [mapped[0, 1], mapped[1, 1], 1.0])
        norm = np.hypot(line[0], line[1])
        if norm < 1e-9:
            continue
        away = abs(line @ np.array([centre[0], centre[1], 1.0])) / norm
        if abs(away - D_CHORD_OFFSET_M) > D_CHORD_TOLERANCE_M:
            continue
        score = abs(away - D_CHORD_OFFSET_M)
        if best is None or score < best[0]:
            best = (score, (x1, y1, x2, y2))
    return None if best is None else best[1]


def shift_along(dx: float) -> np.ndarray:
    out = np.eye(3)
    out[0, 2] = dx
    return out


def anchor_from_penalty_arc(homography, support):
    """Re-anchor a map built on the D, once it is known to be the D.

    Returns the corrected map and which penalty spot it was decided to be.
    """
    points = np.column_stack([support, np.ones(len(support))]).T
    mapped = homography @ points
    if np.any(np.abs(mapped[2]) < 1e-9):
        return None, None
    mapped = mapped[:2] / mapped[2]
    if not np.all(np.isfinite(mapped)):
        return None, None

    # The arc bulges away from its goal, so its centroid sits on the side of
    # the (wrongly placed) centre that faces up the pitch.
    bulge = float(mapped[0].mean()) - PITCH_CENTRE_M[0]
    spot = PENALTY_SPOTS_M[0] if bulge > 0 else PENALTY_SPOTS_M[1]
    return shift_along(spot - PITCH_CENTRE_M[0]) @ homography, spot


# The vanishing line of the ground plane when the perspective is weak enough
# to ignore it. Passing this to `metric_from_circle` makes the map affine.
AT_INFINITY_LINE = np.array([0.0, 0.0, 1.0])

# How far below the circle's centre to probe when settling which way round
# the pitch is. Far enough that the mapped difference is not noise, close
# enough to stay inside the frame.
ORIENTATION_PROBE_PX = 40.0


def turned_pitch() -> np.ndarray:
    """The pitch end for end: (x, y) -> (105 - x, 68 - y)."""
    out = np.eye(3)
    out[:2, :2] = -np.eye(2)
    out[0, 2] = PITCH_LENGTH_M
    out[1, 2] = PITCH_WIDTH_M
    return out


def metric_from_circle(horizon, ellipse, halfway_direction=None,
                       radius_m: float = CENTRE_CIRCLE_RADIUS_M):
    """Image to pitch metres, from a horizon and the centre circle.

    `halfway_direction` is the halfway line's direction in the image. It is
    used only to resolve the rotation the circle cannot -- a circle looks the
    same from every angle -- and never for position, which leaves where the
    halfway line *lands* as a free check on the result.
    """
    rectification = affine_rectify_from_horizon(horizon)
    conic = transform_conic(ellipse_to_conic(ellipse), rectification)
    centre, shape = conic_centre_and_shape(conic)
    if centre is None:
        return None

    linear = _matrix_sqrt(shape)
    if linear is None:
        return None
    linear = radius_m * linear             # now maps the ellipse to a circle

    rotation = np.eye(2)
    if halfway_direction is not None:
        direction = np.asarray(halfway_direction, dtype=float)
        mapped = rectification[:2, :2] @ direction + rectification[:2, 2] * 0
        mapped = linear @ mapped
        norm = np.linalg.norm(mapped)
        if norm > 1e-9:
            mapped = mapped / norm
            # The halfway line runs across the pitch, so it must end up
            # along the y axis.
            angle = np.arctan2(mapped[0], mapped[1])
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])

    metric = np.eye(3)
    metric[:2, :2] = rotation @ linear
    metric[:2, 2] = (np.array(PITCH_CENTRE_M)
                     - rotation @ linear @ centre)
    result = metric @ rectification

    # Settle which way round the pitch is, which nothing above has done.
    #
    # A circle is unchanged by turning it through 180 degrees and so is the
    # halfway line, so the two of them together leave the orientation
    # ambiguous. The line above resolved it with arctan2 of the line's
    # direction vector -- and that vector is built from a detected segment's
    # endpoints in whatever order the detector listed them, which HoughLinesP
    # makes no promise about. The orientation of the entire map was being
    # decided by an array index.
    #
    # Nothing caught it because a pitch is exactly symmetric under that
    # rotation: 105 minus each line across it gives the same seven numbers,
    # 68 minus each line along it gives the same six. A flipped anchor puts
    # every marking on a real pitch line, so `marking_error` scores it
    # identically. It is invisible to every accuracy figure here and it is
    # the error that matters most, because a shot flipped end for end is
    # attributed to the other goal.
    #
    # The absolute orientation cannot be recovered from markings -- by that
    # same symmetry there is no telling one end of a bare pitch from the
    # other, and it does not matter, so long as every frame agrees. The
    # camera does not orbit the pitch; it sits on one side, so the near
    # touchline stays near. Requiring that moving DOWN the image moves
    # toward increasing y pins the choice to that, and it is the same choice
    # on every frame of every clip.
    probe = np.array([[ellipse[0][0], ellipse[0][0]],
                      [ellipse[0][1], ellipse[0][1] + ORIENTATION_PROBE_PX],
                      [1.0, 1.0]])
    mapped_probe = result @ probe
    if np.all(np.abs(mapped_probe[2]) > 1e-9):
        here, below = (mapped_probe[:2] / mapped_probe[2]).T
        if below[1] < here[1]:
            result = turned_pitch() @ result
    return result


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
