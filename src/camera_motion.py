"""Compensate player positions for camera movement.

Without this a pan is indistinguishable from every player sprinting sideways:
it inflates distance covered, corrupts velocities, and breaks the test at the
heart of carry detection, which asks whether a player and the ball are moving
together.

Estimation is Lucas-Kanade optical flow on the static background, fitted with
a partial affine and RANSAC so a pan's rotation and scale are captured rather
than translation alone. Features are seeded on everything that is *not* pitch
-- the stands and stadium structure are fixed to the ground, while the grass
is what players move across.

The validator is the part that took two attempts to get right, and it matters
more than the estimator: a wrong estimate subtracted from every position is
worse than no compensation at all.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .detect_track_hybrid import _grass_mask

# Movement below this many pixels is treated as camera noise, not a pan.
MIN_MOVEMENT_PX = 1.0

# Reseed tracked features at least this often. Carrying the same points for
# hundreds of frames lets them drift onto poor texture, and the degraded
# matches drive RANSAC toward a near-zero translation: the estimator reported
# a median step of 0.00 px on footage whose true median is ~1.6 px, while
# still accumulating drift. Reseeding only when points are depleted is not
# enough, because a depleted-but-stale set still tracks.
RESEED_INTERVAL_FRAMES = 10

# Displacement, in pixels, that a validation interval should span. Large
# enough that accumulated bias shows above per-frame noise, small enough that
# most of the scene is still in frame.
VALIDATION_SPAN_PX = 100.0

MIN_VALIDATION_GAP = 5
MAX_VALIDATION_GAP = 60

# An interval needs at least this many RANSAC inliers for its direct
# measurement to count. Sparse matches on low-texture background produce
# confident nonsense: intervals with 6-13 inliers disagreed with
# well-supported ones by a factor of five on the same video.
MIN_VALIDATION_INLIERS = 20

# How far accumulated motion may differ from an independent measurement of the
# same interval, as a fraction of that measurement. This is a residual test,
# not a ratio of magnitudes: an estimate that jitters in place has the right
# total distance and the wrong displacement, and comparing magnitudes passes
# it.
#
# Measured separation on the broadcast clip -- the true estimate sits at 0.16,
# every deliberately corrupted one at 1.0 or above:
#
#   true motion                        0.16   accept
#   inflated 3x                        2.02   reject
#   steps shuffled, magnitudes kept    1.01   reject
#   random walk, same step size        1.04   reject
#   constant 5 px/frame drift          1.12   reject
#   all zeros                          1.00   reject
MAX_RESIDUAL_FRACTION = 0.5

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)

FEATURE_PARAMS = dict(
    maxCorners=400,
    qualityLevel=0.01,
    minDistance=8,
    blockSize=7,
)


# --- Burned-in graphics ------------------------------------------------------
# Scoreboards, clocks and watermarks are painted into the frame after the
# camera has moved, so they sit still while everything real slides past. The
# background mask -- "anything that is not grass" -- happily kept 90% of this
# clip's scoreboard, and optical flow then tracked those corners as though
# they were the far stand. Features that never move report zero motion, which
# drags the camera estimate toward zero: this is consistent with the
# validator finding accumulated motion undershooting a direct measurement by
# 25%, and with `auto_tune` reading 0.28 px/frame where the truth is 2.07.
#
# The giveaway is generic, so nothing here is hard-coded to one broadcaster:
# a burned-in graphic has **structure that does not move while the view
# does**. Real scenery has structure and moves; flat grass does not move much
# in appearance but has no structure to seed features on.

# Frames sampled across the clip to find pixels that never change.
GRAPHICS_SAMPLES = 32

# A pixel varies less than this across the clip (0-255) to count as static.
#
# Swept against the validator's residual, which is the one independent check
# available -- accumulated motion against a direct measurement of the same
# interval:
#
#     threshold   frame masked   median step   residual
#         6            1.4%        2.25 px       0.21
#        14            2.9%        2.29 px       0.20
#        25            5.9%        2.30 px       0.20
#
# Widening buys 0.01 of residual for four times as much of the frame, so the
# conservative value takes essentially all of the benefit. It leaves the
# translucent watermark on this clip uncaught: a semi-transparent graphic
# varies with whatever passes behind it, so it fails a test built on pixels
# that do not change. An opaque scoreboard is the damaging case and it is
# caught.
GRAPHICS_MAX_TEMPORAL_STD = 6.0

# ...and must carry at least this much local gradient to be worth excluding.
# Without it a flat patch of sky or a uniform stand qualifies, and masking
# those costs nothing but means nothing either.
GRAPHICS_MIN_GRADIENT = 12.0

# If more than this share of the frame looks static, the camera itself is
# probably still and the test cannot separate graphics from scenery. Masking
# half the frame on that basis would be worse than leaving it alone.
GRAPHICS_MAX_SHARE = 0.20


def static_graphics_mask(video_path: str, samples: int = GRAPHICS_SAMPLES,
                         verbose: bool = False) -> np.ndarray | None:
    """Pixels holding structure that never moves: burned-in graphics.

    Returns a mask that is 255 where the frame is usable and 0 over the
    graphics, or None when the test cannot be trusted -- too few frames, or
    so much of the view static that the camera is evidently not moving.
    """
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames < samples * 2:
        cap.release()
        return None

    picks = np.linspace(0, n_frames - 1, samples).astype(int)
    stack = []
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            stack.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(stack) < samples // 2:
        return None

    arr = np.stack(stack).astype(np.float32)
    temporal = arr.std(axis=0)

    # Structure, measured on the median frame so a moving player in one
    # sample cannot invent an edge.
    median = np.median(arr, axis=0).astype(np.uint8)
    gx = cv2.Sobel(median, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(median, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.GaussianBlur(np.hypot(gx, gy), (5, 5), 0)

    graphics = ((temporal < GRAPHICS_MAX_TEMPORAL_STD)
                & (gradient > GRAPHICS_MIN_GRADIENT)).astype(np.uint8) * 255
    # Join the strokes of a caption into one region, then take in its
    # surroundings: a feature sits on a corner, and the corner of a glyph is
    # a pixel or two from the glyph itself.
    graphics = cv2.morphologyEx(graphics, cv2.MORPH_CLOSE,
                                np.ones((15, 15), np.uint8))
    graphics = cv2.dilate(graphics, np.ones((9, 9), np.uint8), iterations=1)

    share = float((graphics > 0).mean())
    if share > GRAPHICS_MAX_SHARE:
        if verbose:
            print(f"  [graphics] {share:.0%} of the frame looks static; the "
                  "camera is probably still, so nothing is masked")
        return None
    if verbose:
        print(f"  [graphics] burned-in overlays cover {share:.1%} of the frame")
    return cv2.bitwise_not(graphics)


def _background_mask(frame: np.ndarray) -> np.ndarray:
    """Everything that is not pitch: stands, roof, hoardings.

    These are fixed relative to the ground, so their apparent motion is the
    camera's. Seeding features on the grass would instead track the players
    moving across it.
    """
    background = cv2.bitwise_not(_grass_mask(frame))
    # Pull back from the pitch boundary so features are not seeded on the
    # touchline, where grass and background meet.
    return cv2.erode(background, np.ones((9, 9), np.uint8), iterations=1)


def estimate_camera_motion(video_path: str, max_frames: int | None = None,
                           verbose: bool = True) -> np.ndarray:
    """Per-frame cumulative camera displacement, as (n_frames, 2) pixels."""
    cap = cv2.VideoCapture(video_path)
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return np.zeros((0, 2))

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        n_frames = min(n_frames, max_frames)

    # Burned-in graphics never move, so features seeded on them report no
    # motion and pull the estimate toward zero. Found once for the clip.
    graphics = static_graphics_mask(video_path, verbose=verbose)

    def seed_mask(frame: np.ndarray) -> np.ndarray:
        m = _background_mask(frame)
        return m if graphics is None else cv2.bitwise_and(m, graphics)

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray, mask=seed_mask(prev), **FEATURE_PARAMS)

    cumulative = np.zeros(2, dtype=float)
    out = [cumulative.copy()]
    reseeds = 0
    frames_since_seed = 0

    for _ in range(1, n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        step = np.zeros(2, dtype=float)
        if prev_pts is not None and len(prev_pts) >= 8:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **LK_PARAMS)
            if nxt is not None and status is not None:
                good_new = nxt[status.flatten() == 1]
                good_old = prev_pts[status.flatten() == 1]
                if len(good_new) >= 8:
                    matrix, _ = cv2.estimateAffinePartial2D(
                        good_old, good_new, method=cv2.RANSAC,
                        ransacReprojThreshold=3.0)
                    if matrix is not None:
                        # Where the image centre went, not the affine
                        # translation term. They agree while rotation is
                        # negligible (0.006 deg/frame measured here) and
                        # diverge sharply when it is not, because the
                        # translation term is measured about the origin.
                        h, w = gray.shape
                        centre = np.array([w / 2.0, h / 2.0])
                        moved = matrix @ np.array([centre[0], centre[1], 1.0])
                        step = moved - centre
                    else:
                        # Robust fallback: median displacement, not the
                        # maximum -- one badly tracked corner should not set
                        # the estimate.
                        step = np.median(
                            good_new.reshape(-1, 2) - good_old.reshape(-1, 2),
                            axis=0)
                    prev_pts = good_new.reshape(-1, 1, 2)

        if np.hypot(*step) < MIN_MOVEMENT_PX:
            step[:] = 0.0

        cumulative = cumulative + step
        out.append(cumulative.copy())

        frames_since_seed += 1
        if (prev_pts is None or len(prev_pts) < 40
                or frames_since_seed >= RESEED_INTERVAL_FRAMES):
            frames_since_seed = 0
            prev_pts = cv2.goodFeaturesToTrack(
                gray, mask=seed_mask(frame), **FEATURE_PARAMS)
            reseeds += 1

        prev_gray = gray

    cap.release()
    motion = np.array(out)
    if verbose and len(motion) > 1:
        net = float(np.hypot(*(motion[-1] - motion[0])))
        steps = np.linalg.norm(np.diff(motion, axis=0), axis=1)
        print(f"  [camera] {len(motion)} frames, median step "
              f"{np.median(steps):.2f} px, net drift {net:.0f} px, "
              f"{reseeds} reseeds")
    return motion


def _direct_displacement(gray_a: np.ndarray, gray_b: np.ndarray,
                         mask: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Displacement of the image centre between two frames, by descriptor match.

    Deliberately not Lucas-Kanade, which is what the estimator itself uses.
    Validating a measurement with the method that produced it is circular, and
    LK additionally fails at exactly the displacements worth checking: it
    searches locally from each point's previous position, so once the scene
    moves beyond its pyramid reach the points settle on whatever nearby
    texture resembles them and the median collapses toward zero. On broadcast
    footage that panned a measured 118 px over 21 frames, LK reported 6.6 px
    -- and a validator built on it rejected a correct estimate.

    ORB matches by descriptor, so displacement magnitude does not matter. On
    the same intervals it returns 118.4 px against the accumulated 119.3.
    """
    orb = cv2.ORB_create(nfeatures=3000)
    kp_a, desc_a = orb.detectAndCompute(gray_a, mask)
    kp_b, desc_b = orb.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None or len(kp_a) < 10 or len(kp_b) < 10:
        return None, 0

    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < MIN_VALIDATION_INLIERS:
        return None, len(good)

    src = np.float32([kp_a[m.queryIdx].pt for m in good])
    dst = np.float32([kp_b[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None:
        return None, len(good)

    n_inliers = int(inliers.sum()) if inliers is not None else len(good)
    h, w = gray_a.shape
    centre = np.array([w / 2.0, h / 2.0])
    moved = matrix @ np.array([centre[0], centre[1], 1.0])
    return moved - centre, n_inliers


def validate_motion(video_path: str, motion: np.ndarray,
                    n_anchors: int = 12, verbose: bool = True) -> tuple[bool, str]:
    """Check accumulated motion against an independent measurement.

    Frame-to-frame estimates accumulate: a small bias each step compounds into
    a large fabricated drift, which would then be subtracted from every player
    position. Measuring displacement directly does not accumulate, so
    comparing the two exposes it.

    The interval over which "directly" is measured decides whether this works
    at all. It used to be the whole video split into six, ~375 frames apart.
    That is fine for a camera that barely moves and impossible for one that
    pans: at 5.2 px/frame the scene shifts ~1950 px between anchors, so every
    feature has left a 1280 px frame. The interval is now chosen from the
    motion itself.
    """
    if len(motion) < 2:
        return False, "no motion estimated"

    steps = np.linalg.norm(np.diff(motion, axis=0), axis=1)
    typical = float(np.median(steps)) if len(steps) else 0.0
    gap = (MAX_VALIDATION_GAP if typical < 1e-6
           else int(np.clip(round(VALIDATION_SPAN_PX / typical),
                            MIN_VALIDATION_GAP, MAX_VALIDATION_GAP)))

    # The validator must exclude the same burned-in graphics the
    # estimator does, or it measures a different quantity.
    graphics = static_graphics_mask(video_path)

    cap = cv2.VideoCapture(video_path)
    starts = np.linspace(0, len(motion) - 1 - gap, n_anchors).astype(int)

    direct_vecs, accum_vecs = [], []
    for a in starts:
        b = int(a) + gap
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(a))
        ok1, first = cap.read()
        if not ok1:
            continue
        # Read forward rather than seeking: seeking compressed video is not
        # frame-accurate, and a mis-seek reads as disagreement.
        second = None
        for _ in range(gap):
            ok2, second = cap.read()
            if not ok2:
                second = None
                break
        if second is None:
            continue

        vmask = _background_mask(first)
        if graphics is not None:
            vmask = cv2.bitwise_and(vmask, graphics)
        disp, n_inliers = _direct_displacement(
            cv2.cvtColor(first, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(second, cv2.COLOR_BGR2GRAY),
            vmask)
        if disp is None or n_inliers < MIN_VALIDATION_INLIERS:
            continue

        direct_vecs.append(disp)
        accum_vecs.append(motion[b] - motion[a])

    cap.release()

    if not direct_vecs:
        return False, "could not measure displacement directly"

    direct = np.array(direct_vecs)
    accum = np.array(accum_vecs)
    direct_mag = float(np.sum(np.linalg.norm(direct, axis=1)))
    accum_mag = float(np.sum(np.linalg.norm(accum, axis=1)))
    residual = float(np.sum(np.linalg.norm(accum - direct, axis=1)))

    fraction = residual / max(direct_mag, 1e-6)
    reason = (f"direct {direct_mag:.0f} px vs accumulated {accum_mag:.0f} px "
              f"over {len(direct_vecs)} intervals of {gap} frames, "
              f"residual {residual:.0f} px ({fraction:.2f} of direct)")

    if direct_mag < 1.0 and accum_mag < 1.0:
        return True, f"static camera ({accum_mag:.0f} px, confirmed)"

    if fraction > MAX_RESIDUAL_FRACTION:
        return False, f"disagrees with direct measurement: {reason}"
    return True, f"consistent: {reason}"


def compensate(tracks: pd.DataFrame, motion: np.ndarray) -> pd.DataFrame:
    """Subtract camera displacement from track positions.

    Positions become expressed in a frame fixed to the ground rather than to
    the camera, so a pan no longer reads as player movement.
    """
    if len(motion) == 0:
        return tracks.copy()

    out = tracks.copy()
    idx = out.frame.to_numpy().astype(int).clip(0, len(motion) - 1)
    out["px"] = out.px.to_numpy(dtype=float) - motion[idx, 0]
    out["py"] = out.py.to_numpy(dtype=float) - motion[idx, 1]
    return out
