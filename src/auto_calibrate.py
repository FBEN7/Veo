"""Automatic pitch homography calibration using white line detection.

Replaces the manual ``src/calibrate.py`` click-based workflow.

Algorithm
---------
1. Sample several frames and compute a median background (removes moving players).
2. Build a "white-on-green" mask: pixels that are bright/low-saturation AND
   sit on a green background → pitch markings.
3. Apply a Hough probabilistic line transform to find long straight segments.
4. Classify segments as **horizontal** (touchlines) or **vertical** (goal lines).
5. Pick the extreme lines: topmost + bottommost horizontal, leftmost + rightmost
   vertical.  These are the far touchline, near touchline, and the two goal lines.
6. Compute the 4 intersection points → pitch corners in image coordinates.
7. Map those corners to FIFA world coordinates (105 × 68 m) via
   ``cv2.getPerspectiveTransform`` and save the 3×3 matrix.

Usage
-----
    python -m src.auto_calibrate data/match.mp4
    python -m src.auto_calibrate data/match.mp4 --debug
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M  =  68.0

PITCH_WORLD_PTS = np.array([
    [0.0,            0.0           ],   # top-left
    [PITCH_LENGTH_M, 0.0           ],   # top-right
    [PITCH_LENGTH_M, PITCH_WIDTH_M ],   # bottom-right
    [0.0,            PITCH_WIDTH_M ],   # bottom-left
], dtype=np.float32)

# Number of frames to blend for the background estimate
N_SAMPLE_FRAMES = 12
SAMPLE_STEP_S   = 5.0


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def _sample_frames(video_path: str) -> list[np.ndarray]:
    """Return sampled frames spread through the video timeline."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(SAMPLE_STEP_S * fps))

    frames = []
    for i in range(N_SAMPLE_FRAMES):
        pos = min(step * i, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def _median_frame(video_path: str) -> np.ndarray:
    """Return a pixel-wise median of sampled frames."""
    frames = _sample_frames(video_path)
    if not frames:
        raise RuntimeError(f"Could not read any frames from {video_path}")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def _best_line_frame(video_path: str) -> np.ndarray:
    """Return the sampled frame with strongest white-line evidence.

    This helps when median blending weakens markings too much for Hough.
    """
    frames = _sample_frames(video_path)
    if not frames:
        raise RuntimeError(f"Could not read any frames from {video_path}")

    best_idx = 0
    best_score = -1
    for i, fr in enumerate(frames):
        mask = _white_on_green_mask(fr)
        score = int(mask.sum())
        if score > best_score:
            best_score = score
            best_idx = i

    return frames[best_idx]


# ---------------------------------------------------------------------------
# White-on-green mask
# ---------------------------------------------------------------------------

def _white_on_green_mask(bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask of white pitch-marking pixels on green grass."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Green grass
    green_mask = cv2.inRange(
        hsv,
        np.array([25, 25, 50], np.uint8),
        np.array([95, 255, 255], np.uint8),
    )

    # White: low saturation, high brightness
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 175], np.uint8),
        np.array([180, 50, 255], np.uint8),
    )

    # Dilate green to catch line pixels right on the border
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    green_dilated = cv2.dilate(green_mask, k, iterations=2)

    # White lines are on the pitch
    return cv2.bitwise_and(white_mask, green_dilated)


# ---------------------------------------------------------------------------
# Hough line detection and corner extraction
# ---------------------------------------------------------------------------

def _line_angle_deg(x1: int, y1: int, x2: int, y2: int) -> float:
    return abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))


def _intersect(l1: tuple, l2: tuple) -> tuple[float, float] | None:
    """Return the intersection point of two infinite lines (each given as
    (x1, y1, x2, y2) from a detected segment)."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None  # parallel
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return float(x), float(y)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def _corners_are_plausible(corners: np.ndarray, frame_h: int, frame_w: int) -> bool:
    """Reject clearly wrong quadrilaterals (stands/scoreboard instead of pitch)."""
    if corners.shape != (4, 2):
        return False

    tl, tr, br, bl = corners
    top_y = float((tl[1] + tr[1]) * 0.5)
    bot_y = float((bl[1] + br[1]) * 0.5)
    height = bot_y - top_y

    top_w = float(np.hypot(*(tr - tl)))
    bot_w = float(np.hypot(*(br - bl)))

    area = cv2.contourArea(corners.reshape(-1, 1, 2))
    if area < frame_h * frame_w * 0.10:
        return False
    if height < frame_h * 0.30:
        return False
    if bot_y < frame_h * 0.60:
        return False
    if top_w < frame_w * 0.25 or bot_w < frame_w * 0.25:
        return False
    return True


def _detect_corners(
    mask: np.ndarray,
    frame_h: int,
    frame_w: int,
    min_line_len_frac: float = 0.18,
) -> np.ndarray | None:
    """Find the 4 pitch corners from white-line Hough segments.

    Returns shape (4, 2) float32 array: TL, TR, BR, BL — or None on failure.
    """
    min_len = int(min(frame_h, frame_w) * min_line_len_frac)

    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 360,   # 0.5° resolution
        threshold=40,
        minLineLength=min_len,
        maxLineGap=25,
    )

    if lines is None or len(lines) < 2:
        return None

    horizontal, vertical = [], []
    for seg in lines:
        seg = seg.flatten()
        x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
        ang = _line_angle_deg(x1, y1, x2, y2)
        length = np.hypot(x2 - x1, y2 - y1)
        if ang < 20 and length > min_len:         # nearly horizontal
            horizontal.append((x1, y1, x2, y2, length))
        elif ang > 70 and length > min_len * 0.6: # nearly vertical
            vertical.append((x1, y1, x2, y2, length))

    if not horizontal or not vertical:
        return None

    # ------------------------------------------------------------------
    # For each set, find the two EXTREME lines (outermost boundaries).
    # We rank by the y-coordinate of midpoint for horizontal,
    # and x-coordinate of midpoint for vertical.
    # ------------------------------------------------------------------

    # Sort horizontal lines by midpoint y (ascending = top of image = far touchline)
    horizontal.sort(key=lambda s: (s[1] + s[3]) / 2)
    vertical.sort(key=lambda s: (s[0] + s[2]) / 2)

    far_line  = horizontal[0]          # topmost horizontal
    near_line = horizontal[-1]         # bottommost horizontal
    left_line = vertical[0]            # leftmost vertical
    right_line = vertical[-1]          # rightmost vertical

    # Compute 4 corner intersections
    tl = _intersect(far_line[:4], left_line[:4])
    tr = _intersect(far_line[:4], right_line[:4])
    br = _intersect(near_line[:4], right_line[:4])
    bl = _intersect(near_line[:4], left_line[:4])

    if any(c is None for c in (tl, tr, br, bl)):
        return None

    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    # Sanity check: corners must form a quadrilateral that covers a
    # significant portion of the frame and isn't degenerate.
    area = cv2.contourArea(corners.reshape(4, 1, 2))
    if area < frame_h * frame_w * 0.1:
        return None

    return corners


def _fallback_corners_from_green(bgr: np.ndarray) -> np.ndarray:
    """Estimate pitch corners from dominant green pitch contour.

    This is more robust than row/column scans and avoids mapping the full frame
    as the pitch when touchlines are not fully visible.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(
        hsv,
        np.array([30, 45, 35], np.uint8),
        np.array([95, 255, 255], np.uint8),
    )
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k_open, iterations=1)

    h, w = green_mask.shape
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Final guard fallback
        return np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32,
        )

    largest = max(contours, key=cv2.contourArea)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    # Build a trapezoid from left/right grass envelope by scanlines.
    rows = np.where(mask.max(axis=1) > 0)[0]
    if len(rows) == 0:
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    widths = []
    lr = {}
    for y in rows:
        xs = np.where(mask[y] > 0)[0]
        if len(xs) < 10:
            continue
        x_l, x_r = int(xs.min()), int(xs.max())
        widths.append((y, x_r - x_l + 1))
        lr[y] = (x_l, x_r)

    if not widths:
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    # Estimate the tribune->pitch transition from row-wise green density.
    row_frac = mask.mean(axis=1) / 255.0
    k = np.ones(11, dtype=np.float32) / 11.0
    row_s = np.convolve(row_frac, k, mode="same")

    far_y = None
    y0 = int(h * 0.05)
    y1 = int(h * 0.90)
    for y in range(y0, y1):
        if row_s[y] > 0.30 and np.mean(row_s[y:min(h, y + 18)]) > 0.34:
            far_y = y
            break
    if far_y is None:
        # fallback to first row with meaningful grass occupancy
        cand = np.where(row_s > 0.30)[0]
        far_y = int(cand[0]) if len(cand) else int(rows[0])

    # Near touchline candidate: last sufficiently wide row, prefer lower frame.
    min_near_width = int(w * 0.28)
    near_y = None
    for y, ww in reversed(widths):
        if y >= int(h * 0.55) and ww >= min_near_width:
            near_y = y
            break
    if near_y is None:
        near_y = widths[-1][0]

    # Enforce minimum vertical spread to avoid tiny top rectangles.
    min_span = int(h * 0.45)
    if near_y - far_y < min_span:
        near_y = min(h - 1, far_y + min_span)
        if near_y not in lr:
            y_probe = near_y
            while y_probe > far_y and y_probe not in lr:
                y_probe -= 1
            near_y = y_probe if y_probe in lr else widths[-1][0]

    # Robust left/right at far/near by taking median over a small vertical band.
    def _band_lr(yc: int, half: int = 5) -> tuple[int, int]:
        vals = [lr[y] for y in range(max(0, yc - half), min(h, yc + half + 1)) if y in lr]
        if not vals:
            return (0, w - 1)
        lefts = sorted(v[0] for v in vals)
        rights = sorted(v[1] for v in vals)
        return int(np.median(lefts)), int(np.median(rights))

    xlt, xrt = _band_lr(int(far_y), half=6)
    xlb, xrb = _band_lr(int(near_y), half=6)

    quad = np.array(
        [[xlt, far_y], [xrt, far_y], [xrb, near_y], [xlb, near_y]],
        dtype=np.float32,
    )
    quad = _order_corners(quad)
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    return quad


# ---------------------------------------------------------------------------
# Main calibration function
# ---------------------------------------------------------------------------

def auto_calibrate(
    video_path: str,
    out_path: str = "output/homography.npy",
    debug: bool = False,
) -> np.ndarray:
    """Detect pitch corners automatically and compute the homography matrix.

    Returns the 3×3 homography H such that world_pt = H @ image_pt_homog.
    """
    print(f"[AutoCal] Sampling frames from {video_path} …")
    frame = _median_frame(video_path)
    h, w = frame.shape[:2]

    print("[AutoCal] Detecting white pitch lines …")
    wog = _white_on_green_mask(frame)
    corners = _detect_corners(wog, h, w)

    method = "Hough lines (median frame)"
    if corners is not None:
        corners = _order_corners(corners)
        if not _corners_are_plausible(corners, h, w):
            print("[AutoCal] Hough corners on median frame rejected (implausible geometry).")
            corners = None

    if corners is None:
        print("[AutoCal] Hough failed on median frame — trying best single frame.")
        frame_best = _best_line_frame(video_path)
        wog_best = _white_on_green_mask(frame_best)
        corners = _detect_corners(wog_best, frame_best.shape[0], frame_best.shape[1])
        # Keep this frame as fallback source as it is usually less ghosted than median.
        frame = frame_best
        wog = wog_best
        if corners is not None:
            corners = _order_corners(corners)
            if _corners_are_plausible(corners, frame_best.shape[0], frame_best.shape[1]):
                method = "Hough lines (best single frame)"
            else:
                print("[AutoCal] Hough corners on best frame rejected (implausible geometry).")
                corners = None

    if corners is None:
        print("[AutoCal] Hough detection failed — falling back to green-region scan.")
        corners = _fallback_corners_from_green(frame)
        method = "green-region scan"

    print(f"[AutoCal] Method used: {method}")
    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
    for label, pt in zip(labels, corners):
        print(f"  {label}: ({pt[0]:.0f}, {pt[1]:.0f})")

    H = cv2.getPerspectiveTransform(corners, PITCH_WORLD_PTS)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, H)
    print(f"[AutoCal] Homography saved → {out_path}")

    if debug:
        vis = frame.copy()
        for i, pt in enumerate(corners):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(vis, labels[i], (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        pts_draw = corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(vis, [pts_draw], True, (0, 255, 0), 2)
        wog_color = cv2.cvtColor(wog, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([vis, cv2.addWeighted(frame, 0.6, wog_color, 0.4, 0)])
        debug_path = Path(out_path).parent / "auto_calibrate_debug.png"
        cv2.imwrite(str(debug_path), combined)
        print(f"[AutoCal] Debug image → {debug_path}")

    return H


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Auto-calibrate pitch homography.")
    ap.add_argument("video", help="Path to the input video.")
    ap.add_argument("--out", default="output/homography.npy")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    auto_calibrate(args.video, args.out, args.debug)

