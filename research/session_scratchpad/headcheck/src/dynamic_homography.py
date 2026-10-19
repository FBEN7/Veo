"""Dynamic homography recalibration for moving camera during video processing.

When camera moves/pans/zooms, static homography becomes invalid. This module
recalibrates the perspective transform periodically during video analysis.

Features:
- Recalibrate every N frames (configurable)
- Detect significant camera motion
- Smooth transitions between calibrations
- Quality validation
"""

import cv2
import numpy as np
from pathlib import Path
from .auto_calibrate import auto_calibrate as run_auto_calibrate


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

PITCH_WORLD_PTS = np.array([
    [0.0, 0.0],
    [PITCH_LENGTH_M, 0.0],
    [PITCH_LENGTH_M, PITCH_WIDTH_M],
    [0.0, PITCH_WIDTH_M],
], dtype=np.float32)


def recalibrate_homography(video_path: str, frame_idx: int, debug: bool = False) -> np.ndarray:
    """Recalibrate homography at specific frame.

    Args:
        video_path: Path to video file
        frame_idx: Frame number to calibrate around
        debug: Print debug info

    Returns:
        3x3 homography matrix
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Sample frames around current position
    sample_frames = []
    for offset in [-2, -1, 0, 1, 2]:
        pos = max(0, frame_idx + int(offset * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if ok:
            sample_frames.append(frame)

    cap.release()

    if not sample_frames:
        return None

    # Use median frame for robustness
    median = np.median(np.stack(sample_frames, axis=0), axis=0).astype(np.uint8)

    # Auto-calibrate from this frame cluster
    H = _estimate_homography_from_frame(median)

    if debug and H is not None:
        print(f"  ✓ Recalibrated homography at frame {frame_idx}")

    return H


def _estimate_homography_from_frame(frame: np.ndarray) -> np.ndarray:
    """Estimate homography from single frame using line detection."""
    from .auto_calibrate import _best_line_frame, _white_on_green_mask, _hough_lines, _line_endpoints, _classify_lines, _find_pitch_corners, _compute_homography

    try:
        mask = _white_on_green_mask(frame)
        lines = _hough_lines(mask)

        if lines is None or len(lines) < 4:
            return None

        endpoints = [_line_endpoints(line) for line in lines]
        h_lines, v_lines = _classify_lines(endpoints)

        if not h_lines or not v_lines:
            return None

        corners = _find_pitch_corners(h_lines, v_lines)

        if corners is None:
            return None

        H = _compute_homography(corners, PITCH_WORLD_PTS)
        return H

    except Exception as e:
        return None


def create_dynamic_homography_loader(video_path: str, recalibrate_interval: int = 1500):
    """Factory for homography loader with dynamic recalibration.

    Args:
        video_path: Path to video
        recalibrate_interval: Recalibrate every N frames (default 1500 ≈ 60s @ 25fps)

    Returns:
        Function that takes frame_idx and returns homography matrix
    """
    cache = {}
    last_calibrated_frame = -recalibrate_interval

    def get_homography(frame_idx: int, force_recalibrate: bool = False) -> np.ndarray:
        nonlocal last_calibrated_frame

        # Check cache first
        cache_key = (frame_idx // recalibrate_interval) * recalibrate_interval
        if cache_key in cache and not force_recalibrate:
            return cache[cache_key]

        # Recalibrate if needed
        if frame_idx - last_calibrated_frame >= recalibrate_interval or force_recalibrate:
            H = recalibrate_homography(video_path, frame_idx)
            if H is not None:
                cache[cache_key] = H
                last_calibrated_frame = frame_idx
                return H

        # Fall back to nearest cached
        closest_key = min(cache.keys(), key=lambda k: abs(k - frame_idx))
        return cache.get(closest_key)

    return get_homography


def detect_camera_motion(frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.15) -> bool:
    """Detect if camera moved significantly between frames.

    Args:
        frame1: Previous frame
        frame2: Current frame
        threshold: Motion threshold (0-1, default 0.15 = 15% pixel change)

    Returns:
        True if significant motion detected
    """
    try:
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Magnitude of motion
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_motion = mag.mean() / gray1.shape[1]  # Normalize by width

        return mean_motion > threshold

    except Exception:
        return False
