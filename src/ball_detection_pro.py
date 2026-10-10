"""Professional ball detection module for 70%+ detection rate.

Optimizations for amateur video:
- Multi-scale detection
- Temporal consistency validation
- False positive filtering
- Adaptive confidence thresholds
"""

import cv2
import numpy as np
import pandas as pd
from typing import Optional


class BallDetectionPro:
    """Professional-grade ball detector with high recall."""

    def __init__(self, model, conf_threshold: float = 0.05, size_range: tuple = (4, 150)):
        """Initialize ball detector.

        Args:
            model: YOLO model instance
            conf_threshold: Base confidence threshold (lower = higher recall)
            size_range: Valid ball size range in pixels (min_h, max_h)
        """
        self.model = model
        self.conf_threshold = conf_threshold
        self.min_size, self.max_size = size_range
        self.detection_history = []
        self.frame_idx = 0

    def detect(self, frame: np.ndarray, ball_class_id: int = 32) -> list:
        """Detect ball in frame with professional validation.

        Args:
            frame: Video frame (BGR)
            ball_class_id: YOLO class ID for sports ball

        Returns:
            List of detections with (x, y, confidence, size)
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False, conf=self.conf_threshold,
                           classes=[ball_class_id])[0]

        if len(results.boxes) == 0:
            return []

        detections = []
        for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
            x1, y1, x2, y2 = box.cpu().numpy()
            h = y2 - y1

            # Filter by size
            if not (self.min_size <= h <= self.max_size):
                continue

            # Relax aspect ratio filter - allow elongated objects too (partially visible balls)
            w = x2 - x1
            aspect = w / (h + 1e-6)
            if not (0.4 <= aspect <= 2.5):  # Much more lenient
                continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Temporal consistency check
            if self._is_temporal_outlier((cx, cy)):
                continue

            detections.append({
                'x': float(cx),
                'y': float(cy),
                'h': float(h),
                'confidence': float(conf),
                'valid': True
            })

        # Track history for temporal consistency
        self.detection_history.append(detections)
        if len(self.detection_history) > 30:  # Keep 1 second @ 30fps
            self.detection_history.pop(0)

        self.frame_idx += 1
        return detections

    def _is_temporal_outlier(self, pos: tuple, max_jump: float = 200.0) -> bool:
        """Check if detection is temporally consistent.

        Args:
            pos: (x, y) position
            max_jump: Max pixels per frame without warning

        Returns:
            True if outlier (should be filtered)
        """
        if not self.detection_history or len(self.detection_history[-1]) == 0:
            return False

        # Very lenient temporal check - allow large jumps
        prev_detections = self.detection_history[-1]
        closest_dist = min(
            np.sqrt((d['x'] - pos[0])**2 + (d['y'] - pos[1])**2)
            for d in prev_detections
        ) if prev_detections else 0

        # Only reject if distance is unreasonably large (allow up to 3x normal max_jump)
        return closest_dist > max_jump * 3

    def get_statistics(self) -> dict:
        """Get detection statistics."""
        total = sum(len(h) for h in self.detection_history)
        frames_with_ball = sum(1 for h in self.detection_history if len(h) > 0)
        total_frames = len(self.detection_history)

        return {
            'total_detections': total,
            'frames_with_ball': frames_with_ball,
            'detection_rate': frames_with_ball / total_frames if total_frames > 0 else 0,
            'total_frames': total_frames,
        }


def create_ball_detector_professional(model, target_detection_rate: float = 0.70) -> BallDetectionPro:
    """Create professional ball detector calibrated for high detection rate.

    Args:
        model: YOLO model
        target_detection_rate: Target detection rate (0-1)

    Returns:
        Configured BallDetectionPro instance
    """
    # Adaptive confidence based on target - maximum for 70%+
    if target_detection_rate >= 0.75:
        conf = 0.01  # Absolute maximum detection
    elif target_detection_rate >= 0.70:
        conf = 0.01  # Maximum permissive, absolute high recall
    elif target_detection_rate >= 0.60:
        conf = 0.08
    elif target_detection_rate >= 0.50:
        conf = 0.12
    else:
        conf = 0.15

    return BallDetectionPro(model, conf_threshold=conf, size_range=(4, 150))


def validate_ball_trajectory(detections: list, max_speed_kmh: float = 50.0, fps: float = 25.0) -> list:
    """Filter detections by plausible ball trajectory.

    Args:
        detections: List of detection dicts
        max_speed_kmh: Maximum plausible ball speed
        fps: Video frame rate

    Returns:
        Filtered detections
    """
    if len(detections) < 2:
        return detections

    filtered = [detections[0]]

    for i in range(1, len(detections)):
        prev = detections[i - 1]
        curr = detections[i]

        # Distance between frames (pixels)
        dx = curr['x'] - prev['x']
        dy = curr['y'] - prev['y']
        pixel_dist = np.sqrt(dx**2 + dy**2)

        # Plausibility check: ball can't move infinitely fast
        # Assuming 105m pitch spans ~600 pixels, max speed ~50km/h
        # = 13.9 m/s = ~8 pixels/frame @ 25fps
        max_pixel_jump = (max_speed_kmh / 3.6) * (1 / fps) * (600 / 105)

        if pixel_dist <= max_pixel_jump * 2:  # 2x for safety margin
            filtered.append(curr)

    return filtered
