"""Where the grass is, in one frame.

Split out of `detect_track_hybrid` so that asking "is this point on the
pitch" does not import a detector. That module imports ultralytics and
supervision at the top, which pull in torch, so everything that reached
these three functions -- the line probe, and through it the anchor, the shot
detector, the crossing detector and their synthetic controls -- needed two
gigabytes of machine learning to answer a question about the colour green.

The functions are unchanged. `detect_track_hybrid` re-exports them, so every
existing import still works.
"""

from __future__ import annotations

import cv2
import numpy as np

from .constants import (
    GRASS_H_MIN, GRASS_H_MAX, GRASS_S_MIN, GRASS_S_MAX,
    GRASS_V_MIN, GRASS_V_MAX,
)


def _grass_mask(frame: np.ndarray) -> np.ndarray:
    """Return a loose pitch mask to reject crowd/stands false positives."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        (GRASS_H_MIN, GRASS_S_MIN, GRASS_V_MIN),
        (GRASS_H_MAX, GRASS_S_MAX, GRASS_V_MAX)
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _on_pitch(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = mask.shape[:2]
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    return bool(mask[yi, xi] > 0)


def _near_pitch(mask: np.ndarray, x: float, y: float, radius: int = 12) -> bool:
    """Is there pitch anywhere in a neighbourhood of this point?

    `_on_pitch` samples one pixel, which is the wrong question for anything
    that sits *on* the grass rather than being grass. A white ball is never
    green, and the single-pixel test rejected 96.7% of genuine ball
    detections. A player's feet land on a boot, a shadow or a painted line
    often enough to matter too.

    Asking whether the point is surrounded by pitch keeps the property that
    actually distinguishes play from the crowd -- the stands have no grass
    anywhere near them -- without requiring the object itself to be green.
    """
    h, w = mask.shape[:2]
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    r = max(1, int(radius))
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    window = mask[y0:y1, x0:x1]
    return bool(window.size and window.max() > 0)
