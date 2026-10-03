"""Reject ball candidates that are not on the playing surface.

YOLO's `sports ball` class fires on plenty of things that are not the match
ball: heads in the crowd, advertising boards, bright patches in the stands.
Those candidates are not merely wrong, they are attractive -- a stationary
false positive is easy for a tracker to follow, and it will happily track the
crowd for a hundred frames while the real ball is in play.

The discriminator is the surroundings rather than the object. A ball is white
and never green, so testing its own pixels fails; but a ball in play always
sits in a neighbourhood that is overwhelmingly grass, and a candidate in the
stands sits in one with almost none. Measured on the broadcast clip, the two
populations barely overlap: candidates in the stands have a median grass
fraction of 0.07, candidates on the pitch 0.99.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .detect_track_hybrid import _grass_mask

# Half-width of the neighbourhood sampled around a candidate.
WINDOW_PX = 30

# Minimum share of that neighbourhood that must be pitch. Read off the gap
# between the two populations (0.07 against 0.99), so its exact value is not
# delicate -- anything from roughly 0.2 to 0.8 separates them equally well.
MIN_GRASS_FRACTION = 0.40


def grass_fraction(mask: np.ndarray, x: float, y: float,
                   window: int = WINDOW_PX) -> float:
    """Share of pitch pixels in a square neighbourhood of a point."""
    h, w = mask.shape[:2]
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    r = max(1, int(window))
    patch = mask[max(0, yi - r):min(h, yi + r + 1),
                 max(0, xi - r):min(w, xi + r + 1)]
    if patch.size == 0:
        return 0.0
    return float((patch > 0).mean())


def annotate_grass_fraction(tracks: pd.DataFrame, video_path: str,
                            verbose: bool = True) -> pd.DataFrame:
    """Add a `grass_frac` column for every ball candidate.

    Reads the video once in order rather than seeking per row: seeking
    compressed video is slow and not frame-accurate, and a mis-seek here
    samples the wrong frame's pitch.
    """
    out = tracks.copy()
    out["grass_frac"] = np.nan
    if out.empty or "cls" not in out.columns:
        return out

    ball_rows = out.index[out.cls == "ball"]
    if len(ball_rows) == 0:
        return out

    by_frame: dict[int, list[int]] = {}
    for idx in ball_rows:
        by_frame.setdefault(int(out.at[idx, "frame"]), []).append(idx)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    wanted = set(by_frame)
    last = max(wanted)

    while frame_idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in wanted:
            mask = _grass_mask(frame)
            for idx in by_frame[frame_idx]:
                out.at[idx, "grass_frac"] = grass_fraction(
                    mask, out.at[idx, "px"], out.at[idx, "py"])
        frame_idx += 1
    cap.release()

    if verbose:
        got = out.loc[ball_rows, "grass_frac"].dropna()
        if len(got):
            print(f"  [pitch] {len(got)} ball candidates, grass fraction "
                  f"median {got.median():.2f}, "
                  f"{(got >= MIN_GRASS_FRACTION).mean() * 100:.1f}% on pitch")
    return out


def filter_ball_by_pitch(tracks: pd.DataFrame,
                         min_fraction: float = MIN_GRASS_FRACTION,
                         verbose: bool = False) -> pd.DataFrame:
    """Drop ball candidates whose surroundings are not mostly pitch.

    Candidates with no grass fraction recorded are kept: an absent measurement
    is not evidence against a detection, and dropping them would silently
    delete every ball when annotation has not been run.
    """
    if tracks.empty or "grass_frac" not in tracks.columns:
        return tracks.copy()

    is_ball = tracks.cls == "ball"
    frac = tracks.grass_frac
    drop = is_ball & frac.notna() & (frac < min_fraction)

    if verbose and drop.any():
        print(f"  [pitch] dropping {int(drop.sum())} of {int(is_ball.sum())} "
              f"ball candidates below {min_fraction:.2f} grass")

    return tracks[~drop].copy()
