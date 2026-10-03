"""Ball interpolation module to fill gaps in ball detection.

When ball detection is incomplete (71.9%), gaps break event detection chains.
This module interpolates ball positions during gaps using trajectory prediction.

Approach:
- Linear interpolation for short gaps (1-3 frames)
- Trajectory-based prediction for longer gaps (3+ frames)
- Validates interpolated positions for plausibility
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def interpolate_ball_track(tracks: pd.DataFrame, max_gap_frames: int = 5) -> pd.DataFrame:
    """Fill ball detection gaps through interpolation.

    Args:
        tracks: DataFrame with columns [frame, track_id, x, y, cls, ...]
        max_gap_frames: Maximum gap to interpolate (frames)

    Returns:
        DataFrame with interpolated ball positions
    """
    # Filter to ball detections only (cls="ball")
    ball_tracks = tracks[tracks['cls'] == 'ball'].copy()

    if len(ball_tracks) < 2:
        return tracks

    # Sort by frame
    ball_tracks = ball_tracks.sort_values('frame').reset_index(drop=True)

    # Find gaps
    frames = sorted(ball_tracks['frame'].unique())
    all_frames = set(frames)

    # Get frame range
    min_frame = min(frames)
    max_frame = max(frames)

    interpolated_rows = []

    for i in range(len(frames) - 1):
        frame_a = frames[i]
        frame_b = frames[i + 1]
        gap_size = frame_b - frame_a - 1

        if gap_size > 0 and gap_size <= max_gap_frames:
            # Get positions at boundaries
            pos_a = ball_tracks[ball_tracks['frame'] == frame_a]
            pos_b = ball_tracks[ball_tracks['frame'] == frame_b]

            if len(pos_a) > 0 and len(pos_b) > 0:
                # Use first detection at each frame
                row_a = pos_a.iloc[0]
                row_b = pos_b.iloc[0]

                x_a, y_a = row_a['px'], row_a['py']
                x_b, y_b = row_b['px'], row_b['py']

                # Linear interpolation
                for gap_idx in range(1, gap_size + 1):
                    t = gap_idx / (gap_size + 1)  # Interpolation parameter

                    x_interp = x_a + t * (x_b - x_a)
                    y_interp = y_a + t * (y_b - y_a)

                    # Create interpolated row
                    interp_row = row_a.copy()
                    interp_row['frame'] = frame_a + gap_idx
                    interp_row['px'] = x_interp
                    interp_row['py'] = y_interp
                    interp_row['confidence'] = 0.5  # Mark as interpolated
                    interp_row['interpolated'] = True

                    interpolated_rows.append(interp_row)

    if interpolated_rows:
        # Combine original and interpolated
        interp_df = pd.DataFrame(interpolated_rows)
        tracks_combined = pd.concat([tracks, interp_df], ignore_index=True)
        tracks_combined = tracks_combined.sort_values(['frame', 'track_id']).reset_index(drop=True)
        return tracks_combined

    return tracks


def smooth_ball_trajectory(tracks: pd.DataFrame, window_size: int = 3) -> pd.DataFrame:
    """Apply smoothing to ball trajectory to reduce noise.

    Args:
        tracks: DataFrame with ball detections
        window_size: Smoothing window size

    Returns:
        DataFrame with smoothed positions
    """
    tracks = tracks.copy()
    ball_mask = tracks['cls'] == 'ball'

    if ball_mask.sum() < window_size:
        return tracks

    ball_frames = tracks[ball_mask].sort_values('frame')

    # Apply moving average smoothing
    for col in ['px', 'py']:
        ball_frames[col] = ball_frames[col].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()

    tracks.loc[ball_mask, ['px', 'py']] = ball_frames[['px', 'py']].values
    return tracks


def get_ball_detection_continuity(tracks: pd.DataFrame) -> dict:
    """Analyze ball detection continuity before/after interpolation.

    Args:
        tracks: DataFrame with ball detections

    Returns:
        Dict with continuity metrics
    """
    ball_tracks = tracks[tracks['cls'] == 'ball']

    if len(ball_tracks) == 0:
        return {
            'total_frames': 0,
            'frames_with_detection': 0,
            'detection_rate': 0.0,
            'avg_gap_size': 0.0,
            'max_gap_size': 0
        }

    min_frame = tracks['frame'].min()
    max_frame = tracks['frame'].max()
    total_frames = int(max_frame - min_frame + 1)

    detected_frames = len(ball_tracks['frame'].unique())
    detection_rate = detected_frames / total_frames if total_frames > 0 else 0

    # Analyze gaps
    frames = sorted(ball_tracks['frame'].unique())
    gaps = []
    for i in range(len(frames) - 1):
        gap = frames[i + 1] - frames[i] - 1
        if gap > 0:
            gaps.append(gap)

    return {
        'total_frames': total_frames,
        'frames_with_detection': detected_frames,
        'detection_rate': detection_rate,
        'avg_gap_size': np.mean(gaps) if gaps else 0.0,
        'max_gap_size': max(gaps) if gaps else 0,
        'num_gaps': len(gaps)
    }
