"""Project ball coordinates from pixel space to pitch space (meters)."""

import numpy as np
import pandas as pd


def project_ball_to_pitch(ball_px_py: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project ball coordinates from pixel space to pitch space using homography.
    
    Args:
        ball_px_py: Array of shape (N, 2) with pixel coordinates [px, py]
        H: Homography matrix (3x3) for perspective transform
    
    Returns:
        Array of shape (N, 2) with pitch coordinates [x, y] in meters
    """
    if len(ball_px_py) == 0:
        return ball_px_py
    
    # Add homogeneous coordinate
    ones = np.ones((len(ball_px_py), 1))
    pts_h = np.hstack([ball_px_py, ones])
    
    # Apply homography: proj = H @ pt
    proj_h = (H @ pts_h.T).T
    
    # Normalize by third coordinate (perspective division)
    proj = proj_h[:, :2] / proj_h[:, 2:3]
    
    return proj


def project_ball_track_to_pitch(ball_track: pd.DataFrame, H: np.ndarray) -> pd.DataFrame:
    """Project ball tracking data from pixel to pitch coordinates.

    Takes ball tracking with columns [frame, time_s, x, y, vx, vy] in pixel space
    and returns same with x, y projected to pitch meters.

    Note: Velocity (vx, vy) is NOT transformed (remains in pixel/frame units).

    Args:
        ball_track: DataFrame with ball kinematics in pixel space
        H: Homography matrix for projection

    Returns:
        DataFrame with x, y in pitch meters
    """
    if ball_track.empty:
        return ball_track

    out = ball_track.copy()

    # Project position coordinates
    pts = out[['x', 'y']].to_numpy(dtype=np.float64)
    proj_pts = project_ball_to_pitch(pts, H)
    out['x'] = proj_pts[:, 0]
    out['y'] = proj_pts[:, 1]

    # Filter out-of-pitch positions (with 5m tolerance like players)
    PITCH_X, PITCH_Y = 105.0, 68.0
    mask = out.x.between(-5, PITCH_X + 5) & out.y.between(-5, PITCH_Y + 5)

    return out[mask].reset_index(drop=True)


def project_ball_kinematics_to_pitch(ball_kinematics: pd.DataFrame, H: np.ndarray) -> pd.DataFrame:
    """Project ball kinematics from pixel to pitch coordinates.

    This is a specialized version for the output of _ball_kinematics() which uses
    columns [frame, time_s, bx, by, vel_x, vel_y, speed_kmh] in pixel space.

    Args:
        ball_kinematics: DataFrame from _ball_kinematics() with bx, by in pixel space
        H: Homography matrix for projection

    Returns:
        DataFrame with bx, by projected to pitch meters
    """
    if ball_kinematics.empty:
        return ball_kinematics

    out = ball_kinematics.copy()

    # Project position coordinates from pixel to pitch
    pts = out[['bx', 'by']].to_numpy(dtype=np.float64)
    proj_pts = project_ball_to_pitch(pts, H)
    out['bx'] = proj_pts[:, 0]
    out['by'] = proj_pts[:, 1]

    # Filter out-of-pitch positions (with 5m tolerance like players)
    PITCH_X, PITCH_Y = 105.0, 68.0
    mask = out.bx.between(-5, PITCH_X + 5) & out.by.between(-5, PITCH_Y + 5)

    return out[mask].reset_index(drop=True)
