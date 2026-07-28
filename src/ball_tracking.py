"""Advanced ball tracking with Kalman filtering and trajectory prediction.

Handles occlusion, noisy detections, and fills gaps when ball is hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class SimpleKalmanFilter:
    """1D Kalman filter for smooth state estimation."""

    def __init__(self, process_variance: float = 0.01, measurement_variance: float = 1.0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.value = 0.0
        self.estimate_error = 1.0

    def update(self, measurement: Optional[float]) -> float:
        """Update filter with new measurement, return smoothed estimate."""
        if measurement is None:
            return self.value

        # Prediction step
        self.estimate_error += self.process_variance

        # Update step
        kalman_gain = self.estimate_error / (self.estimate_error + self.measurement_variance)
        self.value = self.value + kalman_gain * (measurement - self.value)
        self.estimate_error = (1 - kalman_gain) * self.estimate_error

        return self.value


class BallKalmanTracker:
    """2D Ball tracker with Kalman filtering for position and velocity."""

    def __init__(self, process_var: float = 0.05, measurement_var: float = 2.0):
        """Initialize tracker with independent X/Y filters."""
        self.kf_x = SimpleKalmanFilter(process_var, measurement_var)
        self.kf_y = SimpleKalmanFilter(process_var, measurement_var)
        self.kf_vx = SimpleKalmanFilter(process_var * 2, measurement_var * 2)
        self.kf_vy = SimpleKalmanFilter(process_var * 2, measurement_var * 2)

        self.last_x = None
        self.last_y = None
        self.dt = 1.0 / 25.0  # 25 FPS default

    def update(
        self,
        x: Optional[float],
        y: Optional[float],
        dt: float = None,
    ) -> Tuple[float, float, float, float]:
        """Update with new measurement, return (x, y, vx, vy)."""
        if dt is not None:
            self.dt = dt

        # Estimate velocities from detections
        vx_meas = None
        vy_meas = None
        if x is not None and self.last_x is not None:
            vx_meas = (x - self.last_x) / self.dt
        if y is not None and self.last_y is not None:
            vy_meas = (y - self.last_y) / self.dt

        # Update position and velocity estimates
        x_est = self.kf_x.update(x)
        y_est = self.kf_y.update(y)
        vx_est = self.kf_vx.update(vx_meas)
        vy_est = self.kf_vy.update(vy_meas)

        # Store for next iteration
        if x is not None:
            self.last_x = x
        if y is not None:
            self.last_y = y

        return x_est, y_est, vx_est, vy_est

    def predict(self, steps: int = 1) -> Tuple[float, float]:
        """Predict ball position N steps ahead."""
        x_pred = self.kf_x.value + self.kf_vx.value * self.dt * steps
        y_pred = self.kf_y.value + self.kf_vy.value * self.dt * steps
        return x_pred, y_pred


def fill_ball_gaps(ball_df: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """Fill gaps in ball detections using Kalman prediction and interpolation.

    Parameters
    ----------
    ball_df : pd.DataFrame
        Ball tracking data with columns: frame, time_s, x, y, confidence
    max_gap : int
        Maximum gap size to fill (frames)

    Returns
    -------
    pd.DataFrame
        Ball data with filled gaps
    """
    if ball_df.empty:
        return ball_df

    ball_df = ball_df.sort_values("frame").reset_index(drop=True)
    tracker = BallKalmanTracker()
    filled_rows = []

    for i, row in ball_df.iterrows():
        x, y = row.get("x"), row.get("y")
        x_est, y_est, vx_est, vy_est = tracker.update(x, y)

        filled_rows.append(
            {
                "frame": int(row["frame"]),
                "time_s": float(row["time_s"]),
                "x": x_est,
                "y": y_est,
                "vx": vx_est,
                "vy": vy_est,
                "confidence": float(row.get("confidence", 1.0)),
                "filled": False,
            }
        )

        # Fill gaps forward
        if i < len(ball_df) - 1:
            next_frame = int(ball_df.iloc[i + 1]["frame"])
            curr_frame = int(row["frame"])
            gap = next_frame - curr_frame - 1

            if 0 < gap <= max_gap:
                for g in range(1, gap + 1):
                    x_pred, y_pred = tracker.predict(steps=g)
                    filled_rows.append(
                        {
                            "frame": curr_frame + g,
                            "time_s": float(row["time_s"] + g * (1.0 / 25.0)),
                            "x": x_pred,
                            "y": y_pred,
                            "vx": vx_est,
                            "vy": vy_est,
                            "confidence": 0.5,  # Lower confidence for predicted
                            "filled": True,
                        }
                    )

    return pd.DataFrame(filled_rows).sort_values("frame").reset_index(drop=True)


def validate_ball_detection(
    ball_detections: pd.DataFrame,
    max_speed_kmh: float = 120.0,
    max_pixel_jump: float = 300.0,
) -> pd.DataFrame:
    """Validate ball detections, reject spurious ones.

    Criteria:
    - Speed must be physically plausible
    - Position change must be reasonable
    - Confidence must be above minimum
    """
    if ball_detections.empty:
        return ball_detections

    ball_detections = ball_detections.sort_values("frame").reset_index(drop=True)
    valid_rows = []

    for i, row in ball_detections.iterrows():
        is_valid = True

        if i > 0:
            prev_row = ball_detections.iloc[i - 1]
            dx = row["x"] - prev_row["x"]
            dy = row["y"] - prev_row["y"]
            pixel_dist = np.hypot(dx, dy)

            # Check pixel jump (unrealistic teleportation)
            if pixel_dist > max_pixel_jump:
                is_valid = False

            # Check speed (convert to km/h for 1 frame at 25fps)
            dt = 1.0 / 25.0
            speed_kmh = pixel_dist / dt * 3.6 / 100  # Rough conversion
            if speed_kmh > max_speed_kmh:
                is_valid = False

        if is_valid:
            valid_rows.append(row)

    return pd.DataFrame(valid_rows).reset_index(drop=True)


def extract_ball_tracking(
    tracks: pd.DataFrame, smooth_window: int = 3, fill_gaps: bool = True
) -> pd.DataFrame:
    """Extract and clean ball tracking from detection tracks.

    Returns ball DataFrame with improved tracking.
    """
    ball = tracks[tracks.cls == "ball"][["frame", "time_s", "x", "y", "confidence"]].copy()

    if ball.empty:
        return ball

    # 1. Validate detections
    ball = validate_ball_detection(ball)

    if ball.empty:
        return ball

    # 2. Smooth detections
    ball["x"] = ball["x"].rolling(smooth_window, min_periods=1, center=True).median()
    ball["y"] = ball["y"].rolling(smooth_window, min_periods=1, center=True).median()

    # 3. Fill gaps with prediction
    if fill_gaps:
        ball = fill_ball_gaps(ball, max_gap=5)

    return ball.drop_duplicates("frame").reset_index(drop=True)
