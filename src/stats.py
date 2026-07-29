"""Calcul des métriques à partir des tracks + homographie.

Métriques MVP (fiables -> approximatives, dans cet ordre) :
  - heatmaps par équipe
  - distance parcourue / vitesses / sprints par track
  - field tilt (position moyenne en X par équipe)
  - possession proxy (joueur le plus proche du ballon) -- APPROXIMATIF
"""
import numpy as np
import pandas as pd

PITCH_X, PITCH_Y = 105.0, 68.0
SPRINT_KMH = 20.0
MAX_SPEED_KMH = 40.0  # au-delà = erreur de tracking, on filtre


def to_pitch_coords(tracks: pd.DataFrame, H: np.ndarray) -> pd.DataFrame:
    pts = tracks[["px", "py"]].to_numpy(dtype=np.float64)
    ones = np.ones((len(pts), 1))
    proj = (H @ np.hstack([pts, ones]).T).T
    proj = proj[:, :2] / proj[:, 2:3]
    out = tracks.copy()
    out["x"], out["y"] = proj[:, 0], proj[:, 1]
    # rejette ce qui tombe hors terrain (marge 5m) : spectateurs, bancs
    mask = out.x.between(-5, PITCH_X + 5) & out.y.between(-5, PITCH_Y + 5)
    return out[mask].reset_index(drop=True)


def _smooth(g: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    g_sorted = g.sort_values("time_s").copy()
    g_sorted["x"] = g_sorted.x.rolling(window, min_periods=1, center=True).median()
    g_sorted["y"] = g_sorted.y.rolling(window, min_periods=1, center=True).median()
    return g_sorted


def physical_stats(tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    players = tracks[(tracks.cls == "player") & (tracks.team.isin(["team_A", "team_B"]))]
    for (tid, team), g in players.groupby(["track_id", "team"]):
        if len(g) < 20:
            continue
        g = _smooth(g)
        dx = g.x.diff()
        dy = g.y.diff()
        dt = g.time_s.diff()
        speed_kmh = np.sqrt(dx**2 + dy**2) / dt * 3.6
        valid = speed_kmh < MAX_SPEED_KMH
        dist = float(np.sqrt(dx[valid]**2 + dy[valid]**2).sum())
        rows.append(dict(
            track_id=tid, team=team,
            minutes_tracked=float((g.time_s.max() - g.time_s.min()) / 60),
            distance_m=round(dist, 1),
            top_speed_kmh=round(float(speed_kmh[valid].max()), 1),
            n_sprints=int(((speed_kmh > SPRINT_KMH) & valid).sum()),
        ))
    return pd.DataFrame(rows).sort_values("distance_m", ascending=False)


def field_tilt(tracks: pd.DataFrame) -> dict:
    players = tracks[tracks.team.isin(["team_A", "team_B"])]
    return players.groupby("team").x.mean().round(1).to_dict()


def possession_proxy(tracks: pd.DataFrame, radius_m: float = 3.0) -> dict:
    """Frame par frame : équipe du joueur le plus proche du ballon (< radius).
    Approximatif : dépend fortement de la qualité de détection du ballon."""
    ball = tracks[tracks.cls == "ball"][["frame", "x", "y"]]
    players = tracks[tracks.team.isin(["team_A", "team_B"])]
    merged = ball.merge(players, on="frame", suffixes=("_b", ""))
    merged["d"] = np.sqrt((merged.x - merged.x_b)**2 + (merged.y - merged.y_b)**2)
    nearest = merged.loc[merged.groupby("frame").d.idxmin()]
    nearest = nearest[nearest.d < radius_m]
    if len(nearest) == 0:
        return {"team_A": None, "team_B": None, "frames_used": 0}
    share = nearest.team.value_counts(normalize=True).round(3).to_dict()
    share["frames_used"] = int(len(nearest))
    return share


def ball_detection_rate(tracks: pd.DataFrame) -> float:
    n_frames = tracks.frame.nunique()
    n_ball = tracks[tracks.cls == "ball"].frame.nunique()
    return round(n_ball / max(n_frames, 1), 3)


def ball_velocity(tracks: pd.DataFrame, smooth_window: int = 5) -> pd.DataFrame:
    """Return per-frame ball speed (km/h) and velocity components.

    Returns a DataFrame with columns: frame, time_s, bx, by, speed_kmh.
    Requires that ``to_pitch_coords`` has already been applied (x, y columns
    present).  Frames without ball detection are excluded.
    """
    ball = (
        tracks[tracks.cls == "ball"][["frame", "time_s", "x", "y"]]
        .sort_values("frame")
        .drop_duplicates("frame")
        .reset_index(drop=True)
    )
    if ball.empty:
        return pd.DataFrame(columns=["frame", "time_s", "bx", "by", "speed_kmh"])

    ball["bx"] = ball["x"].rolling(smooth_window, min_periods=1, center=True).median()
    ball["by"] = ball["y"].rolling(smooth_window, min_periods=1, center=True).median()

    dt = ball["time_s"].diff().fillna(1.0 / 30.0)
    dx = ball["bx"].diff().fillna(0.0)
    dy = ball["by"].diff().fillna(0.0)

    raw = np.sqrt(dx**2 + dy**2) / dt * 3.6
    ball["speed_kmh"] = raw.clip(upper=120.0)

    return ball[["frame", "time_s", "bx", "by", "speed_kmh"]].reset_index(drop=True)
