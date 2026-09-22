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

# Percentile of a track's frame-to-frame speeds reported as its top speed.
#
# This was the maximum, and a maximum over a noisy signal measures the noise.
# On the four labelled windows a track's maximum was about double its own 95th
# percentile -- 29-31 km/h against 15-16 -- and 7-17% of tracks reached the
# 40 km/h discard threshold, which is a filter catching tracking error rather
# than a limit football approaches.
#
# There is no ground-truth speed here, so the candidates were scored for
# reliability instead: a statistic that measures a player agrees between the
# first half of their track and the second, one that measures jitter does not.
# Over 320 tracks of at least 3 s (`sweep_top_speed.py`):
#
#     statistic          split-half r   median   pop. max   at 40 km/h
#     maximum (was)              0.50     30.7       40.0          13%
#     95th percentile            0.56     15.9       37.0           0%
#     90th percentile            0.56     13.1       36.3           0%
#     fastest 0.5 s window       0.43     15.8       39.9           5%
#
# The rolling window is what sport science reports and it came last, because
# taking the maximum *over windows* is still an extreme-value statistic: one
# bad frame corrupts every window containing it and the maximum then selects
# the worst of them.
#
# 95 rather than 90 because both are equally reliable and 95 keeps more of the
# spread between quick and slow players (interquartile range 9.3 against 7.9).
# At 25 fps over a typical track it is the speed held for roughly half a
# second, which is close to what "top speed" is meant to mean.
TOP_SPEED_PERCENTILE = 95.0


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

        usable = speed_kmh[valid]
        if usable.empty:
            continue
        top = float(np.percentile(usable, TOP_SPEED_PERCENTILE))

        rows.append(dict(
            track_id=tid, team=team,
            minutes_tracked=float((g.time_s.max() - g.time_s.min()) / 60),
            distance_m=round(dist, 1),
            top_speed_kmh=round(top, 1),
            # n_sprints has the same fault this file just fixed: it counts
            # individual frames over a threshold, and the frames above 20 km/h
            # are the same tail that made the maximum unusable. It is left
            # alone because changing it changes a second published number, and
            # it is not what was asked for -- but it should not be trusted.
            n_sprints=int(((speed_kmh > SPRINT_KMH) & valid).sum()),
        ))
    return pd.DataFrame(rows).sort_values("distance_m", ascending=False)


def top_speed_reliability(tracks: pd.DataFrame, min_track_s: float = 3.0) -> dict:
    """How much of the reported top speed is the player rather than the tracker.

    There is no ground-truth speed, but a statistic that measures a player
    agrees between the first half of their track and the second, and one that
    measures jitter does not. This splits every sufficiently long track and
    correlates the halves.

    It is worth running per clip rather than trusting a number measured
    elsewhere, because the answer depends on the footage. On 720p broadcast
    the reported statistic reaches 0.56; on 640x360 Veo footage nothing
    reaches 0.35, so per-track top speed there is not a player's speed
    whatever statistic is used.
    """
    players = tracks[(tracks.cls == "player")
                     & (tracks.team.isin(["team_A", "team_B"]))]
    first, second = [], []
    for _, g in players.groupby("track_id"):
        if len(g) < 20:
            continue
        g = g.sort_values("time_s")
        t = g.time_s.to_numpy(dtype=float)
        if (t[-1] - t[0]) < min_track_s:
            continue
        mid = len(g) // 2
        a, b = _half_top_speed(g.iloc[:mid]), _half_top_speed(g.iloc[mid:])
        if np.isfinite(a) and np.isfinite(b):
            first.append(a)
            second.append(b)

    if len(first) < 4:
        return dict(r=float("nan"), n=len(first))
    return dict(r=float(np.corrcoef(first, second)[0, 1]), n=len(first))


def _half_top_speed(g: pd.DataFrame) -> float:
    """The reported statistic, on one half of a track."""
    if len(g) < 3:
        return float("nan")
    g = _smooth(g)
    dx, dy, dt = g.x.diff(), g.y.diff(), g.time_s.diff()
    speed = np.sqrt(dx**2 + dy**2) / dt * 3.6
    usable = speed[speed < MAX_SPEED_KMH]
    if usable.empty:
        return float("nan")
    return float(np.percentile(usable, TOP_SPEED_PERCENTILE))


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
