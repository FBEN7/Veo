"""Choose one ball per frame from many candidates, using motion.

Detection now emits every on-pitch ball candidate rather than the most
confident one, because confidence is a poor single-frame guide: a distant
player's head outscores the real ball often enough to break the track. What
distinguishes the ball is not how it looks in one frame but how it moves
across several -- it travels fast, it travels continuously, and it does not
teleport across the pitch and back.

That is a sequence problem, so it is solved as one. A Viterbi pass over the
whole clip picks the chain of candidates with the best total score, trading
per-frame confidence against the plausibility of the movement implied between
consecutive picks. A confident candidate that would require the ball to move
at 400 km/h loses to a weaker one that sits where the ball was heading.

Frames with no candidate are not filled in. A gap is honest about what was
observed, and the transition simply spans it -- a ball may legitimately travel
further across ten missing frames than across one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A struck football reaches roughly 120-130 km/h. Above that the implied
# movement is a detection jumping between different objects, not a ball.
MAX_BALL_SPEED_KMH = 130.0

# Speed is computed between frame centres of detections whose positions carry
# their own error, so the limit is enforced with slack rather than as a wall.
SPEED_TOLERANCE = 1.5

# Weight on the movement penalty relative to detection confidence. High enough
# that confidence cannot buy an impossible jump.
MOTION_WEIGHT = 4.0


def _candidate_frames(balls: pd.DataFrame):
    """Ball candidates grouped by frame, in frame order."""
    frames = []
    for frame, group in balls.groupby("frame", sort=True):
        frames.append((
            int(frame),
            group.index.to_numpy(),
            group.px.to_numpy(dtype=float),
            group.py.to_numpy(dtype=float),
            (group.confidence.to_numpy(dtype=float)
             if "confidence" in group.columns
             else np.ones(len(group), dtype=float)),
        ))
    return frames


def select_single_ball(tracks: pd.DataFrame, px_per_m: float,
                       fps: float = 25.0, verbose: bool = False) -> pd.DataFrame:
    """Keep one ball row per frame: the chain that moves like a ball."""
    if tracks.empty or "cls" not in tracks.columns:
        return tracks.copy()

    balls = tracks[tracks.cls == "ball"]
    if balls.empty:
        return tracks.copy()
    if not np.isfinite(px_per_m) or px_per_m <= 0:
        return tracks.copy()

    frames = _candidate_frames(balls)
    if not frames:
        return tracks.copy()

    max_m_per_frame = (MAX_BALL_SPEED_KMH * SPEED_TOLERANCE) / 3.6 / max(fps, 1e-6)

    # Viterbi forward pass. Scores are additive: confidence rewards a
    # candidate, implausible movement penalises the step that reaches it.
    n0 = len(frames[0][1])
    score = frames[0][4].astype(float).copy()
    back: list[np.ndarray] = [np.full(n0, -1, dtype=int)]

    for t in range(1, len(frames)):
        f_prev, _, px_prev, py_prev, _ = frames[t - 1]
        f_cur, _, px_cur, py_cur, conf_cur = frames[t]

        gap = max(1, f_cur - f_prev)
        budget = max_m_per_frame * gap

        # Distance in metres between every previous candidate and every
        # current one.
        dx = (px_cur[None, :] - px_prev[:, None]) / px_per_m
        dy = (py_cur[None, :] - py_prev[:, None]) / px_per_m
        dist = np.hypot(dx, dy)

        # Zero while the movement is possible, growing once it is not.
        penalty = MOTION_WEIGHT * np.maximum(0.0, dist - budget) / budget

        total = score[:, None] - penalty
        best_prev = np.argmax(total, axis=0)
        score = total[best_prev, np.arange(total.shape[1])] + conf_cur
        back.append(best_prev)

    # Backward pass.
    keep = []
    j = int(np.argmax(score))
    for t in range(len(frames) - 1, -1, -1):
        keep.append(frames[t][1][j])
        j = int(back[t][j]) if t > 0 else -1
        if j < 0 and t > 0:
            break

    keep_idx = set(keep)
    drop = balls.index.difference(list(keep_idx))

    if verbose:
        print(f"  [ball] {len(balls)} candidates over {len(frames)} frames "
              f"-> {len(keep_idx)} selected")

    return tracks.drop(index=drop).copy()


def ball_track_diagnostics(tracks: pd.DataFrame, px_per_m: float,
                           fps: float = 25.0) -> dict:
    """Coverage and speed of the selected ball track."""
    out = dict(ball_frames=0, coverage_pct=0.0,
               median_speed_kmh=float("nan"), p95_speed_kmh=float("nan"))
    if tracks.empty or "cls" not in tracks.columns:
        return out

    balls = tracks[tracks.cls == "ball"].sort_values("frame")
    if balls.empty:
        return out

    total_frames = int(tracks.frame.nunique())
    out["ball_frames"] = int(balls.frame.nunique())
    out["coverage_pct"] = (100.0 * out["ball_frames"] / total_frames
                           if total_frames else 0.0)

    if len(balls) < 2 or not np.isfinite(px_per_m) or px_per_m <= 0:
        return out

    f = balls.frame.to_numpy(dtype=float)
    x = balls.px.to_numpy(dtype=float) / px_per_m
    y = balls.py.to_numpy(dtype=float) / px_per_m
    step = np.diff(f)
    step[step <= 0] = np.nan
    speed = np.hypot(np.diff(x), np.diff(y)) / (step / fps) * 3.6
    speed = speed[np.isfinite(speed)]
    if speed.size:
        out["median_speed_kmh"] = float(np.median(speed))
        out["p95_speed_kmh"] = float(np.percentile(speed, 95))
    return out
