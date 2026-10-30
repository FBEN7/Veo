"""Who made an action: the possession rule chosen in
research/session_scratchpad/passer/possession.json (rule C, -0.4 / +0.2 s).

Over the frames from 0.4 s before an event to 0.2 s after it, at every
players' frame with a ball row within 2 frames, the player detection
nearest the ball within 3 m (pitch metres) gets a vote. The track with
most votes is the actor; ties go to the track whose vote is nearest the
event. On SoccerTrack v2 (matches the ball detector was trained on, inside
the ball's range) it names the labelled player on 0.568 of actions
(passes 0.586), precision 0.68: an in-sample choice, no held-out match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BEFORE_S, AFTER_S = 0.4, 0.2
RADIUS_M = 3.0
BALL_TOL = 2                  # frames between a players' frame and its ball row


def actor(tracks: pd.DataFrame, frame: int, fps: float = 25.0) -> int | None:
    """The actor's track_id at `frame`, or None. `tracks`: pitch-metre rows
    (frame, track_id, cls in {"player", "ball"}, x, y), as
    pixel_scale.to_pitch_metres gives."""
    lo, hi = frame - round(BEFORE_S * fps), frame + round(AFTER_S * fps)
    t = tracks[(tracks.frame >= lo - BALL_TOL) & (tracks.frame <= hi + BALL_TOL)]
    ball = t[t.cls == "ball"].drop_duplicates("frame").set_index("frame")
    players = t[(t.cls == "player") & (t.frame >= lo) & (t.frame <= hi)]
    if ball.empty or players.empty:
        return None
    bf = ball.index.to_numpy()
    votes = []
    for f, q in players.groupby("frame"):
        j = np.argmin(abs(bf - f))
        if abs(bf[j] - f) > BALL_TOL:
            continue
        b = ball.iloc[j]
        d = np.hypot(q.x - b.x, q.y - b.y).to_numpy()
        k = int(np.nanargmin(d)) if np.isfinite(d).any() else None
        if k is not None and d[k] <= RADIUS_M:
            votes.append((int(q.track_id.iloc[k]), abs(int(f) - frame)))
    if not votes:
        return None
    v = pd.DataFrame(votes, columns=["track", "lag"]).groupby("track").agg(n=("lag", "size"), lag=("lag", "min"))
    return int(v.sort_values(["n", "lag"], ascending=[False, True]).index[0])


def attach(events: list[dict], tracks: pd.DataFrame, fps: float = 25.0) -> list[dict]:
    """Events (dicts with timestamp_s) with an `actor_track` key added."""
    return [{**e, "actor_track": actor(tracks, round(e["timestamp_s"] * fps), fps)} for e in events]


def _check():
    """Player 7 holds the ball for 0.3 s before the event, player 9 runs in
    and is nearer only at the event frame: the rule names 7."""
    rows = []
    for f in range(90, 106, 2):
        rows.append((f, -1, "ball", 10.0, 0.0))
        rows.append((f, 7, "player", 10.5, 0.0))
        rows.append((f, 9, "player", 10.2 if f >= 100 else 14.0, 0.0))
    t = pd.DataFrame(rows, columns=["frame", "track_id", "cls", "x", "y"])
    assert actor(t, 100) == 7, actor(t, 100)
    assert actor(t, 300) is None
    print("passer: ok")


if __name__ == "__main__":
    _check()
