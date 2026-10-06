"""Events built from touches: who has the ball, frame by frame, and what
happens between one possession and the next.

The first detector (`src/events.py`) followed the player nearest the ball
over time and emitted an event when that changed. Checked against 139
clicked events it credited the right player on 22 of 66 real ones, and the
person labelling them found one pass counted twice, a team-mate's
reception called a recovery, a carry in the middle of a pass, and passes
stamped 0.4 s after the touch, with the ball already travelling
(PLAYER_IDENTITY.md). All four are what that design does.

This one starts from contact:

1. **Touches.** On a frame the ball is *at* a player when it is within
   `CONTACT` of their feet, in body heights -- in the picture, where both
   are measured. Frames at the same player, allowing `HOLD_GAP` frames
   unseen, are one **possession**, from the first touch to the last.
2. **Actions from consecutive possessions**, in the SPADL format
   (`src/event_schema.py`):

   - the same player again, after the ball left them briefly: one
     possession (a dribble knocks the ball ahead);
   - a team-mate next: a **pass**, from the last touch of the first (the
     release) to the first touch of the second (the reception), with both
     players;
   - an opponent next, after the ball travelled: a failed **pass** and a
     **recovery** by the opponent at their first touch;
   - an opponent next at close quarters (within `TACKLE_GAP` frames): a
     **tackle** by the opponent;
   - a possession lasting `CARRY_MIN_S` in which the player moves at
     least `CARRY_MIN_M`: a **carry** from the first touch to the last.

Tracks known to be one person (`identities.json`) count as one player, so
two fragments of a player cannot pass to each other.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .event_schema import Action

CONTACT = 0.6            # ball within this many body heights of the feet
CONTROL = 0.9            # ... and kept by its player while within this
HOLD_GAP = 3             # frames a possession survives unseen
TACKLE_GAP = 4           # frames between two opponents' touches: a duel
MAX_PASS_S = 4.0         # longer between touches and the ball was lost
CARRY_MIN_S = 0.6
# A carry is the ball kept between a reception and the next action, as in
# StatsBomb's data, however short; 1 m only rules out standing still. Of
# 17 clicked carries most moved 0.3-2 m by the pitch positions, so the
# first threshold, 3 m, found one.
CARRY_MIN_M = 1.0
# A possession shorter than this is the ball passing a player, not a touch:
# 36 passes and 16 tackles a minute came from the ball flickering between
# two players side by side.
MIN_TOUCH_FRAMES = 2
# The ball travels at least this far, in body heights, from release to
# reception for a pass -- else it is two players at the same ball.
MIN_PASS_TRAVEL = 1.0
# The winner of a duel keeps the ball this many frames.
# Tuned on the first round of clicks: at 4 frames, 5.7 tackles a minute
# came out, most of them duels; at 8, 2.9, with no real recovery lost.
TACKLE_HOLD = 8
# A ball won after it travelled counts when the winner keeps it this many
# frames: an opponent the ball merely brushes has not won it.
RECOVERY_HOLD = 4
# A carry moves the player at least this many body heights in the
# camera-compensated picture -- the pitch positions understate movement
# (most clicked carries moved 0.3-2 m by them), the picture does not.
# One body height: on the first round of clicks, false carries 5 -> 3 of
# 29 rejected moments, 9 -> 7 of 17 real ones found, 6.9 -> 4.6 a minute.
CARRY_MIN_BODY = 1.0
UNSURE = {"other", "unknown", None}


def holders(ball: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Per ball frame: the player at the ball, or none. `ball` has frame,
    px, py (picture); `players` frame, track_id, team, px, py (feet),
    crop_h."""
    by = {f: g for f, g in players.groupby("frame")}
    rows, current = [], None
    for r in ball.sort_values("frame").itertuples():
        g = by.get(int(r.frame))
        if g is None or g.empty:
            continue
        # From the ball to the lower body: the ball is played at the feet.
        d = np.hypot(g.px.to_numpy() - r.px,
                     (g.py.to_numpy() - 0.1 * g.crop_h.to_numpy()) - r.py)
        d = d / g.crop_h.to_numpy()
        k = int(np.argmin(d))
        mine = (np.flatnonzero(g.track_id.to_numpy() == current)
                if current is not None else [])
        if d[k] <= CONTACT:
            current = int(g.track_id.iloc[k])
            kk = k
        elif len(mine) and d[mine[0]] <= CONTROL:
            # Hysteresis: a dribbler pushes the ball a metre or two ahead,
            # beyond touching distance, and still has it until someone
            # else touches it.
            kk = int(mine[0])
        else:
            current = None
            continue
        rows.append({"frame": int(r.frame), "track_id": int(g.track_id.iloc[kk]),
                     "team": g.team.iloc[kk], "dist": float(d[kk])})
    return pd.DataFrame(rows)


def possessions(held: pd.DataFrame, person) -> list[dict]:
    """Runs of frames at one player (by `person(track_id)`)."""
    out = []
    for r in held.sort_values("frame").itertuples():
        who = person(r.track_id)
        if (out and out[-1]["who"] == who
                and r.frame - out[-1]["last"] <= HOLD_GAP + 1):
            out[-1]["last"] = r.frame
            out[-1]["frames"] += 1
            continue
        out.append({"who": who, "track_id": r.track_id, "team": r.team,
                    "first": r.frame, "last": r.frame, "frames": 1})
    return out


def _merge_returns(poss: list[dict], fps: float) -> list[dict]:
    """The same player again after the ball left them: one possession."""
    out = []
    for p in poss:
        if (out and out[-1]["who"] == p["who"]
                and (p["first"] - out[-1]["last"]) / fps <= 1.0):
            out[-1]["last"] = p["last"]
            out[-1]["frames"] += p["frames"]
            continue
        out.append(dict(p))
    return out


def actions(poss: list[dict], fps: float, where, name,
            ball_at=None, body_at=None) -> list[Action]:
    """SPADL actions from consecutive possessions. `where(track, frame)`
    gives pitch metres or None; `name(track)` the player's label;
    `ball_at(frame)` the ball in the picture with the player's height, for
    how far it travelled."""
    out = []
    poss = [p for p in poss if p["frames"] >= MIN_TOUCH_FRAMES]
    merged = []
    for p in poss:
        if (merged and merged[-1]["who"] == p["who"]
                and (p["first"] - merged[-1]["last"]) / fps <= 1.0):
            merged[-1]["last"] = p["last"]
            merged[-1]["frames"] += p["frames"]
        else:
            merged.append(dict(p))
    poss = merged
    for i, p in enumerate(poss):
        span = (p["last"] - p["first"]) / fps
        a, b = where(p["track_id"], p["first"]), where(p["track_id"], p["last"])
        moved = (np.hypot(b[0] - a[0], b[1] - a[1])
                 if a is not None and b is not None else 0.0)
        moved_body = None
        if body_at is not None:
            u, v = body_at(p["track_id"], p["first"]), body_at(p["track_id"],
                                                               p["last"])
            if u is not None and v is not None:
                moved_body = (np.hypot(v[0] - u[0], v[1] - u[1])
                              / max((u[2] + v[2]) / 2, 1.0))
        if (span >= CARRY_MIN_S and moved >= CARRY_MIN_M
                and (moved_body is None or moved_body >= CARRY_MIN_BODY)):
            out.append(Action("carry", p["first"], p["last"], p["team"],
                              name(p["track_id"]), None,
                              *(a or (None, None)), *(b or (None, None)),
                              "success", fps))
        if i + 1 == len(poss):
            break
        q = poss[i + 1]
        gap = q["first"] - p["last"]
        if gap / fps > MAX_PASS_S:
            continue
        start, end = where(p["track_id"], p["last"]), where(q["track_id"],
                                                             q["first"])
        same_team = (p["team"] == q["team"] or p["team"] in UNSURE
                     or q["team"] in UNSURE)
        travel = None
        if ball_at is not None:
            a0, a1 = ball_at(p["last"]), ball_at(q["first"])
            if a0 is not None and a1 is not None:
                travel = (np.hypot(a1[0] - a0[0], a1[1] - a0[1])
                          / max((a0[2] + a1[2]) / 2, 1.0))
        if same_team and travel is not None and travel < MIN_PASS_TRAVEL:
            continue                       # two team-mates at one ball
        if same_team:
            out.append(Action("pass", p["last"], q["first"], p["team"],
                              name(p["track_id"]), name(q["track_id"]),
                              *(start or (None, None)), *(end or (None, None)),
                              "success", fps))
        elif gap <= TACKLE_GAP:
            if q["frames"] < TACKLE_HOLD:
                continue                   # a duel nobody won yet
            out.append(Action("tackle", q["first"], q["first"], q["team"],
                              name(q["track_id"]), None,
                              *(end or (None, None)), *(end or (None, None)),
                              "success", fps))
        else:
            out.append(Action("pass", p["last"], q["first"], p["team"],
                              name(p["track_id"]), None,
                              *(start or (None, None)), *(end or (None, None)),
                              "fail", fps))
            if q["frames"] < RECOVERY_HOLD:
                continue                   # brushed, not won
            out.append(Action("recovery", q["first"], q["first"], q["team"],
                              name(q["track_id"]), None,
                              *(end or (None, None)), *(end or (None, None)),
                              "success", fps))
    return out


def for_clip(out_dir: Path) -> list[Action]:
    """Touch-based actions for a clip the full pipeline has run on."""
    import detect_shots

    out_dir = Path(out_dir)
    info = json.loads((out_dir / "clip.json").read_text())
    fps = float(info["fps"])
    merged = pd.read_parquet(out_dir / "tracks_merged.parquet")
    players = merged[merged.cls == "player"].copy()
    players["px"] = players.get("px_raw", players.px)
    players["py"] = players.get("py_raw", players.py)
    ball = detect_shots.ball_track(out_dir, ball_detector="fill")
    held = holders(ball[["frame", "px", "py"]], players)

    ident = {}
    blob = out_dir / "identities.json"
    if blob.exists():
        ident = {int(k): v["identity"]
                 for k, v in json.loads(blob.read_text())["tracks"].items()}
    person = lambda tid: ident.get(int(tid), f"t{tid}")
    name = lambda tid: (f"player {ident[int(tid)]}" if int(tid) in ident
                        else f"track {tid}")

    metric_path = out_dir / "tracks_metric.parquet"
    pos = {}
    if metric_path.exists():
        m = pd.read_parquet(metric_path)
        m = m[m.cls == "player"]
        pos = {(int(r.track_id), int(r.frame)): (float(r.x), float(r.y))
               for r in m.itertuples()}
    where = lambda tid, f: pos.get((int(tid), int(f)))

    size = players.groupby("frame").crop_h.median()
    bxy = {int(r.frame): (float(r.px), float(r.py)) for r in ball.itertuples()}
    ball_at = lambda f: ((*bxy[f], float(size.get(f, 80.0)))
                         if f in bxy else None)
    body = {(int(r.track_id), int(r.frame)): (float(r.px_comp), float(r.py_comp),
                                               float(r.crop_h))
            for r in players.assign(px_comp=merged.loc[players.index, "px"],
                                    py_comp=merged.loc[players.index, "py"])
            .itertuples()}
    body_at = lambda tid, f: body.get((int(tid), int(f)))
    poss = _merge_returns(possessions(held, person), fps)
    return actions(poss, fps, where, name, ball_at, body_at)
