"""Possible goals for the club to confirm (semi-automatic, owner 2026-11-12).

After a goal both teams walk back to their own halves for the kick-off. The
rule looks for that formation at one-second samples of the tracked players
and lists each one as a possible goal, with the team proposed as scorer. The
club confirms or dismisses each entry and can correct the team.

Measured on the 20 SoccerTrack v2 halves (research/session_scratchpad/
goals_v5, goals_v6; development data, the owner's matches are the test):
- held out, the list holds 40 of 41 restartable goals at about 11 entries
  per match (7 false);
- the proposed team is right on 39 of 41.

The rule, frozen from goals_v6 (cell T 0.75, R none, N 3; team rule
attack_end):
- **Defended ends:** per half, the team with the lower mean x over its
  first 10 s defends the x < 0 end. The half must start at its kick-off.
- **Share:** at each 1-s sample, the share of team-labelled players
  standing in their own half.
- **Formation:** >= 3 consecutive samples with a share >= 0.75. Those
  starting in the half's first 60 s are dropped, and those less than 20 s
  apart are merged.
- **Scorer:** the half that held the players' mean x 10-40 s before the
  formation is the conceding team's; the other team scored.
- The goal itself lies 10-60 s before the formation starts.

    python -m src.goal_candidates      # synthetic self-check
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import PITCH_LENGTH_M, PITCH_WIDTH_M

TEAMS = ("team_A", "team_B")
SHARE_T = 0.75          # own-half share
MIN_SAMPLES = 3         # consecutive 1-s samples
START_EXCLUSION_S = 60  # the half's own kick-off
MERGE_S = 20
ENDS_S = 10             # defended ends from the half's first 10 s
ATTACK_S = (10, 40)     # where play was before the formation
GOAL_BEFORE_S = (10, 60)


def _centred(x: np.ndarray, y: np.ndarray, origin: str):
    if origin == "corner":
        return x - PITCH_LENGTH_M / 2, y - PITCH_WIDTH_M / 2
    if origin == "centre":
        return x, y
    raise ValueError(f"origin must be 'corner' or 'centre', got {origin!r}")


def _samples(frames: np.ndarray, start: int, end: int, fps: float) -> np.ndarray:
    """For each second of [start, end), the tracked frame nearest to it
    (within half a second), else -1."""
    targets = np.round(start + np.arange(int((end - start) / fps)) * fps).astype(np.int64)
    if not len(frames) or not len(targets):
        return np.full(len(targets), -1, np.int64)
    i = np.clip(np.searchsorted(frames, targets), 1, len(frames) - 1)
    lo, hi = frames[i - 1], frames[i]
    near = np.where(np.abs(targets - lo) <= np.abs(hi - targets), lo, hi)
    near[np.abs(near - targets) > fps / 2] = -1
    if len(frames) == 1:
        near = np.where(np.abs(frames[0] - targets) <= fps / 2, frames[0], -1)
    return near


def _ends(p: pd.DataFrame, teams) -> dict | None:
    m = p.groupby("team").x.mean()
    if not all(t in m.index for t in teams):
        return None
    lo = min(teams, key=lambda t: m[t])
    return {t: (-1 if t == lo else 1) for t in teams}


def _other(team, teams):
    return teams[1] if team == teams[0] else teams[0]


def _runs(share: np.ndarray) -> list[tuple[int, int]]:
    """Formation runs over the samples (indices = seconds from the half start)."""
    ok = np.nan_to_num(share, nan=-1.0) >= SHARE_T
    raw, i = [], 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if j - i + 1 >= MIN_SAMPLES and i >= START_EXCLUSION_S:
                raw.append([i, j])
            i = j + 1
        else:
            i += 1
    merged: list[list[int]] = []
    for a, b in raw:
        if merged and a - merged[-1][1] < MERGE_S:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def find(tracks: pd.DataFrame, fps: float = 25.0, half_starts_s=(0.0,),
         origin: str = "corner", teams=TEAMS) -> list[dict]:
    """Possible goals from tracked players.

    tracks: one row per player per frame, with frame, x, y (pitch metres)
    and team; rows whose team is not in `teams` are ignored, and so are rows
    with cls other than 'player' when that column exists.
    half_starts_s: the video time of each half's kick-off.
    Returns one dict per candidate, in time order.
    """
    p = tracks
    if "cls" in p.columns:
        p = p[p.cls == "player"]
    p = p[p.team.isin(list(teams)) & p.x.notna() & p.y.notna()]
    x, y = _centred(p.x.to_numpy(float), p.y.to_numpy(float), origin)
    p = pd.DataFrame({"frame": p.frame.to_numpy(np.int64), "x": x, "y": y,
                      "team": p.team.to_numpy()})
    if p.empty:
        return []
    frames = np.unique(p.frame.to_numpy())
    last = int(frames[-1]) + 1
    starts = [int(round(s * fps)) for s in sorted(half_starts_s)]
    by_frame = {f: g for f, g in p.groupby("frame")}
    out = []
    for h, start in enumerate(starts):
        end = starts[h + 1] if h + 1 < len(starts) else last
        smp = _samples(frames, start, end, fps)
        rows = [by_frame[f].assign(k=k) for k, f in enumerate(smp) if f >= 0]
        if not rows:
            continue
        q = pd.concat(rows, ignore_index=True)
        ends = _ends(q[q.k <= ENDS_S], teams)
        ends_from = "first 10 s"
        if ends is None:
            ends, ends_from = _ends(q, teams), "whole half"
        if ends is None:
            continue
        own = q.x.to_numpy() * q.team.map(ends).to_numpy() >= 0
        share = pd.Series(own).groupby(q.k.to_numpy()).mean().reindex(range(len(smp))).to_numpy()
        for a, b in _runs(share):
            w = q[(q.k >= a - ATTACK_S[1]) & (q.k <= a - ATTACK_S[0])]
            scorer = None
            if len(w):
                mx = float(w.x.mean())
                conceded = [t for t, e in ends.items() if mx * e > 0]
                scorer = _other(conceded[0], teams) if conceded else None
            t0 = start / fps + a
            out.append({
                "half": h + 1,
                "formation_from_s": round(t0, 1),
                "formation_to_s": round(start / fps + b, 1),
                "goal_between_s": (round(max(start / fps, t0 - GOAL_BEFORE_S[1]), 1),
                                   round(t0 - GOAL_BEFORE_S[0], 1)),
                "proposed_scorer": scorer,
                "ends_from": ends_from,
            })
    return out


def mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60}:{t % 60:02d}"


# --------------------------------------------------------------------------
# Synthetic self-check: a half with one goal, its kick-off formation, and
# live play around it.

def _synthetic(seed=0, stride=2, swap=False, fps=25, minutes=10):
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * fps)
    a_def, b_def = ("team_B", "team_A") if swap else ("team_A", "team_B")  # a_def defends x < 0
    rows = []
    for f in range(0, n, stride):
        t = f / fps
        for team, sign in ((a_def, -1), (b_def, 1)):
            if t < 8 or 330 <= t < 345:        # kick-offs: everyone in their own half
                xs = sign * rng.uniform(1, 45, 11)
            elif 280 <= t < 320:              # b_def attacks into a_def's half, scores at 315 s
                xs = rng.uniform(-50, -5, 11)
            else:                              # live play: both halves
                xs = rng.uniform(-50, 50, 11)
            for i, xv in enumerate(xs):
                rows.append((f, f"{team}{i}", team, xv + PITCH_LENGTH_M / 2,
                             rng.uniform(0, PITCH_WIDTH_M), "player"))
    return pd.DataFrame(rows, columns=["frame", "track_id", "team", "x", "y", "cls"])


def _check() -> int:
    bad = 0
    for stride in (1, 2, 5):
        c = find(_synthetic(stride=stride))
        ok = (len(c) == 1 and 325 <= c[0]["formation_from_s"] <= 335
              and c[0]["proposed_scorer"] == "team_B"
              and c[0]["goal_between_s"][0] <= 315 <= c[0]["goal_between_s"][1])
        print(f"stride {stride}: {len(c)} candidate(s), {c[:1]}", "OK" if ok else "FAIL")
        bad += not ok
    c = find(_synthetic(swap=True))
    ok = len(c) == 1 and c[0]["proposed_scorer"] == "team_A"
    print("teams swapped: scorer", c[0]["proposed_scorer"] if c else None, "OK" if ok else "FAIL")
    bad += not ok
    d = _synthetic()
    d = d[~((d.frame / 25 >= 330) & (d.frame / 25 < 345))]   # no kick-off formation
    c = find(d)
    ok = len(c) == 0
    print("no formation:", len(c), "candidate(s)", "OK" if ok else "FAIL")
    bad += not ok
    d = _synthetic()
    d2 = d.copy()
    d2["frame"] = d2.frame + 10 * 60 * 25                     # a second half, ends swapped
    d2["x"] = PITCH_LENGTH_M - d2.x
    c = find(pd.concat([d, d2]), half_starts_s=(0.0, 600.0))
    ok = (len(c) == 2 and [x["half"] for x in c] == [1, 2]
          and all(x["proposed_scorer"] == "team_B" for x in c))
    print("two halves:", [(x["half"], x["formation_from_s"], x["proposed_scorer"]) for x in c],
          "OK" if ok else "FAIL")
    bad += not ok
    print("goal candidates self-check:", "FAIL" if bad else "OK")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if _check() else 0)
