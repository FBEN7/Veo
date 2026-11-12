"""The kick-off of each half in a whole recording (goals_v7/plan.json).

src/goal_candidates.py needs the video time of each half's kick-off. A club
recording may start before the match (warm-up, line-up) and hold both halves
and the half-time. This finds the kick-offs from the tracked players alone,
at 1-s samples. Each sample has:
- n: the team-labelled players present;
- s: the share of them in their own half if team 1 defends x < 0;
- sep: max(s, 1 - s), how far apart the teams are.

The steps:
- **Periods of play:** stretches where the centred 61-s rolling median of n
  is >= 6. Stretches less than 180 s apart are merged. A stretch is kept if
  it lasts >= 15 min and its median sep is < 0.8: a warm-up keeps the teams
  apart, while live play mixes them (median 0.58 on the SoccerTrack
  halves).
- **One long period:** a period over 60 min (half-time spent on the pitch)
  is split where the orientation of strongly separated samples flips (the
  teams change ends). Both parts must last >= 20 min, and >= 0.8 of the
  strong samples on each side must agree.
- **Kick-off:** in each period, the end of the first formation (>= 3 samples
  with sep >= 0.75 and n >= 6) starting from 120 s before the period start
  to 180 s after it; else the period start. After a split (addendum 1), the
  end of the first formation in the new ends after the split time.

Checked only on simulated recordings (SoccerTrack halves with synthetic
warm-ups and half-times): the owner's recordings are the test.

    python -m src.match_periods      # synthetic self-check
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.goal_candidates import TEAMS, _centred, _samples

MIN_PRESENT = 6
ROLL_S = 61
MERGE_GAP_S = 180
MIN_PERIOD_S = 15 * 60
WARMUP_SEP = 0.8
LONG_PERIOD_S = 60 * 60
MIN_PART_S = 20 * 60
STRONG = 0.85
AGREE = 0.8
FORMATION_SEP = 0.75
FORMATION_N = 3
SEARCH_S = (-120, 180)


def _per_sample(tracks: pd.DataFrame, fps: float, origin: str, teams):
    p = tracks
    if "cls" in p.columns:
        p = p[p.cls == "player"]
    p = p[p.team.isin(list(teams)) & p.x.notna() & p.y.notna()]
    if p.empty:
        return np.zeros(0), np.zeros(0)
    x, _ = _centred(p.x.to_numpy(float), p.y.to_numpy(float), origin)
    fr = p.frame.to_numpy(np.int64)
    frames = np.unique(fr)
    smp = _samples(frames, 0, int(frames[-1]) + 1, fps)
    own = np.where(p.team.to_numpy() == teams[0], x <= 0, x >= 0)
    g = pd.DataFrame({"frame": fr, "own": own}).groupby("frame").own.agg(["size", "mean"])
    n = np.zeros(len(smp))
    s = np.full(len(smp), np.nan)
    ok = smp >= 0
    n[ok] = g["size"].reindex(smp[ok]).fillna(0).to_numpy()
    s[ok] = g["mean"].reindex(smp[ok]).to_numpy()
    return n, s


def _stretches(mask: np.ndarray) -> list[list[int]]:
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append([i, j + 1])          # [start, end)
            i = j + 1
        else:
            i += 1
    return out


def _split(s, n, a, b):
    """The flip of ends inside [a, b): the split time, or None."""
    o = np.zeros(b - a)
    ss, nn = s[a:b], n[a:b]
    good = nn >= MIN_PRESENT
    o[good & (ss >= STRONG)] = 1
    o[good & (ss <= 1 - STRONG)] = -1
    c = np.r_[0, np.cumsum(o)]
    total = c[-1]
    ts = np.arange(MIN_PART_S, b - a - MIN_PART_S + 1)
    if not len(ts):
        return None
    score = c[ts] - (total - c[ts])          # left sum minus right sum
    k = np.abs(score)
    best = k.max()
    t = int(ts[np.flatnonzero(k == best)[-1]])  # the latest maximiser
    sgn = 1 if score[ts.tolist().index(t)] > 0 else -1
    left, right = o[:t], o[t:]
    la = (left == sgn).sum() / max((left != 0).sum(), 1)
    ra = (right == -sgn).sum() / max((right != 0).sum(), 1)
    if best == 0 or la < AGREE or ra < AGREE:
        return None
    return a + t, sgn


def _kickoff_after_split(s, n, t, end, sgn):
    """Addendum 1: the end of the first formation in the new ends (own-half
    share >= 0.75 under them) starting at or after the split time."""
    ss = np.nan_to_num(s, nan=0.5)
    form = ((ss <= 1 - FORMATION_SEP) if sgn > 0 else (ss >= FORMATION_SEP)) & (n >= MIN_PRESENT)
    i = t
    while i < end:
        if form[i]:
            j = i
            while j + 1 < end and form[j + 1]:
                j += 1
            if j - i + 1 >= FORMATION_N:
                return j, "formation in the new ends"
            i = j + 1
        else:
            i += 1
    return t, "split time"


def _kickoff(sep, n, start, end):
    lo, hi = max(0, start + SEARCH_S[0]), min(len(sep), start + SEARCH_S[1] + 1)
    form = (np.nan_to_num(sep, nan=0) >= FORMATION_SEP) & (n >= MIN_PRESENT)
    i = lo
    while i < len(form):
        if form[i]:
            j = i
            while j + 1 < len(form) and form[j + 1]:
                j += 1
            if j - i + 1 >= FORMATION_N and i < hi:
                return j, "formation"
            i = j + 1
            if i >= hi:
                break
        else:
            i += 1
            if i >= hi:
                break
    return start, "period start"


def find(tracks: pd.DataFrame, fps: float = 25.0, origin: str = "corner",
         teams=TEAMS) -> dict:
    """{'kickoffs_s': [...], 'periods': [{from_s, to_s, kickoff_s, how}],
    'dropped': [...]} from tracked players (as goal_candidates.find)."""
    n, s = _per_sample(tracks, fps, origin, teams)
    if not len(n):
        return {"kickoffs_s": [], "periods": [], "dropped": []}
    sep = np.maximum(s, 1 - s)
    roll = pd.Series(n).rolling(ROLL_S, center=True, min_periods=1).median().to_numpy()
    st = _stretches(roll >= MIN_PRESENT)
    merged: list[list[int]] = []
    for a, b in st:
        if merged and a - merged[-1][1] < MERGE_GAP_S:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    periods, dropped = [], []
    for a, b in merged:
        med = float(np.nanmedian(np.where(n[a:b] >= MIN_PRESENT, sep[a:b], np.nan))) \
            if (n[a:b] >= MIN_PRESENT).any() else 1.0
        if b - a < MIN_PERIOD_S:
            dropped.append({"from_s": a, "to_s": b, "why": "shorter than 15 min"})
            continue
        if med >= WARMUP_SEP:
            dropped.append({"from_s": a, "to_s": b, "why": f"teams kept apart (median sep {med:.2f}): warm-up"})
            continue
        if b - a > LONG_PERIOD_S:
            r = _split(s, n, a, b)
            if r is not None:
                t, sgn = r
                periods += [[a, t, "split", None], [t, b, "split", sgn]]
                continue
        periods.append([a, b, "presence", None])
    out = []
    for a, b, src, sgn in periods:
        if sgn is None:
            k, how = _kickoff(sep, n, a, b)
        else:
            k, how = _kickoff_after_split(s, n, a, b, sgn)
        out.append({"from_s": a, "to_s": b, "kickoff_s": float(k), "how": f"{src}, {how}"})
    return {"kickoffs_s": [p["kickoff_s"] for p in out], "periods": out, "dropped": dropped}


# --------------------------------------------------------------------------
# Synthetic self-check: warm-up, two halves with a goal each, half-time.

def _synthetic(halftime_on_pitch=False, warmup_s=600, seed=0, stride=5, fps=25):
    from src.constants import PITCH_LENGTH_M, PITCH_WIDTH_M
    rng = np.random.default_rng(seed)
    rows = []

    def emit(t0, t1, gen):
        for f in range(int(t0 * fps), int(t1 * fps), stride):
            for team, xs in gen(f / fps):
                for i, xv in enumerate(xs):
                    rows.append((f, team, xv + PITCH_LENGTH_M / 2, rng.uniform(0, PITCH_WIDTH_M), "player"))

    apart = lambda sign_a: (lambda t: [("team_A", sign_a * rng.uniform(1, 45, 11)),
                                       ("team_B", -sign_a * rng.uniform(1, 45, 11))])
    mixed = lambda t: [("team_A", rng.uniform(-50, 50, 11)), ("team_B", rng.uniform(-50, 50, 11))]
    mixed7 = lambda t: [("team_A", rng.uniform(-50, 50, 7)), ("team_B", rng.uniform(-50, 50, 7))]
    emit(0, warmup_s, apart(1))                   # warm-up, in the other halves
    k1 = warmup_s + 360 + 45                      # 6 min empty, then the formation
    emit(k1 - 45, k1, apart(-1))
    emit(k1, k1 + 45 * 60, mixed)                 # half 1
    ht0, ht1 = k1 + 45 * 60, k1 + 45 * 60 + 15 * 60
    if halftime_on_pitch:
        emit(ht0, ht1, mixed7)                    # 14 people milling about
    emit(ht1, ht1 + 45, apart(1))                 # ends swapped
    emit(ht1 + 45, ht1 + 45 + 45 * 60, mixed)
    return pd.DataFrame(rows, columns=["frame", "team", "x", "y", "cls"]), [k1, ht1 + 45]


def _check() -> int:
    bad = 0
    for on_pitch, warm in ((False, 600), (True, 600), (False, 1200), (True, 1200)):
        d, truth = _synthetic(halftime_on_pitch=on_pitch, warmup_s=warm)
        r = find(d)
        k = r["kickoffs_s"]
        ok = len(k) == 2 and all(abs(a - b) <= 10 for a, b in zip(k, truth))
        print(f"warm-up {warm // 60} min, half-time {'on' if on_pitch else 'off'} the pitch: "
              f"kick-offs {k}, truth {truth},", [p["how"] for p in r["periods"]],
              "dropped", [x["why"] for x in r["dropped"]], "OK" if ok else "FAIL")
        bad += not ok
    print("match periods self-check:", "FAIL" if bad else "OK")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if _check() else 0)
