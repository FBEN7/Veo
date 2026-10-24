"""The kick-off formation rule of the goals protocol (protocol.json, GOALS), as
importable functions shared by the ground-truth selection (formation_gt.py)
and the detection runs.

After a goal both teams walk back into their own halves for the kick-off, so
on 1-s samples the share of team-labelled people (players and goalkeepers,
referees excluded) standing in their own half stays high for a while. A
formation run is >= N consecutive samples with that share >= T; runs that
start in the first 60 s of a half (the half's own kick-off) are ignored and
runs less than 20 s apart are merged. A run matches a BAS GOAL of the same
half if it starts 15-60 s after it (one to one). Two rules name the scoring
team: `kick_spot` (the team of the person nearest the centre spot at the
run's last sample kicks off, so the other team scored) and `attack_end` (the
team whose half holds the people's mean x over the 10-40 s before the run
conceded).

Inputs are tables of 1-s samples, one row per person: `frame` (0-indexed
half-video frame at 25 fps, a multiple of 25), `team` (any two labels: GSR
left/right, or team_A/team_B), `x`, `y` (pitch metres, centre origin).

Readings where the protocol is silent (the simplest one each time):
- The end a team defends: "its mean x over the half" is read between the two
  teams -- the team with the lower mean x (over all its 1-s person samples)
  defends the x < 0 end, the other the x > 0 end. (On 4 of the 14 dev GT
  halves both teams' means have the same sign, e.g. M4 1st -24.1 and -11.8,
  so the sign of each mean alone would give both teams the same end.)
- Own half is closed: x * end >= 0, so the halfway line counts for both
  teams (the kicker at the centre spot is in his own half, as in Law 8).
- A sample with no team-labelled person has no share and breaks a run.
  Samples are consecutive when their frames are 25 apart.
- Order: runs of >= N samples are found first, then those starting before
  60 s (frame < 1500) are dropped, then the rest are merged when the next
  run's first sample is less than 20 s after the previous run's last sample;
  a merged run starts at its first run's start and ends at its last run's
  end.
- Matching window is inclusive: 15 s <= run start - goal <= 60 s. All BAS
  GOALs take part in the matching ("a run matches a BAS GOAL"); hits are
  counted on restartable goals (a later action of the half >= 15 s after
  it); a run matched to a non-restartable goal is neither a hit nor a false
  run and is reported apart. One-to-one matching is greedy: goals in time
  order each take the earliest free run in their window, which is a maximum
  matching because every goal's window has the same length.
- kick_spot: nearest to (0, 0) in metres among the team-labelled people at
  the run's last sample.
- attack_end: the pooled mean x of all team-labelled person samples (players
  and goalkeepers, both teams) with run start - 40 s <= frame <= run start -
  10 s; the team whose own half holds it strictly (x * end > 0) conceded.
  No sample, or a mean of exactly 0, names no team (counted wrong).
- Random-time baseline: per half the same number of run starts drawn
  uniformly on [60 s, the half's last sample], matched the same way; hits
  summed over halves per draw; 1000 draws, numpy default_rng(seed 0).

    from formation import half_runs, score_half      # see formation_gt.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FPS = 25
SAMPLE = 25                 # frames between 1-s samples
START_EXCLUSION_S = 60
MERGE_S = 20
MATCH_S = (15, 60)
RESTART_S = 15
ATTACK_S = (10, 40)
TEAM_RULES = ("kick_spot", "attack_end")


def defended_ends(people: pd.DataFrame) -> dict:
    """team -> the end it defends (-1: x < 0, +1: x > 0); the team with the
    lower mean x over the half defends the x < 0 end."""
    m = people.groupby("team").x.mean().sort_values()
    if len(m) != 2:
        raise ValueError(f"need two teams, got {list(m.index)}")
    return {m.index[0]: -1, m.index[1]: +1}


def own_half_share(people: pd.DataFrame, ends: dict,
                   frames=None) -> pd.Series:
    """Per 1-s sample (frame) the share of team-labelled people standing in
    their own half (x * end >= 0); reindexed on `frames` when given (NaN
    where a sample has nobody)."""
    p = people[people.team.isin(list(ends)) & people.x.notna()]
    own = p.x.to_numpy() * p.team.map(ends).to_numpy() >= 0
    share = pd.Series(own, index=p.frame.to_numpy()).groupby(level=0).mean()
    if frames is not None:
        share = share.reindex(frames)
    return share.sort_index()


def raw_runs(share: pd.Series, T: float, N: int) -> list[tuple[int, int]]:
    """Maximal stretches of consecutive samples (frames SAMPLE apart) with
    share >= T, of at least N samples: (first frame, last frame)."""
    ok = share[share >= T]
    fr = ok.index.to_numpy(dtype=np.int64)
    if not len(fr):
        return []
    breaks = np.flatnonzero(np.diff(fr) != SAMPLE) + 1
    out = []
    for seg in np.split(fr, breaks):
        if len(seg) >= N:
            out.append((int(seg[0]), int(seg[-1])))
    return out


def runs(share: pd.Series, T: float, N: int) -> list[tuple[int, int]]:
    """Formation runs of a half: >= N consecutive samples with share >= T;
    those starting in the first 60 s dropped, then those less than 20 s
    apart merged. (start frame, end frame), half-video frames."""
    rs = [r for r in raw_runs(share, T, N) if r[0] >= START_EXCLUSION_S * FPS]
    merged: list[list[int]] = []
    for a, b in rs:
        if merged and a - merged[-1][1] < MERGE_S * FPS:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def restartable(goal_frames, action_frames) -> list[bool]:
    """Per goal: whether a later action of the half is >= 15 s after it."""
    acts = np.asarray(action_frames, dtype=float)
    return [bool((acts >= g + RESTART_S * FPS).any()) for g in goal_frames]


def match_runs(run_starts, goal_frames) -> dict:
    """One-to-one matching of runs to goals: a run matches a goal if it
    starts 15-60 s after it (inclusive). goal index -> run index."""
    lo, hi = MATCH_S[0] * FPS, MATCH_S[1] * FPS
    starts = np.asarray(run_starts, dtype=float)
    used = np.zeros(len(starts), bool)
    out = {}
    for gi in np.argsort(np.asarray(goal_frames, dtype=float), kind="stable"):
        d = starts - goal_frames[gi]
        cand = np.flatnonzero(~used & (d >= lo) & (d <= hi))
        if len(cand):
            ri = cand[np.argmin(starts[cand])]
            used[ri] = True
            out[int(gi)] = int(ri)
    return out


def _other(team, ends: dict):
    return next(t for t in ends if t != team)


def scorer_kick_spot(people: pd.DataFrame, ends: dict, run) -> str | None:
    """The team of the person nearest the centre spot at the run's last
    sample kicks off; the other team scored."""
    p = people[(people.frame == run[1]) & people.team.isin(list(ends))
               & people.x.notna() & people.y.notna()]
    if not len(p):
        return None
    k = np.hypot(p.x.to_numpy(), p.y.to_numpy()).argmin()
    return _other(p.team.iloc[k], ends)


def scorer_attack_end(people: pd.DataFrame, ends: dict, run) -> str | None:
    """The team whose half holds the people's mean x over the 10-40 s before
    the run conceded; the other team scored."""
    a, b = run[0] - ATTACK_S[1] * FPS, run[0] - ATTACK_S[0] * FPS
    p = people[(people.frame >= a) & (people.frame <= b)
               & people.team.isin(list(ends)) & people.x.notna()]
    if not len(p):
        return None
    mx = float(p.x.mean())
    conceded = [t for t, e in ends.items() if mx * e > 0]
    return _other(conceded[0], ends) if conceded else None


SCORERS = {"kick_spot": scorer_kick_spot, "attack_end": scorer_attack_end}


def score_half(people: pd.DataFrame, ends: dict, share: pd.Series,
               goals: pd.DataFrame, action_frames, T: float, N: int,
               team_of_side=None) -> dict:
    """Runs of one half under (T, N), matched to its BAS GOALs (`goals`:
    frame, side). Hits are matched restartable goals; false runs are
    unmatched runs. Per team rule, the hits whose scoring team is right:
    the rule's team mapped by `team_of_side` (BAS side -> team label;
    identity by default) equals the goal's."""
    rs = runs(share, T, N)
    gf = goals.frame.to_numpy()
    rest = restartable(gf, action_frames)
    m = match_runs([r[0] for r in rs], gf)
    matched_runs = set(m.values())
    rows = []
    for gi, ri in sorted(m.items(), key=lambda kv: gf[kv[0]]):
        side = goals.side.iloc[gi]
        want = team_of_side(side) if team_of_side else side
        row = {"goal_frame": int(gf[gi]), "goal_side": side,
               "restartable": rest[gi], "run": list(rs[ri]),
               "delay_s": (rs[ri][0] - gf[gi]) / FPS}
        for name, f in SCORERS.items():
            row[name] = f(people, ends, rs[ri])
            row[name + "_right"] = row[name] == want
        rows.append(row)
    hits = [r for r in rows if r["restartable"]]
    return {
        "runs": [list(r) for r in rs],
        "restartable_goals": [int(g) for g, ok in zip(gf, rest) if ok],
        "missed_by_construction": [int(g) for g, ok in zip(gf, rest) if not ok],
        "matches": rows,
        "hits": len(hits),
        "false_runs": len(rs) - len(matched_runs),
        "matched_non_restartable": len(rows) - len(hits),
        "team_right": {name: int(sum(r[name + "_right"] for r in hits))
                       for name in SCORERS},
        "false_run_list": [list(rs[i]) for i in range(len(rs))
                           if i not in matched_runs],
    }


def random_baseline(halves: list[dict], n_draws: int = 1000,
                    seed: int = 0) -> np.ndarray:
    """Hits of random run times: per half (`n_runs`, `goals` frames,
    `restartable` flags, `last` sample frame) the same number of starts
    drawn uniformly on [60 s, last], matched the same way; total hits per
    draw."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n_draws, int)
    for d in range(n_draws):
        for h in halves:
            if not h["n_runs"]:
                continue
            starts = rng.uniform(START_EXCLUSION_S * FPS, h["last"], h["n_runs"])
            m = match_runs(starts, np.asarray(h["goals"], dtype=float))
            out[d] += sum(h["restartable"][g] for g in m)
    return out


def _check():
    """Synthetic self-check of the run, merge and matching rules."""
    fr = np.arange(0, 400 * SAMPLE, SAMPLE)
    share = pd.Series(0.5, index=fr)
    share.loc[[25 * k for k in range(10, 14)]] = 0.95      # 10-13 s: excluded
    share.loc[[25 * k for k in range(100, 104)]] = 0.95    # run A 100-103 s
    share.loc[[25 * k for k in range(115, 118)]] = 0.95    # 115 s: merged
    share.loc[[25 * k for k in range(200, 202)]] = 0.95    # only 2 samples
    share.loc[[25 * k for k in range(300, 306)]] = 0.95
    share.loc[25 * 303] = np.nan                            # broken by NaN
    assert runs(share, 0.9, 3) == [(2500, 2925), (7500, 7550)]
    assert runs(share, 0.9, 2) == [(2500, 2925), (5000, 5025), (7500, 7625)]
    m = match_runs([2500, 5000], np.array([2500 - 375, 2500 - 400, 5000 - 1501]))
    assert m == {1: 0}
    assert match_runs([2500, 2600], np.array([2000, 2050])) == {0: 0, 1: 1}
    people = pd.DataFrame({"frame": [0, 0, 0, 0], "team": ["a", "a", "b", "b"],
                           "x": [-10.0, 0.0, 5.0, -1.0], "y": [0.0] * 4})
    ends = defended_ends(people)
    assert ends == {"a": -1, "b": +1}
    assert own_half_share(people, ends).iloc[0] == 0.75
    assert scorer_kick_spot(people, ends, (0, 0)) == "b"
    print("formation: self-check passed")


if __name__ == "__main__":
    _check()
