"""Possession share, against the team on every labelled ball action.

Possession is the most prominent number in the HTML report -- "Team A 54%"
-- and until now the least evidenced thing in this repository. It has never
been scored against anything.

It turns out it can be, with data already on disk. Every one of the 1844
ball actions in a labelled match carries a `team` field, left or right. The
team in control between one action and the next is the team of the earlier
action, so the labels are not a set of events but a **possession timeline**,
sampled about once a second, for the whole match.

## Two questions, and the second is the one that can flatter

**Share** is what the report prints: of the time the ball was in play, what
fraction belonged to each team. It is what a reader quotes.

**Per-frame agreement** is whether the right team is named moment to moment.

A share can be right while every frame is wrong -- two errors in opposite
directions cancel exactly in a ratio -- so reporting the share alone would
let a coin flip look competent. Both are reported, and the majority-class
baseline is reported beside them, because on football where one team has 60%
of the ball a detector that always says that team scores 60%.

## Two predictions, since the repository contains two

`stats.possession_proxy` is what the report calls: nearest player within
3 m, counted per frame. `events._possession_per_frame` is the careful one
that drives passes and carries -- hysteresis so two players running together
do not trade the ball, a minimum hold so flicker is not control. Scoring
both says whether the report is using the weaker of the two.

## What is fitted, and it is one bit

Our teams are cluster labels, `team_A` and `team_B`; the truth is `left` and
`right`. The mapping between them is chosen to maximise agreement, which
fits exactly one bit of information and is the convention used elsewhere in
this project. On a timeline of thousands of frames that is negligible, but
it is fitted, so it is said.

## Dead ball

From an OUT or a GOAL until the next action, nobody is in possession. Those
intervals are excluded from both timelines rather than credited to whoever
touched the ball last, which is how possession is normally counted and also
the only honest option -- the alternative gives a team credit for the ball
being in the crowd.

    python check_possession.py [--check]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from detect_ball_events import CLIP_SOURCES
from src import events as ev_module
from src import stats

# Actions after which the ball is dead until the next action.
STOPPERS = ("OUT", "GOAL")

# The report's rule: a player within this many metres of the ball holds it.
PROXY_RADIUS_M = 3.0

# How long a gap between firings may be bridged. 0 keeps the sampled
# behaviour the report ships, which is the baseline this is against.
GAPS_S = (0.0, 1.0, 2.0, 5.0)

LABEL_DIR = Path("/root/.claude/uploads/cd4d7e67-1dd4-5fa1-975c-2f5b3217663b")


def truth_timeline(source: str, offset_s: float, duration_s: float,
                   times: np.ndarray):
    """Which team held the ball at each of `times`, or None when dead.

    The action list is read from before the window as well as inside it, so
    the team in control at t=0 is known rather than guessed.
    """
    path = LABEL_DIR / source
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    actions = sorted(
        ((int(row["position"]) / 1000.0 - offset_s, row["label"],
          row.get("team"))
         for row in blob["annotations"]),
        key=lambda row: row[0])

    held = np.full(times.shape, None, dtype=object)
    for k, (when, label, team) in enumerate(actions):
        nxt = actions[k + 1][0] if k + 1 < len(actions) else duration_s
        if label in STOPPERS or team not in ("left", "right"):
            continue                       # dead, or nobody named
        inside = (times >= when) & (times < nxt)
        held[inside] = team
    return held


def proxy_timeline(metric: pd.DataFrame, frames: np.ndarray,
                   radius_m: float = PROXY_RADIUS_M):
    """The report's rule, frame by frame: nearest player inside the radius."""
    ball = metric[metric.cls == "ball"][["frame", "x", "y"]]
    players = metric[metric.team.isin(("team_A", "team_B"))]
    merged = ball.merge(players, on="frame", suffixes=("_b", ""))
    if merged.empty:
        return np.full(frames.shape, None, dtype=object)
    merged["d"] = np.hypot(merged.x - merged.x_b, merged.y - merged.y_b)
    nearest = merged.loc[merged.groupby("frame").d.idxmin()]
    nearest = nearest[nearest.d < radius_m]
    by_frame = dict(zip(nearest.frame.astype(int), nearest.team))
    return np.array([by_frame.get(int(f)) for f in frames], dtype=object)


def spell_timeline(metric: pd.DataFrame, frames: np.ndarray):
    """The careful rule, from the possession logic that drives the events.

    This wants the ball's kinematics frame rather than its raw rows -- it
    reads the ball's speed to decide whether anyone can be controlling it --
    and it hands back the holding team itself, so there is nothing here to
    reconstruct from track ids.
    """
    ball = ev_module._ball_kinematics(metric)
    if ball.empty:
        return np.full(frames.shape, None, dtype=object)
    poss = ev_module._possession_per_frame(ball, metric)
    if poss.empty or "possessor_team" not in poss.columns:
        return np.full(frames.shape, None, dtype=object)
    by_frame = dict(zip(poss.frame.astype(int), poss.possessor_team))
    return np.array([by_frame.get(int(f)) if by_frame.get(int(f))
                     in ("team_A", "team_B") else None
                     for f in frames], dtype=object)


def fill_gaps(predicted, times, max_gap_s: float):
    """Turn a sampled timeline into a continuous one.

    This is the fix for the selection effect, and it is worth being precise
    about what the effect is. The share is not computed over the match; it is
    computed over the frames where the rule fires -- ball detected, a player
    within three metres of it -- and those frames are not a random sample of
    football. A team that plays long balls spends its possession with nobody
    near the ball, so its possession is disproportionately invisible and the
    ratio is taken over what is left.

    Counting intervals instead of samples removes that, because an interval
    spans its own gaps. Between two firings of the rule:

      * naming the same team, the whole interval is that team's -- the ball
        was theirs before and after, so it was theirs in between;
      * naming different teams, the interval is split down the middle, since
        the change happened somewhere inside it and the middle is the only
        defensible guess;
      * longer than `max_gap_s`, nobody is credited. Beyond some length the
        two ends are not one possession and joining them invents football.

    The truth timeline is built exactly this way -- control runs from one
    labelled action to the next -- so filling makes the two comparable rather
    than comparing a continuous timeline against a sampled one.
    """
    filled = list(predicted)
    known = [k for k, value in enumerate(filled) if value is not None]
    if not known:
        return np.array(filled, dtype=object)

    for start, end in zip(known, known[1:]):
        if end == start + 1:
            continue
        if float(times[end] - times[start]) > max_gap_s:
            continue
        before, after = filled[start], filled[end]
        if before == after:
            for k in range(start + 1, end):
                filled[k] = before
        else:
            middle = (times[start] + times[end]) / 2.0
            for k in range(start + 1, end):
                filled[k] = before if times[k] < middle else after
    return np.array(filled, dtype=object)


def compare(predicted, truth):
    """Agreement and share, under whichever team mapping agrees more."""
    both = np.array([p is not None and t is not None
                     for p, t in zip(predicted, truth)])
    if not both.any():
        return None
    pred, real = predicted[both], truth[both]

    best = None
    for mapping in ({"team_A": "left", "team_B": "right"},
                    {"team_A": "right", "team_B": "left"}):
        mapped = np.array([mapping[p] for p in pred])
        agreement = float(np.mean(mapped == real))
        if best is None or agreement > best[0]:
            best = (agreement, mapped, mapping)
    agreement, mapped, mapping = best

    share_pred = float(np.mean(mapped == "left"))
    share_true = float(np.mean(real == "left"))
    majority = max(share_true, 1.0 - share_true)

    # Per-team recall, which is what a share error is made of. If the rule
    # is equally wrong about both teams the errors cancel in a ratio and the
    # share survives; a share that is out while the frames are mostly right
    # means the mistakes run one way.
    recall = {}
    for side in ("left", "right"):
        here = real == side
        recall[side] = (float(np.mean(mapped[here] == side))
                        if here.any() else float("nan"))

    return {"frames": int(both.sum()), "agreement": agreement,
            "share_pred": share_pred, "share_true": share_true,
            "majority": majority, "mapping": mapping,
            "recall_left": recall["left"], "recall_right": recall["right"],
            "share_error": abs(share_pred - share_true)}


def synthetic_check():
    """A timeline that is right, one that is backwards, and one that is luck.

    The third case has to be built with the ball split evenly, and that is
    not a quirk of the test. Choosing the team mapping to agree more ties
    the share to the agreement whenever one team dominates: get the frames
    backwards and the share comes out backwards too, and the mapping flips
    to fix both at once. Only at 50/50 can a prediction keep the share
    exactly while getting the frames no better than a coin.
    """
    lopsided = np.array(["left"] * 60 + ["right"] * 40, dtype=object)
    even = np.array(["left", "right"] * 50, dtype=object)

    ok = True
    for name, predicted, truth, expect_agree, expect_share in (
            ("exactly right",
             np.array(["team_A"] * 60 + ["team_B"] * 40, dtype=object),
             lopsided, 1.00, 0.60),
            ("labels swapped",
             np.array(["team_B"] * 60 + ["team_A"] * 40, dtype=object),
             lopsided, 1.00, 0.60),
            ("share exact, frames a coin",
             np.array(["team_A"] * 50 + ["team_B"] * 50, dtype=object),
             even, 0.50, 0.50)):
        got = compare(predicted, truth)
        good = (abs(got["agreement"] - expect_agree) < 1e-9
                and abs(got["share_pred"] - expect_share) < 1e-9)
        ok &= good
        print(f"   {name:<28s} agreement {got['agreement']:.2f}, "
              f"share {got['share_pred']:.2f} against "
              f"{got['share_true']:.2f}  {'ok' if good else 'WRONG'}")
    print("\n   The third case is the point: a timeline no better than a "
          "coin flip\n   reports the share exactly right, which is why the "
          "share alone is not a\n   measurement of anything.")
    return ok




def fill_check():
    """Filling a sampled timeline, and refusing to fill too far."""
    times = np.arange(10) / 2.0          # half a second apart
    sampled = np.array(["team_A", None, None, "team_A", None, None,
                        "team_B", None, None, "team_B"], dtype=object)
    ok = True

    filled = fill_gaps(sampled, times, max_gap_s=5.0)
    same = list(filled[:4]) == ["team_A"] * 4
    split = list(filled[4:7]) == ["team_A", "team_B", "team_B"]
    print(f"   {'same team either side':<28s} -> "
          f"{'filled through' if same else 'WRONG'}")
    print(f"   {'different teams':<28s} -> "
          f"{'split at the middle' if split else 'WRONG'}")
    ok &= same and split

    narrow = fill_gaps(sampled, times, max_gap_s=0.6)
    untouched = sum(1 for v in narrow if v is None) == 6
    print(f"   {'gap wider than allowed':<28s} -> "
          f"{'left empty' if untouched else 'WRONG'}")
    ok &= untouched

    # The point of the whole exercise: a team whose possession is sampled
    # less often loses share, and filling gives it back.
    times2 = np.arange(12) / 2.0
    biased = np.array(["team_A"] * 6 + ["team_B", None, None, None, None,
                       "team_B"], dtype=object)
    before = sum(1 for v in biased if v == "team_B") / \
        sum(1 for v in biased if v is not None)
    after_fill = fill_gaps(biased, times2, max_gap_s=5.0)
    after = sum(1 for v in after_fill if v == "team_B") / len(after_fill)
    good = abs(before - 0.25) < 1e-9 and abs(after - 0.5) < 1e-9
    ok &= good
    print(f"   {'sparsely seen team':<28s} -> share {before:.2f} sampled, "
          f"{after:.2f} filled  {'ok' if good else 'WRONG'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        print("Scoring a known timeline, to see what the two numbers do.\n")
        ok = synthetic_check()
        print("\nFilling the gaps between firings, which is the follow-up "
              "this measures.\n")
        ok &= fill_check()
        print(f"\n   {'all pass' if ok else 'SOMETHING IS WRONG'}")
        return

    import score_soccernet as sc

    print("Possession against the team named on every labelled ball "
          "action.\n")
    print(f"  {'clip':>14s} {'rule/gap':>14s} {'frames':>7s} "
          f"{'agreement':>10s} {'majority':>9s} {'share ours':>11s} "
          f"{'share true':>11s} {'error':>6s} {'hit left':>8s} "
          f"{'hit right':>9s}")

    for out_dir, (source, offset) in CLIP_SOURCES.items():
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info = json.loads((path / "clip.json").read_text())
        _, _, metric = sc.run_pipeline(info["path"], path, return_tracks=True)
        frames = np.array(sorted(metric.frame.unique()), dtype=int)
        times = frames / float(info["fps"])
        truth = truth_timeline(source, offset,
                               info["n_frames"] / info["fps"], times)
        if truth is None:
            continue

        for label, predicted in (("proxy", proxy_timeline(metric, frames)),
                                 ("spells", spell_timeline(metric, frames))):
            for gap in GAPS_S:
                filled = (predicted if gap <= 0
                          else fill_gaps(predicted, times, gap))
                got = compare(filled, truth)
                if got is None:
                    continue
                tag = f"{label}/{'sampled' if gap <= 0 else f'{gap:.0f}s'}"
                print(f"  {out_dir[-14:]:>14s} {tag:>14s} {got['frames']:7d} "
                      f"{got['agreement']:10.2f} {got['majority']:9.2f} "
                      f"{got['share_pred']:11.2f} {got['share_true']:11.2f} "
                      f"{got['share_error']:6.2f} "
                      f"{got['recall_left']:8.2f} "
                      f"{got['recall_right']:9.2f}", flush=True)

    print("\n  'agreement' is per frame and 'share' is the number the report "
          "prints.\n  'majority' is what always naming the dominant team "
          "would score, which is\n  the bar agreement has to clear. The team "
          "mapping is chosen to agree more,\n  one fitted bit.")


if __name__ == "__main__":
    main()
