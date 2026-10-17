"""Can the receiving team be decided better than it currently is?

`diagnose_transfer.py` establishes the mechanism. The credited receiver is the
nearest player to the ball 100% of the time, and on the passes we call wrong
the nearest player is an opponent 2.1 m from the ball while the nearest
teammate is 3.7 m. The decision turns on about 1.5 m, which is the same on the
calls we get right (teammate 2.8 m, nearest opponent 4.3 m). The two cases are
symmetric, so no threshold on that separation can tell them apart.

That leaves the question of whether we are looking at the wrong *moment*.
`_settled_possessor` existed in `src/events.py` with a docstring saying exactly
this -- credit the receiver once the ball is under control, not at the instant
it arrives near somebody -- and it was never called. Nothing in the pipeline
used it. This experiment is what decided the parameters of the
`_settled_receiver` that replaced it.

So this sweeps the decision rather than assuming one:

  * **settle**  -- how long after the ball is struck to wait before reading
    off the nearest player, from 0 (what we do now) to 1.5 s;
  * **radius**  -- how close that player must be for the reading to count;
  * **oracle passer** -- the same sweep with the passer's team taken from
    SoccerNet instead of our kit clustering, which isolates the receiver
    decision from team-assignment error at the other end.

Accuracy alone is not the measure: 78 of 94 matched passes kept the ball, so
answering "success" always scores 83%. Reported alongside are the rate we call
turnovers (truth is 17%), balanced accuracy over the two classes, and the
coverage -- what fraction of passes the rule is willing to decide at all.

    python experiment_receiver.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_pass_outcome import (UPLOADS, WINDOWS, TOLERANCE_S, match,
                                  truth_passes, best_team_mapping)
from score_soccernet import run_pipeline
from src import events as ev

# Settle windows, capped at 0.6 s -- and the cap is the point.
#
# The truth this is scored against is derived from the team of the *next*
# labelled ball action, and in these windows the next action follows a pass
# after a median of 1.08 to 1.40 s, with 52-69% of them inside 1.5 s. A rule
# that waits 1.5 s and reads off who has the ball is therefore reading the
# next action itself, and scores well by construction rather than by having
# identified the receiver. That circularity is worth naming because it is
# invisible in the numbers: the 1.5 s rule reached balanced accuracy 0.89 on
# a held-out match, better than anything here, and it is meaningless.
#
# At 0.6 s only 0-5% of next actions have occurred, so the reading is still
# about reception.
SETTLES = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6)

# The current possession radius is 8 m, which is not possession.
RADII = (1.5, 2.0, 3.0, 4.0, 6.0, 8.0)

# The ball is under control below this, not still arriving.
CONTROL_KMH = 20.0

HELD_OUT = "reading 5115"

# Read the geometry at the first controlled frame after the ball arrives
# rather than the last one inside the settle window. Both are measured.
PICK_FIRST = False


def collect():
    """Every matched pass, with the ball and player geometry around it."""
    rows = []
    for name, out_dir, labels_name, offset in WINDOWS:
        clip = json.loads((Path(out_dir) / "clip.json").read_text())["path"]
        events, _, metric = run_pipeline(clip, Path(out_dir), return_tracks=True)

        ball = ev._ball_kinematics(metric)
        players = metric[(metric.cls == "player")
                         & (metric.team.isin(["team_A", "team_B"]))][
            ["frame", "track_id", "team", "px", "py"]]

        ours = [e for e in events if e.get("event_type") == "pass"]
        truth = truth_passes(UPLOADS / labels_name, offset, True)
        pairs = match([e["timestamp_s"] for e in ours],
                      [t["t"] for t in truth], TOLERANCE_S)

        # Map our colour clusters onto SoccerNet's sides on this window, so an
        # oracle passer team can be expressed in our own vocabulary.
        mapping, _ = best_team_mapping(
            [(ours[pi].get("team"), truth[ti]["team"]) for pi, ti in pairs])
        inverse = {v: k for k, v in mapping.items()}

        by_frame = {int(f): g for f, g in players.groupby("frame")}

        for pi, ti in pairs:
            e = ours[pi]
            strike_t = float(e["timestamp_s"])
            arrive_t = strike_t + float(e.get("pass_interval_s") or 0.0)

            readings, separations = {}, {}
            for s in SETTLES:
                target = arrive_t + s
                win = ball[(ball.time_s >= arrive_t) & (ball.time_s <= target + 1e-9)]
                if s > 0 and "speed_kmh" in win.columns:
                    # Prefer a frame where the ball is actually under control.
                    calm = win[win.speed_kmh.fillna(0.0) <= CONTROL_KMH]
                    if not calm.empty:
                        win = calm
                if win.empty:
                    readings[s], separations[s] = (np.nan, None), np.nan
                    continue
                # The first frame in which the ball is under control, not the
                # last one available: at a settle of a second or more the last
                # calm frame can sit beyond the next action entirely, which
                # would credit the receiver of a pass we are not looking at.
                row = win.iloc[0] if PICK_FIRST else win.iloc[-1]
                g = by_frame.get(int(row["frame"]))
                if g is None or g.empty:
                    readings[s], separations[s] = (np.nan, None), np.nan
                    continue
                d = np.hypot(g.px - float(row["bx"]), g.py - float(row["by"]))
                j = int(np.argmin(d.to_numpy()))
                nearest_team = str(g.team.iloc[j])
                readings[s] = (float(d.iloc[j]), nearest_team)

                # How much clearer the nearest player is than the closest
                # player of the other team -- the quantity the call actually
                # turns on.
                rival = d[g.team.to_numpy() != nearest_team]
                separations[s] = (float(rival.min()) - float(d.iloc[j])
                                  if len(rival) else np.inf)

            rows.append(dict(
                window=name,
                truth=truth[ti]["outcome"],
                passer_team_ours=str(e.get("team")),
                passer_team_oracle=inverse.get(truth[ti]["team"]),
                current_call=e.get("outcome"),
                readings=readings,
                separations=separations,
            ))
    return rows


def score(rows, settle, radius, oracle, margin=0.0):
    """Outcome accuracy for one (settle, radius, margin) rule.

    ``margin`` requires the nearest player to be clear of the nearest player
    of the other team by that distance before the call is made. Below it the
    pass is left undecided rather than guessed, which is the honest answer
    when the two candidates are separated by less than the position error.
    """
    tp = fp = fn = tn = undecided = 0
    for r in rows:
        dist, team = r["readings"].get(settle, (np.nan, None))
        sep = r["separations"].get(settle, np.nan)
        passer = r["passer_team_oracle"] if oracle else r["passer_team_ours"]
        if team is None or not np.isfinite(dist) or dist > radius or passer is None:
            undecided += 1
            continue
        if margin > 0 and (not np.isfinite(sep) or sep < margin):
            undecided += 1
            continue
        call = "success" if team == passer else "intercepted"
        truth = r["truth"]
        if truth == "intercepted" and call == "intercepted":
            tp += 1
        elif truth == "success" and call == "intercepted":
            fp += 1
        elif truth == "intercepted" and call == "success":
            fn += 1
        else:
            tn += 1

    n = tp + fp + fn + tn
    if not n:
        return None
    acc = (tp + tn) / n
    baseline = (tn + fp) / n           # always answering "success"
    rec_int = tp / (tp + fn) if (tp + fn) else np.nan
    rec_suc = tn / (tn + fp) if (tn + fp) else np.nan
    bal = np.nanmean([rec_int, rec_suc])
    return dict(settle=settle, radius=radius, margin=margin, n=n,
                coverage=n / (n + undecided), acc=acc, baseline=baseline,
                balanced=bal, called_turnover=(tp + fp) / n,
                true_turnover=(tp + fn) / n, tp=tp, fp=fp, fn=fn, tn=tn)


MARGINS = (0.0, 0.5, 1.0, 1.5)

# A rule that refuses to decide most passes is not a usable rule, however
# accurate it is on the few it keeps.
MIN_COVERAGE = 0.60


def grid(rows, oracle, margins=(0.0,)):
    """Every (settle, radius, margin) rule scored on these passes."""
    out = []
    for s in SETTLES:
        for rad in RADII:
            for mg in margins:
                m = score(rows, s, rad, oracle, mg)
                if m and m["n"] >= 20:
                    out.append(m)
    return out


def best_rule(scored):
    """The most informative rule that still decides most passes.

    Selected on balanced accuracy, not accuracy. Only 17% of these passes
    lost the ball, so answering "success" every time scores 83% while
    carrying no information at all; balanced accuracy scores that constant at
    0.50 and is the only one of the two that can tell an improvement from a
    capitulation.
    """
    usable = [m for m in scored if m["coverage"] >= MIN_COVERAGE]
    if not usable:
        return None
    return max(usable, key=lambda m: (m["balanced"], m["coverage"]))


def table(rows, oracle, title, margins=(0.0,)):
    print(f"\n{title}")
    print("  settle radius margin    n  cover   acc  base   bal  called%  "
          f"(truth {sum(r['truth'] == 'intercepted' for r in rows) / len(rows):.0%})")
    scored = grid(rows, oracle, margins)
    best = best_rule(scored)
    for m in scored:
        flag = " <-" if m is best else ""
        print(f"  {m['settle']:5.1f}s {m['radius']:5.1f}m {m['margin']:5.1f}m "
              f"{m['n']:4d} {m['coverage']:5.0%} {m['acc']:5.0%} "
              f"{m['baseline']:5.0%} {m['balanced']:5.2f} "
              f"{m['called_turnover']:6.0%}{flag}")
    return best


def validate(rows, oracle, label):
    """Fit the rule on three windows, report it on the unseen second match."""
    tune = [r for r in rows if r["window"] != HELD_OUT]
    held = [r for r in rows if r["window"] == HELD_OUT]
    best = best_rule(grid(tune, oracle, MARGINS))
    if not best:
        print(f"  {label}: no rule reaches {MIN_COVERAGE:.0%} coverage")
        return
    m = score(held, best["settle"], best["radius"], oracle, best["margin"])
    base = score(tune, 0.0, 8.0, oracle, 0.0)
    print(f"\n  {label}")
    print(f"    chosen: settle {best['settle']:.1f}s, radius "
          f"{best['radius']:.1f}m, margin {best['margin']:.1f}m")
    print(f"    current behaviour on tuning windows  bal {base['balanced']:.2f}, "
          f"acc {base['acc']:.0%}, calls {base['called_turnover']:.0%} turnovers")
    print(f"    chosen rule on tuning windows        bal {best['balanced']:.2f}, "
          f"acc {best['acc']:.0%}, calls {best['called_turnover']:.0%} turnovers "
          f"(n={best['n']}, cover {best['coverage']:.0%})")
    if m:
        print(f"    chosen rule on the HELD-OUT match    bal {m['balanced']:.2f}, "
              f"acc {m['acc']:.0%}, calls {m['called_turnover']:.0%} turnovers "
              f"(n={m['n']}, cover {m['coverage']:.0%}, truth "
              f"{m['true_turnover']:.0%})")


def main():
    rows = collect()
    print(f"\n{len(rows)} matched passes; "
          f"{sum(r['truth'] == 'intercepted' for r in rows)} truly lost the ball")

    cur = sum(r["current_call"] == "intercepted" for r in rows) / len(rows)
    cur_acc = sum((r["current_call"] == r["truth"]) for r in rows) / len(rows)
    print(f"current pipeline: calls {cur:.0%} turnovers, accuracy {cur_acc:.0%}")

    table(rows, False, "A. our passer team (what the pipeline has)", MARGINS)
    table(rows, True, "B. oracle passer team from SoccerNet "
                      "(isolates the receiver decision)", MARGINS)

    print("\nC. fitted on three windows, checked on the unseen second match")
    validate(rows, False, "our passer team")
    validate(rows, True, "oracle passer team")


if __name__ == "__main__":
    main()
