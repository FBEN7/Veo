"""Score pass outcome against ground truth derived from SoccerNet team labels.

`EVENT_ACCURACY.md` compares our *rate* of intercepted passes against the rate
implied by the labels. Rates can agree while every individual call is wrong, so
this scores the calls themselves.

SoccerNet's ball-action file carries five fields per action -- gameTime, label,
position, team, visibility -- and no player identity. But `team` is enough to
derive the outcome of a pass without any hand annotation: a pass whose next
ball action belongs to the other team lost possession; one whose next action
belongs to the same team kept it. That is the definition our pipeline is
trying to reproduce.

Three things are measured:

  * the confusion matrix of our outcome against that derived truth;
  * team attribution at matched events, mapping our colour clusters to
    SoccerNet's left/right by whichever assignment agrees more;
  * outcome accuracy *conditional* on having attributed the passer's team
    correctly, which separates "the kit clustering is wrong" from "the
    possession transition is wrong".

Run with the cached window directories:

    python analyse_pass_outcome.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from score_soccernet import run_pipeline

DURATION_S = 90.0

# Matching tolerance. The user's standard, not a metrology one: a second of
# timing error costs a coach nothing, so an event that lands within two
# seconds of a real one is the same event.
TOLERANCE_S = 2.0

# Our pass labels for actions that are passes in SoccerNet's vocabulary.
PASS_LABELS = {"PASS", "HIGH PASS", "CROSS"}

# Actions that end the passage rather than continue it. A pass followed by OUT
# has no receiving team, so its outcome is derived from the action after the
# restart -- reported separately because the choice is arguable.
DEAD_BALL = {"OUT", "THROW IN", "FREE KICK", "GOAL"}

WINDOWS = [
    ("w1 stoke 1820", "output_soccernet", "34498b39-Labels-ball.json", 1820.0),
    ("w2 stoke 4010", "output_soccernet_w2", "34498b39-Labels-ball.json", 4010.0),
    ("w3 stoke 320", "output_soccernet_w3", "34498b39-Labels-ball.json", 320.0),
    ("reading 5115", "output_soccernet_reading", "53f06df4-Labels-ball.json", 5115.0),
]

UPLOADS = Path("/root/.claude/uploads/cd4d7e67-1dd4-5fa1-975c-2f5b3217663b")


def truth_passes(labels_path: Path, offset_s: float, skip_dead: bool):
    """Labelled passes inside the window, each with a derived outcome.

    The outcome comes from the team of the following ball action, which is
    read from the full match rather than the window so that a pass near the
    end of the clip is still resolved.
    """
    ann = json.loads(labels_path.read_text())["annotations"]
    ann = sorted(ann, key=lambda a: float(a["position"]))

    out = []
    for i, a in enumerate(ann):
        if a["label"] not in PASS_LABELS:
            continue
        t = float(a["position"]) / 1000.0 - offset_s
        if not (0.0 <= t <= DURATION_S):
            continue

        # Walk forward to the next action that identifies a possessing team.
        j = i + 1
        while j < len(ann) and skip_dead and ann[j]["label"] in DEAD_BALL:
            j += 1
        if j >= len(ann):
            continue

        out.append(dict(
            t=t,
            team=a["team"],
            next_team=ann[j]["team"],
            outcome="success" if ann[j]["team"] == a["team"] else "intercepted",
        ))
    return out


def match(preds_t, truths_t, tol):
    """Greedy nearest-first one-to-one matching; returns (pred_i, truth_i)."""
    pairs = sorted(
        ((abs(p - t), pi, ti)
         for pi, p in enumerate(preds_t) for ti, t in enumerate(truths_t)
         if abs(p - t) <= tol),
        key=lambda x: x[0])
    used_p, used_t, matched = set(), set(), []
    for _, pi, ti in pairs:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        matched.append((pi, ti))
    return matched


def best_team_mapping(pairs):
    """Map team_A/team_B onto left/right by whichever agrees more.

    Our clusters have no intrinsic side, so a mapping must be chosen. Choosing
    the better of the two is generous to us -- it can only raise the agreement
    -- and that is deliberate: the figure is meant to bound how good team
    attribution could be, not to be pessimistic about it.
    """
    best, best_map = -1, {"team_A": "left", "team_B": "right"}
    for mp in ({"team_A": "left", "team_B": "right"},
               {"team_A": "right", "team_B": "left"}):
        agree = sum(1 for ours, truth in pairs if mp.get(ours) == truth)
        if agree > best:
            best, best_map = agree, mp
    return best_map, best / len(pairs) if pairs else float("nan")


def longest_error_run(flags) -> int:
    """Longest consecutive run of team errors, in event order."""
    best = run = 0
    for wrong in flags:
        run = run + 1 if wrong else 0
        best = max(best, run)
    return best


def analyse(name, out_dir, labels_name, offset, skip_dead):
    clip = json.loads((Path(out_dir) / "clip.json").read_text())["path"]
    events, _ = run_pipeline(clip, Path(out_dir))

    ours = [e for e in events if e.get("event_type") == "pass"]
    truth = truth_passes(UPLOADS / labels_name, offset, skip_dead)
    if not ours or not truth:
        print(f"{name}: nothing to score")
        return None

    pairs = match([e["timestamp_s"] for e in ours], [t["t"] for t in truth],
                  TOLERANCE_S)

    # Team mapping is fitted on the matched events of this window.
    team_pairs = [(ours[pi].get("team"), truth[ti]["team"]) for pi, ti in pairs]
    mapping, team_acc = best_team_mapping(team_pairs)

    # Outcome confusion, and the same restricted to events where we named the
    # passer's team correctly.
    conf = {("success", "success"): 0, ("success", "intercepted"): 0,
            ("intercepted", "success"): 0, ("intercepted", "intercepted"): 0}
    right_team_conf = dict(conf)
    team_err_flags = []

    for pi, ti in pairs:
        o = ours[pi].get("outcome")
        t = truth[ti]["outcome"]
        if o not in ("success", "intercepted"):
            continue
        conf[(t, o)] += 1
        team_ok = mapping.get(ours[pi].get("team")) == truth[ti]["team"]
        team_err_flags.append(not team_ok)
        if team_ok:
            right_team_conf[(t, o)] += 1

    def summarise(c):
        n = sum(c.values())
        if not n:
            return dict(n=0, acc=float("nan"), baseline=float("nan"),
                        our_turnover=float("nan"), true_turnover=float("nan"))
        correct = c[("success", "success")] + c[("intercepted", "intercepted")]
        true_succ = c[("success", "success")] + c[("success", "intercepted")]
        our_int = c[("success", "intercepted")] + c[("intercepted", "intercepted")]
        return dict(n=n, acc=correct / n, baseline=true_succ / n,
                    our_turnover=our_int / n, true_turnover=1 - true_succ / n)

    all_s = summarise(conf)
    ok_s = summarise(right_team_conf)

    print(f"\n{name}   ({len(ours)} passes emitted, {len(truth)} labelled, "
          f"{len(pairs)} matched at +/-{TOLERANCE_S:.0f}s)")
    print(f"  team attribution at matched events : {team_acc:.2f} "
          f"({mapping['team_A']}/{mapping['team_B']}), "
          f"longest error run {longest_error_run(team_err_flags)} "
          f"of {len(team_err_flags)}")
    print(f"  truth turnover rate                : {all_s['true_turnover']:.0%}")
    print(f"  our  turnover rate                 : {all_s['our_turnover']:.0%}")
    print(f"  outcome accuracy                   : {all_s['acc']:.0%}  "
          f"(always-success baseline {all_s['baseline']:.0%}, n={all_s['n']})")
    print(f"  ... on the passes whose team we got right: "
          f"{ok_s['acc']:.0%} (baseline {ok_s['baseline']:.0%}, n={ok_s['n']})")
    print("           truth->  success  intercepted")
    print(f"    ours success   {conf[('success','success')]:7d}  "
          f"{conf[('intercepted','success')]:11d}")
    print(f"    ours intercept {conf[('success','intercepted')]:7d}  "
          f"{conf[('intercepted','intercepted')]:11d}")

    return dict(name=name, team_acc=team_acc, **all_s,
                acc_right_team=ok_s["acc"], n_right_team=ok_s["n"],
                conf=conf, conf_right_team=right_team_conf)


def pooled(rows, key="conf"):
    """The four windows' confusion matrices added together.

    Per-window counts are small -- 16 genuinely lost passes across all four --
    so the question of whether the outcome field carries any information at
    all is only answerable pooled.
    """
    total = {k: 0 for k in
             (("success", "success"), ("success", "intercepted"),
              ("intercepted", "success"), ("intercepted", "intercepted"))}
    for r in rows:
        for k, v in r[key].items():
            total[k] += v
    return total


def report_pooled(total, title):
    ts_os = total[("success", "success")]
    ts_oi = total[("success", "intercepted")]
    ti_os = total[("intercepted", "success")]
    ti_oi = total[("intercepted", "intercepted")]

    n_ts, n_ti = ts_os + ts_oi, ti_os + ti_oi
    print(f"\n{title}")
    print("                     we said success   we said intercepted")
    print(f"  pass kept the ball {ts_os:14d} {ts_oi:21d}")
    print(f"  pass lost the ball {ti_os:14d} {ti_oi:21d}")
    if not (n_ts and n_ti):
        return
    print(f"\n  we call 'intercepted' on {ts_oi / n_ts:.0%} of passes that kept "
          f"the ball (n={n_ts})")
    print(f"  we call 'intercepted' on {ti_oi / n_ti:.0%} of passes that lost "
          f"the ball (n={n_ti})")

    try:
        from scipy.stats import fisher_exact
        _, p = fisher_exact([[ts_os, ts_oi], [ti_os, ti_oi]])
        print(f"  Fisher exact p = {p:.2f} -- the call is "
              f"{'independent of' if p > 0.05 else 'associated with'} "
              "what actually happened")
    except ImportError:
        pass


def dead_ball_sensitivity():
    """Does skipping dead-ball actions change any derived outcome?

    Deriving the outcome from the very next action would call a pass that went
    out of play a turnover or not depending on which team took the throw-in.
    Checked rather than assumed -- and in these four windows no labelled pass
    is immediately followed by a dead-ball action, so the choice is moot here.
    """
    print("derived-outcome sensitivity to dead-ball handling")
    for name, _, labels_name, offset in WINDOWS:
        a = truth_passes(UPLOADS / labels_name, offset, True)
        b = truth_passes(UPLOADS / labels_name, offset, False)
        diff = sum(1 for x, y in zip(a, b) if x["outcome"] != y["outcome"])
        print(f"  {name:16s} {len(a):3d} labelled passes, "
              f"{diff} outcomes change")


def main():
    dead_ball_sensitivity()
    print()
    rows = [r for r in (analyse(n, d, l, o, True) for n, d, l, o in WINDOWS)
            if r]
    if not rows:
        return
    print(f"\n  mean team attribution   {np.mean([r['team_acc'] for r in rows]):.2f}")
    print(f"  mean outcome accuracy   {np.mean([r['acc'] for r in rows]):.0%}")
    print(f"  mean baseline           {np.mean([r['baseline'] for r in rows]):.0%}")
    print("  mean outcome accuracy on right-team passes  "
          f"{np.nanmean([r['acc_right_team'] for r in rows]):.0%}")

    report_pooled(pooled(rows), "POOLED over four windows")
    report_pooled(pooled(rows, "conf_right_team"),
                  "POOLED, restricted to passes whose passer team we got right")


if __name__ == "__main__":
    main()
