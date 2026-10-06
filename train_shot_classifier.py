"""Learn which candidate strikes are shots, and test it on footage it never saw.

`src/shot_features.py` turns every placed ball sighting in shooting range
into a row of readings; `score_hand_labels.py --dump-features` writes one
file per clip with the clip's labels attached. Here a candidate is a
positive when a labelled shot or goal is within `POSITIVE_S` of it, and
a gradient boosted tree model learns from the rest.

A clip has a few hundred candidates and a shot is a run of them, so the
model's output is turned back into events the way the rules' is: the most
likely candidate, then nothing else within `EVENT_GAP_S` of it, down to a
threshold. Events are matched one-to-one to the labels within
`score_hand_labels.TOLERANCE_S`, as every other detector here is scored.
A shot and a goal labelled less than `MERGE_S` apart are one attempt.

Tested two ways, each against footage the model never trained on:

  * by match -- trained on Stoke v Huddersfield, tested on Reading v
    Fulham, and back: a new ground and a new camera;
  * by clip -- every clip held out in turn, trained on all the others:
    the same grounds, unseen moments, which is how a Veo camera fixed at
    one ground would be used.

    python train_shot_classifier.py feat/*.parquet [--save model.joblib]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# A few thousand rows fit in seconds on one thread. With the feature dumps
# running beside it, the trees' default of every core spun for twelve
# minutes without finishing.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from score_hand_labels import TOLERANCE_S, match
from src.goal_plane import MERGE_S

# Candidates this close to a labelled attempt are positives. At a second,
# the pass or dribble just before the strike was a positive too, and the
# false detections checked on the footage were exactly that: crosses,
# dribbles and passes around the box. On 32 clips, +-0.4 s against +-1 s
# raised the best F1 held out by match from 0.50 to 0.56.
POSITIVE_S = 0.5
# A labelled attempt with no candidate within this long was never placed.
COVER_S = 1.0
# Detected events are at least this far apart. At two seconds the build-up
# before a shot and the clearance after it came out as events of their own;
# at four, on 32 clips held out by match, precision at threshold 0.5 went
# from 34% to 43% for one shot lost.
EVENT_GAP_S = 4.0
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
NOT_FEATURES = {"clip", "match", "frame", "time_s", "label"}


# What the hand-set rules found per clip, where the dump recorded it.
RULES = {}


def load(paths):
    tables, truth, duration = [], {}, {}
    for p in paths:
        table = pd.read_parquet(p)
        if table.empty:
            continue
        clip = str(table["clip"].iloc[0])
        events = json.loads(table.attrs.get("truth", "[]"))
        # SoccerNet labels a goal twice, the shot and then the goal a second
        # later: one attempt, which one detection should answer.
        attempts = []
        for t in sorted(e["time_s"] for e in events
                        if e["event_type"] in ("shot", "goal")):
            if not attempts or t - attempts[-1] >= MERGE_S:
                attempts.append(t)
        truth[clip] = attempts
        duration[clip] = float(table.attrs.get("duration_s", 0.0))
        if "rule_shots" in table.attrs:
            RULES[clip] = [s["time_s"]
                           for s in json.loads(table.attrs["rule_shots"])]
        t = table.time_s.to_numpy()
        table["label"] = [int(any(abs(x - y) <= POSITIVE_S
                                  for y in truth[clip])) for x in t]
        tables.append(table)
    return pd.concat(tables, ignore_index=True), truth, duration


def features(table):
    return [c for c in table.columns if c not in NOT_FEATURES]


def model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0,
        class_weight="balanced", random_state=0)


def events(table, probs, threshold):
    """Peaks of the model's output, at least `EVENT_GAP_S` apart."""
    order = np.argsort(-probs)
    kept = []
    times = table.time_s.to_numpy()
    for i in order:
        if probs[i] < threshold:
            break
        if all(abs(times[i] - times[j]) >= EVENT_GAP_S for j in kept):
            kept.append(i)
    return sorted(kept, key=lambda i: times[i])


def held_out(data, truth, fold_of, name, verbose):
    """Fit without each fold, predict it; events per clip at each threshold."""
    cols = features(data)
    probs = np.full(len(data), np.nan)
    for fold in sorted(set(fold_of)):
        test = np.array([f == fold for f in fold_of])
        train = data[~test]
        if train.label.sum() == 0:
            continue
        # A reading never taken in the training clips -- no kick-off in any
        # of them -- cannot be learned, and the trees refuse an empty column.
        use = [c for c in cols if train[c].notna().any()]
        m = model().fit(train[use], train.label)
        probs[test] = m.predict_proba(data[test][use])[:, 1]
    print(f"\n  {name}")
    print(f"  {'threshold':>9s} {'labelled':>9s} {'found':>6s} "
          f"{'matched':>8s} {'recall':>7s} {'precision':>10s}")
    best = None
    for thr in THRESHOLDS:
        labelled = found = matched = 0
        for clip, rows in data.groupby("clip"):
            idx = rows.index.to_numpy()
            got = events(rows, probs[idx], thr)
            t = rows.time_s.to_numpy()[got].tolist()
            pairs, _, _ = match(t, truth[clip], TOLERANCE_S)
            labelled += len(truth[clip])
            found += len(t)
            matched += len(pairs)
        recall = matched / max(labelled, 1)
        precision = matched / max(found, 1)
        print(f"  {thr:9.1f} {labelled:9d} {found:6d} {matched:8d} "
              f"{recall:7.0%} {precision:10.0%}")
        f1 = 2 * matched / max(labelled + found, 1)
        if best is None or f1 > best[1]:
            best = (thr, f1)
    if verbose:
        thr = best[0]
        print(f"\n  events at threshold {thr} (the best F1 here, so chosen on "
              f"the test set itself), to check by eye:")
        for clip, rows in data.groupby("clip"):
            idx = rows.index.to_numpy()
            got = events(rows, probs[idx], thr)
            t = rows.time_s.to_numpy()[got].tolist()
            pairs, used, used_truth = match(t, truth[clip], TOLERANCE_S)
            marks = []
            for k, i in enumerate(got):
                r = rows.iloc[i]
                marks.append(f"{r.time_s:.1f}s p={probs[idx][i]:.2f} "
                             f"frame {int(r.frame)}"
                             + ("" if k in used else " FALSE"))
            missed = [f"{x:.1f}s" for j, x in enumerate(truth[clip])
                      if j not in used_truth]
            print(f"    {clip}: " + ("; ".join(marks) or "nothing")
                  + (f"  | missed {', '.join(missed)}" if missed else ""))
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features", nargs="+")
    ap.add_argument("--save", help="fit on everything and save the model")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    data, truth, _ = load([Path(p) for p in args.features])
    clips = sorted(data["clip"].unique())
    print(f"  {len(data)} candidates from {len(clips)} clips; "
          f"{int(data.label.sum())} within {POSITIVE_S:.1f} s of "
          f"{sum(len(truth[c]) for c in clips)} labelled shots and goals")
    for m, rows in data.groupby("match"):
        print(f"    {m}: {rows['clip'].nunique()} clips, "
              f"{sum(len(truth[c]) for c in rows['clip'].unique())} labelled")
    # A labelled attempt with no candidate near it was never placed -- no
    # goal in frame, so no camera pose -- and no classifier can find it.
    unseen = []
    for clip in clips:
        times = data.loc[data["clip"] == clip, "time_s"].to_numpy()
        for t in truth[clip]:
            if not np.any(np.abs(times - t) <= COVER_S):
                unseen.append(f"{clip} {t:.0f}s")
    total = sum(len(truth[c]) for c in clips)
    print(f"  {total - len(unseen)} of {total} labelled attempts have a "
          f"candidate within {COVER_S:.0f} s; never placed: "
          f"{', '.join(unseen) or 'none'}")

    if RULES:
        # The rules learn nothing, so every clip is unseen by them.
        have = [c for c in clips if c in RULES]
        found = sum(len(RULES[c]) for c in have)
        matched = sum(len(match(RULES[c], truth[c], TOLERANCE_S)[0])
                      for c in have)
        labelled = sum(len(truth[c]) for c in have)
        print(f"  hand-set rules (goal_plane.find_shots) on {len(have)} "
              f"clips: {matched} of {labelled} found, {found} detections, "
              f"precision {matched / max(found, 1):.0%}")

    if data["match"].nunique() > 1:
        held_out(data, truth, data["match"].tolist(),
                 "held out by match (trained on the other ground)",
                 not args.quiet)
    held_out(data, truth, data["clip"].tolist(),
             "held out by clip (trained on every other clip)",
             not args.quiet)

    if args.save:
        import joblib

        m = model().fit(data[features(data)], data.label)
        joblib.dump({"model": m, "features": features(data)}, args.save)
        print(f"\n  saved to {args.save}")


if __name__ == "__main__":
    main()
