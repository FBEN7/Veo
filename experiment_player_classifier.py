"""Would a learned appearance classifier beat the distance rule?

Non-player rejection currently works by distance to the nearer kit centre --
AUC 0.97 and 0.91 on the two hand-labelled matches. It is *relative*: it knows
nothing about what a steward looks like, only that they look unlike both kits.

A classifier trained on labelled football footage would know something
absolute: hi-vis lime is never a kit, a dark overcoat is never a kit. That is
the argument for the Roboflow football-players dataset, which labels player,
goalkeeper, referee and ball, and which is CC BY 4.0 rather than SoccerNet's
non-commercial NDA.

Before fetching a dataset, this asks whether the idea works at all, using the
labels already read off the crops. A classifier is trained on one match's
tracks and tested on the other's -- different kits, different non-players
(Reading has hi-vis stewards and dark coats; Stoke has a cyan referee, a
yellow keeper and crowd). If absolute appearance transfers between two
matches it is worth training on many; if it does not, a bigger dataset of the
same kind of feature will not save it.

The comparison is against the distance rule on the same tracks, which is the
incumbent and the thing to beat.

Note on provenance: this trains on SoccerNet-derived crops, so the classifier
it produces cannot ship. It is an instrument for deciding what to build, like
everything else SoccerNet is used for here.

    python experiment_player_classifier.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyse_pass_outcome import WINDOWS
from sweep_roster_rejection import LABELLED, load as load_window

# Hue-saturation-value histogram bins. Coarse on purpose: with 107 and 79
# tracks a finer grid has more dimensions than examples.
H_BINS, S_BINS, V_BINS = 12, 3, 3


def histogram(samples) -> np.ndarray:
    """Absolute colour distribution of a track's torso pixels.

    A distribution rather than a summary statistic, because what marks a
    steward is that some of their pixels are a colour no kit is, and an
    average hides that.
    """
    acc = np.zeros((H_BINS, S_BINS, V_BINS), dtype=float)
    for hsv, _ in samples:
        hb = np.clip((hsv[:, 0].astype(int) * H_BINS) // 180, 0, H_BINS - 1)
        sb = np.clip(hsv[:, 1].astype(int) * S_BINS // 256, 0, S_BINS - 1)
        vb = np.clip(hsv[:, 2].astype(int) * V_BINS // 256, 0, V_BINS - 1)
        np.add.at(acc, (hb, sb, vb), 1.0)
    total = acc.sum()
    return (acc / total).ravel() if total else acc.ravel()


def auc(scores, is_positive) -> float:
    """Chance that a random positive outscores a random negative."""
    pos = scores[is_positive]
    neg = scores[~is_positive]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def build(window):
    """Per-track histogram, label, and the incumbent rule's score."""
    name, df, _, kit, nonplayers = load_window(window)
    out_dir = next(w[1] for w in WINDOWS if w[0] == window)
    clip = json.loads((Path(out_dir) / "clip.json").read_text())["path"]
    raw = pd.read_parquet(Path(out_dir) / "tracks_grass.parquet")

    from sweep_team_variants import pixels_for
    samples = pixels_for(name, clip, raw)

    rows, X, y, resid = [], [], [], []
    for tid in df.index:
        t = int(tid)
        if t not in kit and t not in nonplayers:
            continue
        if not samples.get(t):
            continue
        X.append(histogram(samples[t]))
        y.append(1 if t in nonplayers else 0)     # positive = non-player
        resid.append(float(df.loc[tid, "residual"]))
        rows.append(t)
    return name, np.array(X), np.array(y, dtype=bool), np.array(resid)


def main():
    data = {w: build(w) for w in LABELLED}

    print("Positive class is 'not a player'. AUC 0.50 is no separation.\n")
    print(f"  {'trained on':18s} {'tested on':18s} {'classifier':>11s} "
          f"{'distance rule':>14s}")

    for train_w in LABELLED:
        test_w = [w for w in LABELLED if w != train_w][0]
        ntr, Xtr, ytr, rtr = data[train_w]
        nte, Xte, yte, rte = data[test_w]

        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, C=0.1))
        clf.fit(Xtr, ytr)
        s = clf.predict_proba(Xte)[:, 1]
        print(f"  {ntr:18s} {nte:18s} {auc(s, yte):11.2f} "
              f"{auc(rte, yte):14.2f}")

    print("\nFor reference, the same classifier scored on its own training "
          "match\n(optimistic -- it has seen these tracks):")
    for w in LABELLED:
        n, X, y, r = data[w]
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, C=0.1)).fit(X, y)
        print(f"  {n:18s} {auc(clf.predict_proba(X)[:, 1], y):.2f}  "
              f"({int(y.sum())} non-players of {len(y)})")


if __name__ == "__main__":
    main()
