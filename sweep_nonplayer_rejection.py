"""Keeping non-players out of the roster, measured against the kits.

`score_team_assignment.py` establishes where the fault actually is. Kit
clustering reaches **0.93** on the tracks that are players; the 0.55 reported
for "team assignment" was possession error wearing its clothes. What is
broken is the roster: 25 of 107 tracks are stewards in hi-vis, staff in black
coats, or an advertising hoarding, 16 of them are given a team, and they take
22% of the passer/receiver slots on matched passes.

Three cheap rejections were already measured and failed -- grass underfoot
(non-players score *higher*), movement, and size. This sweeps the one signal
that does distinguish them, which is how they look, now scored against the
hand-read kit labels rather than through possession.

The shipped rule is k=3 with the smallest cluster discarded: it allows
exactly one non-team group where there are several. Raising k and taking the
two largest clusters as the teams lets any number fall out. What that costs
is real players, so both are reported:

    purity    of the tracks given a team, how many are players
    coverage  of the players, how many are given a team
    kit acc   of the players given a team, how many on the right side

    python sweep_nonplayer_rejection.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from analyse_pass_outcome import WINDOWS
from experiment_teams import VARIANTS
from score_team_assignment import best_mapping_accuracy
from sweep_team_variants import pixels_for

KS = (2, 3, 4, 5, 6, 8, 10, 12)


def assign(samples, feature_fn, k, rule, seed=0):
    """Cluster, then pick which clusters are the two teams.

    ``rule`` is "largest2" (the two biggest clusters) or "drop_smallest"
    (the shipped rule: everything except the single smallest).
    """
    tids = [t for t in samples if samples[t]]
    if len(tids) < k:
        return {}
    X = StandardScaler().fit_transform(
        np.array([feature_fn(samples[t]) for t in tids]))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).labels_
    counts = np.bincount(labels, minlength=k)

    if rule == "drop_smallest":
        if k < 3:
            keep = list(range(k))
        else:
            keep = [c for c in range(k) if c != int(np.argmin(counts))]
        if len(keep) != 2:
            return {}
    else:
        keep = list(np.argsort(counts)[::-1][:2])

    mapping = {int(keep[0]): "team_A", int(keep[1]): "team_B"}
    return {int(t): mapping[int(l)] for t, l in zip(tids, labels)
            if int(l) in mapping}


def score(assigned, kit, nonplayers):
    placed = [t for t in assigned if t in kit or t in nonplayers]
    if not placed:
        return None
    real = [t for t in placed if t in kit]
    purity = len(real) / len(placed)
    coverage = len(real) / len(kit)
    acc, n = best_mapping_accuracy(
        {t: assigned[t] for t in real}, {t: kit[t] for t in real})
    # What actually matters downstream is how many player-slots are right:
    # a track kept and put on the correct side. Purity and coverage trade
    # against each other and this combines them on the pipeline's terms.
    usable = acc * len(real) / len(kit) if np.isfinite(acc) else float("nan")
    return dict(purity=purity, coverage=coverage, acc=acc, n=n, usable=usable,
                placed=len(placed))


LABELLED = ("reading 5115", "w1 stoke 1820")

# Chosen on the Reading window and checked on Stoke, whose kits are entirely
# different -- red-and-white stripes against navy, rather than maroon against
# blue hoops.
CHOSEN = ("no_grass", 5, "largest2")


def load(window):
    name, out_dir, _, _ = next(w for w in WINDOWS if w[0] == window)
    out = Path(out_dir)
    labels = json.loads((out / "kit_labels.json").read_text())
    kit = {int(t): "red" for t in labels["red"]}
    kit.update({int(t): "blue" for t in labels["blue"]})
    nonplayers = {int(t) for t in labels["nonplayer"]}
    clip = json.loads((out / "clip.json").read_text())["path"]
    raw = pd.read_parquet(out / "tracks_grass.parquet")
    samples = pixels_for(name, clip, raw)
    shipped = {int(k): v for k, v in
               raw[raw.cls == "player"].drop_duplicates("track_id")
               .set_index("track_id")["team"].to_dict().items()
               if v in ("team_A", "team_B")}
    return name, kit, nonplayers, samples, shipped


def main():
    data = {w: load(w) for w in LABELLED}

    for window in LABELLED:
        name, kit, nonplayers, samples, shipped = data[window]
        s = score(shipped, kit, nonplayers)
        print(f"\n=== {name}: {len(kit)} player tracks, {len(nonplayers)} "
              "non-player tracks, read off the crops")
        print(f"  {'rule':32s} {'purity':>7s} {'cover':>7s} {'kit acc':>8s} "
              f"{'usable':>7s}")
        print(f"  {'shipped (k=3, drop smallest)':32s} {s['purity']:7.2f} "
              f"{s['coverage']:7.2f} {s['acc']:8.2f} {s['usable']:7.2f}")
        for fname in ("raw (shipped)", "circular (v2)", "no_grass", "hist"):
            for rule in ("drop_smallest", "largest2"):
                for k in KS:
                    a = assign(samples, VARIANTS[fname], k, rule)
                    if not a:
                        continue
                    r = score(a, kit, nonplayers)
                    if not r or not np.isfinite(r["acc"]):
                        continue
                    mark = " <-" if (fname, k, rule) == CHOSEN else ""
                    print(f"  {fname[:9]+f' k={k} '+rule[:5]:32s} "
                          f"{r['purity']:7.2f} {r['coverage']:7.2f} "
                          f"{r['acc']:8.2f} {r['usable']:7.2f}{mark}")

    print("\n" + "=" * 62)
    fname, k, rule = CHOSEN
    print(f"chosen on Reading: {fname}, k={k}, two largest clusters are the "
          "teams")
    print(f"  {'window':18s} {'purity':>7s} {'cover':>7s} {'kit acc':>8s}   "
          "(shipped for comparison)")
    for window in LABELLED:
        name, kit, nonplayers, samples, shipped = data[window]
        r = score(assign(samples, VARIANTS[fname], k, rule), kit, nonplayers)
        s = score(shipped, kit, nonplayers)
        tag = "fitted here" if window == LABELLED[0] else "HELD OUT"
        print(f"  {name:18s} {r['purity']:7.2f} {r['coverage']:7.2f} "
              f"{r['acc']:8.2f}   ({s['purity']:.2f} / {s['coverage']:.2f} / "
              f"{s['acc']:.2f})  {tag}")


if __name__ == "__main__":
    main()
