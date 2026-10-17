"""How many clusters the pitch actually contains.

Looking at every track in the Reading window as a strip of crops shows what
no summary statistic had: roughly a fifth of the tracks the pipeline calls
players are not players. There are stewards in hi-vis lime, staff in black
coats, people in the crowd, and one track that is an advertising hoarding.
They are assigned `team_A` or `team_B` like anyone else, they pollute the
clustering, and at least two of them were credited with possession at a
labelled action.

Three cheap ways to exclude them were measured and all three fail:

    grass under the track   non-players 0.99, real players 0.93 -- they are
                            standing on the pitch surface, so this is
                            backwards and separates nothing
    movement                non-player bounding-box extent runs 32 to 1204 px
                            against 48 to 1288 for players; a cut at 50 px
                            removes 2 of 8 at the cost of 1 real player
    size                    median height 57 px against 72, distributions
                            overlapping from 39 to 98

What does distinguish them is how they look, which is the one signal the
clustering already has. The shipped rule is k=3 with the smallest cluster
discarded, so it allows exactly one non-team group -- and there are several:
officials, two goalkeepers, stewards, staff.

So this sweeps k, taking the two *largest* clusters as the teams and calling
everything else 'other'. Everything else is held fixed.

A caveat on the measure. Team accuracy is scored at matched labelled passes,
and those comparisons are themselves contaminated by possession error -- the
crops show both kits appearing under the same truth side. That depresses the
absolute number for every variant equally, so it remains a fair comparison
between them and a poor estimate of how good any of them is.

    python sweep_team_clusters.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from analyse_pass_outcome import (UPLOADS, WINDOWS, TOLERANCE_S, match,
                                  truth_passes)
from experiment_teams import VARIANTS
from sweep_team_variants import (pixels_for, team_accuracy, chance_floor,
                                 outcome_accuracy)
from score_soccernet import run_pipeline

KS = (2, 3, 4, 5, 6, 8)

# Measured in sweep_team_variants: six ways of representing the colour all
# land between 0.69 and 0.73, so the representation is not what matters and
# two are enough to show the k effect is not an artefact of one of them.
FEATURES = ("raw (shipped)", "circular (v2)")


def cluster_largest_two(samples, feature_fn, k, seed=0) -> dict[int, str]:
    """Cluster kit colour; the two biggest groups are the teams.

    The shipped rule discards the *smallest* of three, which assumes exactly
    one non-team group exists. Taking the two largest instead lets any number
    of smaller groups -- officials, keepers, stewards, staff -- fall out
    together, which is what the crops say is actually there.
    """
    tids = [t for t in samples if samples[t]]
    if len(tids) < k:
        return {}
    X = StandardScaler().fit_transform(
        np.array([feature_fn(samples[t]) for t in tids]))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).labels_

    counts = np.bincount(labels, minlength=k)
    big = np.argsort(counts)[::-1][:2]
    mapping = {int(big[0]): "team_A", int(big[1]): "team_B"}
    return {int(t): mapping[int(l)] for t, l in zip(tids, labels)
            if int(l) in mapping}


def main():
    results: dict[tuple, list] = {}
    floors = []

    for name, out_dir, labels_name, offset in WINDOWS:
        clip = json.loads((Path(out_dir) / "clip.json").read_text())["path"]
        raw = pd.read_parquet(Path(out_dir) / "tracks_grass.parquet")
        samples = pixels_for(name, clip, raw)
        truth = truth_passes(UPLOADS / labels_name, offset, True)
        print(f"\n=== {name}")

        events, _ = run_pipeline(clip, Path(out_dir))
        ours = [e for e in events if e.get("event_type") == "pass"]
        pairs = match([e["timestamp_s"] for e in ours],
                      [t["t"] for t in truth], TOLERANCE_S)
        acc, n = team_accuracy(ours, truth, pairs)
        floors.append(chance_floor(ours, truth, pairs))
        results.setdefault(("shipped", 3), []).append(
            dict(team=acc, kept=np.nan, passes=len(ours)))
        print(f"  {'shipped k=3':22s} team {acc:.2f} (n={n:2d})  "
              f"{len(ours)} passes")

        for fname in FEATURES:
            fn = VARIANTS[fname]
            for k in KS:
                override = cluster_largest_two(samples, fn, k)
                if not override:
                    continue
                events, _ = run_pipeline(clip, Path(out_dir),
                                         team_override=override)
                ours = [e for e in events if e.get("event_type") == "pass"]
                pairs = match([e["timestamp_s"] for e in ours],
                              [t["t"] for t in truth], TOLERANCE_S)
                acc, n = team_accuracy(ours, truth, pairs)
                oacc, on = outcome_accuracy(ours, truth, pairs)
                kept = len(override) / len([t for t in samples if samples[t]])
                results.setdefault((fname, k), []).append(
                    dict(team=acc, kept=kept, passes=len(ours),
                         outcome=oacc, on=on))
                print(f"  {fname[:9]:9s} k={k:<2d} largest-2  team {acc:.2f} "
                      f"(n={n:2d})  keeps {kept:.0%} of tracks  "
                      f"{len(ours)} passes")

    print("\n" + "=" * 70)
    print(f"chance floor (random team per event): {np.nanmean(floors):.2f}")
    print(f"\n  {'variant':24s} {'mean team':>10s} {'kept':>7s}   per window")
    for (fname, k), rows in results.items():
        label = f"{fname[:9]} k={k}" if fname != "shipped" else "shipped k=3"
        per = "  ".join(f"{r['team']:.2f}" for r in rows)
        print(f"  {label:24s} {np.nanmean([r['team'] for r in rows]):10.2f} "
              f"{np.nanmean([r['kept'] for r in rows]):7.0%}   {per}")


if __name__ == "__main__":
    main()
