"""Measure alternative team assignments end to end, where the evidence is.

An earlier attempt scored variants against a per-track truth built from
labelled actions. It could not work: after the gates needed to make a
possession call trustworthy, only three to eight tracks per window carried a
side, and on that many tracks a random split already scores 0.63 to 0.69
because the cluster-to-side mapping is chosen in our favour. The measurement
had no power to detect anything.

The evidence is at the events. Every matched labelled pass carries a team,
which gives 20 to 31 comparisons per window instead of 3 to 8, and it is the
quantity that matters anyway -- team assignment is only interesting here
through what it does to the events.

So each variant is swapped into the pipeline and the whole thing re-run.
Detection is cached and nothing between detection and events reads the team
column, so this costs one video pass per window plus the event detection.

Variants differ in how a track's kit colour is represented:

    shipped     the current implementation, k=3, smallest cluster discarded
    raw         its feature -- mean HSV over the torso crop -- at k=2
    circular    hue as a unit vector, median over the most saturated half
    no_value    the same without brightness, which is what a shadow changes
    norm_value  brightness divided by the frame's own median brightness
    no_grass    green pixels discarded before anything is measured
    hist        a hue-saturation histogram instead of a summary statistic

Read every number against the chance row, which permutes the team labels
across tracks and re-scores. The mapping is chosen in our favour, so chance
is not 0.50.

    python sweep_team_variants.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from analyse_pass_outcome import (UPLOADS, WINDOWS, TOLERANCE_S, match,
                                  truth_passes, best_team_mapping)
from experiment_teams import VARIANTS, sample_pixels
from score_soccernet import run_pipeline

CACHE = Path("/tmp/claude-0/-home-user-Veo/"
             "cd4d7e67-1dd4-5fa1-975c-2f5b3217663b/scratchpad/team_pixels")


def pixels_for(window_name: str, clip: str, raw: pd.DataFrame):
    """Torso pixels per track, cached -- the video pass is the expensive part."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{window_name.replace(' ', '_')}.pkl"
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    samples = sample_pixels(clip, raw)
    with path.open("wb") as fh:
        pickle.dump(samples, fh)
    return samples


def cluster(samples, feature_fn, k=2, seed=0) -> dict[int, str]:
    """Assign each track to team_A or team_B by clustering its kit colour."""
    tids = [t for t in samples if samples[t]]
    if len(tids) < k:
        return {}
    X = StandardScaler().fit_transform(
        np.array([feature_fn(samples[t]) for t in tids]))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).labels_

    if k >= 3:
        # The shipped rule: the smallest cluster is assumed to be officials
        # and goalkeepers and is discarded.
        counts = np.bincount(labels, minlength=k)
        other = int(np.argmin(counts))
        team_ids = [c for c in range(k) if c != other]
        mapping = {team_ids[0]: "team_A", team_ids[1]: "team_B"}
    else:
        mapping = {0: "team_A", 1: "team_B"}

    return {int(t): mapping[int(l)] for t, l in zip(tids, labels)
            if int(l) in mapping}


def team_accuracy(events, truth, pairs):
    """Agreement with SoccerNet's side at the matched passes."""
    team_pairs = [(events[pi].get("team"), truth[ti]["team"])
                  for pi, ti in pairs]
    team_pairs = [(a, b) for a, b in team_pairs
                  if a in ("team_A", "team_B")]
    if len(team_pairs) < 4:
        return float("nan"), 0
    _, acc = best_team_mapping(team_pairs)
    return acc, len(team_pairs)


def chance_floor(events, truth, pairs, n_draws=2000, seed=1):
    """The same score with the two teams assigned to events at random."""
    sides = [truth[ti]["team"] for pi, ti in pairs
             if events[pi].get("team") in ("team_A", "team_B")]
    if len(sides) < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws)
    for k in range(n_draws):
        fake = [("team_A" if v else "team_B")
                for v in rng.integers(0, 2, len(sides))]
        _, out[k] = best_team_mapping(list(zip(fake, sides)))
    return float(out.mean())


def outcome_accuracy(events, truth, pairs):
    """Share of decided outcomes that are right, and how many were decided."""
    ok = n = 0
    for pi, ti in pairs:
        o = events[pi].get("outcome")
        if o not in ("success", "intercepted"):
            continue
        n += 1
        ok += (o == truth[ti]["outcome"])
    return (ok / n if n else float("nan")), n


def main():
    results: dict[str, list] = {}
    floors = []

    for name, out_dir, labels_name, offset in WINDOWS:
        clip = json.loads((Path(out_dir) / "clip.json").read_text())["path"]
        raw = pd.read_parquet(Path(out_dir) / "tracks_grass.parquet")
        samples = pixels_for(name, clip, raw)
        truth = truth_passes(UPLOADS / labels_name, offset, True)

        print(f"\n=== {name}")
        plans = [("shipped", None, 3)] + [
            (label, fn, 2) for label, fn in VARIANTS.items()]

        for label, fn, k in plans:
            override = None if fn is None else cluster(samples, fn, k=k)
            if override is not None and not override:
                continue
            events, _ = run_pipeline(clip, Path(out_dir),
                                     team_override=override)
            ours = [e for e in events if e.get("event_type") == "pass"]
            pairs = match([e["timestamp_s"] for e in ours],
                          [t["t"] for t in truth], TOLERANCE_S)
            acc, n = team_accuracy(ours, truth, pairs)
            oacc, on = outcome_accuracy(ours, truth, pairs)
            results.setdefault(label, []).append(
                dict(window=name, team=acc, n=n, outcome=oacc, on=on,
                     passes=len(ours)))
            if label == "shipped":
                floors.append(chance_floor(ours, truth, pairs))
            print(f"  {label:12s} team {acc:.2f} (n={n:2d})   "
                  f"outcome {oacc:.2f} (n={on:2d})   {len(ours)} passes")

    print("\n" + "=" * 68)
    print("mean over four windows        team   outcome   decided   passes")
    print(f"  {'chance':12s}            {np.nanmean(floors):.2f}        -"
          "         -        -")
    for label, rows in results.items():
        print(f"  {label:12s}            {np.nanmean([r['team'] for r in rows]):.2f}"
              f"      {np.nanmean([r['outcome'] for r in rows]):.2f}"
              f"      {np.mean([r['on'] for r in rows]):5.1f}"
              f"    {np.mean([r['passes'] for r in rows]):5.1f}")
    print("\nper-window team accuracy")
    for label, rows in results.items():
        print(f"  {label:12s} " + "  ".join(f"{r['team']:.2f}" for r in rows))


if __name__ == "__main__":
    main()
