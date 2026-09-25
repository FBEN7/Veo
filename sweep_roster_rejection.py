"""Rejecting the tracks that are not players.

A quarter of the tracks the pipeline calls players are stewards in hi-vis,
staff in dark coats, spectators, or in one case an advertising hoarding. They
are given a team, they take 22% of the passer and receiver slots on matched
passes, and they corrupt possession and every per-player statistic. See
`TEAM_ASSIGNMENT.md`.

Four rejections have already been measured and rejected:

    grass underfoot    non-players score 0.99, real players 0.93 -- they
                       stand on the pitch surface, so the test is backwards
    movement           extents 32-1204 px against 48-1288
    size               median height 57 px against 72, ranges overlapping
    more clusters      k=5 taking the two largest lifts purity 0.84 -> 0.99
                       on the window it was fitted on and 0.83 -> 0.85 on the
                       other, then halves team attribution on a third by
                       splitting one side 103 tracks against 38

The last one failed for a specific reason: it assumes the two teams are the
two *biggest* colour groups, and a team that fragments into several clusters
breaks it silently. So the rule here does not count clusters at all.

Both kits form a tight group each, whatever colours they happen to be. A
steward in hi-vis is far from both; so is a stagehand in black; so is a
hoarding. Rejecting by **distance to the nearer of the two team centres** is
therefore kit-agnostic, which is what a rule has to be to transfer between
matches.

Two other signals are measured alongside it, because a rule should be chosen
against alternatives rather than because it sounded right:

    dispersion   how much a track's own colour varies across its samples --
                 a track that wanders between a shirt and a crowd should
                 vary more than one that stays on a player
    isolation    how far the track sits from the other tracks on the pitch,
                 since players are surrounded by players and perimeter staff
                 are not

Everything is scored against the hand-read kit labels, fitted on one window
and checked on the other, whose kits are entirely different.

    python sweep_roster_rejection.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from analyse_pass_outcome import WINDOWS
from experiment_teams import VARIANTS, _drop_grass, _top_saturated, _circular
from score_team_assignment import best_mapping_accuracy
from sweep_team_variants import pixels_for

LABELLED = ("reading 5115", "w1 stoke 1820", "w2 stoke 4010")

# Thresholds swept, in units of the median distance of tracks to their own
# team centre -- a ratio rather than an absolute, so it does not depend on
# how far apart two particular kits happen to be.
CUTS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 1e9)


def per_sample_features(samples):
    """One feature vector per sampled crop, not per track."""
    return np.array([_circular(_top_saturated(_drop_grass(h)))
                     for h, _ in samples])


def track_table(samples, tracks: pd.DataFrame) -> pd.DataFrame:
    """Colour, dispersion and isolation for every track."""
    rows = []
    players = tracks[tracks.cls == "player"]
    pos = players.groupby("track_id")[["px", "py"]].median()
    frames = players.groupby("track_id")["frame"].agg(["min", "max", "size"])

    for tid, s in samples.items():
        if not s:
            continue
        f = per_sample_features(s)
        rows.append(dict(
            track_id=int(tid),
            **{f"f{i}": v for i, v in enumerate(np.median(f, axis=0))},
            # Spread of a track's own colour. Taken as the median absolute
            # deviation rather than the standard deviation so that one bad
            # crop does not decide it.
            dispersion=float(np.median(
                np.linalg.norm(f - np.median(f, axis=0), axis=1))),
            px=float(pos.px.get(tid, np.nan)),
            py=float(pos.py.get(tid, np.nan)),
            n_frames=int(frames["size"].get(tid, 0)),
        ))
    df = pd.DataFrame(rows).set_index("track_id")

    # Isolation: distance to the 5th nearest other track, in pixels. A player
    # on the pitch has other players around them throughout.
    xy = df[["px", "py"]].to_numpy()
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    k = min(5, max(1, len(df) - 1))
    df["isolation"] = np.sort(d, axis=1)[:, k - 1]
    return df


def team_centres(df: pd.DataFrame):
    """Two kit centres, fitted on the colour features alone."""
    cols = [c for c in df.columns if c.startswith("f")]
    X = df[cols].to_numpy()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xs)
    counts = np.bincount(km.labels_, minlength=3)
    keep = np.argsort(counts)[::-1][:2]
    return scaler, km.cluster_centers_[keep], km, keep


def residuals(df: pd.DataFrame):
    """Distance from each track to the nearer team centre, as a ratio.

    Normalised by the median distance among the tracks that sit in the two
    team clusters, so the cut means the same thing whatever the kits are and
    however far apart they happen to be in colour space.
    """
    cols = [c for c in df.columns if c.startswith("f")]
    scaler, centres, km, keep = team_centres(df)
    Xs = scaler.transform(df[cols].to_numpy())
    d = np.linalg.norm(Xs[:, None, :] - centres[None, :, :], axis=2)
    nearest = d.min(axis=1)
    side = np.where(d[:, 0] <= d[:, 1], "team_A", "team_B")
    in_team = np.isin(km.labels_, keep)
    scale = np.median(nearest[in_team]) if in_team.any() else np.median(nearest)
    return nearest / max(scale, 1e-9), side


def evaluate(df, kept_mask, side, kit, nonplayers):
    tids = df.index.to_numpy()
    assigned = {int(t): s for t, s, k in zip(tids, side, kept_mask) if k}
    placed = [t for t in assigned if t in kit or t in nonplayers]
    if not placed:
        return None
    real = [t for t in placed if t in kit]
    acc, _ = best_mapping_accuracy({t: assigned[t] for t in real},
                                   {t: kit[t] for t in real})
    purity = len(real) / len(placed)
    coverage = len(real) / len(kit)
    # Purity and coverage are both costly to lose and trade against each
    # other, so the rule is chosen on their harmonic mean, scaled by how often
    # a kept player is put on the right side. Keeping a stroller costs a
    # possession error; dropping a player costs an event.
    f = (2 * purity * coverage / (purity + coverage)
         if (purity + coverage) else 0.0)
    return dict(purity=purity, coverage=coverage, acc=acc,
                usable=(acc * coverage if np.isfinite(acc) else np.nan),
                score=(acc * f if np.isfinite(acc) else np.nan))


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
    df = track_table(samples, raw)
    res, side = residuals(df)
    df["residual"] = res
    return name, df, side, kit, nonplayers


def separation(df, kit, nonplayers, col):
    a = df.loc[df.index.isin(kit), col].dropna()
    b = df.loc[df.index.isin(nonplayers), col].dropna()
    if not len(a) or not len(b):
        return "no data"
    # Area under the ROC curve, computed directly: the chance that a random
    # non-player scores higher than a random player. 0.5 is no separation.
    auc = float((b.to_numpy()[:, None] > a.to_numpy()[None, :]).mean())
    return (f"players {np.median(a):7.2f}  non-players {np.median(b):7.2f}  "
            f"AUC {auc:.2f}")


def main():
    data = {w: load(w) for w in LABELLED}

    print("How well each signal separates non-players (AUC 0.50 = not at all)")
    for w in LABELLED:
        name, df, _, kit, nonplayers = data[w]
        print(f"\n  {name}  ({len(kit)} players, {len(nonplayers)} not)")
        for col in ("residual", "dispersion", "isolation", "n_frames"):
            print(f"    {col:12s} {separation(df, kit, nonplayers, col)}")

    print("\n\nRejecting by distance to the nearer kit centre")
    curves = {}
    for w in LABELLED:
        name, df, side, kit, nonplayers = data[w]
        print(f"\n  {name}")
        print(f"    {'cut':>6s} {'purity':>7s} {'cover':>7s} {'kit acc':>8s} "
              f"{'score':>7s}")
        curve = {}
        for cut in CUTS:
            mask = df.residual.to_numpy() <= cut
            r = evaluate(df, mask, side, kit, nonplayers)
            if not r or not np.isfinite(r["acc"]):
                continue
            curve[cut] = r
            tag = "  (no rejection)" if cut > 1e8 else ""
            print(f"    {min(cut, 99):6.2f} {r['purity']:7.2f} "
                  f"{r['coverage']:7.2f} {r['acc']:8.2f} {r['score']:7.2f}"
                  f"{tag}")
        curves[w] = curve

    print("\n\nChoosing the cut on one window and checking it on the other")
    for fit_w in LABELLED:
        test_w = [w for w in LABELLED if w != fit_w][0]
        finite = {c: r for c, r in curves[fit_w].items() if c < 1e8}
        best = max(finite, key=lambda c: finite[c]["score"])
        f, t = curves[fit_w][best], curves[test_w].get(best)
        none_f, none_t = curves[fit_w][1e9], curves[test_w][1e9]
        print(f"\n  fitted on {fit_w}: cut {best:.2f}")
        print(f"    {fit_w:16s} purity {f['purity']:.2f} cover "
              f"{f['coverage']:.2f} acc {f['acc']:.2f}   "
              f"(no rejection: purity {none_f['purity']:.2f})")
        if t:
            print(f"    {test_w:16s} purity {t['purity']:.2f} cover "
                  f"{t['coverage']:.2f} acc {t['acc']:.2f}   "
                  f"(no rejection: purity {none_t['purity']:.2f})   HELD OUT")


if __name__ == "__main__":
    main()
