"""Why team assignment sits at 0.55-0.81, and what would move it.

Team assignment is now the binding constraint on pass outcome: the receiver
rebuild reaches balanced accuracy 0.83 on a held-out match given a correct
passer team, and 0.28 with ours. Two things are already known and neither
points anywhere useful:

  * tracks are internally consistent -- zero of 646 ever change team, so this
    is not measurement noise within a track;
  * `src/team_assignment_v2.py` fixed three real faults in how the colour is
    measured (circular hue, seek-based sampling, whole-crop averaging) and
    scored 0.775 against the original's 0.778.

A third is measured here for the first time: on the Reading window, which
scores 0.55, the clusters are the *most balanced* of the four -- 51 tracks
against 52. The split is clean. It is simply not a split along team lines.
That is the signature of clustering on something other than the kit, and the
obvious candidate is illumination: a pitch with a shadow line splits every
shirt into a lit and a shaded version, and if that variation is larger than
the difference between the two kits, k-means finds it first.

So rather than guess a better feature, this samples the torso pixels once per
window and then derives every variant from the same pixels:

    raw        mean HSV over the crop, as the shipped version does
    circular   hue as a unit vector, median over the most saturated half
    no_value   the same without brightness, which is what a shadow changes
    norm_value brightness divided by the frame's own median brightness
    no_grass   green pixels discarded before anything else is measured
    hist       a hue-saturation histogram rather than a summary statistic

Each is scored two ways. Unsupervised is what the pipeline can actually do.
Supervised, leave-one-out, is the ceiling: what the best possible classifier
on these features would reach. The gap between them says whether to fix the
clustering or the features -- and if the ceiling itself is low, neither.

Truth comes from the passer of each matched labelled pass, whose team
SoccerNet gives. Possession errors put noise in those labels, so the ceiling
measured here is a floor on the real ceiling.

    python experiment_teams.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from analyse_pass_outcome import UPLOADS, WINDOWS
from score_soccernet import run_pipeline, load_window
from src import events as ev

SAMPLES_PER_TRACK = 24
MIN_TRACK_FRAMES = 10

# Torso box as a fraction of the bounding box, matching team_assignment_v2.
TORSO_TOP, TORSO_BOTTOM, TORSO_WIDTH = 0.25, 0.55, 0.5

# At most this many pixels kept per sampled crop. A torso is a few hundred
# pixels and the statistics below are medians, so there is nothing to gain
# from keeping all of them and a lot of memory to lose.
MAX_PIXELS_PER_SAMPLE = 200

# OpenCV hue is 0-179. Grass sits around 35-85 at reasonable saturation.
GRASS_HUE_LO, GRASS_HUE_HI, GRASS_MIN_SAT = 30, 90, 40


def _crop_hsv(frame, px, py, h):
    """Torso pixels in HSV, subsampled, or None."""
    if not np.isfinite(h) or h <= 4:
        return None
    w = h * 0.4
    x1, x2 = int(px - w * TORSO_WIDTH / 2), int(px + w * TORSO_WIDTH / 2)
    y_top = py - h
    y1, y2 = int(y_top + h * TORSO_TOP), int(y_top + h * TORSO_BOTTOM)

    H, W = frame.shape[:2]
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    if len(hsv) < 4:
        return None
    if len(hsv) > MAX_PIXELS_PER_SAMPLE:
        idx = np.linspace(0, len(hsv) - 1, MAX_PIXELS_PER_SAMPLE).astype(int)
        hsv = hsv[idx]
    return hsv


def sample_pixels(video_path: str, tracks: pd.DataFrame):
    """One sequential pass: torso pixels per track, plus frame brightness.

    Frame brightness is kept because a feature that divides by it is one of
    the variants, and it can only be measured while the frame is in hand.
    """
    players = tracks[tracks.cls == "player"]
    wanted = defaultdict(list)
    for tid, g in players.groupby("track_id"):
        if len(g) < MIN_TRACK_FRAMES:
            continue
        g = g.sort_values("frame")
        take = g.iloc[np.linspace(0, len(g) - 1,
                                  min(SAMPLES_PER_TRACK, len(g))).astype(int)]
        for r in take.itertuples():
            wanted[int(r.frame)].append(
                (int(r.track_id), float(r.px), float(r.py), float(r.crop_h)))

    samples: dict[int, list] = defaultdict(list)
    cap = cv2.VideoCapture(video_path)
    last, idx = (max(wanted) if wanted else -1), 0
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            frame_v = float(np.median(
                cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]))
            for tid, px, py, h in wanted[idx]:
                hsv = _crop_hsv(frame, px, py, h)
                if hsv is not None:
                    samples[tid].append((hsv, frame_v))
        idx += 1
    cap.release()
    return samples


# --- feature variants -------------------------------------------------------

def _drop_grass(hsv):
    keep = ~((hsv[:, 0] >= GRASS_HUE_LO) & (hsv[:, 0] <= GRASS_HUE_HI)
             & (hsv[:, 1] >= GRASS_MIN_SAT))
    return hsv[keep] if keep.sum() >= 4 else hsv


def _top_saturated(hsv, share=0.5):
    keep = max(2, int(len(hsv) * share))
    return hsv[np.argsort(hsv[:, 1])[-keep:]]


def feat_raw(samples):
    return np.mean([np.mean(hsv, axis=0) for hsv, _ in samples], axis=0)


def _circular(hsv, frame_v=None, use_value=True, norm_value=False):
    ang = hsv[:, 0].astype(float) * (2 * np.pi / 180.0)
    out = [float(np.median(np.cos(ang))), float(np.median(np.sin(ang))),
           float(np.median(hsv[:, 1])) / 255.0]
    if use_value:
        v = float(np.median(hsv[:, 2]))
        out.append(v / max(frame_v, 1.0) if norm_value else v / 255.0)
    return np.array(out)


def feat_circular(samples):
    return np.median([_circular(_top_saturated(h)) for h, _ in samples], axis=0)


def feat_no_value(samples):
    return np.median([_circular(_top_saturated(h), use_value=False)
                      for h, _ in samples], axis=0)


def feat_norm_value(samples):
    return np.median([_circular(_top_saturated(h), fv, norm_value=True)
                      for h, fv in samples], axis=0)


def feat_no_grass(samples):
    return np.median([_circular(_top_saturated(_drop_grass(h)))
                      for h, _ in samples], axis=0)


def feat_hist(samples):
    """Hue-saturation histogram -- a distribution, not a summary statistic.

    A hooped or striped shirt has two colours, and every statistic above
    returns their average, which is a colour neither of them is.
    """
    acc = np.zeros((8, 3), dtype=float)
    for hsv, _ in samples:
        hsv = _drop_grass(hsv)
        hb = np.clip((hsv[:, 0].astype(int) * 8) // 180, 0, 7)
        sb = np.clip(hsv[:, 1].astype(int) // 86, 0, 2)
        np.add.at(acc, (hb, sb), 1.0)
    total = acc.sum()
    return (acc / total).ravel() if total else acc.ravel()


VARIANTS = {
    "raw (shipped)": feat_raw,
    "circular (v2)": feat_circular,
    "no_value": feat_no_value,
    "norm_value": feat_norm_value,
    "no_grass": feat_no_grass,
    "hist": feat_hist,
}


# --- evaluation -------------------------------------------------------------

# A track needs this many labelled actions agreeing before it is treated as
# carrying a known side. One vote is one possession call, and possession calls
# are the thing under suspicion.
MIN_VOTES = 2
MIN_AGREEMENT = 0.75

# How far back from a labelled action to look for the ball at a player's feet.
LOOKBACK_S = 0.6


def label_tracks(metric: pd.DataFrame, labels: list[dict]) -> dict[int, str]:
    """Truth side per track, from every labelled ball action in the window.

    Taking only the passer of a matched pass gave 16-27 labelled tracks, few
    enough that the evaluation could not tell a real effect from the noise of
    picking the better of two cluster-to-side mappings. Every labelled action
    carries a team, so every one of them is a vote, which roughly doubles the
    evidence.

    The vote is only cast when the possession call is one worth trusting: the
    ball slow, a player close, and that player clearly closer than anyone from
    the other cluster. Those are the same gates the receiver decision uses,
    for the same reason -- a contested call is not evidence.
    """
    ball = ev._ball_kinematics(metric)
    if ball.empty:
        return {}

    players = metric[(metric.cls == "player")
                     & (metric.team.isin(["team_A", "team_B"]))]
    by_frame = {int(f): g for f, g in players.groupby("frame")}

    votes: dict[int, Counter] = defaultdict(Counter)
    for lab in labels:
        # A labelled action is the moment the ball is struck, so at that
        # instant it is moving -- reading the possessor there rejected 60-68%
        # of the labels as "ball too fast". The player who struck it had it at
        # their feet just before, which is where to look.
        before = ball[(ball.time_s <= lab["t"])
                      & (ball.time_s >= lab["t"] - LOOKBACK_S)]
        if "speed_kmh" in before.columns:
            before = before[before.speed_kmh.fillna(0.0)
                            <= ev.BALL_CONTROL_SPEED_KMH]
        if before.empty:
            continue
        near = before.iloc[-1]
        g = by_frame.get(int(near["frame"]))
        if g is None or g.empty:
            continue

        d = np.hypot(g.px - float(near["bx"]), g.py - float(near["by"]))
        order = np.argsort(d.to_numpy())
        best = int(order[0])
        if float(d.iloc[best]) > ev.RECEIVER_MAX_DIST_M:
            continue
        best_team = str(g.team.iloc[best])
        rival = d[g.team.to_numpy() != best_team]
        if not len(rival) or (float(rival.min()) - float(d.iloc[best])
                              ) < ev.RECEIVER_TEAM_MARGIN_M:
            continue
        votes[int(g.track_id.iloc[best])][lab["team"]] += 1

    out = {}
    for tid, c in votes.items():
        total = sum(c.values())
        side, n = c.most_common(1)[0]
        if total >= MIN_VOTES and n / total >= MIN_AGREEMENT:
            out[tid] = side
    return out


def chance_two_way(truth_by_track, n_draws=4000, seed=1):
    """What a random split scores under the better-of-two mapping.

    The mapping is chosen to favour us, so a random two-way split of the same
    tracks does not score 0.50 -- on 20 tracks it scores about 0.60. Without
    this number the unsupervised column cannot be read at all.
    """
    tids = sorted(truth_by_track)
    if len(tids) < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws)
    for k in range(n_draws):
        pred = {t: f"c{v}" for t, v in zip(tids, rng.integers(0, 2, len(tids)))}
        acc, _ = two_way_accuracy(pred, truth_by_track)
        out[k] = acc if np.isfinite(acc) else 0.5
    return float(out.mean())


def two_way_accuracy(pred, truth_by_track):
    """Agreement under the better of the two cluster-to-side mappings."""
    tids = [t for t in truth_by_track if t in pred]
    if len(tids) < 4:
        return float("nan"), 0
    labels = sorted({pred[t] for t in tids})
    if len(labels) != 2:
        return float("nan"), len(tids)

    # The clusters carry no intrinsic side, so a mapping must be chosen; the
    # better of the two is taken, which can only flatter the result and is
    # the same convention the rest of the project scores under.
    best = 0
    for sides in (("left", "right"), ("right", "left")):
        mapping = dict(zip(labels, sides))
        best = max(best, sum(1 for t in tids
                             if mapping[pred[t]] == truth_by_track[t]))
    return best / len(tids), len(tids)


def evaluate(samples, truth_by_track, shipped_team):
    rows = []
    for name, fn in VARIANTS.items():
        tids = [t for t in samples if samples[t]]
        feats = {t: fn(samples[t]) for t in tids}

        labelled = [t for t in tids if t in truth_by_track]
        if len(labelled) < 6:
            continue
        X = np.array([feats[t] for t in labelled])
        y = np.array([1 if truth_by_track[t] == "right" else 0
                      for t in labelled])

        # Unsupervised: cluster every track, score on the labelled ones.
        allX = np.array([feats[t] for t in tids])
        allX = StandardScaler().fit_transform(allX)
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(allX)
        pred = {t: f"c{l}" for t, l in zip(tids, km.labels_)}
        unsup, n = two_way_accuracy(pred, truth_by_track)

        # Supervised ceiling, leave-one-out.
        if len(set(y.tolist())) < 2:
            ceiling = float("nan")
        else:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=1.0))
            ceiling = float(cross_val_score(
                clf, X, y, cv=LeaveOneOut()).mean())

        rows.append(dict(variant=name, n=n, unsupervised=unsup,
                         ceiling=ceiling))

    ship, nship = two_way_accuracy(shipped_team, truth_by_track)
    return rows, ship, nship


def main():
    all_rows = defaultdict(list)
    for name, out_dir, labels_name, offset in WINDOWS:
        clip = json.loads((Path(out_dir) / "clip.json").read_text())["path"]
        _, _, metric = run_pipeline(clip, Path(out_dir), return_tracks=True)

        labels = [l for l in load_window(str(UPLOADS / labels_name), offset, 90.0)]
        raw_labels = json.loads((UPLOADS / labels_name).read_text())["annotations"]
        sides = {round(float(a["position"]) / 1000.0 - offset, 3): a["team"]
                 for a in raw_labels}
        for l in labels:
            l["team"] = sides.get(round(l["t"], 3))
        labels = [l for l in labels if l["team"] in ("left", "right")]

        truth_by_track = label_tracks(metric, labels)
        floor = chance_two_way(truth_by_track)

        # Pixel coordinates, before the metric conversion and before re-id
        # renamed anything: merged ids are original ids, so this joins.
        raw = pd.read_parquet(Path(out_dir) / "tracks_grass.parquet")
        shipped = {int(k): v for k, v in
                   raw[raw.cls == "player"].drop_duplicates("track_id")
                   .set_index("track_id")["team"].to_dict().items()
                   if v in ("team_A", "team_B")}

        samples = sample_pixels(clip, raw)
        rows, ship, nship = evaluate(samples, truth_by_track, shipped)

        print(f"\n{name}   {len(truth_by_track)} tracks carry a truth side "
              f"from {len(labels)} labelled actions")
        print(f"  random split under the same mapping  {floor:.2f}   <- read "
              "every unsupervised number against this")
        print(f"  shipped assignment                   {ship:.2f}  (n={nship})")
        print(f"  {'variant':16s} {'n':>3s} {'unsupervised':>13s} "
              f"{'ceiling (LOO)':>14s}")
        for r in rows:
            print(f"  {r['variant']:16s} {r['n']:3d} {r['unsupervised']:13.2f} "
                  f"{r['ceiling']:14.2f}")
            all_rows[r["variant"]].append(r)
        all_rows["__shipped__"].append(dict(unsupervised=ship, ceiling=np.nan))
        all_rows["__floor__"].append(dict(unsupervised=floor, ceiling=np.nan))

    print("\n" + "=" * 60)
    print("mean over the four windows")
    print(f"  {'variant':16s} {'unsupervised':>13s} {'ceiling (LOO)':>14s}")
    for key, label in (("__floor__", "random split"), ("__shipped__", "shipped")):
        rows = all_rows.pop(key)
        print(f"  {label:16s} "
              f"{np.nanmean([r['unsupervised'] for r in rows]):13.2f} "
              f"{'-':>14s}")
    for name, rows in all_rows.items():
        print(f"  {name:16s} "
              f"{np.nanmean([r['unsupervised'] for r in rows]):13.2f} "
              f"{np.nanmean([r['ceiling'] for r in rows]):14.2f}")


if __name__ == "__main__":
    main()
