"""Learn when two tracks are one person, and test the joining it gives.

On SoccerNet game-state clips (SN-GSR-2025) every detection of ours can be
matched to a real person (`eval_identity_gsr.match`), so every pair of
tracks of the same team is labelled: one person or two. A gradient boosted
model learns from `player_identity.pair_features` -- appearance, frames on
screen together, the time gap, how far the second track starts from where
the first ended, size -- and `player_identity.join_by_pairs` clusters with
it. Fitted on some clips and scored on the others, against the joining by
appearance alone.

    python train_identity_pairs.py --fit output_gsr_SNGS-022 ... \\
        --test output_gsr_SNGS-021 ... [--save model.joblib]
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from eval_identity_gsr import match, ours, scores, truth
from src import player_identity as pid


def load(out_dir: Path, gsr: Path):
    tracks = pickle.loads((out_dir / "identity_tracks.pkl").read_bytes())
    merged = pd.read_parquet(out_dir / "tracks_merged.parquet")
    fps = float(json.loads((out_dir / "clip.json").read_text())["fps"])
    geo = pid.geometry(merged)
    gt = truth(gsr / out_dir.name.replace("output_gsr_", ""))
    det = ours(merged)
    person = match(det, gt)
    main = (pd.DataFrame({"t": det.track_id, "p": person}).dropna()
            .groupby("t").p.agg(lambda s: s.mode().iloc[0]))
    return tracks, geo, fps, det, person, gt, main


def pairs(tracks, geo, fps, main):
    rows, ys = [], []
    by_team = {}
    for t in tracks.values():
        if t.track_id in main.index:
            by_team.setdefault(t.team, []).append(t)
    for members in by_team.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                f = pid.pair_features(members[i], members[j], geo, fps)
                rows.append([f[k] for k in pid.PAIR_FEATURES])
                ys.append(int(main[members[i].track_id]
                              == main[members[j].track_id]))
    return rows, ys


def score_idents(idents, det, person, gt):
    key = {t.track_id: i.key for i in idents for t in i.tracks}
    ident = det.track_id.map(lambda t: key.get(int(t), f"t{t}"))
    return scores(ident, person, len(gt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--gsr", default=".cache/gsr")
    ap.add_argument("--save")
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = [], []
    for d in map(Path, args.fit):
        tracks, geo, fps, _, _, _, main_p = load(d, Path(args.gsr))
        r, l = pairs(tracks, geo, fps, main_p)
        X += r
        y += l
    print(f"  fitted on {len(y)} same-team pairs from {len(args.fit)} clips, "
          f"{sum(y)} of them one person")
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=10, class_weight="balanced", random_state=0)
    model.fit(np.array(X), np.array(y))

    for d in map(Path, args.test):
        tracks, geo, fps, det, person, gt, _ = load(d, Path(args.gsr))
        print(f"\n  {d.name}: {gt.person.nunique()} people")
        base = pid.identities(copy.deepcopy(tracks))
        print(f"    re-joined tracks      : "
              f"{score_idents(base, det, person, gt)}")
        look = pid.identities(copy.deepcopy(tracks), look_threshold=0.9)
        print(f"    by look, 0.9          : "
              f"{score_idents(look, det, person, gt)}")
        for thr in (0.3, 0.5, 0.7):
            idents = pid.join_by_pairs(copy.deepcopy(tracks), geo, fps,
                                       model, threshold=thr)
            print(f"    pair model, {thr}       : "
                  f"{score_idents(idents, det, person, gt)}")
    if args.save:
        import joblib

        joblib.dump(model, args.save)
        print(f"\n  saved {args.save}")


if __name__ == "__main__":
    main()
