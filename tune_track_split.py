"""Choose where to cut tracks and when to rejoin them, on game-state clips.

For each SoccerNet game-state clip run through the pipeline, samples the
appearance along every track once (cached as `split_looks.pkl`), then for
each pair of settings -- how unlike the look must become to cut a track
(`track_split`), how alike two pieces must be to rejoin them
(`player_identity.join_by_look`) -- scores the people that result
against the ground truth (`eval_identity_gsr.scores`). The settings are
chosen on the `--fit` clips and reported on the `--test` clips.

    python tune_track_split.py --fit output_gsr_SNGS-022 ... \\
        --test output_gsr_SNGS-021 ... --reid .cache/player_reid.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from eval_identity_gsr import match, ours, scores, truth
from src import player_identity as pid
from src import track_split as ts

SPLITS = (None, 0.4, 0.5, 0.6, 0.7)
JOINS = (None, 0.9, 0.8, 0.7)


def prepare(out_dir: Path, embedder, gsr: Path):
    info = json.loads((out_dir / "clip.json").read_text())
    merged = pd.read_parquet(out_dir / "tracks_merged.parquet")
    cache = out_dir / "split_looks.pkl"
    if cache.exists():
        sampled = pickle.loads(cache.read_bytes())
    else:
        sampled = ts.looks(info["path"], merged, embedder)
        cache.write_bytes(pickle.dumps(sampled))
    gt = truth(gsr / out_dir.name.replace("output_gsr_", ""))
    person = match(ours(merged), gt)
    return merged, sampled, gt, person


def run(merged, sampled, gt, person, cut, join):
    if cut is None:
        df, look = merged, {}
        for tid, (frames, vecs) in sampled.items():
            v = vecs.mean(axis=0)
            look[tid] = v / max(np.linalg.norm(v), 1e-9)
    else:
        df, look = ts.split(merged, sampled, cut)
    tracks = pid.tracks_of(df, min_rows=1)
    for tid, t in tracks.items():
        t.look = look.get(tid)
    idents = pid.identities(tracks, look_threshold=join)
    key = {t.track_id: i.key for i in idents for t in i.tracks}
    det = ours(df)
    ident = det.track_id.map(lambda t: key.get(int(t), f"t{t}"))
    return scores(ident, person.loc[det.index], len(gt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--reid", required=True)
    ap.add_argument("--gsr", default=".cache/gsr")
    args = ap.parse_args()
    import torch

    from train_player_reid import build

    embedder = build()
    embedder.load_state_dict(torch.load(args.reid))
    embedder.eval()
    clips = {d: prepare(Path(d), embedder, Path(args.gsr))
             for d in args.fit + args.test}

    def mean_over(dirs, cut, join):
        rows = [run(*clips[d], cut, join) for d in dirs]
        return {k: float(np.mean([r[k] for r in rows]))
                for k in ("identities", "split", "purity", "IDF1")}

    print("  on the fit clips:")
    best = None
    for cut, join in itertools.product(SPLITS, JOINS):
        m = mean_over(args.fit, cut, join)
        print(f"    cut {cut}, join {join}: " + ", ".join(
            f"{k} {v:.3f}" for k, v in m.items()), flush=True)
        if best is None or m["IDF1"] > best[2]["IDF1"]:
            best = (cut, join, m)
    cut, join, _ = best
    print(f"\n  chosen: cut {cut}, join {join}. On the test clips:")
    for c, j, label in ((None, None, "tracks as they are"),
                        (None, 0.9, "joined by look (before)"),
                        (cut, join, "cut, then joined (chosen)")):
        m = mean_over(args.test, c, j)
        print(f"    {label:26s}: " + ", ".join(f"{k} {v:.3f}"
                                               for k, v in m.items()))


if __name__ == "__main__":
    main()
