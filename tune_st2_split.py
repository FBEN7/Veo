"""Choose the track-cut and look-join thresholds of `identify_players.py
--split` on a SoccerTrack v2 window, for one embedder.

The looks are sampled and embedded once (`track_split.looks`, cached as
`split_looks_<sha8 of the weights>.pkl` like identify_players.py), the
detections matched to ground-truth persons once (by feet, every second
frame, as `eval_soccertrack_v2.py` scores); then every (cut, join) on the
grid is split, joined (`player_identity.identities`, teams strict) and
scored with IDF1. Grid points making more than MAX_PIECES pieces are
skipped and reported -- joining is quadratic in pieces.

The rule, fixed before any result: the highest IDF1; within 0.005 of it,
no cut before any cut, then the lowest cut threshold, then no join before
any join, then the highest join threshold.

    python tune_st2_split.py output_st2_118575_1st_f015000 --reid W.pt
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from eval_identity_gsr import scores
from eval_soccertrack_v2 import boxes, match_feet, truth
from src import player_identity as pid
from src import track_split as ts
from src.paths import DATA_DIR

CUTS = (None, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
JOINS = (None, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97)
MAX_PIECES = 1500
STRIDE = 2


def sampled_looks(out_dir: Path, merged: pd.DataFrame, weights: str):
    from train_player_reid import build
    import torch

    key = hashlib.sha1(Path(weights).read_bytes()).hexdigest()[:8]
    cache = out_dir / f"split_looks_{key}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    model = build()
    model.load_state_dict(torch.load(weights))
    model.eval()
    info = json.loads((out_dir / "clip.json").read_text())
    sampled = ts.looks(info["path"], merged, model)
    cache.write_bytes(pickle.dumps(sampled))
    return sampled


def grid(out_dir: Path, weights: str) -> pd.DataFrame:
    merged = pd.read_parquet(out_dir / "tracks_merged.parquet")
    window = out_dir.name.replace("output_", "")
    gt = truth(DATA_DIR / "soccertrack_v2" / f"{window}_gt.parquet")
    gt = gt[gt.frame % STRIDE == 0]
    sampled = sampled_looks(out_dir, merged, weights)
    rows = []
    for cut in CUTS:
        df, look = ts.split(merged, sampled,
                            threshold=-2.0 if cut is None else cut)
        det = boxes(df)
        det = det[det.frame % STRIDE == 0]
        if cut is None:
            # Splitting keeps every row and its index: match once.
            person = match_feet(det, gt)
        tracks = pid.tracks_of(df, min_rows=1)
        for tid, t in tracks.items():
            t.look = look.get(tid)
        if len(tracks) > MAX_PIECES:
            print(f"  cut {cut}: {len(tracks)} pieces, skipped", flush=True)
            continue
        for join in JOINS:
            idents = pid.identities(copy.deepcopy(tracks), join)
            of = {t.track_id: i.key for i in idents for t in i.tracks}
            ident = det.track_id.map(lambda t: of.get(int(t), f"t{t}"))
            s = scores(ident, person.loc[det.index], len(gt))
            rows.append({"cut": cut, "join": join, "pieces": len(tracks),
                         **s})
            print(f"  cut {cut} join {join}: {len(tracks)} pieces -> "
                  f"{s['identities']} identities, IDF1 {s['IDF1']}, "
                  f"split {s['split']}, purity {s['purity']}", flush=True)
    return pd.DataFrame(rows)


def choose(table: pd.DataFrame) -> dict:
    best = table.IDF1.max()
    near = table[table.IDF1 >= best - 0.005].copy()
    near["_cut"] = near.cut.fillna(-1.0)          # no cut first
    near["_join"] = -near.join.fillna(2.0)        # no join, then highest
    pick = near.sort_values(["_cut", "_join"]).iloc[0]
    return {"cut": None if pd.isna(pick.cut) else float(pick.cut),
            "join": None if pd.isna(pick.join) else float(pick.join),
            "IDF1": float(pick.IDF1), "best IDF1": float(best)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--reid", required=True)
    ap.add_argument("--save", help="write the grid as CSV here")
    args = ap.parse_args()
    table = grid(Path(args.out_dir), args.reid)
    if args.save:
        table.to_csv(args.save, index=False)
    print(json.dumps(choose(table)))


if __name__ == "__main__":
    main()
