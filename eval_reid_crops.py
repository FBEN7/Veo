"""How well an appearance embedding tells team-mates apart on SoccerTrack v2
crops (`build_reid_crops.py`): for each crop of an outfield player, its
nearest crop among the same side's outfield players (goalkeepers left
out) -- rank-1, how often that is the same person, and the mean average
precision. Crops of the same person within `NEAR_S` seconds of the query
are left out of its gallery, so a near-copy a few frames away does not
count as a match. Also by the query's height (far / middle / near side).

`--body` scores the ImageNet ResNet-18 body alone (512-d, no head), the
baseline that does not depend on a random projection.

    python eval_reid_crops.py --reid W.pt --crops A.pkl B.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

FPS = 25
NEAR_S = 10.0
HEIGHTS = ((0, 40), (40, 55), (55, 1e9))


def embed(model, crops, body: bool = False, batch: int = 64):
    import torch

    from train_player_reid import to_tensor

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            x = to_tensor(crops[i:i + batch])
            e = model.body(x) if body else model(x)
            out.append(torch.nn.functional.normalize(e, dim=1).numpy())
    return np.concatenate(out)


def score(feats, who, side, frame, height) -> dict:
    sim = feats @ feats.T
    rows = []
    for i in range(len(feats)):
        far = np.abs(frame - frame[i]) > NEAR_S * FPS
        mask = (side == side[i]) & (np.arange(len(feats)) != i) \
            & ((who != who[i]) | far)
        same = who[mask] == who[i]
        if not same.any():
            continue
        order = np.argsort(-sim[i, mask])
        hits = same[order]
        prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        rows.append((height[i], bool(hits[0]),
                     float((prec * hits).sum() / hits.sum())))
    h = np.array([r[0] for r in rows])
    r1 = np.array([r[1] for r in rows])
    ap = np.array([r[2] for r in rows])
    out = {"queries": len(rows), "rank-1": round(float(r1.mean()), 3),
           "mAP": round(float(ap.mean()), 3)}
    for lo, hi in HEIGHTS:
        m = (h >= lo) & (h < hi)
        name = f"{lo}-{hi:.0f}px" if hi < 1e9 else f">{lo}px"
        out[name] = (f"rank-1 {r1[m].mean():.3f}, mAP {ap[m].mean():.3f}, "
                     f"n {int(m.sum())}") if m.any() else "none"
    return out


def evaluate(model, path: Path, body: bool = False) -> dict:
    blob = pickle.loads(Path(path).read_bytes())
    crops, who, side, frame, height = [], [], [], [], []
    for pid, p in blob["people"].items():
        if p["role"] != "player":
            continue
        crops += p["crops"]
        who += [pid] * len(p["crops"])
        side += [p["side"]] * len(p["crops"])
        frame += p["frame"]
        height += p["h"]
    feats = embed(model, crops, body)
    return score(feats, np.array(who), np.array(side), np.array(frame),
                 np.array(height))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reid", help="weights (train_player_reid.py); "
                                   "with --body, ignored")
    ap.add_argument("--body", action="store_true",
                    help="the ImageNet ResNet-18 body alone, 512-d")
    ap.add_argument("--crops", nargs="+", required=True)
    args = ap.parse_args()
    if not args.reid and not args.body:
        ap.error("give --reid weights, or --body")
    import torch

    from train_player_reid import build

    model = build()
    if args.reid and not args.body:
        model.load_state_dict(torch.load(args.reid))
    for path in args.crops:
        print(f"  {Path(path).stem}: {evaluate(model, path, args.body)}",
              flush=True)


if __name__ == "__main__":
    main()
