"""Train the shirt-number reader on the SoccerNet jersey-number set.

The set is 1,427 player tracklets from broadcast footage, each a folder of
crops with one label for the whole tracklet: the number, or -1 when it is
never legible. Every crop gets its tracklet's label, so a crop of a
player facing the camera is labelled with a number it does not show; the
reader is meant to be judged per tracklet, by `jersey_reader.vote`, where
that noise averages out -- which is also how it is used.

Crops are shrunk at random to 40-110 px tall before being resized for the
network: the windows here show players a median 62 px tall at Stoke and 93
at Reading, smaller than in this set.

Tested on held-out tracklets (`--val-share`) and, with `--test`, on the
set's own test split; reported per tracklet, as it is used: the right
number, a number where there is none, and none where there is one.

    python train_jersey_reader.py --data .cache/jersey/train \\
        [--test .cache/jersey/test]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from src import jersey_reader as jr
from src.paths import CACHE_DIR

CROPS_PER_TRACKLET = 12          # sampled per tracklet per epoch
VOTE_CROPS = 60                  # crops per tracklet when evaluating


def tracklets(root: Path):
    gt_file = next(root.glob("*_gt.json"))
    labels = json.loads(gt_file.read_text())
    out = []
    for key, number in labels.items():
        files = sorted((root / "images" / key).glob("*.jpg"))
        if files:
            out.append((key, int(number), files))
    return out


def augment(image, rng):
    h, w = image.shape[:2]
    target = rng.uniform(40, 110)
    if h > target:
        s = target / h
        image = cv2.resize(image, (max(int(w * s), 4), max(int(target), 8)),
                           interpolation=cv2.INTER_AREA)
    # A little framing noise: the tracker's boxes are not tight.
    h, w = image.shape[:2]
    dx, dy = int(w * rng.uniform(-0.08, 0.08)), int(h * rng.uniform(-0.05, 0.05))
    image = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    shifted = image[8 + dy:8 + dy + h, 8 + dx:8 + dx + w]
    image = shifted if shifted.size else image[8:8 + h, 8:8 + w]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.7, 1.3)
    hsv[..., 2] *= rng.uniform(0.7, 1.3)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8),
                        cv2.COLOR_HSV2BGR)


def evaluate(model, items, label: str):
    right = wrong = missed = false = none_ok = 0
    for _, number, files in items:
        pick = files if len(files) <= VOTE_CROPS else [
            files[i] for i in np.linspace(0, len(files) - 1,
                                          VOTE_CROPS).astype(int)]
        crops = [cv2.imread(str(f)) for f in pick]
        vis, logp = jr.read(model, [c for c in crops if c is not None])
        got, share, _ = jr.vote(vis, logp)
        if number < 0:
            none_ok += got is None
            false += got is not None
        elif got is None:
            missed += 1
        elif got == number:
            right += 1
        else:
            wrong += 1
    numbered = right + wrong + missed
    print(f"  {label}: numbered tracklets {right}/{numbered} right "
          f"({right / max(numbered, 1):.0%}), {wrong} wrong, {missed} "
          f"read as none; no-number tracklets {none_ok}/{none_ok + false} "
          f"read as none", flush=True)
    return right / max(numbered, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--test", help="the set's test split, with its labels")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--val-share", type=float, default=0.1)
    ap.add_argument("--out", default=str(CACHE_DIR / "jersey_reader.pt"))
    args = ap.parse_args()
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    items = tracklets(Path(args.data))
    random.Random(0).shuffle(items)
    n_val = int(len(items) * args.val_share)
    val, train = items[:n_val], items[n_val:]
    print(f"  {len(train)} training tracklets, {len(val)} held out",
          flush=True)

    model = jr.build()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    bce = torch.nn.BCEWithLogitsLoss()
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        batch = []
        for _, number, files in train:
            for f in rng.choice(len(files), min(CROPS_PER_TRACKLET,
                                                len(files)), replace=False):
                batch.append((files[f], number))
        random.Random(epoch).shuffle(batch)
        t0, total = time.time(), 0.0
        for i in range(0, len(batch), 64):
            chunk = batch[i:i + 64]
            crops, tv, tt, tu = [], [], [], []
            for path, number in chunk:
                image = cv2.imread(str(path))
                if image is None:
                    continue
                crops.append(augment(image, rng))
                v, t, u = jr.targets(number)
                tv.append(v), tt.append(t), tu.append(u)
            v, t, u = model(jr.to_tensor(crops))
            loss = (bce(v, torch.tensor(tv)) + ce(t, torch.tensor(tt))
                    + ce(u, torch.tensor(tu)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(crops)
        sched.step()
        print(f"  epoch {epoch + 1}: loss {total / len(batch):.3f}, "
              f"{time.time() - t0:.0f} s", flush=True)
        if epoch % 3 == 2 or epoch == args.epochs - 1:
            score = evaluate(model, val, "held out")
            if score > best:
                best = score
                torch.save(model.state_dict(), args.out)
                print(f"  saved {args.out}", flush=True)
    if args.test:
        model.load_state_dict(torch.load(args.out))
        evaluate(model, tracklets(Path(args.test)), "test split")


if __name__ == "__main__":
    main()
