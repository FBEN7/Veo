"""Fine-tune PARSeq to read shirt numbers (`src/parseq_reader.py`).

Two sources:

- **the SoccerNet jersey set**, whose label is per tracklet: a crop of the
  player facing the camera carries a number it does not show, and trained
  on such crops a reader learns players' looks (`train_jersey_reader.py`).
  So only crops where PARSeq, untrained, already reads the tracklet's
  number are kept (`PER_TRACKLET` tried per tracklet; cached in
  `.cache/parseq_jersey_kept.json`). They teach the look of shirts and
  sizes of this kind of footage, not hard cases;
- **players a person named on our footage** (`dump_named_crops.py`),
  crops within `NEAR` frames of the click, where the number was read:
  the hard cases, from one match.

The weights and the confidence bar are chosen on `--dev`, a clip of the
training match not trained on, with the untrained model as epoch 0 --
fine-tuning must beat it -- and reported once on `--test`, the other
match, whose kits are never seen.

    python train_parseq_reader.py --parseq-repo parseq \\
        --add .cache/jersey_named_stoke_1302 .cache/jersey_named_stoke_4207 \\
        --dev .cache/jersey_named_stoke_7001 \\
        --test .cache/jersey_named_reading_0737 ... --out parseq_stoke.pt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from src import parseq_reader as pr
from src.paths import CACHE_DIR
from train_jersey_reader import tracklets

PER_TRACKLET = 16       # jersey-set crops tried per tracklet for keeping
NEAR = 3                # named crops this many frames from the click
ADD_REPEAT = 3          # each named crop this many times an epoch
JERSEY_PER_EPOCH = 3000
FROZEN_BLOCKS = 8       # of the encoder's 12, kept as published
BARS = (0.5, 0.7, 0.9)
BATCH = 64


def kept_jersey(model, root: Path, cache: Path):
    """Jersey-set crops whose number untrained PARSeq reads: [(path, label)]."""
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]
    rng = np.random.default_rng(0)
    out, t0 = [], time.time()
    items = [t for t in tracklets(root) if t[1] >= 0]
    for k, (_, number, files) in enumerate(items):
        pick = [files[i] for i in sorted(rng.choice(
            len(files), min(PER_TRACKLET, len(files)), replace=False))]
        crops = [cv2.imread(str(f)) for f in pick]
        ok = [(f, c) for f, c in zip(pick, crops) if c is not None]
        labels, _ = pr.read(model, [pr.torso(c) for _, c in ok])
        out += [(str(f), str(number)) for (f, _), g in zip(ok, labels)
                if g == str(number)]
        if k % 200 == 0:
            print(f"    {k}/{len(items)} tracklets, {len(out)} crops kept, "
                  f"{time.time() - t0:.0f} s", flush=True)
    cache.write_text(json.dumps(out))
    return out


def near_click(root: Path, near: int = NEAR):
    """Named crops within `near` frames of the click: [(path, label)]."""
    out = []
    for key, number, files in tracklets(root):
        click = int(key.split("_")[-2])
        out += [(str(f), str(number)) for f in files
                if abs(int(f.stem) - click) <= near]
    return out


def augment(image, rng):
    """Footage-sized, framed a little differently, recoloured."""
    h, w = image.shape[:2]
    target = rng.uniform(40, 110)
    if h > target:
        s = target / h
        image = cv2.resize(image, (max(int(w * s), 4), max(int(target), 8)),
                           interpolation=cv2.INTER_AREA)
    top, bottom, side = pr.TORSO
    region = (top + rng.uniform(-0.03, 0.03), bottom + rng.uniform(-0.05, 0.05),
              side + rng.uniform(-0.04, 0.04))
    c = pr.torso(image, region)
    if c.shape[0] < 4 or c.shape[1] < 4:
        c = pr.torso(image)
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.7, 1.3)
    hsv[..., 2] *= rng.uniform(0.7, 1.3)
    c = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.5:
        c = c[:, :, rng.permutation(3)]
    if rng.random() < 0.3:
        c = cv2.GaussianBlur(c, (3, 3), rng.uniform(0.5, 1.2))
    return np.ascontiguousarray(c)


def evaluate(model, items, label: str, bars=BARS):
    """Per bar: (right, wrong, read as none) over named players, each the
    vote of all their crops."""
    reads = []
    for _, number, files in items:
        crops = [pr.torso(c) for c in (cv2.imread(str(f)) for f in files)
                 if c is not None]
        reads.append((number, *pr.read(model, crops)))
    out = {}
    for bar in bars:
        right = wrong = none = 0
        for number, labels, confs in reads:
            got, _, _ = pr.vote(labels, confs, bar)
            if got is None:
                none += 1
            elif got == number:
                right += 1
            else:
                wrong += 1
        out[bar] = (right, wrong, none)
        print(f"  {label}, bar {bar}: {right}/{len(reads)} right, {wrong} "
              f"wrong, {none} unread", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parseq-repo", required=True)
    ap.add_argument("--jersey", default=str(CACHE_DIR / "jersey" / "train"))
    ap.add_argument("--add", nargs="*", default=[])
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = pr.load(args.parseq_repo)
    jersey = kept_jersey(model, Path(args.jersey),
                         CACHE_DIR / "parseq_jersey_kept.json")
    named = [x for d in args.add for x in near_click(Path(d))]
    dev = [t for d in args.dev for t in tracklets(Path(d))]
    print(f"  {len(jersey)} jersey-set crops kept, {len(named)} named crops "
          f"(x{ADD_REPEAT}), {len(dev)} players to choose on", flush=True)

    scores = evaluate(model, dev, "untrained, chosen on")
    best_bar = max(BARS, key=lambda b: scores[b][0] - scores[b][1])
    best = scores[best_bar][0] - scores[best_bar][1]
    torch.save(model.state_dict(), args.out)
    print(f"  saved {args.out} (untrained, bar {best_bar})", flush=True)

    enc = model.model.encoder
    for p in enc.patch_embed.parameters():
        p.requires_grad = False
    for b in enc.blocks[:FROZEN_BLOCKS]:
        for p in b.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    tf = pr.transform(model)
    for epoch in range(args.epochs):
        model.train()
        pool = [jersey[i] for i in rng.choice(
            len(jersey), min(JERSEY_PER_EPOCH, len(jersey)), replace=False)]
        pool += named * ADD_REPEAT
        random.Random(epoch).shuffle(pool)
        t0, total, n = time.time(), 0.0, 0
        for i in range(0, len(pool), BATCH):
            chunk = pool[i:i + BATCH]
            crops, labels = [], []
            for path, label in chunk:
                image = cv2.imread(path)
                if image is not None:
                    crops.append(augment(image, rng))
                    labels.append(label)
            loss = model.training_step((pr.to_batch(model, crops, tf), labels), i)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(crops)
            n += len(crops)
        sched.step()
        model.eval()
        print(f"  epoch {epoch + 1}: loss {total / max(n, 1):.3f}, "
              f"{time.time() - t0:.0f} s", flush=True)
        scores = evaluate(model, dev, f"epoch {epoch + 1}, chosen on")
        bar = max(BARS, key=lambda b: scores[b][0] - scores[b][1])
        if scores[bar][0] - scores[bar][1] > best:
            best, best_bar = scores[bar][0] - scores[bar][1], bar
            torch.save(model.state_dict(), args.out)
            print(f"  saved {args.out} (epoch {epoch + 1}, bar {bar})",
                  flush=True)
    if args.test:
        model.load_state_dict(torch.load(args.out))
        model.eval()
        test = [t for d in args.test for t in tracklets(Path(d))]
        evaluate(model, test, f"TEST ({len(test)} players)", (best_bar,))
        print("  (untrained, on the same players:)", flush=True)
        evaluate(pr.load(args.parseq_repo), test, "untrained, test",
                 (best_bar,))


if __name__ == "__main__":
    main()
