"""Train the shirt-number reader on the SoccerNet jersey-number set.

The set is 1,427 player tracklets from broadcast footage, each a folder of
crops with one label for the whole tracklet: the number, or -1 when it is
never legible. A crop of a player facing the camera carries a number it
does not show. Trained per crop on those labels (`--plain`, as first
done), the reader read 36 of 42 held-out tracklets -- from the same games
as its training ones -- and 0 of 47 players a person had read on our
footage: it had learnt players' looks, not digits. So by default:

- each tracklet's label is asked of *some* of its crops (`mil_loss`): a
  number is visible if any crop shows one, and the number is the crops'
  readings weighted by how visible each says it is, as `jr.vote` reads;
- colours are scrambled (any hue, channels shuffled, sometimes grey), so
  a kit's colour says nothing about the number.

Crops are shrunk at random to 40-110 px tall before being resized for the
network: the windows here show players a median 62 px tall at Stoke and 93
at Reading, smaller than in this set.

The weights are chosen on `--dev`, a set from other footage
(`dump_named_crops.py`: players a person named), and reported once on
`--test`; reported per tracklet, as the reader is used: the right number,
a wrong one, and none where there is one.

    python train_jersey_reader.py --data .cache/jersey/train \\
        --dev .cache/jersey_named_stoke --test .cache/jersey_named_reading
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
MIL_CROPS = 8                    # crops per tracklet per training step
VISIBLE_BARS = (0.3, 0.5, 0.7, 0.9)  # tried on --dev
ADD_REPEAT = 10


def tracklets(root: Path):
    gt_file = next(root.glob("*_gt.json"))
    labels = json.loads(gt_file.read_text())
    out = []
    for key, number in labels.items():
        files = sorted((root / "images" / key).glob("*.jpg"))
        if files:
            out.append((key, int(number), files))
    return out


def augment(image, rng, colour: bool = True):
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
    if colour:
        # Any hue: a kit's colour must say nothing about the number.
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(0, 180)) % 180
    hsv[..., 1] *= rng.uniform(0.7, 1.3)
    hsv[..., 2] *= rng.uniform(0.7, 1.3)
    image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8),
                         cv2.COLOR_HSV2BGR)
    if colour:
        image = image[:, :, rng.permutation(3)]
        if rng.random() < 0.2:
            image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                 cv2.COLOR_GRAY2BGR)
        if rng.random() < 0.3:
            image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.5, 1.2))
    return np.ascontiguousarray(image)


def evaluate(model, items, label: str, min_visible: float = jr.MIN_VISIBLE):
    """(right, wrong, read as none) over the numbered tracklets."""
    right = wrong = missed = false = none_ok = 0
    for _, number, files in items:
        pick = files if len(files) <= VOTE_CROPS else [
            files[i] for i in np.linspace(0, len(files) - 1,
                                          VOTE_CROPS).astype(int)]
        crops = [cv2.imread(str(f)) for f in pick]
        vis, logp = jr.read(model, [c for c in crops if c is not None])
        got, share, _ = jr.vote(vis, logp, min_visible)
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
    return right, wrong, missed


def mil_loss(v, t, u, numbers, k):
    """Each tracklet's label holds for *some* of its `k` crops: P(a number
    is visible) is the chance that any crop shows one, and P(number) is the
    crops' digit readings weighted by how visible each says the number
    is -- the training form of `jr.vote`. A crop of a player facing the
    camera is not asked to give the number, so the reader is not taught to
    tell a number from the player's look."""
    import torch
    import torch.nn.functional as F

    b = len(numbers)
    v, lt, lu = v.view(b, k), F.log_softmax(t, 1).view(b, k, 11), \
        F.log_softmax(u, 1).view(b, k, 10)
    log_none = F.logsigmoid(-v).sum(1)
    log_any = torch.log(-torch.expm1(log_none.clamp(max=-1e-6)))
    y = torch.tensor([float(n >= 0) for n in numbers])
    loss = -(y * log_any + (1 - y) * log_none).mean()
    idx = [i for i, n in enumerate(numbers) if n >= 0]
    if idx:
        tens = torch.tensor([jr.targets(numbers[i])[1] for i in idx])
        units = torch.tensor([jr.targets(numbers[i])[2] for i in idx])
        logw = F.log_softmax(F.logsigmoid(v[idx]), 1)
        lp = (logw + lt[idx, :, :].gather(2, tens.view(-1, 1, 1).expand(-1, k, 1)).squeeze(2)
              + lu[idx, :, :].gather(2, units.view(-1, 1, 1).expand(-1, k, 1)).squeeze(2))
        loss = loss - torch.logsumexp(lp, 1).mean()
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--test", nargs="*", default=[],
                    help="sets reported once, together, with the chosen "
                    "weights")
    ap.add_argument("--dev", nargs="*", default=[],
                    help="sets from other footage the weights are chosen "
                    "on (`dump_named_crops.py`)")
    ap.add_argument("--add", nargs="*", default=[],
                    help="sets from other footage to train on as well, "
                    "each tracklet `--add-repeat` times an epoch")
    ap.add_argument("--add-repeat", type=int, default=ADD_REPEAT)
    ap.add_argument("--init", help="weights to start from")
    ap.add_argument("--plain", action="store_true",
                    help="per-crop labels and no colour scrambling, as "
                    "first trained")
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
    dev = [t for d in args.dev for t in tracklets(Path(d))] or None
    added = [t for d in args.add for t in tracklets(Path(d))]
    train = train + added * args.add_repeat
    print(f"  {len(train)} training tracklets ({len(added)} from other "
          f"footage, x{args.add_repeat}), {len(val)} held out"
          + (f", {len(dev)} to choose on" if dev else ""), flush=True)

    model = jr.build()
    if args.init:
        model.load_state_dict(torch.load(args.init))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    bce = torch.nn.BCEWithLogitsLoss()
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    best = None
    k = MIL_CROPS
    for epoch in range(args.epochs):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        if args.plain:
            batch = []
            for _, number, files in train:
                for f in rng.choice(len(files), min(CROPS_PER_TRACKLET,
                                                    len(files)), replace=False):
                    batch.append((files[f], number))
            random.Random(epoch).shuffle(batch)
            steps = [batch[i:i + 64] for i in range(0, len(batch), 64)]
        else:
            order = list(train)
            random.Random(epoch).shuffle(order)
            steps = [order[i:i + 64 // k] for i in range(0, len(order), 64 // k)]
        for chunk in steps:
            if args.plain:
                crops, tv, tt, tu = [], [], [], []
                for path, number in chunk:
                    image = cv2.imread(str(path))
                    if image is None:
                        continue
                    crops.append(augment(image, rng, colour=False))
                    v, t, u = jr.targets(number)
                    tv.append(v), tt.append(t), tu.append(u)
                v, t, u = model(jr.to_tensor(crops))
                loss = (bce(v, torch.tensor(tv)) + ce(t, torch.tensor(tt))
                        + ce(u, torch.tensor(tu)))
            else:
                crops, numbers = [], []
                for _, number, files in chunk:
                    pick = rng.choice(len(files), k, replace=len(files) < k)
                    images = [cv2.imread(str(files[f])) for f in pick]
                    images = [im for im in images if im is not None]
                    if not images:
                        continue
                    while len(images) < k:
                        images.append(images[0])
                    crops += [augment(im, rng) for im in images]
                    numbers.append(number)
                v, t, u = model(jr.to_tensor(crops))
                loss = mil_loss(v, t, u, numbers, k)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(crops)
            seen += len(crops)
        sched.step()
        print(f"  epoch {epoch + 1}: loss {total / max(seen, 1):.3f}, "
              f"{time.time() - t0:.0f} s", flush=True)
        model.eval()
        if dev is not None:
            # Chosen on reads right less reads wrong (a wrong number joins
            # two players), with how sure a crop must be that a number
            # shows: trained this way a tracklet needs only one sure crop,
            # and the bar set for the first reader may suit it badly.
            score = None
            for mv in VISIBLE_BARS:
                right, wrong, _ = evaluate(model, dev, f"chosen on, bar {mv}", mv)
                if score is None or right - wrong > score:
                    score, bar = right - wrong, mv
        elif epoch % 3 == 2 or epoch == args.epochs - 1:
            right, wrong, _ = evaluate(model, val, "held out")
            score, bar = right - wrong, jr.MIN_VISIBLE
        else:
            continue
        if best is None or score > best:
            best, best_bar = score, bar
            torch.save(model.state_dict(), args.out)
            print(f"  saved {args.out} (bar {bar})", flush=True)
    model.load_state_dict(torch.load(args.out))
    model.eval()
    evaluate(model, val, "held out (same games as training)", best_bar)
    if args.test:
        test = [t for d in args.test for t in tracklets(Path(d))]
        evaluate(model, test, f"test ({len(args.test)} sets), bar {best_bar}",
                 best_bar)


if __name__ == "__main__":
    main()
