"""A ball detector for fixed panoramas, trained on weak labels
(`build_ball_patches.py`): each patch says only that the ball is within a
radius of a point, or hidden.

A small fully convolutional net (ResNet-18 to stride 8, with its stride-4
features added back) gives a ball logit per 4 x 4 pixels. Multiple-instance
loss: the highest logit inside the label disk should be high -- the ball is
somewhere there -- and every cell further than the radius plus `MARGIN`
should be low, there being one ball. Negatives are everything else in the
patch: other players' feet, socks, lines.

`--val-match` is held out to choose nothing but report: the share of
patches whose highest peak falls inside the disk (`top-1 in disk`), at
actions and between them.

    python train_ball_heatmap.py --patches .cache/soccertrack_v2/ball_patches \\
        --val-match 128058 --out .cache/ball_heatmap.pt
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

STRIDE = 4
MARGIN = 16.0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build():
    import torch
    import torchvision

    class BallNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            r = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            self.stem = torch.nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
            self.layer1, self.layer2 = r.layer1, r.layer2       # /4, /8
            self.lat = torch.nn.Conv2d(128, 64, 1)
            self.head = torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU(),
                torch.nn.Conv2d(64, 1, 1))
            torch.nn.init.constant_(self.head[-1].bias, -4.0)

        def forward(self, x):
            c1 = self.layer1(self.stem(x))
            c2 = self.layer2(c1)
            up = torch.nn.functional.interpolate(self.lat(c2), size=c1.shape[2:],
                                                 mode="nearest")
            return self.head(c1 + up)[:, 0]                     # logits /4

    return BallNet()


def to_tensor(images):
    import torch

    x = np.stack([(im[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
                  for im in images])
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


def disk_masks(rows, shape):
    """Per patch: cells inside the label disk, and cells far enough out to
    be negatives."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (yy + 0.5) * STRIDE, (xx + 0.5) * STRIDE
    inside, outside = [], []
    for r in rows:
        d = np.hypot(cx - r.u, cy - r.v)
        inside.append(d <= r.radius)
        outside.append(d > r.radius + MARGIN)
    return np.stack(inside), np.stack(outside)


def augment(image, row, rng):
    u, v = row.u, row.v
    if rng.random() < 0.5:
        image = image[:, ::-1]
        u = image.shape[1] - u
    image = np.clip(image.astype(np.float32) * rng.uniform(0.8, 1.2)
                    + rng.uniform(-15, 15), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image), row._replace(u=u, v=v)


def loss_fn(logits, inside, outside):
    import torch

    F = torch.nn.functional
    pos = torch.stack([lg[m].max() for lg, m in zip(logits, inside)])
    pos_loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
    neg = logits[outside]
    p = torch.sigmoid(neg)
    neg_loss = (F.binary_cross_entropy_with_logits(
        neg, torch.zeros_like(neg), reduction="none") * p ** 2).sum() \
        / max(len(neg), 1) * 50.0
    return pos_loss + neg_loss


def top1_in_disk(model, table, root, batch: int = 32) -> dict:
    """Share of patches whose highest cell lies inside the label disk."""
    import torch

    model.eval()
    hits = {"action": [], "between": []}
    with torch.no_grad():
        for i in range(0, len(table), batch):
            rows = list(table.iloc[i:i + batch].itertuples())
            ims = [cv2.imread(str(root / r.file)) for r in rows]
            lg = model(to_tensor(ims)).numpy()
            for r, m in zip(rows, lg):
                k = np.unravel_index(m.argmax(), m.shape)
                y, x = (k[0] + 0.5) * STRIDE, (k[1] + 0.5) * STRIDE
                hits[r.kind].append(np.hypot(x - r.u, y - r.v) <= r.radius)
    return {k: round(float(np.mean(v)), 3) for k, v in hits.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True)
    ap.add_argument("--val-match", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch

    torch.manual_seed(0)
    random.seed(0)
    rng = np.random.default_rng(0)
    root = Path(args.patches)
    table = pd.read_parquet(root / "patches.parquet")
    table["match"] = table.match.astype(str)
    val = table[table.match == args.val_match]
    train = table[table.match != args.val_match].reset_index(drop=True)
    print(f"  training on {sorted(train.match.unique())}: {len(train)} patches "
          f"({train.kind.value_counts().to_dict()}); held out "
          f"{args.val_match}: {len(val)}", flush=True)
    model = build()
    print(f"  untrained: top-1 in disk {top1_in_disk(model, val, root)}",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        rows, ims = [], []
        for k in rng.integers(0, len(train), args.batch):
            r = train.iloc[int(k)]
            im, r = augment(cv2.imread(str(root / r.file)),
                            next(train.iloc[[int(k)]].itertuples()), rng)
            ims.append(im)
            rows.append(r)
        logits = model(to_tensor(ims))
        inside, outside = disk_masks(rows, logits.shape[1:])
        loss = loss_fn(logits, torch.from_numpy(inside),
                       torch.from_numpy(outside))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0:
            print(f"  step {step}: loss {loss.item():.3f}  "
                  f"[{time.time() - t0:.0f} s]", flush=True)
        if step % 1000 == 0 or step == args.steps:
            print(f"  held out: top-1 in disk "
                  f"{top1_in_disk(model, val, root)}", flush=True)
            torch.save(model.state_dict(), args.out)
    print(f"  saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
