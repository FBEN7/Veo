"""Train the panorama ball detector (`src/ball_heatmap.py`) on weak labels
(`build_ball_patches.py`, then `build_ball_labels.py`).

Each patch has a code per 4 x 4 cell (1 the ball is somewhere here, 0
ignore, -1 not the ball, -3 not the ball and easily taken for it).
Multiple-instance loss: per positive bag, a smooth maximum of the logits
over its region (log-mean-exp, r = 5) should be high; negative cells low,
by focal loss (gamma 2), hard ones weighted 3, summed over the batch and
divided by its number of bags. Between-action bags carry no negatives and
half the weight. From step 1000 the 15% action bags and 25% between bags
with the highest loss in a batch are dropped -- a hidden ball makes a bag
wrong, and those are the hardest.

Batches of 16: 6 action bags, 2 between, 8 ball-free patches. Flips (the
codes too), brightness, contrast, saturation, blur, JPEG re-encoding. The
stem and layer1 are frozen for the first 800 steps. Checkpoints at 1500,
2250 and 3000 steps (`<out>.<step>.pt`), chosen afterwards on tuning
windows by `eval_ball_detector.py`.

    python train_ball_heatmap.py --patches .cache/soccertrack_v2/ball_patches \\
        --out .cache/ball_heatmap
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.ball_heatmap import build, to_tensor

MIX = {"action": 6, "between": 2, "negative": 8}
CHECKPOINTS = (1500, 2250, 3000)


def augment(image, codes, rng):
    if rng.random() < 0.5:
        image, codes = image[:, ::-1], codes[:, ::-1]
    im = image.astype(np.float32)
    im = (im - im.mean()) * rng.uniform(0.75, 1.25) + im.mean() * rng.uniform(0.75, 1.25)
    hsv = cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32)
                           * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if rng.random() < 0.2:
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.3:
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY,
                                            int(rng.integers(60, 96))])
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(im), np.ascontiguousarray(codes)


def bag_losses(logits, codes, positive_only):
    """Per patch: the bag loss (nan without a bag) and the negative loss."""
    import torch

    F = torch.nn.functional
    pos, neg = [], []
    for lg, c, po in zip(logits, codes, positive_only):
        inside = c == 1
        if inside.any():
            z = torch.logsumexp(5 * lg[inside], 0) / 5 - np.log(
                float(inside.sum())) / 5
            pos.append(F.binary_cross_entropy_with_logits(
                z, torch.ones_like(z)))
        else:
            pos.append(torch.tensor(float("nan")))
        if po:
            neg.append(torch.zeros(()))
            continue
        w = torch.where(c == -3, 3.0, torch.where(c == -1, 1.0, 0.0))
        p = torch.sigmoid(lg)
        bce = F.binary_cross_entropy_with_logits(lg, torch.zeros_like(lg),
                                                 reduction="none")
        neg.append((bce * p ** 2 * w).sum())
    return torch.stack(pos), torch.stack(neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", required=True, help="checkpoint prefix")
    ap.add_argument("--exclude-matches", nargs="*", default=["117092"],
                    help="never trained on (M1: another camera)")
    args = ap.parse_args()
    import torch

    torch.manual_seed(0)
    random.seed(0)
    rng = np.random.default_rng(0)
    root = Path(args.patches)
    table = pd.read_parquet(root / "labels.parquet")
    codes_all = np.load(root / "labels_codes.npy")
    table["match"] = table.match.astype(str)
    keep = ~table.match.isin(args.exclude_matches)
    table, codes_all = table[keep].reset_index(drop=True), codes_all[keep.to_numpy()]
    pools = {k: np.flatnonzero(table.kind.to_numpy() == k) for k in MIX}
    print(f"  {sorted(table.match.unique())}: " + ", ".join(
        f"{k} {len(v)}" for k, v in pools.items()), flush=True)
    torch.set_num_threads(4)
    model = build()
    early = list(model.stem.parameters()) + list(model.layer1.parameters())
    for p in early:
        p.requires_grad_(False)
    opt = torch.optim.AdamW([
        {"params": list(model.lat.parameters()) + list(model.head.parameters()),
         "lr": 3e-4},
        {"params": model.layer2.parameters(), "lr": 1e-4},
        {"params": early, "lr": 3e-5}], weight_decay=1e-4)
    base = [g["lr"] for g in opt.param_groups]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if step == 801:
            for p in early:
                p.requires_grad_(True)
        scale = min(1.0, step / 100) * 0.5 * (1 + np.cos(np.pi * step / args.steps))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * scale
        model.train()
        idx = np.concatenate([rng.choice(pools[k], n) for k, n in MIX.items()])
        ims, cs = [], []
        for i in idx:
            im, c = augment(cv2.imread(str(root / table.file[i])),
                            codes_all[i], rng)
            ims.append(im)
            cs.append(torch.from_numpy(c.astype(np.int64)))
        logits = model(to_tensor(ims))
        pos, neg = bag_losses(logits, cs, table.positive_only.to_numpy()[idx])
        kinds = table.kind.to_numpy()[idx]
        w = torch.ones(len(idx))
        w[torch.from_numpy(kinds == "between")] = 0.5
        has = ~torch.isnan(pos)
        if step > 1000:
            for kind, share in (("action", 0.15), ("between", 0.25)):
                m = torch.from_numpy(kinds == kind) & has
                n = int(round(share * int(m.sum())))
                if n:
                    worst = torch.topk(torch.where(m, pos.detach(),
                                                   torch.tensor(-1.0)), n).indices
                    has[worst] = False
        n_bags = max(int(has.sum()), 1)
        loss = ((pos[has] * w[has]).sum() + neg.sum()) / n_bags
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  step {step}: loss {loss.item():.3f}  "
                  f"[{time.time() - t0:.0f} s]", flush=True)
        if step in CHECKPOINTS or step == args.steps:
            torch.save(model.state_dict(), f"{args.out}.{step}.pt")
            print(f"  saved {args.out}.{step}.pt", flush=True)


if __name__ == "__main__":
    main()
