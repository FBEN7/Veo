"""Train an appearance embedding that tells team-mates apart.

Joining a clip's tracks into people needs to know whether two tracks of
the same team -- same shirt, same shorts -- are the same person. A shirt
number settles it when it can be read, which on 720p broadcast is rarely.
The rest is in the details a person carries all clip long: build, skin,
hair, boots, sleeves, socks.

Trained on SoccerNet game-state clips (SN-GSR-2025, public), where every
person has one id for the whole 30 s clip. Each batch is drawn from one
clip -- P people, K crops each -- so the negatives are the people a
track could be confused with, team-mates in the same kit, not the other
team. Batch-hard triplet loss on a 128-d embedding. Crops are shrunk to
the sizes players have in this footage (40-110 px tall).

Scored on held-out clips as it is used: per clip, each person's crops
against every other person's of the same team -- how often the nearest
crop is the same person (rank-1), and the mean average precision.

    python train_player_reid.py --clips .cache/gsr_train --held-out 3
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from src.paths import CACHE_DIR

SIZE = (64, 128)
PER_PERSON = 30
P, K = 8, 4
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build(dim: int = 128):
    import torch
    import torchvision

    class Embed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            net.fc = torch.nn.Identity()
            self.body = net
            self.head = torch.nn.Linear(512, dim)

        def forward(self, x):
            return torch.nn.functional.normalize(self.head(self.body(x)), dim=1)

    return Embed()


def to_tensor(crops):
    import torch

    out = []
    for c in crops:
        c = cv2.resize(c, SIZE, interpolation=cv2.INTER_AREA
                       if c.shape[0] > SIZE[1] else cv2.INTER_LINEAR)
        c = (c[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        out.append(c.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(out))


def shrink(crop, rng):
    h, w = crop.shape[:2]
    target = rng.uniform(40, 110)
    if h > target:
        s = target / h
        crop = cv2.resize(crop, (max(int(w * s), 4), int(target)),
                          interpolation=cv2.INTER_AREA)
    return crop


def load_clip(clip_dir: Path, rng):
    """{person: (team, [crops])} for one game-state clip."""
    blob = json.loads((clip_dir / "Labels-GameState.json").read_text())
    file_of = {im["image_id"]: im["file_name"] for im in blob["images"]}
    by = {}
    for a in blob["annotations"]:
        if a.get("category_id") not in (1, 2) or "bbox_image" not in a:
            continue
        b = a["bbox_image"]
        if b["h"] < 30:
            continue
        team = (a.get("attributes") or {}).get("team")
        by.setdefault(a["track_id"], (team, []))[1].append(
            (file_of[a["image_id"]], b))
    out, wanted = {}, {}
    for person, (team, boxes) in by.items():
        pick = [boxes[i] for i in np.linspace(0, len(boxes) - 1,
                                              min(PER_PERSON, len(boxes))
                                              ).astype(int)]
        for name, b in pick:
            wanted.setdefault(name, []).append((person, b))
        out[person] = (team, [])
    for name, items in wanted.items():
        image = cv2.imread(str(clip_dir / "img1" / name))
        if image is None:
            continue
        for person, b in items:
            x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
            crop = image[max(y, 0):y + h, max(x, 0):x + w]
            if crop.size:
                # A copy: a slice keeps the whole 1080p frame alive, and
                # 9,000 of them filled 12 GB.
                out[person][1].append(shrink(crop, rng).copy())
    return {p: v for p, v in out.items() if len(v[1]) >= K}


def evaluate(model, clips, label):
    import torch

    model.eval()
    r1 = []
    aps = []
    with torch.no_grad():
        for people in clips:
            feats, who, team = [], [], []
            for person, (t, crops) in people.items():
                e = model(to_tensor(crops)).numpy()
                feats.append(e)
                who += [person] * len(e)
                team += [t] * len(e)
            f = np.concatenate(feats)
            who, team = np.array(who), np.array(team, dtype=object)
            sim = f @ f.T
            for i in range(len(f)):
                mask = (team == team[i]) & (np.arange(len(f)) != i)
                if not mask.any():
                    continue
                s, same = sim[i, mask], who[mask] == who[i]
                if not same.any():
                    continue
                order = np.argsort(-s)
                r1.append(bool(same[order[0]]))
                hits = np.cumsum(same[order])
                prec = hits / np.arange(1, len(order) + 1)
                aps.append(float((prec * same[order]).sum() / same.sum()))
    print(f"  {label}: rank-1 among team-mates {np.mean(r1):.1%}, "
          f"mAP {np.mean(aps):.1%}", flush=True)
    return float(np.mean(aps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--held-out", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", default=str(CACHE_DIR / "player_reid.pt"))
    args = ap.parse_args()
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dirs = sorted(p for p in Path(args.clips).iterdir() if p.is_dir())
    t0 = time.time()
    clips = [load_clip(d, rng) for d in dirs]
    print(f"  {len(clips)} clips, {sum(len(c) for c in clips)} people, "
          f"loaded in {time.time() - t0:.0f} s", flush=True)
    val, train = clips[:args.held_out], clips[args.held_out:]
    model = build()
    evaluate(model, val, "held out, ImageNet features only")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    best, t0 = -1.0, time.time()
    for step in range(1, args.steps + 1):
        model.train()
        people = random.choice(train)
        chosen = random.sample(list(people), min(P, len(people)))
        crops, labels = [], []
        for k, person in enumerate(chosen):
            for c in random.sample(people[person][1], K):
                crops.append(c)
                labels.append(k)
        e = model(to_tensor(crops))
        lab = torch.tensor(labels)
        d = torch.cdist(e, e)
        same = lab[:, None] == lab[None, :]
        hardest_pos = (d * same).max(1).values
        hardest_neg = (d + same * 10.0).min(1).values
        loss = torch.relu(hardest_pos - hardest_neg + 0.3).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0:
            print(f"  step {step}: loss {loss.item():.3f}, "
                  f"{time.time() - t0:.0f} s", flush=True)
        if step % 1000 == 0 or step == args.steps:
            score = evaluate(model, val, "held out")
            if score > best:
                best = score
                torch.save(model.state_dict(), args.out)
                print(f"  saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
