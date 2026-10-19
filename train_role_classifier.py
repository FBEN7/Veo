"""Tell referees and goalkeepers from players, by how they look.

Team assignment clusters kit colours and calls the two biggest clusters
the teams; everything else is "other". A referee in black next to a team
in dark kits falls into that team, and then wins tackles: on the first
round of event clicks, two of five false "tackles" went to the referee
standing by the ball (PLAYER_IDENTITY.md).

This learns the role from the crop -- player, goalkeeper, referee -- on
SoccerNet game-state clips (SN-GSR-2025, public), where every person's
role is labelled. Referees' kits vary between matches, so it learns from
many: trained on the training clips, tested on the validation clips,
whole matches it never saw. Classes are balanced (54 referees and 32
goalkeepers against 495 players across 25 clips), and a person's role is
the vote of their crops.

    python train_role_classifier.py --train .cache/gsr_train --test .cache/gsr
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
from train_player_reid import shrink, to_tensor

ROLES = {1: 0, 2: 1, 3: 2}          # player, goalkeeper, referee
NAMES = ("player", "goalkeeper", "referee")
PER_PERSON = 20


def build():
    import torch
    import torchvision

    net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    net.fc = torch.nn.Linear(512, len(NAMES))
    return net


def people(clip_dir: Path, rng):
    """[(role, [crops])] per person of one clip."""
    blob = json.loads((clip_dir / "Labels-GameState.json").read_text())
    file_of = {im["image_id"]: im["file_name"] for im in blob["images"]}
    by = {}
    for a in blob["annotations"]:
        if a.get("category_id") not in ROLES or "bbox_image" not in a:
            continue
        b = a["bbox_image"]
        if b["h"] < 30:
            continue
        by.setdefault(a["track_id"], (ROLES[a["category_id"]], []))[1].append(
            (file_of[a["image_id"]], b))
    wanted = {}
    out = {}
    for pid, (role, boxes) in by.items():
        pick = [boxes[i] for i in np.linspace(0, len(boxes) - 1, min(
            PER_PERSON, len(boxes))).astype(int)]
        for name, b in pick:
            wanted.setdefault(name, []).append((pid, b))
        out[pid] = (role, [])
    for name, items in wanted.items():
        image = cv2.imread(str(clip_dir / "img1" / name))
        if image is None:
            continue
        for pid, b in items:
            x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
            crop = image[max(y, 0):y + h, max(x, 0):x + w]
            if crop.size:
                out[pid][1].append(shrink(crop, rng).copy())
    return [v for v in out.values() if v[1]]


def vote(model, crops):
    """Mean class probabilities over a person's crops."""
    import torch

    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(to_tensor(crops)), 1).numpy()
    return p.mean(axis=0)


def evaluate(model, clips, label):
    conf = np.zeros((3, 3), dtype=int)
    for persons in clips:
        for role, crops in persons:
            conf[role, int(np.argmax(vote(model, crops)))] += 1
    print(f"  {label}: people by true role (rows) and predicted (columns)")
    for i, n in enumerate(NAMES):
        print(f"    {n:10s} " + "  ".join(f"{conf[i, j]:4d}" for j in range(3)))
    ref_recall = conf[2, 2] / max(conf[2].sum(), 1)
    ref_precision = conf[2, 2] / max(conf[:, 2].sum(), 1)
    print(f"    referees: {conf[2, 2]}/{conf[2].sum()} found ({ref_recall:.0%}), "
          f"{conf[:, 2].sum() - conf[2, 2]} others called referee; "
          f"goalkeepers {conf[1, 1]}/{conf[1].sum()}", flush=True)
    return ref_recall * ref_precision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", default=str(CACHE_DIR / "role_classifier.pt"))
    args = ap.parse_args()
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    train = [people(d, rng) for d in sorted(Path(args.train).iterdir())
             if d.is_dir()]
    test = [people(d, rng) for d in sorted(Path(args.test).iterdir())
            if d.is_dir()]
    print(f"  {sum(len(c) for c in train)} training people, "
          f"{sum(len(c) for c in test)} test people, loaded in "
          f"{time.time() - t0:.0f} s", flush=True)
    pool = {r: [c for clip in train for role, crops in clip if role == r
                for c in crops] for r in range(3)}
    model = build()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    for step in range(1, args.steps + 1):
        model.train()
        # Balanced: the same number of crops of each role per batch.
        crops, labels = [], []
        for r in range(3):
            for c in random.sample(pool[r], min(16, len(pool[r]))):
                crops.append(c)
                labels.append(r)
        loss = loss_fn(model(to_tensor(crops)), torch.tensor(labels))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0:
            print(f"  step {step}: loss {loss.item():.3f}, "
                  f"{time.time() - t0:.0f} s", flush=True)
    torch.save(model.state_dict(), args.out)
    evaluate(model, test, "test clips (matches never seen)")
    print(f"  saved {args.out}")


if __name__ == "__main__":
    main()
