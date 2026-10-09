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

Fixed panoramas (SoccerTrack v2): `--crops` takes the crop sets written
by `build_reid_crops.py` for the training windows, pooled per match by
the dataset's persistent `player_id` (so a player's two halves, at
different depths, are one person), and `--val-crops` the held-out
windows. `--exclude` names ground-truth tables whose people must not be
trained on: the same players appear in several matches, so anyone in a
tuning or report window is dropped from training altogether, also as a
negative. Goalkeepers are left out (their kits make them trivial). The
crops keep their own size (mostly 30-95 px tall, as the pipeline cuts them),
with flips, blur by down-sizing and brightness changes each step; the
final weights are saved.

    python train_player_reid.py --crops W1.pkl W2.pkl ... --val-crops V.pkl \
        --exclude M2_gt.parquet ... --out .cache/player_reid_st2.pt
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


def load_crops(paths, exclude: set = frozenset(), pool: bool = True):
    """Clips from build_reid_crops.py sets: {person: (team, [crops])} per
    match when `pool` (a person's windows merged by player_id; team from
    the first window read, which only evaluation uses), else per window.
    Goalkeepers, people in `exclude` and people with fewer than K crops
    are dropped."""
    import pickle

    clips: dict[str, dict] = {}
    for path in paths:
        blob = pickle.loads(Path(path).read_bytes())
        clip = clips.setdefault(blob["match"] if pool else blob["window"], {})
        for pid, p in blob["people"].items():
            if p["role"] != "player" or pid in exclude:
                continue
            team, crops = clip.setdefault(pid, (p["side"], []))
            crops.extend(p["crops"])
    out = []
    for name, people in clips.items():
        people = {p: v for p, v in people.items() if len(v[1]) >= K}
        print(f"    {name}: {len(people)} people, "
              f"{sum(len(v[1]) for v in people.values())} crops", flush=True)
        out.append(people)
    return out


def augment(crop, rng):
    """A training view of a panorama crop: mirrored half the time, blurred
    by shrinking to between 25 px and its height and back (so sharpness,
    which goes with depth, does not tell people apart), brightness
    +-15%."""
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    h, w = crop.shape[:2]
    if h > 25:
        t = rng.uniform(25, h)
        small = cv2.resize(crop, (max(int(w * t / h), 4), int(t)),
                           interpolation=cv2.INTER_AREA)
        crop = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    crop = np.clip(crop.astype(np.float32) * rng.uniform(0.85, 1.15), 0, 255)
    return np.ascontiguousarray(crop.astype(np.uint8))


def excluded_ids(tables) -> set:
    """player_ids of everyone in the given ground-truth tables (window
    _gt.parquet or a half's GSR table)."""
    import pandas as pd

    ids = set()
    for t in tables:
        g = pd.read_parquet(t, columns=["player_id"])
        ids |= set(g.player_id.dropna().astype(str))
    return ids


def evaluate(model, clips, label):
    import torch

    model.eval()
    r1 = []
    aps = []
    apart = []
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
            apart.append(sim[who[:, None] != who[None, :]])
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
    # Collapse shows as everyone alike: the median cosine of different
    # people's crops near 1.
    print(f"  {label}: rank-1 among team-mates {np.mean(r1):.1%}, "
          f"mAP {np.mean(aps):.1%}, median cosine of different people "
          f"{np.median(np.concatenate(apart)) if apart else float('nan'):.2f}",
          flush=True)
    return float(np.mean(aps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", help="SoccerNet game-state clip dirs")
    ap.add_argument("--crops", nargs="+", help="build_reid_crops.py sets "
                                               "(fixed panoramas) to train on")
    ap.add_argument("--val-crops", nargs="+", help="held-out crop sets "
                                                   "(needed with --crops)")
    ap.add_argument("--disjoint", nargs="*", default=[],
                    help="crop sets (e.g. the report windows') none of whose "
                         "people may be trained on: checked, so a forgotten "
                         "--exclude fails loudly")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="ground-truth tables whose people are not trained on")
    ap.add_argument("--save-untrained", metavar="PATH",
                    help="write the untrained embedding (ImageNet ResNet-18 "
                         "body, seed-0 random head) to PATH and stop: on "
                         "SoccerTrack v2 nothing trained here beat it")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-all", action="store_true",
                    help="average the soft-margin loss over every triplet in "
                         "the batch instead of each anchor's hardest pair, "
                         "which is less prone to collapse")
    ap.add_argument("--soft-margin", action="store_true",
                    help="softplus(d+ - d-) instead of the 0.3 hinge: on "
                         "SoccerTrack v2 crops the hinge collapsed (every "
                         "crop to nearly one vector, the loss stuck at the "
                         "margin), as Hermans et al. 2017 report it can")
    ap.add_argument("--held-out", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", default=str(CACHE_DIR / "player_reid.pt"))
    args = ap.parse_args()
    import torch

    torch.manual_seed(0)
    random.seed(0)
    rng = np.random.default_rng(0)
    if args.save_untrained:
        torch.save(build().state_dict(), args.save_untrained)
        print(f"  saved {args.save_untrained}", flush=True)
        return
    t0 = time.time()
    if args.crops:
        if not args.val_crops:
            ap.error("--crops needs --val-crops")
        drop = excluded_ids(args.exclude)
        print(f"  training on {args.crops}\n  held out {args.val_crops}\n"
              f"  excluding {len(drop)} people from {args.exclude}",
              flush=True)
        train = load_crops(args.crops, drop)
        val = load_crops(args.val_crops, pool=False)
        import pickle

        seen = {p for c in train for p in c}
        # Everyone in the held-out and --disjoint windows, before any filter.
        for path in list(args.val_crops) + list(args.disjoint):
            people = set(pickle.loads(Path(path).read_bytes())["people"])
            assert not seen & people, f"people of {path} are in training"
        print(f"  {len(seen)} distinct people trained on", flush=True)
        clips = train + val
    else:
        dirs = sorted(p for p in Path(args.clips).iterdir() if p.is_dir())
        clips = [load_clip(d, rng) for d in dirs]
        val, train = clips[:args.held_out], clips[args.held_out:]
    print(f"  {len(clips)} clips, {sum(len(c) for c in clips)} people, "
          f"loaded in {time.time() - t0:.0f} s", flush=True)
    model = build()
    evaluate(model, val, "held out, ImageNet features only")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    best, t0 = -1.0, time.time()
    for step in range(1, args.steps + 1):
        model.train()
        people = random.choice(train)
        chosen = random.sample(list(people), min(P, len(people)))
        crops, labels = [], []
        for k, person in enumerate(chosen):
            for c in random.sample(people[person][1], K):
                crops.append(augment(c, rng) if args.crops else c)
                labels.append(k)
        e = model(to_tensor(crops))
        lab = torch.tensor(labels)
        d = torch.cdist(e, e)
        same = lab[:, None] == lab[None, :]
        hardest_pos = (d * same).max(1).values
        hardest_neg = (d + same * 10.0).min(1).values
        if args.batch_all:
            # every (anchor, positive, negative) triplet of the batch
            other = ~same
            pos = same & ~torch.eye(len(lab), dtype=torch.bool)
            t = d[:, :, None] - d[:, None, :]          # d(a,p) - d(a,n)
            valid = pos[:, :, None] & other[:, None, :]
            loss = torch.nn.functional.softplus(t[valid]).mean()
        elif args.soft_margin:
            loss = torch.nn.functional.softplus(hardest_pos - hardest_neg).mean()
        else:
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
            # Panoramas: the final weights, decided beforehand -- the
            # held-out window also tunes the thresholds, so it does not
            # pick the checkpoint too.
            if args.crops or score > best:
                best = score
                torch.save(model.state_dict(), args.out)
                print(f"  saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
