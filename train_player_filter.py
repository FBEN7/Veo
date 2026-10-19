#!/usr/bin/env python3
"""Train a filter that tells a footballer from everyone else on the touchline.

A quarter of the tracks the pipeline calls players are stewards in hi-vis,
staff in dark coats, spectators and one advertising hoarding. Rejecting them
by distance to the nearer kit centre gets purity to 0.96 and 0.89 on the two
hand-labelled matches; 15% of passer/receiver slots still land on someone who
is not a player. See `TEAM_ASSIGNMENT.md`.

The distance rule is *relative*: it knows only that a steward looks unlike
both kits. Something trained on labelled football would know absolutely that
hi-vis is never a kit.

**A colour classifier is not that something**, and that is measured rather
than assumed. `experiment_player_classifier.py` trains one on hue-saturation
histograms from one match's hand-read labels and tests it on the other's: it
fits its training match at AUC 1.00 and transfers at 0.65 and 0.87, against
the distance rule's 0.91 and 0.97. Absolute colour memorises kits. So this
model sees the crop itself and can learn shape and texture, not just
what colour it is.

Input is the archive `prepare_player_crops.py` writes on a machine that can
reach Roboflow, whose football-players dataset is CC BY 4.0. Its negatives
are mined with the same COCO detector this pipeline deploys, keeping every
`person` detection the annotators left unlabelled -- which is precisely the
population that contaminates the roster.

    python train_player_filter.py --crops player_crops.npz

The result is judged on transfer, not on held-out accuracy within the
dataset. The number that matters is whether it separates non-players on our
own footage, which `--kit-labels` scores against the hand-read labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

CROP_W, CROP_H = 32, 64
DEFAULT_WEIGHTS = Path("models/player_filter.pt")


class SmallNet(nn.Module):
    """Three convolutions and a linear head.

    Deliberately small. There are a few tens of thousands of crops and four
    CPU cores; a larger model would take longer to overfit and no longer to
    be wrong.
    """

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 1)),
        )
        self.head = nn.Linear(64 * 2, 1)

    def forward(self, x):
        return self.head(self.body(x).flatten(1)).squeeze(1)


def to_tensor(crops: np.ndarray) -> torch.Tensor:
    """BGR uint8 (N, H, W, 3) -> float tensor (N, 3, H, W) in 0-1."""
    x = torch.from_numpy(crops[..., ::-1].copy()).float().div_(255.0)
    return x.permute(0, 3, 1, 2).contiguous()


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    pos, neg = scores[positive], scores[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def train(crops, labels, sources, epochs=12, seed=0, quiet=False):
    """Fit the net; hold out the dataset's own validation split if it has one."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    held = np.isin(sources, ("valid", "test"))
    if held.sum() < 20 or (~held).sum() < 20:
        # No usable split in the archive, so make one. Random rather than by
        # image, which is optimistic: crops from the same frame land on both
        # sides. The transfer score below is the honest number regardless.
        held = rng.random(len(labels)) < 0.2

    Xtr, ytr = to_tensor(crops[~held]), torch.from_numpy(labels[~held]).float()
    Xva, yva = to_tensor(crops[held]), labels[held].astype(bool)

    model = SmallNet()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    # Participants outnumber the people the annotators ignored, so the loss is
    # weighted rather than the data resampled.
    pos = float(ytr.sum())
    weight = torch.tensor([(len(ytr) - pos) / max(pos, 1.0)])
    lossfn = nn.BCEWithLogitsLoss(pos_weight=weight)

    n, batch = len(Xtr), 256
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            xb = Xtr[idx]
            # Mirroring is the one augmentation that is certainly safe here:
            # a footballer facing left is a footballer.
            if rng.random() < 0.5:
                xb = torch.flip(xb, dims=[3])
            opt.zero_grad()
            loss = lossfn(model(xb), ytr[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)

        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(Xva)).numpy()
        if not quiet:
            print(f"  epoch {epoch + 1:2d}  loss {total / n:.4f}  "
                  f"held-out AUC {auc(s, yva):.3f}")

    model.eval()
    with torch.no_grad():
        s = torch.sigmoid(model(Xva)).numpy()
    return model, auc(s, yva)


def score_tracks(model, samples_crops: dict[int, np.ndarray]) -> dict[int, float]:
    """Median participant-probability per track."""
    out = {}
    model.eval()
    with torch.no_grad():
        for tid, arr in samples_crops.items():
            if arr is None or not len(arr):
                continue
            s = torch.sigmoid(model(to_tensor(arr))).numpy()
            out[int(tid)] = float(np.median(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="player_crops.npz")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--kit-labels", action="store_true",
                    help="also score transfer onto the hand-read kit labels")
    args = ap.parse_args()

    path = Path(args.crops)
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Produce it with prepare_player_crops.py on a "
            "machine that can reach roboflow.com, then upload it -- it is a "
            "few megabytes, unlike the dataset.")

    d = np.load(path, allow_pickle=True)
    crops, labels = d["crops"], d["labels"].astype(np.uint8)
    sources = d["sources"] if "sources" in d else np.array(["train"] * len(labels))
    print(f"{len(crops)} crops: {int(labels.sum())} participants, "
          f"{int((1 - labels).sum())} others")

    model, va = train(crops, labels, sources, epochs=args.epochs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"\nheld-out AUC within the dataset: {va:.3f}")
    print(f"weights written to {args.out}")

    if args.kit_labels:
        transfer_report(model)


def transfer_report(model):
    """The number that matters: does it separate on our own footage?"""
    import pandas as pd
    from analyse_pass_outcome import WINDOWS
    from sweep_roster_rejection import LABELLED, load as load_window

    print("\nTransfer onto the hand-read kit labels "
          "(positive = IS a player; AUC against the distance rule):")
    for w in LABELLED:
        name, df, _, kit, nonplayers = load_window(w)
        out_dir = next(x[1] for x in WINDOWS if x[0] == w)
        clip = json.loads((Path(out_dir) / "clip.json").read_text())["path"]
        raw = pd.read_parquet(Path(out_dir) / "tracks_grass.parquet")
        by_track = crops_for_tracks(clip, raw, set(kit) | set(nonplayers))
        scored = score_tracks(model, by_track)

        tids = [t for t in scored if t in kit or t in nonplayers]
        if len(tids) < 8:
            continue
        s = np.array([scored[t] for t in tids])
        is_player = np.array([t in kit for t in tids])
        resid = np.array([float(df.loc[t, "residual"]) for t in tids])
        print(f"  {name:18s} classifier {auc(s, is_player):.2f}   "
              f"distance rule {auc(-resid, is_player):.2f}   (n={len(tids)})")


def crops_for_tracks(clip: str, tracks, wanted: set[int], per_track: int = 8):
    """Full-body crops for the given tracks, one sequential pass of the video."""
    from collections import defaultdict

    players = tracks[tracks.cls == "player"]
    want = defaultdict(list)
    for tid, g in players.groupby("track_id"):
        if int(tid) not in wanted or len(g) < 5:
            continue
        g = g.sort_values("frame")
        take = g.iloc[np.linspace(0, len(g) - 1,
                                  min(per_track, len(g))).astype(int)]
        for r in take.itertuples():
            want[int(r.frame)].append(
                (int(r.track_id), float(r.px), float(r.py), float(r.crop_h)))

    got = defaultdict(list)
    cap = cv2.VideoCapture(clip)
    idx, last = 0, (max(want) if want else -1)
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, px, py, h in want.get(idx, ()):
            w = h * 0.4
            x1, x2 = int(px - w / 2), int(px + w / 2)
            y1, y2 = int(py - h), int(py)
            H, W = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 >= 4 and y2 - y1 >= 12:
                got[tid].append(cv2.resize(frame[y1:y2, x1:x2],
                                           (CROP_W, CROP_H),
                                           interpolation=cv2.INTER_AREA))
        idx += 1
    cap.release()
    return {t: np.stack(v) for t, v in got.items() if v}


if __name__ == "__main__":
    main()
