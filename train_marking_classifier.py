"""Learn what kind of marking a patch of pitch is showing.

The geometry has had five attempts at this and lost all of them. What is
left is to look at the picture: a centre circle sits in open grass with a
line through its middle, a penalty arc sits against the edge of a box with a
goal behind it. Obvious to the eye, and not a property of the arc's own
shape, which is all the previous attempts were allowed to measure.

## How it is scored, and why by clip

Held out by *clip*, never by crop. Crops from the same clip share a pitch, a
camera, a broadcaster's colour grading and often the same few metres of
paint; a model tested on crops from a clip it trained on is being asked
whether it can recognise a picture it has already seen. Splitting by clip
asks the only question worth asking -- whether it works on footage it has
never met.

The floor is the majority class, and it is a high floor: touchlines and goal
lines are most of what a football pitch has painted on it, so a model that
names the commonest class every time already scores well. Balanced accuracy
is reported alongside for that reason, and the two classes this exists for
are reported on their own, because they are rare and everything else
succeeding would hide them failing.

## Horizontal flips are free

The classes are symmetric by construction -- a touchline flipped is a
touchline -- so mirroring a crop gives another honest training example. That
is the same symmetry that made every attempt to name *ends* of the pitch
fail, finally being useful rather than fatal.

    python train_marking_classifier.py [--epochs 12] [--hold-out Veo]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import pitch_model as pm

CROPS = Path("marking_crops")
CLASSES = len(pm.MARKING_CLASSES)


class Small(nn.Module):
    """About twenty thousand parameters, which is what four threads allow."""

    def __init__(self, classes=CLASSES):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        return self.head(self.body(x).flatten(1))


def load():
    """Every clip's crops, kept apart so they can be held out whole."""
    out = {}
    for path in sorted(CROPS.glob("*.npz")):
        blob = np.load(path, allow_pickle=True)
        name = str(blob["clip"])
        out[name] = (blob["crops"], blob["labels"])
    return out


def as_tensors(crops, labels, flip=False):
    images = crops.astype(np.float32) / 255.0
    if flip:
        images = np.concatenate([images, images[:, :, ::-1]])
        labels = np.concatenate([labels, labels])
    images = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    return images, torch.from_numpy(labels)


def evaluate(model, images, labels):
    model.eval()
    predicted = []
    with torch.no_grad():
        for start in range(0, len(images), 512):
            predicted.append(model(images[start:start + 512]).argmax(1))
    predicted = torch.cat(predicted).numpy()
    truth = labels.numpy()

    overall = float((predicted == truth).mean())
    per_class, seen = {}, np.bincount(truth, minlength=CLASSES)
    for index in range(1, CLASSES):
        if seen[index]:
            here = truth == index
            per_class[pm.MARKING_CLASSES[index]] = (
                float((predicted[here] == index).mean()), int(seen[index]))
    balanced = (float(np.mean([value for value, _ in per_class.values()]))
                if per_class else float("nan"))
    majority = float((truth == np.bincount(truth,
                                           minlength=CLASSES).argmax()).mean())
    return overall, balanced, majority, per_class, predicted, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--hold-out", default=None,
                    help="clip to keep back; default is each in turn")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)

    data = load()
    if not data:
        print("no crops found -- run build_marking_crops.py first")
        return
    print("Held out by clip. The floor is naming the commonest class every "
          "time.\n")
    print(f"  {'held out':>14s} {'train':>7s} {'test':>6s} {'accuracy':>9s} "
          f"{'balanced':>9s} {'majority':>9s} {'circle':>8s} {'arc':>8s}")

    order = [args.hold_out] if args.hold_out else list(data)
    summary = []
    for held in order:
        if held not in data:
            continue
        train_crops = np.concatenate([data[k][0] for k in data if k != held])
        train_labels = np.concatenate([data[k][1] for k in data if k != held])
        images, labels = as_tensors(train_crops, train_labels, flip=True)
        test_images, test_labels = as_tensors(*data[held])

        counts = np.bincount(train_labels, minlength=CLASSES).astype(
            np.float32)
        # Cast explicitly: np.where against a Python float promotes the
        # whole thing to float64, and cross_entropy will not take a double
        # weight beside float activations.
        weight = torch.from_numpy(
            (np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
             / max(1, (counts > 0).sum())).astype(np.float32))

        model = Small()
        optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(args.epochs):
            model.train()
            shuffle = torch.randperm(len(images))
            for start in range(0, len(images), 128):
                batch = shuffle[start:start + 128]
                optimiser.zero_grad()
                loss = F.cross_entropy(model(images[batch]), labels[batch],
                                       weight=weight)
                loss.backward()
                optimiser.step()

        overall, balanced, majority, per_class, _, _ = evaluate(
            model, test_images, test_labels)
        circle = per_class.get("centre circle", (float("nan"), 0))
        arc = per_class.get("penalty arc", (float("nan"), 0))
        print(f"  {held:>14s} {len(images):7d} {len(test_images):6d} "
              f"{overall:8.0%} {balanced:8.0%} {majority:8.0%} "
              f"{circle[0]:7.0%} {arc[0]:7.0%}")
        summary.append((held, overall, balanced, majority, circle, arc))

    if summary:
        print(f"\n  {'mean':>14s} {'':7s} {'':6s} "
              f"{np.mean([s[1] for s in summary]):8.0%} "
              f"{np.nanmean([s[2] for s in summary]):8.0%} "
              f"{np.mean([s[3] for s in summary]):8.0%}")
        arcs = [s[5] for s in summary if s[5][1] > 0]
        if arcs:
            seen = sum(n for _, n in arcs)
            got = sum(value * n for value, n in arcs)
            print(f"\n  Penalty arcs: {got:.0f} of {seen} recognised "
                  f"({got / seen:.0%}), on clips never trained on.")

    print("\n  Beating the majority class is the minimum. The number that "
          "matters is\n  the last one: every geometric attempt at telling "
          "those two apart failed,\n  so anything well above chance there is "
          "new.")


if __name__ == "__main__":
    main()
