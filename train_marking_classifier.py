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

## Crop size decides this, and it decides it differently per question

With 24 m of pitch per crop instead of 9, the distinction that defeated
every geometric attempt becomes learnable. Balanced, so chance is 50%:

    held out   9 m crops   24 m crops
    w1               72%          98%
    reading          50%          50%
    w2               50%          50%
    w3               50%          98%
    mean             55%          74%

Repeating each fold with four initialisations, reporting every one:

    held out   seed 0   seed 1   seed 2   seed 3
    w1            98%      97%     100%      99%
    w3            98%     100%      97%      51%
    w2            52%      53%      84%      50%
    reading       52%      50%      50%      50%

That separates instability from inability, and the answer is both. On w1 it
works on every seed and works nearly perfectly, which is what a real signal
looks like rather than a lucky draw. On w3 it works on three seeds of four.
On w2 it mostly collapses and once reaches 84%. On reading it never leaves
chance.

So the distinction is learnable and does not yet transfer to every clip.
Two folds near perfect is the finding. A centre circle and a penalty arc are
separable from the picture, and the missing ingredient was context: 9 m
around an arc pixel does not reach the penalty-area line that cuts it, and
24 m does. The other two folds collapse to naming one class for every input.

The same change makes the NINE-class problem worse, 61% to 50% against a 60%
floor. That is not a contradiction. A 24 m crop centred on a touchline pixel
contains a great deal that is not the touchline, so a fine-grained "which
line is this" gets diluted, while circle-against-arc is entirely a question
about surroundings. The right footprint depends on the question, and one
classifier for both was the wrong shape.

## What was measured before, at 9 m



Nine classes, held out by clip:

    held out   accuracy   majority   circle    arc
    w1              78%        61%      84%    11%
    reading         50%        61%      81%     0%
    w2              72%        55%      75%    61%
    w3              81%        50%      82%    79%
    Veo             24%        74%       0%      -
    mean            61%        60%

On the majority floor. Balanced accuracy 37%. The class it exists for swings
between 0% and 79% depending on which clip is held out, which is a model
that sometimes happens to match rather than one that has learned. Veo is
worst and matters most: 24% against a 74% floor, not one centre circle
recognised -- four clips of 720p broadcast do not transfer to 640x360.

Reduced to the single distinction, balanced so chance is exactly 50%:

    held out   accuracy   circle    arc
    w1              72%       43%   100%
    reading         50%      100%     0%
    w2              50%      100%     0%
    w3              50%      100%     0%
    mean            55%

Three folds of four land on exactly 50% with a 100%/0% split, which is a
model predicting one class for every input. It has not learned the
difference; it has learned which guess is safer.

## Why, and what would change it

The binding constraint is the number of penalty arcs: 217 crops in total,
about 350 per class once balanced. That is very little for a convolutional
network, and it is not an accident of sampling. The labels come from
anchored frames, a confident anchor needs a centre circle, and a centre
circle is at midfield -- so the labelled set is midfield-biased by
construction. Borrowing anchors from neighbouring frames took penalty arcs
from zero to 217 and cannot take them much further, because propagation
reaches only as far as the camera went.

Not established, and worth trying before concluding the idea is wrong: a
larger crop, since 160 px at 720p may not reach from an arc to the box edge
that distinguishes it, and a larger model, since this one has about twenty
thousand parameters. Neither was explored, because with 350 examples a
class the data is the limit either way.

What would actually change it is more footage. For the product that means
more Veo matches -- this clip contributes zero penalty arcs, so a model
trained on Veo alone, which is what the licence permits, currently has
nothing to learn the class from at all.

    python train_marking_classifier.py [--epochs 12] [--binary]
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
    ap.add_argument("--seeds", type=int, default=1,
                    help="repeat each fold with different initialisations; "
                         "reported as a mean and a spread, never selected on")
    ap.add_argument("--binary", action="store_true",
                    help="centre circle against penalty arc only, balanced, "
                         "which is the one distinction this exists for")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)

    data = load()
    if not data:
        print("no crops found -- run build_marking_crops.py first")
        return

    if args.binary:
        # Nine classes with this much imbalance can hide the only question
        # that matters. Reduced to centre circle against penalty arc, with
        # the two balanced by subsampling, chance is exactly 50% and there is
        # nowhere for a majority class to hide.
        circle = pm.MARKING_CLASS_INDEX["centre circle"]
        arc = pm.MARKING_CLASS_INDEX["penalty arc"]
        picker = np.random.default_rng(0)
        reduced = {}
        for name, (crops, labels) in data.items():
            keep = np.isin(labels, (circle, arc))
            crops, labels = crops[keep], labels[keep]
            if len(np.unique(labels)) < 2:
                continue
            smallest = min((labels == value).sum() for value in (circle, arc))
            chosen = np.concatenate([
                picker.choice(np.flatnonzero(labels == value), smallest,
                              replace=False) for value in (circle, arc)])
            reduced[name] = (crops[chosen],
                             (labels[chosen] == arc).astype(np.int64))
        data = reduced
        if not data:
            print("no clip has both classes")
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

        width = 2 if args.binary else CLASSES
        counts = np.bincount(train_labels, minlength=width).astype(np.float32)
        # Cast explicitly: np.where against a Python float promotes the
        # whole thing to float64, and cross_entropy will not take a double
        # weight beside float activations.
        weight = torch.from_numpy(
            (np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
             / max(1, (counts > 0).sum())).astype(np.float32))

        runs = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = Small(2 if args.binary else CLASSES)
            optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
            for _ in range(args.epochs):
                model.train()
                shuffle = torch.randperm(len(images))
                for start in range(0, len(images), 128):
                    batch = shuffle[start:start + 128]
                    optimiser.zero_grad()
                    loss = F.cross_entropy(model(images[batch]),
                                           labels[batch], weight=weight)
                    loss.backward()
                    optimiser.step()
            runs.append(model)

        # Every seed is reported. Picking the best on the held-out clip would
        # be choosing a model by the answer, which is the thing this whole
        # file exists to avoid.
        if args.seeds > 1 and args.binary:
            scores = []
            for candidate in runs:
                candidate.eval()
                with torch.no_grad():
                    guess = candidate(test_images).argmax(1).numpy()
                scores.append(float((guess == test_labels.numpy()).mean()))
            print(f"  {held:>14s} {len(images):7d} {len(test_images):6d} "
                  f"{np.mean(scores):8.0%} {'':8s} {0.5:8.0%} "
                  f"  seeds " + " ".join(f"{s:.0%}" for s in scores))
            summary.append((held, float(np.mean(scores)), float("nan"), 0.5,
                            (float("nan"), 0), (float("nan"), 0)))
            continue
        model = runs[0]

        if args.binary:
            model.eval()
            with torch.no_grad():
                guess = model(test_images).argmax(1).numpy()
            truth = test_labels.numpy()
            overall = float((guess == truth).mean())
            circle_right = float((guess[truth == 0] == 0).mean())
            arc_right = float((guess[truth == 1] == 1).mean())
            print(f"  {held:>14s} {len(images):7d} {len(test_images):6d} "
                  f"{overall:8.0%} {'':8s} {0.5:8.0%} "
                  f"{circle_right:7.0%} {arc_right:7.0%}")
            summary.append((held, overall, float("nan"), 0.5,
                            (circle_right, int((truth == 0).sum())),
                            (arc_right, int((truth == 1).sum()))))
            continue

        overall, balanced, majority, per_class, _, _ = evaluate(
            model, test_images, test_labels)
        circle = per_class.get("centre circle", (float("nan"), 0))
        arc = per_class.get("penalty arc", (float("nan"), 0))
        print(f"  {held:>14s} {len(images):7d} {len(test_images):6d} "
              f"{overall:8.0%} {balanced:8.0%} {majority:8.0%} "
              f"{circle[0]:7.0%} {arc[0]:7.0%}")
        summary.append((held, overall, balanced, majority, circle, arc))

    if summary:
        balanced = [s[2] for s in summary if np.isfinite(s[2])]
        print(f"\n  {'mean':>14s} {'':7s} {'':6s} "
              f"{np.mean([s[1] for s in summary]):8.0%} "
              f"{(np.mean(balanced) if balanced else float('nan')):8.0%} "
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
