"""Shirt numbers, read where they can be and voted across a player's frames.

A number is legible on a few frames of a player's time on screen -- the
back turned to the camera, close enough -- and on the rest it is not there
to read. So the reader is asked two things per crop: is a number visible,
and if so which digits; and a person's number is the vote of every crop of
theirs, each weighted by how sure it is that a number is visible
(`vote`). One clear back view among fifty crops decides it, which is the
point: an identity seen on half the frames needs its number read once.

Digits are read separately -- the tens (or blank, for a one-digit number)
and the units -- so a number rarely seen in training, 47 say, is read from
digits seen often.

Trained on the SoccerNet jersey-number set (`train_jersey_reader.py`),
which, like the footage, is under a non-commercial agreement: the weights
are not in this repository and are passed in.
"""

from __future__ import annotations

import numpy as np

# Crops are resized to this, width x height, the player's whole box.
SIZE = (64, 128)
BLANK = 10                      # the tens class of a one-digit number
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build():
    """ResNet-18 with three heads: number visible, tens (0-9 or blank),
    units (0-9)."""
    import torch
    import torchvision

    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__()
            net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            net.fc = torch.nn.Identity()
            self.body = net
            self.visible = torch.nn.Linear(512, 1)
            self.tens = torch.nn.Linear(512, 11)
            self.units = torch.nn.Linear(512, 10)

        def forward(self, x):
            f = self.body(x)
            return self.visible(f).squeeze(1), self.tens(f), self.units(f)

    return Reader()


def to_tensor(crops):
    """BGR crops (any size) to a normalised batch."""
    import cv2
    import torch

    out = []
    for c in crops:
        c = cv2.resize(c, SIZE, interpolation=cv2.INTER_AREA
                       if c.shape[0] > SIZE[1] else cv2.INTER_LINEAR)
        c = (c[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        out.append(c.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(out))


def targets(number: int):
    """(visible, tens, units) for a label; -1 is no number."""
    if number < 0:
        return 0.0, -100, -100            # digits ignored
    return 1.0, (number // 10 if number >= 10 else BLANK), number % 10


def number_probs(tens_logp, units_logp):
    """log P(number) for 0..99 from the two digit heads, shape (n, 100)."""
    n = tens_logp.shape[0]
    out = np.empty((n, 100), dtype=np.float32)
    for k in range(100):
        t = k // 10 if k >= 10 else BLANK
        out[:, k] = tens_logp[:, t] + units_logp[:, k % 10]
    return out


def read(model, crops, batch: int = 64):
    """Per crop: P(number visible) and log P(number) over 0..99."""
    import torch

    model.eval()
    vis, logp = [], []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            v, t, u = model(to_tensor(crops[i:i + batch]))
            vis.append(torch.sigmoid(v).numpy())
            logp.append(number_probs(torch.log_softmax(t, 1).numpy(),
                                     torch.log_softmax(u, 1).numpy()))
    if not vis:
        return np.zeros(0), np.zeros((0, 100))
    return np.concatenate(vis), np.concatenate(logp)


# Tuned on 142 held-out tracklets of the jersey set: a crop counts when the
# reader is this sure a number is visible. With src/player_identity.py's
# 60% share and 5 crops, 82 of 101 numbered tracklets read right, 4 wrong;
# 27 of 41 that show no number are left unread.
MIN_VISIBLE = 0.95


def vote(visible, logp, min_visible: float = MIN_VISIBLE):
    """A player's number from all their crops: (number, share, crops used).

    Each crop judged to show a number casts its number distribution,
    weighted by how sure it is that a number is visible; `share` is the
    winning number's part of the total. None when no crop shows a number.
    """
    keep = visible >= min_visible
    if not keep.any():
        return None, 0.0, 0
    w = visible[keep][:, None]
    total = (w * np.exp(logp[keep])).sum(axis=0)
    best = int(np.argmax(total))
    return best, float(total[best] / total.sum()), int(keep.sum())
