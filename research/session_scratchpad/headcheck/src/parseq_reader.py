"""Shirt numbers read as text, by PARSeq, from the upper part of the shirt.

The ResNet reader (`jersey_reader.py`), trained on the SoccerNet jersey
set, read 0 of 47 numbers a person had read on our footage: it had learnt
players' looks. PARSeq (Bautista & Atienza, ECCV 2022) is a scene-text
recogniser trained on millions of words; it reads digits it has never
seen on any kit. Untrained for shirts, on the upper torso of the players
a person named (`dump_named_crops.py`), read only as digits and voted
over each player's crops at confidence 0.9, it read 21 of 55 right at
Reading (14 wrong) and 6 of 47 at Stoke (9 wrong) -- where the ResNet
reader, even trained on Stoke's own named players, read 4 of 55 at
Reading. `train_parseq_reader.py` fine-tunes it.

The crop is the box's upper torso (`TORSO`), stretched to the model's
32 x 128 input: padding it to the input's shape instead, so the digits
keep their proportions, read fewer (11 and 2 of 55 at 2:1 and 4:1).

PARSeq's code is not in this repository: clone
https://github.com/baudm/parseq (Apache 2.0) and pass its folder; it
needs timm, pytorch-lightning, hydra-core, lmdb and nltk besides torch.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

# Share of the player's box: top, bottom, and the margin cut from each side.
TORSO = (0.10, 0.50, 0.15)
DIGITS = "0123456789"
MIN_CONF = 0.9


def torso(crop, region=TORSO):
    """The upper torso of a player's box (BGR), where the number is."""
    top, bottom, side = region
    h, w = crop.shape[:2]
    return crop[int(h * top):int(h * bottom), int(w * side):int(w * (1 - side))]


def load(repo: str, weights: str | None = None):
    """PARSeq from a clone of its repository, with the published weights,
    or fine-tuned ones."""
    import torch

    if repo not in sys.path:
        sys.path.insert(0, repo)
    model = torch.hub.load(repo, "parseq", pretrained=weights is None,
                           source="local", trust_repo=True)
    if weights:
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.eval()


def transform(model):
    from strhub.data.module import SceneTextDataModule

    return SceneTextDataModule.get_transform(model.hparams.img_size)


def to_batch(model, crops, tf=None):
    """BGR torso crops to the model's input batch."""
    import cv2
    import torch
    from PIL import Image

    tf = tf or transform(model)
    return torch.stack([tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                        for c in crops])


def digit_mask(model):
    """Over the model's characters: True for digits and end-of-text."""
    import torch

    tok = model.tokenizer
    keep = torch.zeros(len(tok._itos), dtype=torch.bool)
    for i, ch in enumerate(tok._itos):
        keep[i] = ch in DIGITS or i == tok.eos_id
    return keep


def read(model, crops, batch: int = 64):
    """Per torso crop: the digits read (at most two, else '') and the
    confidence, the product of the characters' probabilities."""
    import torch

    tf, keep = transform(model), digit_mask(model)
    labels, confs = [], []
    for i in range(0, len(crops), batch):
        chunk = [c for c in crops[i:i + batch]]
        with torch.no_grad():
            logits = model(to_batch(model, chunk, tf))
        logits[..., ~keep[:logits.shape[-1]]] = -1e9
        got, conf = model.tokenizer.decode(logits.softmax(-1))
        for g, c in zip(got, conf):
            ok = 0 < len(g) <= 2 and not (len(g) == 2 and g[0] == "0")
            labels.append(g if ok else "")
            confs.append(float(c.prod()) if ok else 0.0)
    return labels, np.array(confs)


def vote(labels, confs, min_conf: float = MIN_CONF):
    """A player's number from all their crops: (number, share, crops used).
    Each crop read at `min_conf` or more votes with its confidence."""
    votes = defaultdict(float)
    used = 0
    for g, c in zip(labels, confs):
        if g and c >= min_conf:
            votes[g] += c
            used += 1
    if not votes:
        return None, 0.0, 0
    best = max(votes, key=votes.get)
    return int(best), votes[best] / sum(votes.values()), used
