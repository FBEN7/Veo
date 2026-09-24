"""Ask a trained classifier what an arc is, and let it decline to answer.

The measurement behind this is mixed and the wiring has to respect that. On
two clips of four the centre-circle-against-penalty-arc classifier lands at
97-100% on every initialisation; on the other two it sits at chance. So it
is a source of evidence that is excellent when it works and worthless when
it does not, and the job here is to use it only in the first case.

Three things make that possible.

**It may abstain.** A prediction is used only when the network is confident.
Anything less and the arc is refused, which returns the anchor to where it
already was rather than to a guess -- and a wrong answer here is a shot at
the wrong end of the pitch, which nothing downstream would notice.

**It votes.** An arc is not one crop. Crops are taken along its supporting
pixels and classified separately, and the arc's identity is what most of
them say. A single crop can be unlucky; a dozen spread along the arc are
unlucky together only if the model has genuinely misread the frame.

**It is asked at the right scale.** Crops cover a fixed 24 m of pitch rather
than a fixed number of pixels, converted through the anchor's own Jacobian.
That is what made the classifier work at all -- 9 m around an arc pixel does
not reach the penalty-area line that identifies it -- and getting it wrong
at inference would silently undo the training.

## A licence line that matters

Weights trained on the SoccerNet clips inherit their terms, which are
non-commercial and no-redistribution, so they are as unshippable as the
frames they came from. Nothing here is committed. `load` returns None when
no weights are present and every caller must carry on without it, which is
also what makes it safe to wire in: the anchor's behaviour with no model is
exactly its behaviour before this existed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODELS = Path("models")

# How sure the network has to be before its answer is used at all.
MIN_CONFIDENCE = 0.80

# How much of the vote has to agree before the arc is called.
MIN_AGREEMENT = 0.70

# Crops along the arc, and the footprint each covers. The footprint must
# match what the model was trained at or the scale it learned is wrong.
VOTES = 12
FOOTPRINT_M = 24.0
CROP_OUT = 64

CIRCLE, PENALTY_ARC = 0, 1


def _network(classes=2):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, classes))


def load(exclude: str | None = None):
    """The arc classifier, or None if there is none to load.

    `exclude` asks for the model trained without a particular clip, which is
    the only honest one to use when measuring on that clip. Without it, a
    model that has seen the clip would be marking its own homework.
    """
    try:
        import torch
    except ImportError:
        return None
    if exclude is not None:
        path = MODELS / f"arc_without_{exclude.replace(' ', '_')}.pt"
        if not path.exists():
            return None
    else:
        found = sorted(MODELS.glob("arc_without_*.pt"))
        if not found:
            return None
        path = found[0]

    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = _network()
    # The training module builds the same layers as a body and a head; the
    # names differ, so map by order rather than by key.
    state = blob["state"]
    ordered = [state[key] for key in state]
    own = list(model.state_dict().keys())
    if len(ordered) != len(own):
        return None
    model.load_state_dict({key: value for key, value in zip(own, ordered)})
    model.eval()
    return model


def pixels_per_metre(homography, x, y):
    """Pixels covering a metre of pitch, here, from the map's own Jacobian."""
    here = np.array([[x, x + 1.0, x], [y, y, y + 1.0], [1.0, 1.0, 1.0]])
    mapped = homography @ here
    if np.any(np.abs(mapped[2]) < 1e-9):
        return None
    mapped = mapped[:2] / mapped[2]
    if not np.all(np.isfinite(mapped)):
        return None
    jacobian = np.column_stack([mapped[:, 1] - mapped[:, 0],
                                mapped[:, 2] - mapped[:, 0]])
    area = abs(float(np.linalg.det(jacobian)))
    return None if area < 1e-12 else 1.0 / np.sqrt(area)


def _crop(frame, x, y, scale):
    size = int(round(FOOTPRINT_M * scale))
    if not (48 <= size <= 900):
        return None
    half = size // 2
    padded = cv2.copyMakeBorder(frame, half, half, half, half,
                                cv2.BORDER_REPLICATE)
    patch = padded[int(y):int(y) + 2 * half, int(x):int(x) + 2 * half]
    if patch.shape[0] < 4 or patch.shape[1] < 4:
        return None
    return cv2.resize(patch, (CROP_OUT, CROP_OUT),
                      interpolation=cv2.INTER_AREA)


def classify_arc(model, frame, support, homography, rng=None):
    """Is this arc a penalty D or a centre circle? None means it will not say.

    Returns (is_penalty_arc, share_agreeing) or (None, share) when too few
    crops were confident, or the vote was too close, to be acted on.
    """
    if model is None or len(support) == 0:
        return None, 0.0
    import torch

    rng = rng or np.random.default_rng(0)
    pick = rng.choice(len(support), min(VOTES, len(support)), replace=False)
    patches = []
    for x, y in np.asarray(support)[pick]:
        scale = pixels_per_metre(homography, float(x), float(y))
        if scale is None:
            continue
        patch = _crop(frame, float(x), float(y), scale)
        if patch is not None:
            patches.append(patch)
    if len(patches) < 3:
        return None, 0.0

    batch = torch.from_numpy(
        np.stack(patches).astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    with torch.no_grad():
        probability = torch.softmax(model(batch), dim=1).numpy()

    confident = probability.max(axis=1) >= MIN_CONFIDENCE
    if confident.sum() < 3:
        return None, 0.0
    votes = probability[confident].argmax(axis=1)
    share = float((votes == PENALTY_ARC).mean())
    if share >= MIN_AGREEMENT:
        return True, share
    if (1.0 - share) >= MIN_AGREEMENT:
        return False, 1.0 - share
    return None, max(share, 1.0 - share)
