"""Is this ball candidate a ball? A small classifier on the object itself.

The detector's `sports ball` class takes printed logos, heads and boots for
the ball, and the ball path then follows them: on three unplaced labelled
shots the path ran along the advertising hoardings while the ball was among
the players. Two fixes from outside the object failed -- preferring grass
around a candidate dropped the frames where events happen (the ball against
the net, the stands, the hoardings past the line), and more candidates at
full resolution meant more look-alikes. This one looks at the candidate.

A crop centred on the candidate, `CONTEXT` times its box across -- enough to
see a round object and its edge, little enough that the background behind
it does not decide -- resized to `SIZE` pixels, scored by a small CNN.

Trained by `train_ball_classifier.py` on the Roboflow football-players
dataset (CC BY 4.0, roboflow-jvuqo/football-players-detection-3zvbc):
hand-labelled balls as positives, and as negatives everything the same
detector proposes in those images that is not the labelled ball.

## Measured, and not used

Held out on three whole source matches it scores AUC 0.956. Wired into the
ball path on six labelled windows it lost every event -- shots 3 of 10 to
0, goals 1 of 2 to 0, out of play 2 of 4 to 1 with a false one -- and the
reason is the one the grass weighting had. On reading_1155's goal it scores
the ball at the strike, on grass, 1.0, and the same ball in the top corner
of the net and lying in it afterwards 0.0, while look-alikes around it
score 0.5-0.7. Nearly every training ball is on grass, so grass became part
of what a ball is; the held-out test agreed because its balls were on grass
too. What it lacks are balls against the net, the stands and the hoardings,
which these public labels barely contain. It is kept, unwired, for that:
retrained with such examples it is the same experiment with the gap closed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Crop side as a multiple of the candidate's box, and the network's input.
CONTEXT = 2.5
SIZE = 32
# Smallest crop side in pixels: a 4 px box still gets some surroundings.
MIN_CROP_PX = 12

WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "ball_classifier.pt"


def crop(image, cx: float, cy: float, box_px: float):
    """A SIZE x SIZE RGB crop centred on (cx, cy), or None off the image."""
    import cv2

    side = max(MIN_CROP_PX, CONTEXT * float(box_px))
    half = side / 2.0
    h, w = image.shape[:2]
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return None
    # Pad rather than shift at the edges, so the object stays centred.
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                   cv2.BORDER_REFLECT)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    patch = cv2.resize(patch, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)


def network():
    """A small CNN: three conv blocks and a linear head, about 60k weights."""
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                             nn.BatchNorm2d(cout), nn.ReLU(),
                             nn.MaxPool2d(2))

    return nn.Sequential(block(3, 16), block(16, 32), block(32, 64),
                         nn.Flatten(), nn.Dropout(0.3),
                         nn.Linear(64 * (SIZE // 8) ** 2, 1))


def to_tensor(crops):
    import torch

    arr = np.stack(crops).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


class BallClassifier:
    """Loads the trained weights and scores crops: probability of a ball."""

    def __init__(self, weights: Path = WEIGHTS):
        import torch

        self.model = network()
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()

    def score(self, crops) -> np.ndarray:
        import torch

        if not len(crops):
            return np.empty(0)
        with torch.no_grad():
            logits = self.model(to_tensor(crops)).squeeze(1)
        return torch.sigmoid(logits).numpy()
