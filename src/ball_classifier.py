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

Held out on three whole source matches it scores AUC 0.956. Retrained with
the balls clicked on the six labelled windows, each clip held out in turn,
it scores 0.92-1.00, and wherever the stored candidates include the clicked
ball it ranks it first, 63 times of 63. Wired into the ball path it still
lost every shot and goal (EVENT_ACCURACY.md, "The ball clicked by hand"):
the detector proposes the ball on only 63 of the 138 clicked frames, so
scoring down the look-alikes leaves the path nothing to follow through the
ball's flight. It is kept, unwired, for when the detector finds more of
the ball -- full resolution finds 90 of the 138, and brings back the
look-alikes this classifier is for.
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

    def __init__(self, weights: Path = WEIGHTS, model=None):
        import torch

        if model is None:
            model = network()
            model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model = model
        self.model.eval()

    def score(self, crops) -> np.ndarray:
        import torch

        if not len(crops):
            return np.empty(0)
        with torch.no_grad():
            logits = self.model(to_tensor(crops)).squeeze(1)
        return torch.sigmoid(logits).numpy()


SCORES_FILE = "ball_scores.parquet"


def score_clip(out_dir: Path, info: dict, verbose: bool = True,
               classifier: "BallClassifier | None" = None,
               overwrite: bool = False):
    """Score every stored ball candidate in a clip, cached next to the tracks.

    Returns the cache path, or None without trained weights. The scores are
    derived from the clip's footage, so like every run output they are not
    committed.
    """
    import cv2
    import pandas as pd

    out_dir = Path(out_dir)
    cache = out_dir / SCORES_FILE
    if cache.exists() and not overwrite:
        return cache
    if classifier is None and not WEIGHTS.exists():
        if verbose:
            print(f"  [ball classifier] no weights at {WEIGHTS}; run "
                  f"train_ball_classifier.py")
        return None
    from .ball_densify import with_fullres

    # The full-resolution candidates too, where a run has cached them.
    tracks = with_fullres(pd.read_parquet(out_dir / "tracks.parquet"),
                          out_dir)
    balls = tracks[tracks.cls == "ball"]
    by_frame = {int(f): g for f, g in balls.groupby("frame")}
    model = classifier or BallClassifier()
    cap = cv2.VideoCapture(info["path"])
    rows, index = [], -1
    for frame_no in sorted(by_frame):
        while index < frame_no:
            ok, image = cap.read()
            index += 1
            if not ok:
                break
        if not ok:
            break
        group = by_frame[frame_no]
        crops, keep = [], []
        for row in group.itertuples():
            patch = crop(image, float(row.px), float(row.py),
                         float(row.crop_h))
            if patch is not None:
                crops.append(patch)
                keep.append(row)
        for row, p in zip(keep, model.score(crops)):
            rows.append({"frame": frame_no, "px": float(row.px),
                         "py": float(row.py), "p_ball": float(p)})
    cap.release()
    pd.DataFrame(rows).to_parquet(cache)
    if verbose:
        print(f"  [ball classifier] scored {len(rows)} ball candidates")
    return cache


def with_scores(tracks, out_dir: Path):
    """The tracks with a `p_ball` column where a cached score exists."""
    import pandas as pd

    cache = Path(out_dir) / SCORES_FILE
    if not cache.exists():
        return tracks
    scores = pd.read_parquet(cache)
    return tracks.merge(scores, on=["frame", "px", "py"], how="left")
