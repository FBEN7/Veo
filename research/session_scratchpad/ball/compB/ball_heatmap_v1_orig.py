"""A ball detector for fixed panoramas: a stride-4 heatmap, trained on weak
labels (`build_ball_patches.py`, `train_ball_heatmap.py`).

On a SoccerTrack v2 panorama the ball is 4-18 px across and the COCO
detector finds it within 60 px of the dataset's ball track on 21% of
action frames even at confidence 0.02. No ball boxes are released; the
labels here are regions in metres around where the actions say the ball
was (`regions`).

- `build()`: ResNet-18 to stride 8 with its stride-4 features added back,
  one logit per 4 x 4 pixels;
- `local_scale()`: pixels per metre along the pitch and in depth at a
  point, from the match's calibration;
- `detect()`: the ball candidates of a frame -- the pitch area only,
  local maxima above a threshold, refined to the sub-pixel.
"""

from __future__ import annotations

import numpy as np

STRIDE = 4
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BALL_M = 0.22


def build():
    import torch
    import torchvision

    nn = torch.nn

    class BallNet(nn.Module):
        def __init__(self):
            super().__init__()
            r = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
            self.layer1, self.layer2 = r.layer1, r.layer2       # /4, /8
            self.lat = nn.Conv2d(128, 64, 1)
            self.head = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, 1))
            nn.init.constant_(self.head[-1].bias, -4.6)

        def forward(self, x):
            c1 = self.layer1(self.stem(x))
            up = nn.functional.interpolate(self.lat(self.layer2(c1)),
                                           size=c1.shape[2:], mode="bilinear",
                                           align_corners=False)
            return self.head(c1 + up)[:, 0]

    return BallNet()


def to_tensor(images):
    """BGR uint8 images -> normalised RGB batch."""
    import torch

    x = np.stack([(im[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
                  for im in images])
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


def pitch_to_image(xy, cal: dict) -> np.ndarray:
    """(N, 2) pitch metres (centre origin) to panorama pixels; the inverse
    of `soccertrack_v2.image_to_pitch`."""
    import cv2

    H = np.linalg.inv(cal["Hinv"])
    p = np.c_[np.asarray(xy, float).reshape(-1, 2) + (52.5, 34.0),
              np.ones(len(np.asarray(xy).reshape(-1, 2)))] @ H.T
    und = p[:, :2] / p[:, 2:]
    norm = (und - cal["Knew"][:2, 2]) / np.diag(cal["Knew"])[:2]
    return cv2.fisheye.distortPoints(
        np.ascontiguousarray(norm.reshape(-1, 1, 2)), cal["K"],
        cal["D"]).reshape(-1, 2)


def local_scale(xy, cal: dict, step: float = 0.5) -> tuple[float, float]:
    """Pixels per metre at pitch point `xy`: along the pitch (x) and in
    depth (y), by finite differences."""
    xy = np.asarray(xy, float).reshape(2)
    p = pitch_to_image(np.array([xy, xy + (step, 0), xy + (0, step)]), cal)
    return (float(np.hypot(*(p[1] - p[0])) / step),
            float(np.hypot(*(p[2] - p[0])) / step))


def ellipse(shape, uv, half_w: float, below: float, above: float) -> np.ndarray:
    """Cells (stride 4, centres) of a `shape` grid inside an ellipse around
    pixel `uv` with the given half-width and extents below / above."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (xx + 0.5) * STRIDE - uv[0], (yy + 0.5) * STRIDE - uv[1]
    ry = np.where(cy >= 0, below, above)
    return (cx / max(half_w, 1e-6)) ** 2 + (cy / np.maximum(ry, 1e-6)) ** 2 <= 1


def detect(model, frame, outline, tau: float = 0.15, top_k: int = 4,
           margin_px: float = 10.0, pad_above: int = 64,
           pad_below: int = 32, cal: dict | None = None):
    """Ball candidates in a frame: (u, v, p, size_px) rows, at most `top_k`
    local maxima with p >= tau, inside the pitch outline (+ `margin_px`).
    `size_px` is the ball's diameter there when `cal` is given."""
    import cv2
    import torch

    poly = np.asarray(outline, np.float32)
    H, W = frame.shape[:2]
    x0 = max(int(poly[:, 0].min()) - 32, 0)
    x1 = min(int(poly[:, 0].max()) + 32, W)
    y0 = max(int(poly[:, 1].min()) - pad_above, 0)
    y1 = min(int(poly[:, 1].max()) + pad_below, H)
    crop = frame[y0:y1, x0:x1]
    ph, pw = (-crop.shape[0]) % 8, (-crop.shape[1]) % 8
    crop = cv2.copyMakeBorder(crop, 0, ph, 0, pw, cv2.BORDER_CONSTANT)
    model.eval()
    with torch.inference_mode():
        logit = model(to_tensor([crop]))[0]
        prob = torch.sigmoid(logit)
        peak = torch.nn.functional.max_pool2d(prob[None, None], 5, 1, 2)[0, 0]
    prob, peak = prob.numpy(), peak.numpy()
    gy, gx = np.nonzero((prob >= tau) & (prob >= peak))
    if not len(gy):
        return np.zeros((0, 4))
    u = x0 + (gx + 0.5) * STRIDE
    v = y0 + (gy + 0.5) * STRIDE
    inside = np.array([cv2.pointPolygonTest(poly.reshape(-1, 1, 2),
                                            (float(a), float(b)), True)
                       >= -margin_px for a, b in zip(u, v)], dtype=bool)
    gy, gx, u, v = gy[inside], gx[inside], u[inside], v[inside]
    order = np.argsort(-prob[gy, gx])[:top_k]
    rows = []
    for k in order:
        y, x = gy[k], gx[k]
        ys, xs = slice(max(y - 1, 0), y + 2), slice(max(x - 1, 0), x + 2)
        w = prob[ys, xs]
        yy, xx = np.mgrid[ys, xs]
        cu = x0 + ((xx + 0.5) * STRIDE * w).sum() / w.sum()
        cv = y0 + ((yy + 0.5) * STRIDE * w).sum() / w.sum()
        size = np.nan
        if cal is not None:
            from .soccertrack_v2 import image_to_pitch

            g = image_to_pitch([[cu, cv]], cal)[0]
            if np.isfinite(g).all():
                size = BALL_M * local_scale(g, cal)[0]
        rows.append((cu, cv, float(prob[y, x]), size))
    return np.array(rows)
