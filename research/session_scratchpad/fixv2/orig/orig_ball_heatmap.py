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

Version 2 adds motion (`motion_bg.py`): `build(motion=True)` gives the
stem a second 7 x 7 / 2 convolution on three luminance channels -- the
differences to frames t-2 and t+2 and to a background median -- whose
output is added to the RGB one before bn1. Its weights start at zero, so
at step 0 it is the RGB model. `detect_v2()` runs it on a frame with its
neighbours and background (same crop, peaks and refinement as `detect`)
and keeps the raw logit, for the Platt calibration in `<ckpt>.calib.json`
(`load_calibration`, `p_cal`). A v2 checkpoint records what it was trained
on in `<ckpt>.provenance.json` (`write_provenance`), and `check_unseen`
refuses to measure it on those matches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STRIDE = 4
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BALL_M = 0.22

# Scales of the motion channels (`motion_bg.motion_channels`), in units of
# 255 grey levels: a robust std of the frame differences (D) and of the
# difference to the background (F) on training strips' pitch pixels.
# Placeholders until `label_ball_strips.py --sigma` measures them; the
# measured values are written here by hand (one line each) before training.
SIGMA_D = 0.02
SIGMA_F = 0.05

# v2 training data (protocol v3): matches and halves, absolute frame ranges
# left out, and matches never to be trained on (tuning U, report V, M1 on
# another camera, the test matches).
V2_TRAIN = {"118575": (1, 2), "118576": (1, 2), "118577": (1, 2),
            "128058": (1, 2)}
V2_EXCLUDE = {("118576", 2): (13500, 20999), ("128058", 2): (13500, 20999)}
NEVER = {"118578", "117093", "132877", "117092", "128057", "132831"}


def build(motion: bool = False):
    """The heatmap net, from ImageNet weights. With `motion`, input is N x 6
    x H x W (RGB normalised as `to_tensor`, then the three motion channels)
    and `conv1_motion` (3 -> 64, 7 x 7 / 2, no bias, zero at the start)
    adds its output to the RGB conv1's (`stem[0]`) before bn1 (`stem[1]`);
    an N x 3 input is zero motion, through the RGB stem alone (exactly
    the RGB net). Without, it is v1's net, and v1's state dicts load
    into it."""
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
            self.motion = motion
            if motion:
                self.conv1_motion = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
                nn.init.zeros_(self.conv1_motion.weight)

        def forward(self, x):
            s = self.stem
            if self.motion and x.shape[1] == 6:
                # The two stems' sum as one 6-channel convolution: about
                # 1.06 x v1's full-frame forward, against 1.15-1.20 x as two.
                w = torch.cat([s[0].weight, self.conv1_motion.weight], 1)
                x = s[3](s[2](s[1](nn.functional.conv2d(x, w, None, 2, 3))))
            else:
                x = s(x)            # v1, or zero motion: the RGB stem alone
            c1 = self.layer1(x)
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


def to_tensor_v2(rgb_list, motion_list):
    """BGR uint8 images and their motion channels (H x W x 3 float32, from
    `motion_bg.motion_channels`; None for zero) -> N x 6 x H x W."""
    import torch

    h, w = rgb_list[0].shape[:2]
    x = np.empty((len(rgb_list), 6, h, w), np.float32)
    for i, (im, mo) in enumerate(zip(rgb_list, motion_list)):
        x[i, :3] = ((im[:, :, ::-1].astype(np.float32) / 255.0 - MEAN)
                    / STD).transpose(2, 0, 1)
        x[i, 3:] = 0.0 if mo is None else np.asarray(mo).transpose(2, 0, 1)
    return torch.from_numpy(x)


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


def pitch_box(shape, outline, pad_above: int = 64,
              pad_below: int = 32) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the part of a frame of `shape` the detectors
    look at: the pitch outline's bounding box, 32 px wider on each side,
    `pad_above` px higher (balls in the air) and `pad_below` lower."""
    poly = np.asarray(outline, np.float32)
    H, W = shape[:2]
    x0 = max(int(poly[:, 0].min()) - 32, 0)
    x1 = min(int(poly[:, 0].max()) + 32, W)
    y0 = max(int(poly[:, 1].min()) - pad_above, 0)
    y1 = min(int(poly[:, 1].max()) + pad_below, H)
    return x0, y0, x1, y1


def _peaks(logit, x0: int, y0: int, poly, tau: float, top_k: int,
           margin_px: float):
    """Local maxima (5 x 5) of a logit map with p >= tau, inside the outline
    (+ `margin_px`), best first, at most `top_k`, each refined to the
    p-weighted centre of its 3 x 3 cells: (u, v, p, logit) in frame pixels."""
    import cv2
    import torch

    prob = torch.sigmoid(logit)
    peak = torch.nn.functional.max_pool2d(prob[None, None], 5, 1, 2)[0, 0]
    prob, peak, logit = prob.numpy(), peak.numpy(), logit.numpy()
    gy, gx = np.nonzero((prob >= tau) & (prob >= peak))
    if not len(gy):
        return None
    u = x0 + (gx + 0.5) * STRIDE
    v = y0 + (gy + 0.5) * STRIDE
    inside = np.array([cv2.pointPolygonTest(poly.reshape(-1, 1, 2),
                                            (float(a), float(b)), True)
                       >= -margin_px for a, b in zip(u, v)], dtype=bool)
    gy, gx, u, v = gy[inside], gx[inside], u[inside], v[inside]
    order = np.argsort(-prob[gy, gx])[:top_k]
    out = []
    for k in order:
        y, x = gy[k], gx[k]
        ys, xs = slice(max(y - 1, 0), y + 2), slice(max(x - 1, 0), x + 2)
        w = prob[ys, xs]
        yy, xx = np.mgrid[ys, xs]
        cu = x0 + ((xx + 0.5) * STRIDE * w).sum() / w.sum()
        cv = y0 + ((yy + 0.5) * STRIDE * w).sum() / w.sum()
        out.append((cu, cv, float(prob[y, x]), float(logit[y, x])))
    return out


def _ground(cu: float, cv: float, cal: dict):
    from .soccertrack_v2 import image_to_pitch

    g = image_to_pitch([[cu, cv]], cal)[0]
    return g if np.isfinite(g).all() else None


def detect(model, frame, outline, tau: float = 0.15, top_k: int = 4,
           margin_px: float = 10.0, pad_above: int = 64,
           pad_below: int = 32, cal: dict | None = None):
    """Ball candidates in a frame: (u, v, p, size_px) rows, at most `top_k`
    local maxima with p >= tau, inside the pitch outline (+ `margin_px`).
    `size_px` is the ball's diameter there when `cal` is given (along the
    pitch, as in v1)."""
    import cv2
    import torch

    poly = np.asarray(outline, np.float32)
    x0, y0, x1, y1 = pitch_box(frame.shape, outline, pad_above, pad_below)
    crop = frame[y0:y1, x0:x1]
    ph, pw = (-crop.shape[0]) % 8, (-crop.shape[1]) % 8
    crop = cv2.copyMakeBorder(crop, 0, ph, 0, pw, cv2.BORDER_CONSTANT)
    model.eval()
    with torch.inference_mode():
        peaks = _peaks(model(to_tensor([crop]))[0], x0, y0, poly, tau,
                       top_k, margin_px)
    if peaks is None:
        return np.zeros((0, 4))
    rows = []
    for cu, cv, p, _ in peaks:
        size = np.nan
        if cal is not None:
            g = _ground(cu, cv, cal)
            if g is not None:
                size = BALL_M * local_scale(g, cal)[0]
        rows.append((cu, cv, p, size))
    return np.array(rows)


_SCRATCH: dict = {}


def _input_v2(crop, prev, nxt, bg, ph: int, pw: int):
    """`to_tensor_v2` of the crop and its motion channels, both zero-padded
    by (ph, pw), built in place in one buffer kept between calls: fresh
    full-frame temporaries (a 100 MB input) cost about as much in page
    faults as the motion stem computes. The tensor is valid until the
    next call."""
    import torch

    from .motion_bg import motion_planes

    h, w = crop.shape[:2]
    shape = (1, 6, h + ph, w + pw)
    x = _SCRATCH.get(shape)
    if x is None:
        _SCRATCH.clear()
        x = _SCRATCH[shape] = np.empty(shape, np.float32)
    for c in range(3):                      # as to_tensor: RGB, normalised
        d = x[0, c]
        d[:h, :w] = crop[:, :, 2 - c]
        d[h:], d[:h, w:] = 0.0, 0.0
        d /= 255.0
        d -= MEAN[c]
        d /= STD[c]
    x[0, 3:, h:], x[0, 3:, :h, w:] = 0.0, 0.0
    motion_planes(prev, crop, nxt, bg, out=x[0, 3:, :h, :w])
    return torch.from_numpy(x)


def detect_v2(model, prev, cur, nxt, bg, outline, tau: float = 0.05,
              top_k: int = 8, margin_px: float = 10.0, cal: dict | None = None,
              motion_zero: bool = False, pad_above: int = 64,
              pad_below: int = 32) -> np.ndarray:
    """Ball candidates of the motion model (`build(motion=True)`) in frame
    `cur` (BGR, full frame): (u, v, p_raw, size_px, logit) rows, at most
    `top_k` local maxima with p_raw >= tau inside the outline (+
    `margin_px`), cropped, peaked and refined as `detect`.

    `prev` / `nxt` are frames t-2 / t+2 (full frames, None at an edge);
    `bg` the grey background B_t (None if it has too few samples), either
    full-frame or already cropped to `pitch_box(cur.shape, outline)` --
    told apart by its shape; `motion_bg.BackgroundCache(crop=pitch_box(..))`
    stores the latter. `size_px` is 0.22 m at the larger of the two local
    scales (`cal` given; NaN otherwise). `motion_zero` feeds zero motion
    (a diagnostic), as the 3-channel input: the RGB branch alone, exactly."""
    import cv2
    import torch

    if not getattr(model, "motion", False):
        raise ValueError("detect_v2 needs a motion model (build(motion=True))")
    poly = np.asarray(outline, np.float32)
    x0, y0, x1, y1 = pitch_box(cur.shape, outline, pad_above, pad_below)

    def cut(im):
        return None if im is None else im[y0:y1, x0:x1]

    if bg is not None:
        if bg.shape[:2] == cur.shape[:2]:
            bg = cut(bg)
        elif bg.shape[:2] != (y1 - y0, x1 - x0):
            raise ValueError(f"background of shape {bg.shape[:2]}: neither the "
                             f"frame {cur.shape[:2]} nor its pitch box "
                             f"{(y1 - y0, x1 - x0)}")
    crop = cut(cur)
    ph, pw = (-crop.shape[0]) % 8, (-crop.shape[1]) % 8
    if motion_zero:
        x = to_tensor([cv2.copyMakeBorder(crop, 0, ph, 0, pw,
                                          cv2.BORDER_CONSTANT)])
    else:
        x = _input_v2(crop, cut(prev), cut(nxt), bg, ph, pw)
    model.eval()
    with torch.inference_mode():
        peaks = _peaks(model(x)[0], x0, y0, poly, tau, top_k, margin_px)
    if not peaks:
        return np.zeros((0, 5))
    rows = []
    for cu, cv, p, z in peaks:
        size = np.nan
        if cal is not None:
            g = _ground(cu, cv, cal)
            if g is not None:
                size = BALL_M * max(local_scale(g, cal))
        rows.append((cu, cv, p, size, z))
    return np.array(rows)


# ---- Calibration and provenance of v2 checkpoints --------------------------------

def calib_path(ckpt) -> Path:
    return Path(f"{ckpt}.calib.json")


def provenance_path(ckpt) -> Path:
    return Path(f"{ckpt}.provenance.json")


def load_calibration(ckpt) -> dict | None:
    """The Platt calibration of a checkpoint, `<ckpt>.calib.json` ({"a",
    "b", "tau", ...}; a path ending in .json is read as is), or None."""
    path = Path(ckpt) if str(ckpt).endswith(".json") else calib_path(ckpt)
    return json.loads(path.read_text()) if path.exists() else None


def p_cal(logit, calib: dict | None):
    """sigmoid(a * logit + b); without a calibration, sigmoid(logit)."""
    a, b = (1.0, 0.0) if calib is None else (float(calib["a"]), float(calib["b"]))
    z = a * np.asarray(logit, dtype=np.float64) + b
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(_jsonable(v) for v in x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def write_provenance(ckpt, info: dict) -> Path:
    """Write `<ckpt>.provenance.json`. `info["train"]` is required, as
    {match: halves} (`V2_TRAIN`); `info["exclude"]` may be `V2_EXCLUDE`
    ({(match, half): (first, last)}) and is stored as [match, half, first,
    last] rows. Refuses a training set that includes a `NEVER` match."""
    if "train" not in info:
        raise ValueError("provenance without 'train' {match: halves}")
    train = {str(m): sorted(int(h) for h in hs) for m, hs in info["train"].items()}
    bad = sorted(set(train) & NEVER)
    if bad:
        raise ValueError(f"trained on never-train matches {bad}")
    out = dict(info, train=train)
    if isinstance(info.get("exclude"), dict):
        out["exclude"] = [[str(m), int(h), int(a), int(b)]
                          for (m, h), (a, b) in sorted(info["exclude"].items())]
    path = provenance_path(ckpt)
    path.write_text(json.dumps(_jsonable(out), indent=1))
    return path


def read_provenance(ckpt) -> dict | None:
    path = provenance_path(ckpt)
    return json.loads(path.read_text()) if path.exists() else None


def check_unseen(ckpt, match, allow_seen: bool = False):
    """Raise if `match` is in the checkpoint's training provenance (any
    half), or if the checkpoint has no provenance; `allow_seen` (the T
    diagnostic) lets both pass."""
    if allow_seen:
        return
    prov = read_provenance(ckpt)
    if prov is None:
        raise ValueError(f"{ckpt}: no {provenance_path(ckpt).name}; cannot tell "
                         f"whether match {match} was trained on")
    halves = prov["train"].get(str(match))
    if halves:
        raise ValueError(f"{ckpt} was trained on match {match} (halves "
                         f"{halves}); allowed only with --allow-seen")


# ---- Self-check ---------------------------------------------------------------

# Names and shapes of v1's state dict (.cache/ball_heatmap.3000.pt), hashed.
V1_STATE_SHA = "d19c3ceaa0af8f71"


def _state_sha(state) -> str:
    import hashlib

    items = [(k, tuple(v.shape)) for k, v in state.items()]
    return hashlib.sha256(repr(items).encode()).hexdigest()[:16]


def _check():
    """v1's net is unchanged and v1 checkpoints load; the motion net is the
    RGB net when its motion input is zero (and at the start, whatever the
    input) while its motion weights get a gradient at once; `detect_v2`
    with `motion_zero` gives `detect`'s candidates; peaks, outline and
    refinement on a known logit map; ball size from the larger local scale;
    calibration and provenance files."""
    import tempfile

    import cv2
    import torch

    torch.set_num_threads(4)
    # v1: the same state dict names and shapes, and it loads strictly.
    torch.manual_seed(0)
    v1 = build()
    assert _state_sha(v1.state_dict()) == V1_STATE_SHA
    torch.manual_seed(0)
    m = build(motion=True)
    assert list(m.state_dict()) == list(v1.state_dict()) + ["conv1_motion.weight"]
    assert not m.conv1_motion.weight.any()
    r = m.load_state_dict(v1.state_dict(), strict=False)
    assert r.missing_keys == ["conv1_motion.weight"] and not r.unexpected_keys
    # A frame: grass texture, a white ball, a dark player.
    rng = np.random.default_rng(0)
    H, W = 200, 480
    frame = np.clip(rng.normal(90, 12, (H, W, 3)), 0, 255).astype(np.uint8)
    frame[:, :, 1] += 30
    prev, nxt = frame.copy(), frame.copy()
    for im, x in ((prev, 150), (frame, 170), (nxt, 190)):
        cv2.circle(im, (x, 110), 3, (250, 250, 250), -1)
    cv2.circle(frame, (300, 120), 9, (20, 20, 60), -1)
    outline = [[40, 70], [440, 70], [460, 180], [20, 180]]
    # At the start the motion net is the RGB net, motion input or not (to
    # rounding: one 6-channel convolution sums in another order), and its
    # motion weights get a gradient at once.
    mo = rng.normal(0, 1, (H, W, 3)).astype(np.float32)
    x6 = to_tensor_v2([frame], [mo])
    assert torch.equal(x6[:, :3], to_tensor([frame]))
    assert torch.equal(x6[0, 3:], torch.from_numpy(mo.transpose(2, 0, 1).copy()))
    with torch.no_grad():
        v1.eval(), m.eval()
        assert torch.allclose(m(x6), v1(to_tensor([frame])), atol=1e-4)
    m.train()
    m(x6).sum().backward()
    assert m.conv1_motion.weight.grad.abs().sum() > 0
    m.zero_grad()
    # Trained-looking weights: motion non-zero, outputs near 0.5 so there are
    # peaks. Zero motion is the RGB path (exactly as a 3-channel input);
    # other motion is not.
    with torch.no_grad():
        m.conv1_motion.weight.normal_(0, 0.05)
        m.head[-1].bias.fill_(0.0)
    v1.load_state_dict({k: v for k, v in m.state_dict().items()
                        if k != "conv1_motion.weight"})
    m.eval(), v1.eval()
    with torch.no_grad():
        z = v1(to_tensor([frame]))
        assert torch.equal(m(to_tensor([frame])), z)
        assert torch.allclose(m(to_tensor_v2([frame], [None])), z, atol=1e-4)
        assert not torch.allclose(m(x6), z, atol=1e-2)
    # The in-place input of detect_v2 is to_tensor_v2 of the padded crop
    # and its padded motion channels, bit for bit.
    from .motion_bg import motion_channels

    bgf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) // 2 + 40
    cr, pr, nr, br = (im[3:198, 5:470] for im in (frame, prev, nxt, bgf))
    pad = lambda im: np.pad(im, ((0, 5), (0, 3)) + ((0, 0),) * (im.ndim - 2))
    ref = to_tensor_v2([pad(cr)], [pad(motion_channels(pr, cr, nr, br))])
    assert torch.equal(_input_v2(cr, pr, nr, br, 5, 3), ref)
    assert torch.equal(_input_v2(cr, None, nr, None, 5, 3)[:, 3:4], 0 * ref[:, 3:4])
    # detect_v2 with zero motion gives detect's rows (and the logit); with
    # motion it differs; a full-frame and a cropped background agree.
    a = detect(v1, frame, outline, tau=0.05, top_k=8)
    b = detect_v2(m, prev, frame, nxt, None, outline, tau=0.05, top_k=8,
                  motion_zero=True)
    assert len(a) == 8 and b.shape == (8, 5)
    assert np.array_equal(a, b[:, :4], equal_nan=True)
    assert np.allclose(1 / (1 + np.exp(-b[:, 4])), b[:, 2], atol=1e-6)
    bg = bgf
    c = detect_v2(m, prev, frame, nxt, bg, outline, tau=0.05, top_k=8)
    x0, y0, x1, y1 = pitch_box(frame.shape, outline)
    d = detect_v2(m, prev, frame, nxt, bg[y0:y1, x0:x1], outline, tau=0.05,
                  top_k=8)
    assert np.array_equal(c, d, equal_nan=True) and not np.array_equal(c, b)
    for bad in (lambda: detect_v2(m, prev, frame, nxt, bg[:50, :50], outline),
                lambda: detect_v2(v1, prev, frame, nxt, None, outline)):
        try:
            bad()
            raise AssertionError("a wrong background or model was accepted")
        except ValueError:
            pass
    # Peaks on a known logit map: a 5 x 5 maximum inside the outline is
    # kept and refined to its cells' centre; one outside is dropped.
    lg = torch.full((50, 120), -6.0)
    lg[20, 30], lg[20, 29], lg[20, 31] = 3.0, 1.0, 1.0
    lg[45, 2] = 4.0                                 # outside the 2nd outline
    fake = _Fixed(lg)
    rows = detect(fake, np.zeros((200, 480, 3), np.uint8),
                  [[0, 0], [480, 0], [480, 200], [0, 200]], tau=0.5)
    out = detect(fake, np.zeros((200, 480, 3), np.uint8),
                 [[60, 40], [400, 40], [400, 150], [60, 150]], tau=0.5)
    assert rows.shape == (2, 4) and out.shape == (1, 4)
    assert np.allclose(rows[1, :2], (122, 82))      # 2nd best; box from 0, 0
    assert np.allclose(out[0, :3], (150, 82, 1 / (1 + np.exp(-3))))  # x 28
    # Size: 0.22 m at the larger local scale, through a camera.
    cal = {"K": np.array([[400.0, 0, 240], [0, 400, 100], [0, 0, 1]]),
           "D": np.zeros((4, 1)),
           "Knew": np.array([[400.0, 0, 240], [0, 400, 100], [0, 0, 1]]),
           "Hinv": np.linalg.inv(np.array([[4.0, 0, 30], [0, 2, 20], [0, 0, 1]]))}
    e = detect_v2(m, prev, frame, nxt, bg, outline, top_k=3, cal=cal)
    from .soccertrack_v2 import image_to_pitch

    for u, v, _, size, _ in e:
        sx, sy = local_scale(image_to_pitch([[u, v]], cal)[0], cal)
        assert sy != sx and np.isclose(size, BALL_M * max(sx, sy))
    assert np.isnan(b[:, 3]).all()
    # Calibration and provenance.
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ball_v2.4000.pt"
        assert load_calibration(ckpt) is None and read_provenance(ckpt) is None
        calib_path(ckpt).write_text(json.dumps({"a": 2.0, "b": -1.0, "tau": 0.3}))
        calib = load_calibration(ckpt)
        assert calib["tau"] == 0.3 and load_calibration(calib_path(ckpt)) == calib
        assert np.isclose(p_cal(0.5, calib), 0.5)
        assert np.allclose(p_cal([-1.0, 2.0], None), 1 / (1 + np.exp([1.0, -2.0])))
        try:
            check_unseen(ckpt, "118578")
            raise AssertionError("a checkpoint without provenance passed")
        except ValueError:
            pass
        write_provenance(ckpt, {"train": V2_TRAIN, "exclude": V2_EXCLUDE,
                                "seed": np.int64(0), "sigma": (SIGMA_D, SIGMA_F)})
        prov = read_provenance(ckpt)
        assert prov["train"]["118576"] == [1, 2] and prov["seed"] == 0
        assert ["128058", 2, 13500, 20999] in prov["exclude"]
        check_unseen(ckpt, "118578")
        check_unseen(ckpt, "118576", allow_seen=True)
        for bad in (lambda: check_unseen(ckpt, "118576"),
                    lambda: check_unseen(ckpt, 128058),
                    lambda: write_provenance(ckpt, {"train": {"117093": (1,)}}),
                    lambda: write_provenance(ckpt, {"seed": 0})):
            try:
                bad()
                raise AssertionError("a seen match or bad provenance passed")
            except ValueError:
                pass
    print("  ball_heatmap: v1 net unchanged, motion net = RGB net at zero motion "
          "(and at the start), detect_v2 = detect with motion zeroed, peaks "
          "refined and outlined, size at the larger scale, calibration and "
          "provenance guard")


class _Fixed:
    """A stand-in model returning a fixed logit map (for `_check`)."""

    def __init__(self, logit):
        self.logit = logit

    def eval(self):
        return self

    def __call__(self, x):
        return self.logit[None]


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _check()
