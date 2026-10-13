"""Motion inputs for the panorama ball detector, v2 (`ball_heatmap.build(
motion=True)`).

The camera does not move, so what moves stands out against the frames
around it. A ball in flight is a few pixels across and looks like a
shirt badge or a line mark in one frame; between frames it is the small
thing that moved, and against the median of the 28 s around it, it is not
part of the pitch. Three channels on luminance only (colour stays in
the RGB branch):

    D- = (g_t - g_{t-2}) / 255 / SIGMA_D
    D+ = (g_t - g_{t+2}) / 255 / SIGMA_D
    F  = (g_t - B_t)     / 255 / SIGMA_F

g is cv2's BGR2GRAY (uint8, then float); the sigmas are
`ball_heatmap.SIGMA_D` / `SIGMA_F`, read at call time. A missing t-2 or
t+2 (an edge of a half or window) is taken as t itself (D = 0), a missing
background as F = 0.

The background B_t of absolute frame t is the per-pixel median of the
grey frames that are multiples of `BG_STEP` within `BG_HALF_SPAN` of the
centre c = 50 * round(t / 50) (rounded half up) and inside the available
range (the half when cutting training strips, the window at inference);
fewer than `BG_MIN` such frames give none. Training and inference use the
same rule through `BackgroundCache`.

- `bg_centre()`, `bg_sample_frames()`: the rule;
- `BackgroundCache`: grey samples kept while decoding (optionally a
  crop), the median per centre, `ready()` once decoding has passed them;
- `motion_channels()`: the three channels, H x W x 3 float32
  (`motion_planes()`: 3 x H x W, optionally in place);
- `RingBuffer`: the last frames of a sequential decode, for t-2 / t+2.
"""

from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np

from . import ball_heatmap as bh

BG_STEP = 50
BG_HALF_SPAN = 350
BG_MIN = 8
OFFSET = 2


def bg_centre(t: int) -> int:
    """The background centre of absolute frame `t`: the nearest multiple of
    `BG_STEP`, a tie (t = 25 mod 50) going up."""
    return (int(t) + BG_STEP // 2) // BG_STEP * BG_STEP


def bg_sample_frames(t: int, first: int, last: int) -> list[int]:
    """Absolute frames whose median is B_t: multiples of `BG_STEP` within
    `BG_HALF_SPAN` of the centre, inside [first, last]."""
    c = bg_centre(t)
    return [f for f in range(c - BG_HALF_SPAN, c + BG_HALF_SPAN + 1, BG_STEP)
            if first <= f <= last]


def grey(image) -> np.ndarray:
    """uint8 luminance of a BGR image; a 2-D image is taken as grey already
    (a grey image stored as 3 equal channels comes back unchanged)."""
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2GRAY)


def median_u8(images: list[np.ndarray], rows: int = 32) -> np.ndarray:
    """Per-pixel median of same-shape uint8 images; for an even count the
    mean of the middle two, rounded half up. An odd-even transposition sort
    (n rounds of min / max) over bands of `rows` rows that stay in cache:
    0.07 s for 15 x 1000 x 4096, against 0.6-1.6 s for numpy's or torch's
    median."""
    n, k = len(images), len(images) // 2
    out = np.empty_like(images[0])
    for r0 in range(0, out.shape[0], rows):
        a = [im[r0:r0 + rows].copy() for im in images]
        tmp = np.empty_like(a[0])
        for r in range(n):
            for i in range(r % 2, n - 1, 2):
                np.minimum(a[i], a[i + 1], out=tmp)
                np.maximum(a[i], a[i + 1], out=a[i + 1])
                a[i], tmp = tmp, a[i]
        out[r0:r0 + rows] = (a[k] if n % 2 else
                             (a[k - 1].astype(np.uint16) + a[k] + 1) // 2)
    return out


class BackgroundCache:
    """Background samples of a decode of absolute frames [first, last].

    `add()` every decoded frame (or only the multiples of `BG_STEP`, as a
    seeking first pass does): the multiples are kept as grey uint8, cropped
    to `crop` = (x0, y0, x1, y1) of the full frame when given, the rest
    only advance the decode position. `ready(t)` once the decode has passed
    every sample frame of t (or `finish()` was called): a sample that was
    never added then simply counts as missing. `get(t)` gives (B_t, number
    of samples), B_t None below `BG_MIN`, cached per centre. `drop_before()`
    frees memory; asking afterwards for a background that needs a dropped
    sample raises rather than silently using fewer."""

    def __init__(self, first: int, last: int, crop=None):
        self.first, self.last = int(first), int(last)
        self.crop = None if crop is None else tuple(int(v) for v in crop)
        self.samples: dict[int, np.ndarray] = {}
        self.medians: dict[int, tuple[np.ndarray | None, int]] = {}
        self.seen = self.first - 1        # latest frame offered to add()
        self.dropped = self.first         # samples before it were freed
        self.finished = False

    def add(self, abs_frame: int, bgr_or_grey) -> bool:
        """Offer a decoded frame; True if it was kept as a sample."""
        f = int(abs_frame)
        self.seen = max(self.seen, f)
        if f % BG_STEP or not self.first <= f <= self.last or f < self.dropped:
            return False
        im = np.asarray(bgr_or_grey)
        if self.crop is not None:
            x0, y0, x1, y1 = self.crop
            im = im[y0:y1, x0:x1]
        self.samples[f] = np.ascontiguousarray(grey(im))
        return True

    def finish(self):
        """No more frames will come: every background is ready."""
        self.finished = True

    def ready(self, t: int) -> bool:
        need = bg_sample_frames(t, self.first, self.last)
        return self.finished or not need or self.seen >= need[-1]

    def get(self, t: int) -> tuple[np.ndarray | None, int]:
        c = bg_centre(t)
        if c in self.medians:
            return self.medians[c]
        if not self.ready(t):
            raise RuntimeError(f"background of frame {t} asked for before "
                               f"frame {bg_sample_frames(t, self.first, self.last)[-1]} "
                               f"was decoded")
        need = bg_sample_frames(t, self.first, self.last)
        if need and need[0] < self.dropped:
            raise RuntimeError(f"background of frame {t} needs frame {need[0]}, "
                               f"dropped (drop_before({self.dropped}))")
        have = [self.samples[f] for f in need if f in self.samples]
        out = (median_u8(have) if len(have) >= BG_MIN else None, len(have))
        self.medians[c] = out
        return out

    def drop_before(self, abs_frame: int):
        """Free the samples before `abs_frame` and the cached backgrounds
        that used one. Safe with t - BG_HALF_SPAN - BG_STEP when no frame
        before t will be asked for."""
        self.dropped = max(self.dropped, int(abs_frame))
        for f in [f for f in self.samples if f < self.dropped]:
            del self.samples[f]
        for c in list(self.medians):
            need = bg_sample_frames(c, self.first, self.last)
            if need and need[0] < self.dropped:
                del self.medians[c]

    @property
    def nbytes(self) -> int:
        return (sum(s.nbytes for s in self.samples.values())
                + sum(m.nbytes for m, _ in self.medians.values() if m is not None))


def motion_channels(prev_bgr, cur_bgr, nxt_bgr, bg_grey,
                    sigma_d: float | None = None,
                    sigma_f: float | None = None) -> np.ndarray:
    """(D-, D+, F) of frame t as H x W x 3 float32, from frames t-2, t,
    t+2 (BGR, or grey) and the background (grey, or BGR / 3 equal
    channels). None for t-2 / t+2 / B gives a zero channel. The sigmas
    default to `ball_heatmap.SIGMA_D` / `SIGMA_F`; pass 1.0 for the raw
    differences (in units of 255)."""
    planes = motion_planes(prev_bgr, cur_bgr, nxt_bgr, bg_grey, sigma_d=sigma_d,
                           sigma_f=sigma_f)
    return np.ascontiguousarray(planes.transpose(1, 2, 0))


def motion_planes(prev_bgr, cur_bgr, nxt_bgr, bg_grey, out=None,
                  sigma_d: float | None = None,
                  sigma_f: float | None = None) -> np.ndarray:
    """The same channels as 3 x H x W float32, written into `out` when given
    (e.g. a view of a network input, as `ball_heatmap.detect_v2` does)."""
    sd = bh.SIGMA_D if sigma_d is None else sigma_d
    sf = bh.SIGMA_F if sigma_f is None else sigma_f
    g = grey(cur_bgr).astype(np.float32)
    if out is None:
        out = np.empty((3,) + g.shape, np.float32)
    for k, other, s in ((0, prev_bgr, sd), (1, nxt_bgr, sd), (2, bg_grey, sf)):
        if other is None:
            out[k] = 0.0
            continue
        o = grey(other)
        if o.shape != g.shape:
            raise ValueError(f"motion input of shape {o.shape}, frame {g.shape}")
        np.subtract(g, o, out=out[k])
        out[k] /= np.float32(255.0 * s)
    return out


class RingBuffer:
    """The last `n` frames of a sequential decode by absolute frame: with
    n = 5, frames t-2 .. t+2 are at hand once t+2 is pushed."""

    def __init__(self, n: int = 5):
        self.n = n
        self.frames: OrderedDict[int, np.ndarray] = OrderedDict()

    def push(self, abs_frame: int, bgr):
        self.frames[int(abs_frame)] = bgr
        while len(self.frames) > self.n:
            self.frames.popitem(last=False)

    def get(self, abs_frame: int):
        return self.frames.get(int(abs_frame))

    def triple(self, t: int):
        """(t - OFFSET, t, t + OFFSET), None where not held."""
        return self.get(t - OFFSET), self.get(t), self.get(t + OFFSET)


# ---- Self-check ---------------------------------------------------------------

def _check():
    """A static textured pitch with a ball crossing it and a player standing
    still: the background drops the ball (it is never at one place for
    long) and keeps the player; the difference channels light up at the
    ball's positions and are zero at the edges of the range."""
    import time

    assert bg_centre(0) == 0 and bg_centre(24) == 0 and bg_centre(25) == 50
    assert bg_centre(74) == 50 and bg_centre(15000) == 15000
    s = bg_sample_frames(15030, 15000, 19499)                # centre 15050
    assert s == list(range(15000, 15401, BG_STEP)), s
    assert len(bg_sample_frames(15000, 15000, 19499)) == BG_MIN
    assert len(bg_sample_frames(10000, 0, 67000)) == 2 * BG_HALF_SPAN // BG_STEP + 1
    assert bg_sample_frames(19499, 15000, 19499) == list(range(19150, 19451, BG_STEP))
    # Medians: odd and even counts.
    a = [np.full((2, 2), v, np.uint8) for v in (5, 1, 9)]
    assert (median_u8(a) == 5).all()
    assert (median_u8(a + [np.full((2, 2), 6, np.uint8)]) == 6).all()   # (5+6)/2 up

    rng = np.random.default_rng(0)
    H, W, first, last = 60, 200, 1000, 1999
    pitch = rng.integers(60, 120, (H, W, 3)).astype(np.uint8)

    def ball_x(t):
        return 30 + 7 * (t - first) % 160    # 14 px per 2 frames; samples apart

    def frame(t):
        im = pitch.copy()
        cv2.rectangle(im, (20, 20), (26, 40), (30, 30, 200), -1)       # player
        cv2.circle(im, (ball_x(t), 30), 2, (255, 255, 255), -1)
        return im

    crop = (10, 5, 190, 55)
    cache, ring = BackgroundCache(first, last, crop=crop), RingBuffer()
    got = {}
    for t in range(first, 1500):
        im = frame(t)
        cache.add(t, im)
        ring.push(t, im)
        if t == 1102:
            # Frame 1100's background (centre 1100) needs frames up to 1450.
            assert not cache.ready(1100)
            try:
                cache.get(1100)
                raise AssertionError("an unready background was given")
            except RuntimeError:
                pass
            assert ring.get(1097) is None
            got["triple"] = ring.triple(1100)
    assert cache.ready(1100) and not cache.ready(1130)    # centre 1150 needs 1500
    bg, n = cache.get(1100)
    assert n == 10, n                                     # 750 .. 950 are outside
    assert bg.shape == (50, 180) and bg.dtype == np.uint8
    x0, y0, x1, y1 = crop
    bare = grey(pitch)[y0:y1, x0:x1]
    still = grey(frame(first))[y0:y1, x0:x1]
    # The ball's row away from the player is the bare pitch; the player stays.
    assert (bg[25, 20:] == bare[25, 20:]).all(), "the ball is left in B"
    assert (bg[15:35, 10:17] == still[15:35, 10:17]).all()
    assert cache.get(1100)[0] is bg                       # cached per centre
    # Too few samples at the start of a range: frame 1000's centre has 7
    # in a range starting at 1050.
    short = BackgroundCache(1050, 1999)
    for t in range(1050, 1400, BG_STEP):
        short.add(t, frame(t))
    short.finish()
    assert short.get(1000) == (None, 7)
    # Motion channels: D- / D+ at the ball now and where it was / will be,
    # F at the ball, zero where nothing moves.
    p, c, nx = got["triple"]
    assert p is not None and nx is not None
    m = motion_channels(p, c, nx, None, sigma_d=1.0, sigma_f=1.0)
    assert m.shape == (H, W, 3) and m.dtype == np.float32
    xb = ball_x(1100)
    assert m[30, xb, 0] > 0.3 and m[30, ball_x(1098), 0] < -0.3
    assert m[30, xb, 1] > 0.3 and m[30, ball_x(1102), 1] < -0.3
    assert np.abs(m[:, :, 2]).max() == 0                  # no background
    assert np.abs(m[:, :28, :2]).max() == 0               # player and pitch still
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = bg
    mf = motion_channels(None, c, None, full, sigma_d=1.0, sigma_f=1.0)
    assert np.abs(mf[:, :, :2]).max() == 0                # t-2, t+2 missing
    assert mf[30, xb, 2] > 0.3 and mf[30, xb + 40, 2] == 0
    # Default sigmas come from ball_heatmap, read at call time; a background
    # stored as three equal channels reads as the grey it was.
    md = motion_channels(p, c, nx, np.repeat(full[:, :, None], 3, 2))
    assert np.allclose(md[:, :, 0] * bh.SIGMA_D, m[:, :, 0], atol=1e-5)
    assert np.allclose(md[:, :, 2] * bh.SIGMA_F, mf[:, :, 2], atol=1e-5)
    # Dropping: a cached background goes with its first sample, and is
    # refused afterwards; after finish(), a sample never added is missing.
    cache.drop_before(1050)
    assert min(cache.samples) == 1050 and 1100 not in cache.medians
    try:
        cache.get(1100)
        raise AssertionError("a background with dropped samples was given")
    except RuntimeError:
        pass
    cache.finish()
    assert cache.get(1400)[1] == 9                        # 1050 .. 1450 of 1050 .. 1750
    # The sorting network is numpy's median (rounded half up), any count,
    # any number of rows; then the speed of one full-width median.
    for n in (8, 9, 14, 15):
        imgs = [rng.integers(0, 256, (45, 7), dtype=np.uint8) for _ in range(n)]
        ref = np.floor(np.median(np.stack(imgs), axis=0) + 0.5).astype(np.uint8)
        assert (median_u8(imgs) == ref).all(), n
    big = [rng.integers(0, 255, (1000, 4096), dtype=np.uint8) for _ in range(15)]
    t0 = time.time()
    median_u8(big)
    dt = time.time() - t0
    print(f"  motion_bg: background drops the moving ball and keeps the still "
          f"player, unready/dropped/short backgrounds refused, D and F at the "
          f"ball, zero at edges; median of 15 x 1000 x 4096 in {dt:.2f} s")


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _check()
