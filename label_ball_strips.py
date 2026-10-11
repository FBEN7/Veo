"""Codes for the ball strips (`build_ball_strips.py`), for
`train_ball_heatmap.py --strips`: one code per 4 x 4 cell of the strip's
t crop (80 x 80 int8),

    1 the ball is somewhere here (a bag), 0 ignore, -1 not the ball,
    -3 not the ball, and easily taken for it.

The codes come from each cell centre's ground point (`image_to_pitch`),
the players on frame t (GSR feet, with v1's body box sized through the
calibration: 1.9 m up, 0.3 m + 10 px to each side), the ball actions and
the release's ball track. Per kind:

- flight: the bag is the stretch of the pass line A->B where a ball that
  slows down should be (projections at [lo, hi] around t_hat,
  `build_ball_strips.bag_window`), 2.5 m to either side of the line or
  20 px of its image (the far side has 2 px per metre in depth), plus up
  to 1 m above. Bag cells on a body or within 1 m of feet are 0 (the ball
  may be there, hidden or not). Around it an ignore band: within 6 m or
  48 px of the whole line, and a 2 m curtain above it. Hard negatives,
  over the band: the bodies of all players 1 m or more from the bag's
  ground (passer and receiver too), and pitch landmarks (0.75 m, at
  least 12 px) outside the band. A strip is kept if 85% of its bag (on a
  640 px canvas around the crop) is inside it, else with the narrower
  stretch, else dropped.
- lofted (HIGH PASS, CROSS): no bag. The ball is somewhere above the
  line: cells up to clip(0.3 L, 3, 15) m above it or within 4 m / 32 px
  of it are 0, so are the players there; other players are -3, the rest
  -1.
- action: v1's rule (`build_ball_labels.py`), on 320 px: 1.5 m and 0.5 m
  up around the actor's feet and the track point (dropped if the two are
  more than 3 m apart), an ignore band of 0.5 m, other players' bodies
  and the actor's head -3. At a THROW IN the ball is in the hands: 2.6 m
  up, and the head is not a negative.
- dribble: 1.5 m around the dribbler; positive only, weight 0.5.
- free: -1, bodies and landmarks -3. With `--coco`, a ball the COCO
  detector sees (yolov8s, confidence 0.25, on the t crop upsampled x 2)
  away from the landmarks (1 m) gets a 1 m ignore disk: a spare ball is
  never called "not a ball".

Writes `labels.parquet` (file, match, half, frame, kind, category, x0,
y0, positive_only, weight, holdout, and the strip's n_bg, has_prev,
has_next), `labels_codes.npy` (row-aligned) and `labels_meta.json`
(`LABEL_CONSTANTS`). 300 strips (1.5%, stratified by kind) are held out
for the training monitor.

Also:
- `--sigma`: the scales of the motion channels (`motion_bg`) on 1,000
  strips' pitch pixels, for `ball_heatmap.SIGMA_D` / `SIGMA_F` (printed,
  not written);
- `--compare-window W --strips-a A --strips-b B`: the same statistics for
  strips cut at the same places from the source stream (A) and from the
  local window (B), and the ratio of the D scales;
- `--t-stats DIR`: where the visually confirmed balls in flight of the T
  judgements (DIR/tiles_ff) are, relative to their pass;
- `--montage PNG`: strips with their codes drawn over the t crop.

    python label_ball_strips.py --strips DIR [--coco] [--holdout 300]
    python label_ball_strips.py --strips DIR --sigma
    python label_ball_strips.py --check
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import build_ball_strips as bs
from build_ball_labels import HEIGHT, MIN_INSIDE, R_ACTION, R_DRIBBLE
from build_ball_strips import (BAG_HEIGHT_M, BAG_PERP_M, BAG_PX, CELLS,
                               LANDMARKS, SIZE, THROW_HEIGHT, Half, bag_window,
                               scales)
from src import soccertrack_v2 as st
from src.ball_heatmap import (BALL_M, NEVER, STRIDE, V2_TRAIN, _jsonable,
                              ellipse, pitch_to_image)
from src.motion_bg import BG_MIN, grey, motion_channels
from src.paths import DATA_DIR

CANVAS_PAD = 40                   # cells around the crop: a 640 px canvas
FEET_M = 1.0                      # bag cells this near feet: 0
IGNORE_M, IGNORE_PX, CURTAIN_M = 6.0, 48.0, 2.0
HARD_FROM_BAG_M = 1.0             # bodies this far from the bag: -3
MIN_INSIDE_FLIGHT = 0.85
LOFT_H = (0.3, 3.0, 15.0)         # curtain height clip(0.3 L, 3, 15) m
LOFT_M, LOFT_PX = 4.0, 32.0
LANDMARK_M, LANDMARK_PX = 0.75, 12.0
COCO_CONF, COCO_UP, COCO_M, COCO_PX, COCO_LANDMARK_M = 0.25, 2, 1.0, 12.0, 1.0
V1_CAP_PX = 112.0                 # v1's cap on a region's extent
WEIGHT = {"flight": 1.0, "lofted": 1.0, "action": 1.0, "dribble": 0.5,
          "free": 1.0}
POSITIVE_ONLY = {"dribble"}
HOLDOUT, HOLDOUT_FRAC = 300, 0.015
SIGMA_STRIPS, SIGMA_PIXELS = 1000, 4096
ZERO_SHARE = 0.25                 # above: the std, not the MAD, scales a channel

LABEL_CONSTANTS = {
    "strip": {"size_px": SIZE, "cell_px": STRIDE, "jpeg_q": bs.QUALITY,
              "jitter_px": bs.JITTER, "slack_margin_px": bs.SLACK_PX},
    "flight": {"s": bs.FLIGHT_S, "s_jitter": bs.S_JITTER, "s_range": bs.S_RANGE,
               "t_hat": "s + 0.4 s (1 - s)", "window": bs.WINDOW,
               "narrow": bs.NARROW, "clear_passer_m": bs.CLEAR_A_M,
               "clear_receiver_m": bs.CLEAR_B_M, "t_bounds": bs.T_BOUNDS,
               "min_pass_m": bs.MIN_PASS_M, "min_flight_s": bs.MIN_FLIGHT_S,
               "max_intercept_s": bs.MAX_FLIGHT_S,
               "intercept_labels": sorted(bs.INTERCEPT),
               "bag_perp_m": BAG_PERP_M, "bag_px": BAG_PX,
               "bag_height_m": BAG_HEIGHT_M, "feet_m": FEET_M,
               "ignore_m": IGNORE_M, "ignore_px": IGNORE_PX,
               "curtain_m": CURTAIN_M, "hard_from_bag_m": HARD_FROM_BAG_M,
               "min_inside": MIN_INSIDE_FLIGHT,
               "canvas_px": SIZE + 2 * CANVAS_PAD * STRIDE},
    "lofted": {"labels": sorted(bs.LOFTED), "s": bs.LOFTED_S,
               "max_flight_s": bs.MAX_FLIGHT_S, "next_labels": sorted(bs.RECEIVE),
               "curtain_h": "clip(%.1f L, %.0f, %.0f) m" % LOFT_H,
               "ground_m": LOFT_M, "ground_px": LOFT_PX},
    "action": {"labels": sorted(bs.ACTION), "radius_m": R_ACTION, "height_m": HEIGHT,
               "throw_in_height_m": THROW_HEIGHT, "band_m": 0.5,
               "max_actor_track_m": bs.MAX_ACTOR_TRACK_M, "min_inside": MIN_INSIDE,
               "cap_px": V1_CAP_PX, "other_bodies_beyond_m": R_ACTION + 1.0},
    "dribble": {"gap_s": bs.DRIBBLE_GAP_S, "radius_m": R_DRIBBLE, "height_m": HEIGHT,
                "same_player_next": True, "min_inside": MIN_INSIDE},
    "free": {"per_frame": bs.FREE_PER_FRAME, "every": 2, "mix": bs.FREE_MIX,
             "min_m": bs.FREE_MIN_M, "tries": bs.FREE_TRIES,
             "cluster": [bs.CLUSTER_M, bs.CLUSTER_N],
             "boundary": [bs.BOUNDARY_IN_M, bs.BOUNDARY_OUT_PX]},
    "landmarks": {"radius_m": LANDMARK_M, "min_px": LANDMARK_PX, "points": LANDMARKS},
    "coco": {"model": "yolov8s.pt", "conf": COCO_CONF, "upsample": COCO_UP,
             "radius_m": COCO_M, "min_px": COCO_PX, "landmark_m": COCO_LANDMARK_M},
    "weight": WEIGHT, "positive_only": sorted(POSITIVE_ONLY),
    "holdout": {"n": HOLDOUT, "frac": HOLDOUT_FRAC},
    "sigma": {"strips": SIGMA_STRIPS, "pixels_per_strip": SIGMA_PIXELS,
              "scale": "1.4826 MAD; std if > %.0f%% exact zeros" % (100 * ZERO_SHARE)},
}


# ---- Cells ------------------------------------------------------------------------

class Canvas:
    """The cells of a crop at (x0, y0) with `pad` cells around it: centres
    in frame pixels, ground points (NaN above the horizon), local scale
    along the pitch (px per metre)."""

    def __init__(self, cal: dict, x0: int, y0: int, pad: int = 0):
        self.pad, self.m = pad, CELLS + 2 * pad
        self.ox, self.oy = x0 - pad * STRIDE, y0 - pad * STRIDE
        jj, self.ii = np.meshgrid(np.arange(self.m), np.arange(self.m))
        self.u = self.ox + (jj + 0.5) * STRIDE
        self.v = self.oy + (self.ii + 0.5) * STRIDE
        P = st.image_to_pitch(np.c_[self.u.ravel(), self.v.ravel()], cal)
        self.P = P.reshape(self.m, self.m, 2)
        self.ok = np.isfinite(self.P).all(axis=2)
        _, sx, _ = scales(np.where(self.ok[..., None], self.P, 0.0).reshape(-1, 2), cal)
        self.sx = np.where(self.ok, sx.reshape(self.m, self.m), np.nan)

    def inner(self, a: np.ndarray) -> np.ndarray:
        return a[self.pad:self.pad + CELLS, self.pad:self.pad + CELLS]

    def ground_dist(self, xy) -> np.ndarray:
        """Metres from each cell's ground point to the nearest of `xy`
        (inf above the horizon)."""
        d = np.full((self.m, self.m), np.inf)
        for p in np.asarray(xy, float).reshape(-1, 2):
            d = np.minimum(d, np.hypot(*(self.P - p).transpose(2, 0, 1)))
        return np.where(self.ok, d, np.inf)

    def line_dist(self, uv) -> np.ndarray:
        """Pixels from each cell centre to the polyline `uv`; pieces far
        outside the canvas left out."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        if len(uv) == 1:
            return np.hypot(self.u - uv[0, 0], self.v - uv[0, 1])
        lo = np.array([self.ox, self.oy]) - 64
        hi = lo + self.m * STRIDE + 128
        d = np.full((self.m, self.m), np.inf)
        for p, q in zip(uv[:-1], uv[1:]):
            if (np.maximum(p, q) < lo).any() or (np.minimum(p, q) > hi).any():
                continue
            e = q - p
            t = np.clip(((self.u - p[0]) * e[0] + (self.v - p[1]) * e[1])
                        / max(e @ e, 1e-12), 0, 1)
            d = np.minimum(d, np.hypot(self.u - p[0] - t * e[0],
                                       self.v - p[1] - t * e[1]))
        return d

    def above(self, uv, h_px) -> np.ndarray:
        """Cells whose centres are at most `h_px` above one of the points
        `uv` in the same column (a point below the canvas reaches up into
        it)."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        h = np.broadcast_to(np.asarray(h_px, float), (len(uv),))
        j = np.floor((uv[:, 0] - self.ox) / STRIDE).astype(int)
        r = (uv[:, 1] - self.oy) / STRIDE - 0.5           # in cell-centre rows
        i = np.floor(r).astype(int)
        top = r - h / STRIDE
        keep = (j >= 0) & (j < self.m) & (i >= 0) & (top <= self.m - 1) & np.isfinite(top)
        T = np.full((self.m, self.m), np.inf)
        np.minimum.at(T, (np.minimum(i[keep], self.m - 1), j[keep]), top[keep])
        return self.ii >= np.minimum.accumulate(T[::-1], axis=0)[::-1]

    def disk(self, uv, half_w: float, half_h: float) -> np.ndarray:
        return (((self.u - uv[0]) / half_w) ** 2
                + ((self.v - uv[1]) / half_h) ** 2) <= 1


def body_slices(uv, sx: float, sy: float, ox: float, oy: float, m: int,
                head_only: bool = False):
    """Cells (row and column slices of an m x m grid at (ox, oy)) of a
    player's body box with feet at pixel `uv` -- v1's `body_box`: 1.9 m up,
    0.3 m + 10 px to each side and 0.3 m + 10 px below; or the head band
    (1.2-1.9 m up). None if off the grid."""
    u, v = uv[0] - ox, uv[1] - oy
    if head_only:
        hw, top, bot = 0.3 * sx, v - 1.9 * sx, v - 1.2 * sx
    else:
        hw, top, bot = 0.3 * sx + 10, v - 1.9 * sx, v + 0.3 * sy + 10
    c0, c1 = int(max((u - hw) // STRIDE, 0)), int(min((u + hw) // STRIDE + 1, m))
    r0, r1 = int(max(top // STRIDE, 0)), int(min(bot // STRIDE + 1, m))
    return (slice(r0, r1), slice(c0, c1)) if c0 < c1 and r0 < r1 else None


def line_uv(A, B, t0: float, t1: float, cal: dict, step_m: float = 0.05):
    """The image of A->B over [t0, t1], every `step_m` metres, with the
    local scale along the pitch at each point."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    L = float(np.hypot(*(B - A)))
    n = int(np.clip(np.ceil(L * (t1 - t0) / step_m), 1, 4000)) + 1
    uv, sx, _ = scales(A + np.linspace(t0, t1, n)[:, None] * (B - A), cal)
    return uv, sx


def _coarse(uv, every: int = 10):
    """Every `every`-th point of a polyline, and its last."""
    return np.vstack([uv[::every], uv[-1:]])


def landmark_mask(h: Half, cv: Canvas) -> np.ndarray:
    """Landmark disks: 0.75 m (at least 12 px) in each image direction."""
    if not hasattr(h, "_landmarks"):
        h._landmarks = scales(LANDMARKS, h.cal)
    uv, sx, sy = h._landmarks
    out = np.zeros((cv.m, cv.m), bool)
    for p, a, b in zip(uv, np.maximum(LANDMARK_M * sx, LANDMARK_PX),
                       np.maximum(LANDMARK_M * sy, LANDMARK_PX)):
        if (cv.ox - a <= p[0] <= cv.ox + cv.m * STRIDE + a
                and cv.oy - b <= p[1] <= cv.oy + cv.m * STRIDE + b):
            out |= cv.disk(p, a, b)
    return out


def _people(h: Half, f: int):
    """Feet (metres), their pixels and scales, ids, on frame f."""
    xy, ids = h.people.at(int(f))
    if not len(xy):
        return xy, np.zeros((0, 2)), np.zeros(0), np.zeros(0), ids
    uv, sx, sy = scales(xy, h.cal)
    return xy, uv, sx, sy, ids


# ---- Codes per kind -------------------------------------------------------------

def flight_codes(r, h: Half):
    """(codes, narrow) or None; see the module docstring."""
    A, B = np.array([r.a_x, r.a_y]), np.array([r.b_x, r.b_y])
    d = B - A
    L = float(np.hypot(*d))
    cv = Canvas(h.cal, r.x0, r.y0, pad=CANVAS_PAD)
    rel = cv.P - A
    tp = (rel @ d) / (L * L)
    perp = np.abs(rel[..., 0] * d[1] - rel[..., 1] * d[0]) / L
    seg = np.hypot(*(rel - np.clip(tp, 0, 1)[..., None] * d).transpose(2, 0, 1))
    whole, wsx = line_uv(A, B, 0.0, 1.0, h.cal)
    band = ((cv.ok & (seg <= IGNORE_M)) | (cv.line_dist(_coarse(whole)) <= IGNORE_PX)
            | cv.above(whole, CURTAIN_M * wsx))
    xy, uv, sx, sy, _ = _people(h, r.frame)
    boxes = [body_slices(uv[k], sx[k], sy[k], cv.ox, cv.oy, cv.m) for k in range(len(xy))]
    on_body = np.zeros((cv.m, cv.m), bool)
    for b in boxes:
        if b is not None:
            on_body[b] = True
    near_feet = cv.ground_dist(xy) <= FEET_M if len(xy) else on_body & False
    marks = landmark_mask(h, cv) & ~band
    for narrow in (False, True):
        lo, hi = bag_window(r.t_hat, L, narrow)
        if lo >= hi:
            continue
        sub, _ = line_uv(A, B, lo, hi, h.cal, step_m=0.5)
        base = cv.ok & (tp >= lo) & (tp <= hi) & (
            (perp <= BAG_PERP_M) | (cv.line_dist(sub) <= BAG_PX))
        if not base.any():
            continue
        bag = base | cv.above(np.c_[cv.u[base], cv.v[base]], BAG_HEIGHT_M * cv.sx[base])
        if cv.inner(bag).sum() < MIN_INSIDE_FLIGHT * bag.sum():
            continue
        codes = np.full((cv.m, cv.m), -1, np.int8)
        codes[band] = 0
        G = cv.P[base]
        for k, b in enumerate(boxes):
            if b is not None and np.hypot(*(G - xy[k]).T).min() >= HARD_FROM_BAG_M:
                codes[b] = -3
        codes[marks] = -3
        codes[bag] = 1
        codes[bag & (on_body | near_feet)] = 0
        c = cv.inner(codes)
        if (c == 1).any():
            return np.ascontiguousarray(c), narrow
    return None


def lofted_codes(r, h: Half) -> np.ndarray:
    A, B = np.array([r.a_x, r.a_y]), np.array([r.b_x, r.b_y])
    d = B - A
    L = float(np.hypot(*d))
    cv = Canvas(h.cal, r.x0, r.y0)
    uv, lsx = line_uv(A, B, 0.0, 1.0, h.cal)
    up = float(np.clip(LOFT_H[0] * L, LOFT_H[1], LOFT_H[2]))
    rel = cv.P - A
    tp = np.clip((rel @ d) / (L * L), 0, 1)
    seg = np.hypot(*(rel - tp[..., None] * d).transpose(2, 0, 1))
    curtain = (cv.above(uv, up * lsx) | (cv.ok & (seg <= LOFT_M))
               | (cv.line_dist(_coarse(uv)) <= LOFT_PX))
    codes = np.full((CELLS, CELLS), -1, np.int8)
    xy, puv, sx, sy, _ = _people(h, r.frame)
    inside = []
    for k in range(len(xy)):
        b = body_slices(puv[k], sx[k], sy[k], cv.ox, cv.oy, cv.m)
        if b is None:
            continue
        t = np.clip((xy[k] - A) @ d / (L * L), 0, 1)
        i = int((puv[k][1] - cv.oy) // STRIDE)
        j = int((puv[k][0] - cv.ox) // STRIDE)
        if np.hypot(*(xy[k] - A - t * d)) <= LOFT_M or (
                0 <= i < cv.m and 0 <= j < cv.m and curtain[i, j]):
            inside.append(b)
        else:
            codes[b] = -3
    codes[curtain] = 0
    for b in inside:
        codes[b] = 0
    return codes


def v1_region(cal, xy, x0, y0, radius, height=HEIGHT, grow=0.0, cap_above=V1_CAP_PX):
    """v1's region (`build_ball_labels.region`) on 320 px crops: cells of a
    grid of the crop and one crop around it inside the image of a ground
    disk of `radius` m around `xy`, `height` m up, grown by `grow` m."""
    uv, sx, sy = scales([xy], cal)
    uv, sx, sy = uv[0] - (x0 - SIZE, y0 - SIZE), sx[0], sy[0]
    r = radius + grow
    half_w = min(r * sx + 8, V1_CAP_PX + grow * sx)
    below = min(r * sy + 8, V1_CAP_PX + grow * sx)
    above = min(r * sy + height * sx + 8, cap_above + grow * sx)
    return ellipse((3 * CELLS, 3 * CELLS), uv, half_w, below, above)


def _inner(mask):
    return mask[CELLS:2 * CELLS, CELLS:2 * CELLS]


def action_codes(r, h: Half):
    f = int(r.frame)
    A = h.people.of(f, r.player_a)
    T, _ = h.track(f)
    if A is None or T is None or np.hypot(*(A - T)) > bs.MAX_ACTOR_TRACK_M:
        return None
    throw = r.label == "THROW IN"
    up = THROW_HEIGHT if throw else HEIGHT
    cap = V1_CAP_PX + (up - HEIGHT) * scales([A], h.cal)[1][0]

    def reg(xy, grow=0.0):
        return v1_region(h.cal, xy, r.x0, r.y0, R_ACTION, up, grow, cap)

    region, band = reg(T) | reg(A), reg(T, 0.5) | reg(A, 0.5)
    if _inner(region).sum() < MIN_INSIDE * region.sum():
        return None
    codes = np.where(_inner(band), 0, -1).astype(np.int8)
    codes[_inner(region)] = 1
    xy, uv, sx, sy, ids = _people(h, f)
    for k in range(len(xy)):
        actor = ids[k] == str(r.player_a)
        if actor and throw:
            continue
        if actor or np.hypot(*(xy[k] - A)) > R_ACTION + 1.0:
            b = body_slices(uv[k], sx[k], sy[k], r.x0, r.y0, CELLS, head_only=actor)
            if b is not None:
                block = codes[b]
                block[block == -1] = -3
    codes[_inner(region)] = 1
    return codes


def dribble_codes(r, h: Half):
    D = h.people.of(int(r.frame), r.player_a)
    if D is None:
        return None
    region = v1_region(h.cal, D, r.x0, r.y0, R_DRIBBLE)
    if _inner(region).sum() < MIN_INSIDE * region.sum():
        return None
    codes = np.zeros((CELLS, CELLS), np.int8)
    codes[_inner(region)] = 1
    return codes


def coco_balls(model, crop: np.ndarray) -> np.ndarray:
    """COCO 'sports ball' boxes on a crop upsampled x COCO_UP: (u, v,
    height, confidence) in crop pixels."""
    big = cv2.resize(crop, None, fx=COCO_UP, fy=COCO_UP, interpolation=cv2.INTER_LINEAR)
    res = model(big, imgsz=SIZE * COCO_UP, conf=COCO_CONF, classes=[32], verbose=False)[0]
    rows = [((x0 + x1) / 2 / COCO_UP, (y0 + y1) / 2 / COCO_UP, (y1 - y0) / COCO_UP, c)
            for (x0, y0, x1, y1), c in zip(res.boxes.xyxy.cpu().numpy(),
                                           res.boxes.conf.cpu().numpy())]
    return np.array(rows, float).reshape(-1, 4)


def free_codes(r, h: Half, balls=None) -> tuple[np.ndarray, int]:
    """(codes, number of COCO balls given an ignore disk); `balls` are
    COCO rows in crop pixels (`coco_balls`)."""
    cv = Canvas.__new__(Canvas)                  # only the grid is needed
    cv.m, cv.pad, cv.ox, cv.oy = CELLS, 0, r.x0, r.y0
    jj, cv.ii = np.meshgrid(np.arange(CELLS), np.arange(CELLS))
    cv.u, cv.v = r.x0 + (jj + 0.5) * STRIDE, r.y0 + (cv.ii + 0.5) * STRIDE
    codes = np.full((CELLS, CELLS), -1, np.int8)
    xy, uv, sx, sy, _ = _people(h, r.frame)
    for k in range(len(xy)):
        b = body_slices(uv[k], sx[k], sy[k], r.x0, r.y0, CELLS)
        if b is not None:
            codes[b] = -3
    codes[landmark_mask(h, cv)] = -3
    n = 0
    for u, v, size, _ in ([] if balls is None else balls):
        u, v = u + r.x0, v + r.y0
        g = st.image_to_pitch([[u, v + size / 2]], h.cal)[0]
        if np.isfinite(g).all() and np.hypot(*(np.asarray(LANDMARKS) - g).T).min() \
                < COCO_LANDMARK_M:
            continue                                  # a spot or a corner, not a ball
        a = b = COCO_PX
        if np.isfinite(g).all():
            _, gsx, gsy = scales([g], h.cal)
            a, b = max(COCO_M * gsx[0], COCO_PX), max(COCO_M * gsy[0], COCO_PX)
        codes[cv.disk((u, v), a, b)] = 0
        n += 1
    return codes, n


def split(strip: np.ndarray):
    """A strip's four crops: t-2, t, t+2 (BGR) and the background (grey)."""
    return (strip[:, :SIZE], strip[:, SIZE:2 * SIZE], strip[:, 2 * SIZE:3 * SIZE],
            grey(strip[:, 3 * SIZE:]))


# ---- Labelling --------------------------------------------------------------------

def holdout_mask(kinds: pd.Series, n: int = HOLDOUT, seed: int = 0) -> np.ndarray:
    """`n` strips (at most HOLDOUT_FRAC of them, at least one), stratified by
    kind (largest remainders)."""
    N = len(kinds)
    out = np.zeros(N, bool)
    if not N:
        return out
    n = min(n, max(1, int(np.ceil(HOLDOUT_FRAC * N))))
    counts = kinds.value_counts().sort_index()
    quota = counts / N * n
    take = np.floor(quota).astype(int)
    for k in (quota - take).sort_values(ascending=False).index[:n - take.sum()]:
        take[k] += 1
    rng = np.random.default_rng(seed)
    for kind, k in take.items():
        idx = np.flatnonzero(kinds.to_numpy() == kind)
        out[rng.choice(idx, min(k, len(idx)), replace=False)] = True
    return out


def label(root: Path, coco: bool = False, holdout: int = HOLDOUT, seed: int = 0,
          allow_any: bool = False) -> pd.DataFrame:
    table = pd.read_parquet(root / "strips.parquet")
    table["match"] = table.match.astype(str)
    bad = table[[int(h) not in V2_TRAIN.get(m, ()) for m, h in zip(table.match, table.half)]
                ] if len(table) else table
    if len(bad) and not allow_any:
        raise SystemExit(f"{len(bad)} strips of halves not trained on "
                         f"({sorted(set(bad.match))}); --allow-any to label them anyway")
    model = None
    if coco:
        from ultralytics import YOLO

        model = YOLO("yolov8s.pt")
    rows, codes, stats = [], [], {}
    t0 = time.time()
    for (match, half, source), part in table.groupby(["match", "half", "source"]):
        h = Half(match, int(half), window=None if source == "stream" else source)
        for r in part.itertuples():
            st_k = stats.setdefault(r.kind, {"strips": 0, "kept": 0, "narrow": 0,
                                             "coco": 0, "pos": [], "hard": []})
            st_k["strips"] += 1
            narrow, n_coco = False, 0
            if r.kind == "flight":
                out = flight_codes(r, h)
                c, narrow = (None, False) if out is None else out
            elif r.kind == "lofted":
                c = lofted_codes(r, h)
            elif r.kind == "action":
                c = action_codes(r, h)
            elif r.kind == "dribble":
                c = dribble_codes(r, h)
            else:
                balls = None
                if model is not None:
                    strip = cv2.imread(str(root / r.file))
                    balls = coco_balls(model, split(strip)[1])
                c, n_coco = free_codes(r, h, balls)
            if c is None:
                continue
            st_k["kept"] += 1
            st_k["narrow"] += int(narrow)
            st_k["coco"] += n_coco
            st_k["pos"].append(int((c == 1).sum()))
            st_k["hard"].append(int((c == -3).sum()))
            rows.append({"file": r.file, "match": match, "half": int(half),
                         "frame": int(r.frame), "kind": r.kind, "category": r.category,
                         "label": r.label, "x0": int(r.x0), "y0": int(r.y0),
                         "positive_only": r.kind in POSITIVE_ONLY,
                         "weight": WEIGHT[r.kind], "n_bg": int(r.n_bg),
                         "has_prev": bool(r.has_prev), "has_next": bool(r.has_next),
                         "narrow": bool(narrow), "coco_balls": n_coco,
                         "source": source})
            codes.append(c)
        print(f"  {match} half {half} ({source}): {len(part)} strips, "
              f"{time.time() - t0:.0f} s", flush=True)
    out = pd.DataFrame(rows)
    out["holdout"] = holdout_mask(out.kind, holdout, seed) if len(out) else []
    out.to_parquet(root / "labels.parquet")
    np.save(root / "labels_codes.npy",
            np.stack(codes) if codes else np.zeros((0, CELLS, CELLS), np.int8))
    (root / "labels_meta.json").write_text(json.dumps(_jsonable({
        "constants": LABEL_CONSTANTS, "coco": coco, "seed": seed,
        "strips": int(len(table)), "labelled": int(len(out)),
        "holdout": int(out.holdout.sum()) if len(out) else 0}), indent=1))
    print_stats(stats, out)
    return out


def print_stats(stats: dict, out: pd.DataFrame):
    for kind in bs.KINDS:
        s = stats.get(kind)
        if not s:
            continue
        extra = ""
        if kind == "flight":
            extra = f", {s['narrow']} on the narrower stretch"
        if kind == "free":
            extra = f", {s['coco']} COCO ignore disks"
        pos = f"bag cells median {np.median(s['pos']):.0f}" if s["pos"] and \
            kind not in ("lofted", "free") else ""
        hard = f"-3 cells median {np.median(s['hard']):.0f}" if s["hard"] else ""
        print(f"  {kind}: {s['kept']} of {s['strips']} kept{extra}; "
              f"{', '.join(x for x in (pos, hard) if x)}", flush=True)
    if len(out):
        print(f"  {len(out)} strips labelled, {int(out.holdout.sum())} held out "
              f"({out[out.holdout].kind.value_counts().to_dict()})", flush=True)


# ---- Motion scales ----------------------------------------------------------------

def pitch_mask(poly, x0: int, y0: int) -> np.ndarray:
    m = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly) - (x0, y0)).astype(np.int32)], 1)
    return m.astype(bool)


def channel_values(root: Path, rows: pd.DataFrame, rng, per_strip: int = SIGMA_PIXELS,
                   pick=None):
    """Raw motion channels (sigma 1: units of 255) at pitch pixels of each
    strip: {"D": values of D- and D+ where t-2 / t+2 exist, "F": where the
    background has >= BG_MIN samples}. `pick(r)` gives the pixel indices
    (default: up to `per_strip` random pitch pixels)."""
    out = {"D": [], "F": []}
    polys = {}
    for r in rows.itertuples():
        key = (str(r.match), int(r.half), int(r.width), int(r.height))
        if key not in polys:
            polys[key] = bs.outline(*key)
        prev, cur, nxt, bg = split(cv2.imread(str(root / r.file)))
        m = motion_channels(prev if r.has_prev else None, cur,
                            nxt if r.has_next else None,
                            bg if r.n_bg >= BG_MIN else None, sigma_d=1.0, sigma_f=1.0)
        if pick is None:
            idx = np.flatnonzero(pitch_mask(polys[key], r.x0, r.y0))
            if not len(idx):
                continue
            idx = rng.choice(idx, min(per_strip, len(idx)), replace=False)
        else:
            idx = pick(r, polys[key])
        m = m.reshape(-1, 3)[idx]
        if r.has_prev:
            out["D"].append(m[:, 0])
        if r.has_next:
            out["D"].append(m[:, 1])
        if r.n_bg >= BG_MIN:
            out["F"].append(m[:, 2])
    return {k: np.concatenate(v) if v else np.zeros(0, np.float32) for k, v in out.items()}


def scale_stats(x: np.ndarray) -> dict:
    """Robust std (1.4826 MAD), std, |x| quantiles and the share of exact
    zeros (x264 copies still blocks; JPEG does not)."""
    if not len(x):
        return {"n": 0}
    med = np.median(x)
    a = np.abs(x)
    return {"n": int(len(x)), "robust": float(1.4826 * np.median(np.abs(x - med))),
            "std": float(x.std()), "mean_abs": float(a.mean()),
            "p99": float(np.quantile(a, 0.99)), "p999": float(np.quantile(a, 0.999)),
            "zero": float((x == 0).mean())}


def _fmt(s: dict) -> str:
    if not s.get("n"):
        return "no values"
    return (f"robust std {s['robust']:.4f}, std {s['std']:.4f}, mean |x| "
            f"{s['mean_abs']:.4f}, |x| 99% {s['p99']:.3f} / 99.9% {s['p999']:.3f}, "
            f"exactly 0 {s['zero']:.0%} ({s['n']} px)")


def measure_sigma(root: Path, n: int = SIGMA_STRIPS, seed: int = 0) -> dict:
    """The motion channels' scales on `n` random strips' pitch pixels:
    printed as the two lines of `src/ball_heatmap.py` to edit by hand."""
    table = pd.read_parquet(root / "strips.parquet")
    rows = table.sample(min(n, len(table)), random_state=seed)
    vals = channel_values(root, rows, np.random.default_rng(seed))
    out = {k: scale_stats(v) for k, v in vals.items()}
    print(f"  motion channels on {len(rows)} strips' pitch pixels (units of 255):")
    for k, name in (("D", "D (t-2, t+2)"), ("F", "F (background)")):
        print(f"    {name}: {_fmt(out[k])}")
    if out["D"].get("n") and out["F"].get("n"):
        print("  for src/ball_heatmap.py (edit by hand):")
        for k in ("D", "F"):
            value, why = sigma_of(out[k])
            print(f"SIGMA_{k} = {value:.4f}    # {why}")
    return out


def sigma_of(s: dict) -> tuple[float, str]:
    """The scale to normalise a channel by: its robust std, unless more than
    ZERO_SHARE of the values are exactly 0 (codec-copied still blocks), when
    the MAD is 0 or nearly and the std is used instead."""
    if s["zero"] > ZERO_SHARE or s["robust"] <= 0:
        return s["std"], (f"std: {s['zero']:.0%} exact zeros, robust std "
                          f"{s['robust']:.4f}")
    return s["robust"], f"robust std (1.4826 MAD); std {s['std']:.4f}"


def compare_window(window: str, dir_a: Path, dir_b: Path, seed: int = 0) -> dict:
    """Strips cut at the same frames and places from the source stream (A)
    and from the local window (B, `build_ball_strips.py --window W --like
    A`): the motion channels' statistics on the same pitch pixels, the
    ratio of the D scales (B / A), and the codec gap between the t crops."""
    info = json.loads((DATA_DIR / "soccertrack_v2" / f"{window}.json").read_text())
    match, half = str(info["match"]), int(info["half"])
    lo, hi = int(info["start_frame"]), int(info["start_frame"]) + int(info["frames"]) - 1
    key = ["match", "half", "frame", "kind", "x0", "y0"]

    def load(d):
        t = pd.read_parquet(Path(d) / "strips.parquet")
        t["match"] = t.match.astype(str)
        return t[(t.match == match) & (t.half == half) & t.frame.between(lo, hi)]

    a, b = load(dir_a), load(dir_b)
    pairs = a.merge(b, on=key, suffixes=("_a", "_b"))
    if not len(pairs):
        print(f"  {window}: no strips in common")
        return {}
    rng = np.random.default_rng(seed)
    picks = {}

    def pick(r, poly):
        k = (r.frame, r.x0, r.y0)
        if k not in picks:
            idx = np.flatnonzero(pitch_mask(poly, r.x0, r.y0))
            picks[k] = rng.choice(idx, min(SIGMA_PIXELS, len(idx)), replace=False)
        return picks[k]

    def side(s):
        cols = {c[:-2]: pairs[c] for c in pairs.columns if c.endswith(s)}
        return pd.DataFrame({**{k: pairs[k] for k in key}, **cols})

    # Both sides on the same pixels, and only channels present on both.
    both = side("_a")
    for c in ("has_prev", "has_next"):
        both[c] = pairs[c + "_a"] & pairs[c + "_b"]
    both["n_bg"] = np.minimum(pairs.n_bg_a, pairs.n_bg_b)
    va = channel_values(Path(dir_a), both.assign(file=pairs.file_a), rng, pick=pick)
    vb = channel_values(Path(dir_b), both.assign(file=pairs.file_b), rng, pick=pick)
    sa = {k: scale_stats(v) for k, v in va.items()}
    sb = {k: scale_stats(v) for k, v in vb.items()}
    gap = []
    for r in pairs.itertuples():
        ga = grey(split(cv2.imread(str(Path(dir_a) / r.file_a)))[1]).astype(int)
        gb = grey(split(cv2.imread(str(Path(dir_b) / r.file_b)))[1]).astype(int)
        gap.append(np.abs(ga - gb).mean())
    print(f"  {window}: {len(pairs)} strips at the same frames and places "
          f"(A {len(a)}, B {len(b)}); t crops differ by {np.mean(gap):.2f} grey "
          f"levels on average")
    for k in ("D", "F"):
        print(f"    {k} stream (A): {_fmt(sa[k])}")
        print(f"    {k} window (B): {_fmt(sb[k])}")
    ratio = {}
    if sa["D"].get("n") and sb["D"].get("n"):
        a_, b_ = sa["D"], sb["D"]
        ratio = {"std": b_["std"] / max(a_["std"], 1e-12),
                 "mean_abs": b_["mean_abs"] / max(a_["mean_abs"], 1e-12),
                 "robust": (b_["robust"] / a_["robust"]
                            if a_["robust"] > 0 and b_["robust"] > 0 else None)}
        rob = ("n/a (a MAD is 0: exact zeros)" if ratio["robust"] is None
               else f"{ratio['robust']:.2f}")
        print(f"  D-scale ratio window / stream: {ratio['std']:.2f} (std), "
              f"{ratio['mean_abs']:.2f} (mean |D|), {rob} (robust std); exact zeros "
              f"{a_['zero']:.0%} / {b_['zero']:.0%}. Outside 1/1.5-1.5, the "
              f"critique's x264 augmentation is due")
    return {"a": sa, "b": sb, "ratio": ratio, "codec_gap": float(np.mean(gap))}


# ---- The T judgements -------------------------------------------------------------

def t_stats(scratch: Path, tiles: str = "tiles_ff") -> pd.DataFrame:
    """Where the confirmed balls in flight of a judged tile set are,
    relative to their pass: the tiles judged "B" (not a resting ball "B*",
    not decoys) on frames of a completed ground PASS (10 m, 0.8 s, s in
    0.2-0.8), one per frame. Ground point = image_to_pitch(u, v + size/2),
    size = 0.22 m at the larger local scale. Windows of never-train matches
    (U, V) are left out, and only their tile counts are printed."""
    key = json.loads((scratch / tiles / "key.json").read_text())
    judged = {}
    for line in (scratch / tiles / "judged.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            k, j = line.split()[:2]
            judged[int(k)] = j
    rows, skipped, ctx = [], {}, {}
    for k in key:
        if k["source"] == "decoy" or judged.get(k["tile"]) != "B":
            continue
        w = k["window"]
        if w.split("_")[1] in NEVER:
            skipped[w] = skipped.get(w, 0) + 1
            continue
        if w not in ctx:
            info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
            ev = pd.read_csv(DATA_DIR / "soccertrack_v2" / f"{w}_events.csv") \
                .sort_values("frame", kind="stable").reset_index(drop=True)
            gt = pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{w}_gt.parquet")
            ctx[w] = (st.calibration(info["match"]), ev,
                      {f: g[["x", "y"]].to_numpy(float) for f, g in gt.groupby("frame")})
        cal, ev, people = ctx[w]
        f = int(k["frame"])
        before, after = ev[ev.frame <= f], ev[ev.frame > f]
        if before.empty or after.empty:
            continue
        a, b = before.iloc[-1], after.iloc[0]
        if a.label != "PASS" or b.side != a.side or b.player_id == a.player_id \
                or pd.isna(a.x) or pd.isna(b.x):
            continue
        A, B = np.array([a.x, a.y], float), np.array([b.x, b.y], float)
        L, dur = float(np.hypot(*(B - A))), (b.frame - a.frame) / st.FPS
        s = (f - a.frame) / (b.frame - a.frame)
        if L < 10 or dur < 0.8 or not 0.2 <= s <= 0.8:
            continue
        g0 = st.image_to_pitch([[k["u"], k["v"]]], cal)[0]
        if not np.isfinite(g0).all():
            continue
        _, sx0, sy0 = scales([g0], cal)
        size = BALL_M * max(sx0[0], sy0[0])
        g = st.image_to_pitch([[k["u"], k["v"] + size / 2]], cal)[0]
        if not np.isfinite(g).all():
            continue
        d = B - A
        t = float((g - A) @ d / (L * L))
        perp = float(np.hypot(*(g - A - np.clip(t, 0, 1) * d)))
        uv, _ = line_uv(A, B, 0, 1, cal, step_m=0.25)
        e = np.diff(uv, axis=0)
        q = np.clip(((k["u"] - uv[:-1, 0]) * e[:, 0] + (k["v"] - uv[:-1, 1]) * e[:, 1])
                    / np.maximum((e ** 2).sum(1), 1e-12), 0, 1)
        px = float(np.hypot(k["u"] - uv[:-1, 0] - q * e[:, 0],
                            k["v"] - uv[:-1, 1] - q * e[:, 1]).min())
        near = float(np.hypot(*(people[f] - g).T).min()) if f in people else np.nan
        rows.append({"window": w, "frame": f, "perp_m": perp, "px": px, "t": t,
                     "s": s, "t_minus_s": t - s, "near_m": near,
                     "far_side": bool(g[1] < 0)})
    df = pd.DataFrame(rows).drop_duplicates(["window", "frame"]) if rows else pd.DataFrame()
    print(f"  {tiles}: {len(df)} confirmed balls in flight on "
          f"{sorted(df.window.unique()) if len(df) else []}; left out (never-train "
          f"matches): {skipped or 'none'}")
    if len(df):
        q = lambda c, p: float(np.nanquantile(df[c], p))   # noqa: E731
        print(f"    perpendicular to the segment: median {q('perp_m', .5):.2f} m, "
              f"90% {q('perp_m', .9):.2f} m (far side median "
              f"{df[df.far_side].perp_m.median():.2f}, near "
              f"{df[~df.far_side].perp_m.median():.2f})")
        print(f"    pixels from the segment's image: median {q('px', .5):.1f}, "
              f"90% {q('px', .9):.1f}")
        print(f"    t - s: median {q('t_minus_s', .5):+.2f}, 5-95% "
              f"[{q('t_minus_s', .05):+.2f}, {q('t_minus_s', .95):+.2f}]")
        print(f"    nearest player: median {q('near_m', .5):.2f} m; "
              f">= 1 m from every player: {(df.near_m >= 1.0).mean():.0%}")
    return df


# ---- Montage ------------------------------------------------------------------------

COLOURS = {1: (0, 220, 0), 0: (0, 210, 255), -3: (0, 0, 255)}


def montage(root: Path, out: Path, n: int = 8, seed: int = 0) -> Path:
    """`n` labelled strips (spread over the kinds), one per row: the four
    crops, the codes drawn over the t crop (green bag, yellow ignore, red
    hard negative) and, for passes, the line A->B (cyan) with t_hat."""
    lab = pd.read_parquet(root / "labels.parquet")
    codes = np.load(root / "labels_codes.npy")
    strips = pd.read_parquet(root / "strips.parquet").set_index("file")
    rng = np.random.default_rng(seed)
    order = []
    pools = {k: list(rng.permutation(np.flatnonzero(lab.kind.to_numpy() == k)))
             for k in bs.KINDS if (lab.kind == k).any()}
    while len(order) < min(n, len(lab)):
        for k in list(pools):
            if pools[k] and len(order) < n:
                order.append(pools[k].pop())
    cals = {}
    rows = []
    for i in order:
        r = lab.iloc[i]
        s = strips.loc[r.file]
        strip = cv2.imread(str(root / r.file))
        t = strip[:, SIZE:2 * SIZE].copy()
        over = t.copy()
        big = np.kron(codes[i], np.ones((STRIDE, STRIDE), np.int8))
        for c, col in COLOURS.items():
            over[big == c] = col
        t = cv2.addWeighted(t, 0.55, over, 0.45, 0)
        if r.kind in ("flight", "lofted") and np.isfinite(s.a_x):
            cal = cals.setdefault(r.match, st.calibration(r.match))
            uv, _ = line_uv((s.a_x, s.a_y), (s.b_x, s.b_y), 0, 1, cal, step_m=0.25)
            pts = np.round(uv - (r.x0, r.y0)).astype(np.int32)
            cv2.polylines(t, [pts.reshape(-1, 1, 2)], False, (255, 255, 0), 1)
            if np.isfinite(s.t_hat):
                p = pitch_to_image([np.array([s.a_x, s.a_y]) + s.t_hat
                                    * np.array([s.b_x - s.a_x, s.b_y - s.a_y])], cal)[0]
                cv2.circle(t, tuple(np.round(p - (r.x0, r.y0)).astype(int)), 4,
                           (255, 0, 255), 1)
        row = strip.copy()
        row[:, SIZE:2 * SIZE] = t
        cv2.line(row, (SIZE, 0), (SIZE, SIZE), (255, 255, 255), 1)
        cv2.line(row, (2 * SIZE, 0), (2 * SIZE, SIZE), (255, 255, 255), 1)
        cv2.line(row, (3 * SIZE, 0), (3 * SIZE, SIZE), (255, 255, 255), 1)
        bar = np.zeros((26, 4 * SIZE, 3), np.uint8)
        text = (f"{r.kind} {r.category} | {r.match}-{r.half} f{r.frame} "
                f"({r.x0},{r.y0}) | n_bg {r.n_bg} prev {int(r.has_prev)} "
                f"next {int(r.has_next)} | bag {(codes[i] == 1).sum()} "
                f"ign {(codes[i] == 0).sum()} hard {(codes[i] == -3).sum()}"
                + (" narrow" if r.get("narrow") else "")
                + (f" s {s.s:.2f} t_hat {s.t_hat:.2f}" if np.isfinite(s.t_hat) else ""))
        cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([bar, row]))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strips", help="a build_ball_strips.py output dir")
    ap.add_argument("--coco", action="store_true",
                    help="ignore disks at COCO balls on ball-free strips (yolov8s)")
    ap.add_argument("--holdout", type=int, default=HOLDOUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-any", action="store_true",
                    help="label strips of halves not trained on (smoke tests)")
    ap.add_argument("--sigma", action="store_true",
                    help="measure the motion channels' scales instead of labelling")
    ap.add_argument("--compare-window")
    ap.add_argument("--strips-a")
    ap.add_argument("--strips-b")
    ap.add_argument("--t-stats", help="the scratchpad dir holding tiles_ff")
    ap.add_argument("--montage", help="draw labelled strips into this PNG")
    ap.add_argument("--montage-n", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        _check()
        return
    if args.t_stats:
        t_stats(Path(args.t_stats))
        return
    if args.compare_window:
        if not (args.strips_a and args.strips_b):
            ap.error("--compare-window needs --strips-a and --strips-b")
        compare_window(args.compare_window, Path(args.strips_a), Path(args.strips_b),
                       args.seed)
        return
    if not args.strips:
        ap.error("--strips DIR is needed")
    root = Path(args.strips)
    if args.sigma:
        measure_sigma(root, seed=args.seed)
        return
    if not args.montage or not (root / "labels.parquet").exists():
        label(root, args.coco, args.holdout, args.seed, args.allow_any)
    if args.montage:
        print(f"  montage: {montage(root, Path(args.montage), args.montage_n, args.seed)}")


# ---- Self-check -------------------------------------------------------------------

def _check():
    """Codes on the synthetic half of `build_ball_strips._check`: a flight
    strip's bag lies along the pass around t_hat, clear of feet and bodies,
    with players far from it hard negatives and an ignore band around; a
    crop that misses the bag is dropped; the lofted curtain is ignored; the
    v1 action rule (and the throw-in's tall bag); a positive-only dribble;
    free strips with bodies, landmarks and a COCO ignore disk; the holdout
    by kind; the motion scales of known noise."""
    import tempfile
    from types import SimpleNamespace as NS

    t_start = time.time()
    h, _ = bs.synthetic_half()
    cal = h.cal
    poly = np.array([[700, 120], [1700, 120], [2340, 660], [60, 660]], np.float32)
    box = bs.pitch_box((700, 2400), poly)
    rows = pd.DataFrame(bs.samples(h, bs.KINDS, box, poly, seed=0))
    # A flight strip: the completed pass p1 (-10, 0) -> p2 (10, 5).
    r = next(rows[(rows.kind == "flight") & (rows.category == "completed")].itertuples())
    c, narrow = flight_codes(r, h)
    assert c.shape == (CELLS, CELLS) and c.dtype == np.int8 and not narrow
    cv = Canvas(cal, r.x0, r.y0)
    A, B = np.array([r.a_x, r.a_y]), np.array([r.b_x, r.b_y])
    d = B - A
    pos = c == 1
    assert pos.sum() > 5
    tp = ((cv.P[pos] - A) @ d) / (d @ d)
    lo, hi = bag_window(r.t_hat, r.L)
    # Ground bag cells project inside [lo, hi]; cells added above may
    # project further (they are higher in the picture, i.e. beyond).
    assert np.quantile(tp, 0.5) > lo - 0.05 and np.quantile(tp, 0.5) < hi + 0.05
    assert (tp >= lo - 0.3).all()
    xy, ids = h.people.at(r.frame)
    assert (cv.ground_dist(xy)[pos] > FEET_M).all()
    # The bag's middle is near t_hat's point.
    mid = pitch_to_image([A + r.t_hat * d], cal)[0] - (r.x0, r.y0)
    ii, jj = np.nonzero(pos)
    assert np.hypot(np.median(jj) * STRIDE - mid[0], np.median(ii) * STRIDE - mid[1]) < 60
    # Ignore around the bag; hard negatives exist only off the band.
    assert (c == 0).sum() > pos.sum() and (c == -1).any()
    seg = np.hypot(*(cv.P - A - np.clip((cv.P - A) @ d / (d @ d), 0, 1)[..., None]
                     * d).transpose(2, 0, 1))
    tq = (cv.P - A) @ d / (d @ d)
    assert (c[cv.ok & (seg <= 1.0) & (tq > 0.2) & (tq < 0.8)] >= 0).all()
    # A crop far from the pass: no bag inside -> dropped.
    away = r._replace(x0=box[2] - SIZE, y0=box[1])
    assert flight_codes(away, h) is None
    # A crop that cuts the bag: only the narrower stretch (or nothing) fits.
    bb = bs.flight_bbox(A, B, lo, hi, cal)
    edge = r._replace(x0=int(np.clip(bb[2] - 0.6 * (bb[2] - bb[0]) - 0, box[0],
                                     box[2] - SIZE)))
    got = flight_codes(edge, h)
    assert got is None or got[1] or (got[0] == 1).sum() > 0
    # A player standing on the bag's line in the crop: his feet zone is 0,
    # his body not -3; a far one is -3.
    on_line = A + r.t_hat * d
    far = on_line + (0, -8.0)
    h2 = bs.Half(h.match, h.half, cal=cal, ev=h.ev, ball=h.ball, people=bs.People(
        pd.DataFrame([{"frame": r.frame, "player_id": p, "role": "player",
                       "x": x, "y": y} for p, (x, y) in
                      (("q1", on_line), ("q2", far))])))
    c2, _ = flight_codes(r, h2)
    uv, sx, sy = scales([on_line, far], cal)
    b_on = body_slices(uv[0], sx[0], sy[0], r.x0, r.y0, CELLS)
    b_far = body_slices(uv[1], sx[1], sy[1], r.x0, r.y0, CELLS)
    assert (c2[b_on] != -3).all() and (c2[b_on] != 1).all()
    if b_far is not None:
        assert (c2[b_far] == -3).all()

    # Lofted: the curtain above the line and the line itself are 0; no bag.
    r = next(rows[rows.kind == "lofted"].itertuples())
    c = lofted_codes(r, h)
    assert not (c == 1).any() and (c == 0).sum() > 50
    cv = Canvas(cal, r.x0, r.y0)
    A, B = np.array([r.a_x, r.a_y]), np.array([r.b_x, r.b_y])
    p = pitch_to_image([A + r.s * (B - A)], cal)[0] - (r.x0, r.y0)
    i, j = int(p[1] // STRIDE), int(p[0] // STRIDE)
    if 0 <= i < CELLS and 0 <= j < CELLS:
        assert (c[max(i - 3, 0):i + 1, j] == 0).all()

    # Action, v1's rule: bag at the actor, the actor's head and other
    # bodies -3; a throw-in's bag reaches higher and its head is not -3.
    acts = rows[rows.kind == "action"]
    r = next(acts[acts.label == "PASS"].itertuples())
    c = action_codes(r, h)
    assert (c == 1).sum() > 4 and (c == 0).any()
    r_t = next(acts[acts.label == "THROW IN"].itertuples())
    c_t = action_codes(r_t, h)
    c_p = action_codes(r_t._replace(label="PASS"), h)
    assert c_t is not None and c_p is not None
    assert (c_t == 1).sum() > (c_p == 1).sum()
    top = lambda m: np.nonzero((m == 1).any(axis=1))[0].min()   # noqa: E731
    assert top(c_t) < top(c_p)
    # Dribble: positive only.
    r = next(rows[rows.kind == "dribble"].itertuples())
    c = dribble_codes(r, h)
    assert set(np.unique(c)) == {0, 1}

    # Free: -1, bodies -3, a landmark -3 (centre spot), and a COCO ball
    # away from the landmarks gets an ignore disk; one on the spot does not.
    spot = pitch_to_image([(0.0, 0.0)], cal)[0]
    x0, y0 = int(spot[0] - 160), int(spot[1] - 160)
    r = NS(x0=x0, y0=y0, frame=1100)
    c, n = free_codes(r, h)
    assert n == 0 and c[40, 40] == -3 and (c == -1).sum() > 0.8 * CELLS * CELLS
    ball = pitch_to_image([(3.0, -3.0)], cal)[0] - (x0, y0)
    balls = np.array([[160.0, 160.0, 6.0, 0.9], [ball[0], ball[1], 6.0, 0.5]])
    c, n = free_codes(r, h, balls)
    i, j = int(ball[1] // STRIDE), int(ball[0] // STRIDE)
    assert n == 1 and c[i, j] == 0 and c[40, 40] == -3

    # Holdout: by kind, about 1.5%, at least one.
    kinds = pd.Series(["flight"] * 600 + ["free"] * 300 + ["dribble"] * 100)
    m = holdout_mask(kinds, 300, 0)
    assert m.sum() == 15 and kinds[m].value_counts().to_dict() == \
        {"flight": 9, "free": 4, "dribble": 2}
    assert holdout_mask(kinds[:20], 300, 0).sum() == 1

    # Motion scales: strips of a still texture with gaussian noise of known
    # sigma (on top of JPEG's own) -> D's robust std near sqrt(2) sigma.
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = cv2.GaussianBlur(rng.integers(60, 140, (SIZE, SIZE, 3)).astype(np.uint8),
                                (0, 0), 3)
        meta = []
        for k in range(6):
            crops = [np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
                     for _ in range(3)]
            strip = bs.assemble(crops, grey(base))
            name = f"free_118575_1_{1000 + k:06d}_700_200.jpg"
            cv2.imwrite(str(tmp / name), strip, [cv2.IMWRITE_JPEG_QUALITY, 98])
            meta.append({"file": name, "match": "118575", "half": 1, "frame": 1000 + k,
                         "kind": "free", "x0": 700, "y0": 200, "n_bg": 15,
                         "has_prev": True, "has_next": True, "width": 2400,
                         "height": 700})
        pd.DataFrame(meta).to_parquet(tmp / "strips.parquet")
        bs_outline = bs.outline
        bs.outline = lambda *a: poly.tolist()        # the synthetic pitch
        try:
            vals = channel_values(tmp, pd.DataFrame(meta), rng)
        finally:
            bs.outline = bs_outline
        s = scale_stats(vals["D"])
        # Grey noise of colour noise 4 is ~2.4-4 levels; D adds two frames'.
        assert 0.008 < s["robust"] < 0.03 and s["zero"] < 0.1, s
        f = scale_stats(vals["F"])
        assert f["robust"] < s["robust"]
        assert sigma_of(s)[0] == s["robust"]
        z = scale_stats(np.r_[np.zeros(600), rng.normal(0, 0.05, 400)])
        assert z["robust"] == 0 and sigma_of(z)[0] == z["std"]
        # A montage of what was just labelled renders.
        lab = pd.DataFrame([dict(m, category="random", label="PASS", positive_only=False,
                                 weight=1.0, narrow=False, holdout=False, coco_balls=0)
                            for m in meta])
        lab.to_parquet(tmp / "labels.parquet")
        np.save(tmp / "labels_codes.npy", np.zeros((len(lab), CELLS, CELLS), np.int8))
        st_meta = pd.DataFrame(meta).assign(a_x=np.nan, a_y=np.nan, b_x=np.nan,
                                            b_y=np.nan, s=np.nan, t_hat=np.nan)
        st_meta.to_parquet(tmp / "strips.parquet")
        img = cv2.imread(str(montage(tmp, tmp / "m.png", n=3)))
        assert img.shape == (3 * (SIZE + 26), 4 * SIZE, 3)
    print(f"  label_ball_strips: flight bag along the pass at t_hat, off feet "
          f"and bodies, far bodies -3, dropped when off the crop; lofted curtain; "
          f"v1 action rule and the throw-in; dribble positive only; free with "
          f"landmarks and a COCO ignore disk; holdout by kind; motion scales "
          f"of known noise; {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
