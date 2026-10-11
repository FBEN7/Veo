"""Training strips for the panorama ball detector v2 (`ball_heatmap.build(
motion=True)`, `train_ball_heatmap.py --strips`), from SoccerTrack v2.

A strip is one JPEG (q98) of 1280 x 320 px: four 320 px crops at the same
place (x0, y0), of frames t-2, t, t+2 and of the background B_t
(`motion_bg`: the per-pixel median of the frames at 50-frame steps within
14 s, stored grey in three equal channels), so that training computes the
motion channels from the same inputs as inference. Where the ball is, or
is not, is decided later (`label_ball_strips.py`); here are the places
and frames, chosen from the ball actions (BAS), the players (GSR) and the
release's ball track:

- flight: a ground PASS received by a team-mate (completed), or taken by
  an opponent's PASS, DRIVE or HIGH PASS 0.8-4 s later (intercepted),
  10 m or longer; at s = 0.25, 0.5, 0.75 of its time (each +-0.08,
  within 0.17-0.83). A passed ball slows down, so it is expected at
  t_hat = s + 0.4 s (1 - s) of the way from A (the passer's feet at the
  kick) to B (the receiver's at the next action);
- lofted: a HIGH PASS or CROSS at s = 0.35, 0.65 (no positives: the ball
  is somewhere above the line);
- action: the frame of a PASS, DRIVE, HIGH PASS, CROSS, FREE KICK, tackle
  (v1's actions) or THROW IN;
- dribble: the middle of a gap of more than 1.5 s opened by a DRIVE and
  closed by the same player;
- free: ball-free strips, two at every second action frame where the
  ball track is in play and within 3 m of the actor, centred on a player
  cluster (35%), a pitch landmark (25%), the boundary (25%: up to 2 m
  inside a touch or goal line or 60 px outside it -- spare balls, ball
  boys) or a random pitch point (15%); every cell's ground point is 12 m
  or more from the actor and the track point (up to 10 draws).

The crop is centred on where the ball should be (the bag), shifted on
each axis by up to min(100, slack) px, slack = (320 - bag extent) / 2 - 8,
so the ball is not always central, and kept inside the frame and the
pitch box the detector looks at (`ball_heatmap.pitch_box`).

A half is streamed once, decoded in order: each sample takes its three
crops as its frames go by and is written once its background is complete
(frame c + 350 decoded, or the end); only the background samples (grey,
pitch box) and the queued crops are held. A missing t-2 / t+2 (an edge of
the range) is stored as t, a background of fewer than 8 samples as grey t
(both read as zero motion); `has_prev`, `has_next` and `n_bg` say which.
Training halves only (`ball_heatmap.V2_TRAIN`), less `V2_EXCLUDE`. A local
window (`--window`, absolute frame = start_frame + local, background
within the window) is for smoke tests and the stream-vs-window check
(`--allow-any` for a match not trained on; `--like DIR` recuts DIR's
strips at the same frames and places).

A half streams for 20 min or more, and the CDN's signed address lasts an
hour: the range requests (`open_half`) sign the address again on a 401 /
403 and try again, with a doubling back-off, on a 5xx, a connection error
or a timeout (6 tries). The finished rows are saved every 200 strips to
`strips_<match>_<half>_<source>_<first>-<last>.partial.parquet`; the same
command run again after a crash keeps the strips whose JPEG and row
exist, decodes from the first frame the others need, and gives the same
strips.parquet. A run replaces the earlier rows of its half, frame range,
kinds and source only. A directory holds strips of one source: the stream
(training) or windows (checks), never both, since their file names are
the same.

Output in DIR: `<kind>_<match>_<half>_<frame>_<x0>_<y0>.jpg` and
`strips.parquet` -- file, match, half, frame, kind, category (flight:
completed / intercepted; free: cluster / landmark / boundary / random;
else the action's label), label, x0, y0, A and B in pitch metres (a_x,
a_y, b_x, b_y), s, t_hat, L, event_a / event_b (rows of
`soccertrack_v2.events(match)`), player_a / player_b, n_bg, has_prev,
has_next, width, height, source ("stream" or the window), allow_any.

    python build_ball_strips.py --match 118575 --half 1 --out DIR
    python build_ball_strips.py --window st2_118577_1st_f030000 --out DIR_W \\
        [--frames 30000 30600] [--like DIR_A] [--allow-any]
    python build_ball_strips.py --check
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from build_ball_labels import HEIGHT, KEEP_ACTION, R_ACTION, R_DRIBBLE
from src import soccertrack_v2 as st
from src.ball_heatmap import (NEVER, STRIDE, V2_EXCLUDE, V2_TRAIN, pitch_box,
                              pitch_to_image)
from src.motion_bg import (BG_MIN, BG_STEP, OFFSET, BackgroundCache,
                           bg_sample_frames, grey)
from src.paths import DATA_DIR

SIZE = 320
CELLS = SIZE // STRIDE
QUALITY = 98
JITTER = 100                      # px, at most, per axis
SLACK_PX = 8
KINDS = ("flight", "lofted", "action", "dribble", "free")
TEST = {"128057", "132831"}       # never opened
TRIES, BACKOFF_S = 6, 2.0         # range requests: tries, first back-off (doubling)
FLUSH_EVERY = 200                 # strips between saves of a run's rows

# Flight and lofted passes.
FLIGHT_S = (0.25, 0.5, 0.75)
S_JITTER = 0.08
S_RANGE = (0.17, 0.83)
MIN_PASS_M = 10.0
MIN_FLIGHT_S = 0.8
MAX_FLIGHT_S = 4.0                # intercepted and lofted: next action within
INTERCEPT = {"PASS", "DRIVE", "HIGH PASS"}
LOFTED = {"HIGH PASS", "CROSS"}
LOFTED_S = (0.35, 0.65)
RECEIVE = {"PASS", "DRIVE", "HIGH PASS", "CROSS", "HEADER", "SHOT",
           "PLAYER SUCCESSFUL TACKLE", "BALL PLAYER BLOCK"}
# The flight bag (label_ball_strips): projections on A->B in [lo, hi] =
# t_hat - WINDOW[0] .. t_hat + WINDOW[1] (NARROW on the retry), clear of
# the passer and the receiver and within T_BOUNDS; within BAG_PERP_M of
# the line or BAG_PX of its image, and up to BAG_HEIGHT_M high.
WINDOW, NARROW = (0.22, 0.28), (0.15, 0.20)
CLEAR_A_M, CLEAR_B_M, T_BOUNDS = 2.0, 1.5, (0.08, 0.95)
BAG_PERP_M, BAG_PX, BAG_HEIGHT_M = 2.5, 20.0, 1.0

# Actions (v1's, `build_ball_labels.py`, and throw-ins, the ball in the
# hands), dribbles.
ACTION = KEEP_ACTION | {"THROW IN"}
THROW_HEIGHT = 2.6
MAX_ACTOR_TRACK_M = 3.0
DRIBBLE_GAP_S = 1.5

# Ball-free strips.
FREE_PER_FRAME = 2
FREE_MIX = {"cluster": 0.35, "landmark": 0.25, "boundary": 0.25,
            "random": 0.15}
FREE_MIN_M = 12.0
FREE_TRIES = 10
CLUSTER_M, CLUSTER_N = 5.0, 2
BOUNDARY_IN_M, BOUNDARY_OUT_PX = 2.0, 60.0
# Pitch landmarks (metres, centre origin): centre spot, penalty spots,
# corners, goal posts, penalty-box and 6-yard-box corners (on the field
# and on the goal line), halfway line on the touchlines.
LANDMARKS = ([(0.0, 0.0), (-41.5, 0.0), (41.5, 0.0), (0.0, -34.0), (0.0, 34.0)]
             + [(sx * 52.5, sy * 34.0) for sx in (-1, 1) for sy in (-1, 1)]
             + [(sx * 52.5, sy * 3.66) for sx in (-1, 1) for sy in (-1, 1)]
             + [(sx * x, sy * 20.16) for x in (36.0, 52.5) for sx in (-1, 1)
                for sy in (-1, 1)]
             + [(sx * x, sy * 9.16) for x in (47.0, 52.5) for sx in (-1, 1)
                for sy in (-1, 1)])


# ---- Geometry -------------------------------------------------------------------

def t_hat(s):
    """Where a pass that slows down is, as a fraction of A->B, at the
    fraction s of its time."""
    return s + 0.4 * s * (1 - s)


def bag_window(th: float, L: float, narrow: bool = False) -> tuple[float, float]:
    """The flight bag's stretch [lo, hi] of A->B around t_hat (`narrow`:
    the retry when the wider one does not fit the crop); 2 m clear of the
    passer and 1.5 m of the receiver."""
    a, b = NARROW if narrow else WINDOW
    return (max(th - a, CLEAR_A_M / L, T_BOUNDS[0]),
            min(th + b, 1 - CLEAR_B_M / L, T_BOUNDS[1]))


def scales(xy, cal: dict, step: float = 0.5):
    """Image points of pitch points `xy` (N x 2) and the pixels per metre
    there along the pitch (sx) and in depth (sy)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    p = pitch_to_image(np.concatenate([xy, xy + (step, 0.0), xy + (0.0, step)]),
                       cal).reshape(3, -1, 2)
    return (p[0], np.hypot(*(p[1] - p[0]).T) / step,
            np.hypot(*(p[2] - p[0]).T) / step)


def flight_bbox(A, B, lo, hi, cal) -> tuple[float, float, float, float]:
    """Image box (u0, v0, u1, v1) of the flight bag: the band of A->B over
    [lo, hi], BAG_PERP_M to either side, BAG_PX around, BAG_HEIGHT_M up."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    d = B - A
    n = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-9)
    pts = [A + t * d + k * BAG_PERP_M * n for t in np.linspace(lo, hi, 9)
           for k in (-1, 0, 1)]
    uv, sx, _ = scales(pts, cal)
    return (uv[:, 0].min() - BAG_PX, (uv[:, 1] - BAG_HEIGHT_M * sx).min() - BAG_PX,
            uv[:, 0].max() + BAG_PX, uv[:, 1].max() + BAG_PX)


def disc_bbox(centres, radius: float, height: float, cal):
    """Image box of ground discs of `radius` m around `centres`, `height`
    m up (v1's regions)."""
    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = radius * np.c_[np.cos(a), np.sin(a)]
    pts = np.concatenate([np.asarray(c, float) + ring for c in centres])
    uv, sx, _ = scales(pts, cal)
    return (uv[:, 0].min(), (uv[:, 1] - height * sx).min(),
            uv[:, 0].max(), uv[:, 1].max())


def place(bbox, rng, limits) -> tuple[int, int]:
    """(x0, y0) of a crop centred on `bbox`, shifted on each axis by up to
    min(JITTER, slack) px, inside `limits` (x0, y0, x1, y1)."""
    out = []
    for a, b, lo, hi in ((bbox[0], bbox[2], limits[0], limits[2]),
                         (bbox[1], bbox[3], limits[1], limits[3])):
        j = min(JITTER, max((SIZE - (b - a)) / 2 - SLACK_PX, 0.0))
        c = (a + b) / 2 + rng.uniform(-j, j)
        out.append(int(np.clip(round(c - SIZE / 2), lo, hi - SIZE)))
    return out[0], out[1]


def crop_ground(cal, x0: int, y0: int, step: int = 2) -> np.ndarray:
    """Ground points of a crop's cell centres, every `step` cells and the
    last row and column (those above the horizon left out)."""
    k = np.unique(np.r_[np.arange(0, CELLS, step), CELLS - 1])
    jj, ii = np.meshgrid(k, k)
    uv = np.c_[x0 + (jj.ravel() + 0.5) * STRIDE, y0 + (ii.ravel() + 0.5) * STRIDE]
    P = st.image_to_pitch(uv, cal)
    return P[np.isfinite(P).all(axis=1)]


# ---- What a half's strips are cut from ------------------------------------------

def _has(x) -> bool:
    return x is not None and bool(pd.notna(x))


class People:
    """A half's players (GSR) by absolute frame: feet in pitch metres and
    player ids."""

    def __init__(self, gsr: pd.DataFrame, shift: int = 0):
        g = gsr[gsr.role.isin(("player", "goalkeeper")) & gsr.x.notna()
                & gsr.y.notna()]
        f = g.frame.to_numpy(np.int64) + shift
        o = np.argsort(f, kind="stable")
        self.frame = f[o]
        self.xy = g[["x", "y"]].to_numpy(float)[o]
        self.pid = g.player_id.astype(object).where(g.player_id.notna(), "") \
            .astype(str).to_numpy()[o]

    def at(self, f: int) -> tuple[np.ndarray, np.ndarray]:
        i0, i1 = np.searchsorted(self.frame, [f, f + 1])
        return self.xy[i0:i1], self.pid[i0:i1]

    def of(self, f: int, pid) -> np.ndarray | None:
        if not _has(pid) or str(pid) == "":
            return None
        xy, ids = self.at(int(f))
        k = np.flatnonzero(ids == str(pid))
        return xy[k[0]] if len(k) else None


def events_of(match: str, half: int) -> pd.DataFrame:
    """A half's ball actions in order, with `event_id` = their row in
    `soccertrack_v2.events(match)`."""
    ev = st.events(match)
    ev = ev.assign(event_id=ev.index)
    return (ev[ev.half == half].sort_values(["frame", "event_id"], kind="stable")
            .reset_index(drop=True))


def people_of(match: str, half: int, window: str | None = None) -> People:
    """The half's GSR (cached parquet); for a window whose half is not
    cached, the window's own ground truth."""
    cached = st.cache_dir() / "gsr" / f"{match}_{st._half(half)}.parquet"
    if window and not cached.exists():
        info = json.loads((DATA_DIR / "soccertrack_v2" / f"{window}.json").read_text())
        gt = pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{window}_gt.parquet")
        return People(gt, shift=int(info["start_frame"]))
    return People(st.gsr_table(match, half))


def ball_track(match: str, half: int) -> pd.DataFrame:
    """The release's ball track of a half (x, y in metres, status) by
    0-based frame; the file is kept in the cache."""
    name = f"{match}_{st._half(half)}_ball.npz"
    path = st.cache_dir() / "ball" / name
    if not path.exists():
        old = st.cache_dir() / name
        data = (old.read_bytes() if old.exists()
                else st.open_stream(f"ball/{name}").read())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    z = np.load(path, allow_pickle=False)
    return pd.DataFrame({"x": z["x"].astype(float), "y": z["y"].astype(float),
                         "status": z["status"].astype(int)},
                        index=z["frame"].astype(int) - 1)


def outline(match: str, half: int, width: int, height: int) -> list:
    """The pitch outline (pixels) of a half's fixed view: from a local
    window of that half, else from the GSR's pitch record (cached)."""
    for p in sorted((DATA_DIR / "soccertrack_v2").glob(
            f"st2_{match}_{st._half(half)}_f*.json")):
        info = json.loads(p.read_text())
        if info.get("pitch") and (info.get("width"), info.get("height")) == (width, height):
            return info["pitch"]
    path = st.cache_dir() / "outline" / f"{match}_{st._half(half)}_{width}x{height}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(st.pitch_outline(match, half, width, height)))
    return json.loads(path.read_text())


class Half:
    """Everything the samples of one half are chosen from."""

    def __init__(self, match: str, half: int, window: str | None = None,
                 cal: dict | None = None, ev=None, people=None, ball=None):
        self.match, self.half = str(match), int(half)
        self.cal = st.calibration(self.match) if cal is None else cal
        self.ev = events_of(self.match, self.half) if ev is None else ev
        self.people = people_of(self.match, self.half, window) if people is None else people
        self.ball = ball_track(self.match, self.half) if ball is None else ball

    def track(self, f: int) -> tuple[np.ndarray | None, int]:
        if f not in self.ball.index:
            return None, 0
        r = self.ball.loc[f]
        xy = np.array([r.x, r.y], float)
        return (xy if np.isfinite(xy).all() else None), int(r.status)


# ---- Samples --------------------------------------------------------------------

def _row(kind, frame, x0, y0, category=None, label=None, A=None, B=None,
         s=np.nan, th=np.nan, L=np.nan, a=None, b=None):
    A = (np.nan, np.nan) if A is None else A
    B = (np.nan, np.nan) if B is None else B
    return {"frame": int(frame), "kind": kind, "category": category,
            "label": label, "x0": int(x0), "y0": int(y0),
            "a_x": float(A[0]), "a_y": float(A[1]), "b_x": float(B[0]),
            "b_y": float(B[1]), "s": float(s), "t_hat": float(th), "L": float(L),
            "event_a": -1 if a is None else int(a.event_id),
            "event_b": -1 if b is None else int(b.event_id),
            "player_a": None if a is None else a.player_id,
            "player_b": None if b is None else b.player_id}


def _pairs(ev):
    for i in range(len(ev) - 1):
        yield ev.iloc[i], ev.iloc[i + 1]


def flight_samples(h: Half, limits, rng_for) -> list[dict]:
    out = []
    for a, b in _pairs(h.ev):
        if a.label != "PASS" or not (_has(a.side) and _has(b.side)):
            continue
        dt = (b.frame - a.frame) / st.FPS
        if b.side == a.side and b.player_id != a.player_id and dt >= MIN_FLIGHT_S:
            cat = "completed"
        elif b.side != a.side and b.label in INTERCEPT and \
                MIN_FLIGHT_S <= dt <= MAX_FLIGHT_S:
            cat = "intercepted"
        else:
            continue
        A, B = h.people.of(a.frame, a.player_id), h.people.of(b.frame, b.player_id)
        if A is None or B is None:
            continue
        L = float(np.hypot(*(B - A)))
        if L < MIN_PASS_M:
            continue
        rng = rng_for("flight", a.event_id)
        n = b.frame - a.frame
        k0, k1 = int(np.ceil(S_RANGE[0] * n)), int(np.floor(S_RANGE[1] * n))
        for s0 in FLIGHT_S:
            s = s0 + rng.uniform(-S_JITTER, S_JITTER)
            f = int(a.frame + np.clip(round(s * n), k0, k1))   # s in S_RANGE, on a frame
            s = (f - a.frame) / n
            th = t_hat(s)
            x0, y0 = place(flight_bbox(A, B, *bag_window(th, L), h.cal), rng, limits)
            out.append(_row("flight", f, x0, y0, cat, a.label, A, B, s, th, L, a, b))
    return out


def lofted_samples(h: Half, limits, rng_for) -> list[dict]:
    out = []
    for a, b in _pairs(h.ev):
        dt = (b.frame - a.frame) / st.FPS
        if a.label not in LOFTED or b.label not in RECEIVE or \
                not MIN_FLIGHT_S <= dt <= MAX_FLIGHT_S:
            continue
        A, B = h.people.of(a.frame, a.player_id), h.people.of(b.frame, b.player_id)
        if A is None or B is None:
            continue
        rng = rng_for("lofted", a.event_id)
        for s0 in LOFTED_S:
            f = int(a.frame + round(s0 * (b.frame - a.frame)))
            s = (f - a.frame) / (b.frame - a.frame)
            u, v = pitch_to_image([A + s * (B - A)], h.cal)[0]
            x0, y0 = place((u, v, u, v), rng, limits)
            out.append(_row("lofted", f, x0, y0, a.label, a.label, A, B, s,
                            np.nan, float(np.hypot(*(B - A))), a, b))
    return out


def action_samples(h: Half, limits, rng_for) -> list[dict]:
    out, seen = [], set()
    for a in h.ev.itertuples():
        if a.label not in ACTION or a.frame in seen:
            continue
        A = h.people.of(a.frame, a.player_id)
        T, _ = h.track(a.frame)
        if A is None or T is None or np.hypot(*(A - T)) > MAX_ACTOR_TRACK_M:
            continue
        seen.add(a.frame)
        up = THROW_HEIGHT if a.label == "THROW IN" else HEIGHT
        x0, y0 = place(disc_bbox([A, T], R_ACTION, up, h.cal),
                       rng_for("action", a.event_id), limits)
        out.append(_row("action", a.frame, x0, y0, a.label, a.label, A, T, a=a))
    return out


def dribble_samples(h: Half, limits, rng_for) -> list[dict]:
    out = []
    for a, b in _pairs(h.ev):
        if a.label != "DRIVE" or b.player_id != a.player_id or \
                b.frame - a.frame <= DRIBBLE_GAP_S * st.FPS:
            continue
        f = int((a.frame + b.frame) // 2)
        D = h.people.of(f, a.player_id)
        if D is None:
            continue
        x0, y0 = place(disc_bbox([D], R_DRIBBLE, HEIGHT, h.cal),
                       rng_for("dribble", a.event_id), limits)
        out.append(_row("dribble", f, x0, y0, "DRIVE", a.label, D, a=a, b=b))
    return out


def free_centre(h: Half, cat: str, f: int, poly, rng) -> tuple[float, float] | None:
    """An image point to centre a ball-free strip on, of category `cat`."""
    cal = h.cal
    if cat == "cluster":
        xy, _ = h.people.at(f)
        if len(xy) < CLUSTER_N + 1:
            return None
        d = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
        busy = np.flatnonzero((d <= CLUSTER_M).sum(axis=1) - 1 >= CLUSTER_N)
        if not len(busy):
            return None
        return tuple(pitch_to_image([xy[rng.choice(busy)]], cal)[0])
    if cat == "landmark":
        return tuple(pitch_to_image([LANDMARKS[rng.integers(len(LANDMARKS))]], cal)[0])
    if cat == "boundary":
        # A point on a touch or goal line (by length), then up to 2 m in
        # or up to 60 px out.
        k = rng.choice(4, p=np.array([105, 105, 68, 68]) / 346)
        q = rng.uniform(-1, 1)
        p, n = [((52.5 * q, -34.0), (0, -1)), ((52.5 * q, 34.0), (0, 1)),
                ((-52.5, 34.0 * q), (-1, 0)), ((52.5, 34.0 * q), (1, 0))][k]
        p, n = np.array(p), np.array(n, float)
        if rng.random() < 0.5:
            return tuple(pitch_to_image([p - n * rng.uniform(0, BOUNDARY_IN_M)], cal)[0])
        a, b = pitch_to_image([p, p + n], cal)
        out = (b - a) / max(np.hypot(*(b - a)), 1e-9)
        return tuple(a + out * rng.uniform(0, BOUNDARY_OUT_PX))
    lo, hi = poly.min(axis=0), poly.max(axis=0)           # random, in the picture
    for _ in range(100):
        uv = rng.uniform(lo, hi)
        if cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(uv[0]), float(uv[1])),
                                False) >= 0:
            return tuple(uv)
    return None


def free_samples(h: Half, limits, rng_for, poly) -> list[dict]:
    out, frames = [], []
    seen = set()
    for a in h.ev.itertuples():
        if a.frame in seen:
            continue
        A = h.people.of(a.frame, a.player_id)
        T, status = h.track(a.frame)
        if A is None or T is None or status == 0 or \
                np.hypot(*(A - T)) > MAX_ACTOR_TRACK_M:
            continue
        seen.add(a.frame)
        frames.append((a, A, T))
    cats, mix = list(FREE_MIX), np.array(list(FREE_MIX.values()))
    for a, A, T in frames[::2]:
        for j in range(FREE_PER_FRAME):
            rng = rng_for("free", a.event_id, j)
            cat = cats[rng.choice(len(cats), p=mix)]
            for _ in range(FREE_TRIES):
                c = free_centre(h, cat, a.frame, poly, rng)
                if c is None or not np.isfinite(c).all():
                    continue
                x0, y0 = place((c[0], c[1], c[0], c[1]), rng, limits)
                P = crop_ground(h.cal, x0, y0)
                if not len(P) or min(np.hypot(*(P - A).T).min(),
                                     np.hypot(*(P - T).T).min()) >= FREE_MIN_M:
                    out.append(_row("free", a.frame, x0, y0, cat, a.label, A, T, a=a))
                    break
    return out


SAMPLERS = {"flight": flight_samples, "lofted": lofted_samples,
            "action": action_samples, "dribble": dribble_samples,
            "free": free_samples}


def samples(h: Half, kinds, limits, poly, seed: int = 0) -> list[dict]:
    """Every sample of the half: frame, kind, category, crop (x0, y0) and
    the pass or action it comes from. Each sample's draws come from its
    own generator (seed, match, half, kind, event), so a sample does not
    change with the frame range or the other kinds."""
    def rng_for(kind, event_id, j=0):
        return np.random.default_rng([seed, int(h.match), h.half,
                                      KINDS.index(kind), int(event_id), j])

    out = []
    for kind in KINDS:
        if kind in kinds:
            fn = SAMPLERS[kind]
            out += (fn(h, limits, rng_for, poly) if kind == "free"
                    else fn(h, limits, rng_for))
    return out


def excluded(frame: int, match: str, half: int, exclude=None) -> bool:
    """Whether frames t-2 .. t+2 touch a left-out range (`V2_EXCLUDE`)."""
    r = (V2_EXCLUDE if exclude is None else exclude).get((str(match), int(half)))
    return r is not None and frame + OFFSET >= r[0] and frame - OFFSET <= r[1]


def check_allowed(match: str, half: int, window: bool, allow_any: bool):
    if str(match) in TEST:
        raise SystemExit(f"match {match} is a test match: never opened")
    if int(half) in V2_TRAIN.get(str(match), ()):
        return
    if window and allow_any:
        print(f"  match {match} half {half} is not a v2 training half "
              f"(--allow-any{'; never-train match' if str(match) in NEVER else ''}): "
              f"these strips are not for training", flush=True)
        return
    raise SystemExit(f"match {match} half {half} is not a v2 training half "
                     f"(ball_heatmap.V2_TRAIN); only --window with --allow-any")


def check_source(out: Path, source: str):
    """Refuse to mix stream strips (training) and window strips (smoke
    tests, the stream-vs-window check) in one directory: their file names
    are the same (`strip_name`), so one would overwrite the other's JPEGs.
    Looks at strips.parquet and at the rows of interrupted runs."""
    have = set()
    if (out / "strips.parquet").exists():
        t = pd.read_parquet(out / "strips.parquet")
        if "source" in t:
            have |= set(t.source.astype(str))
    for p in out.glob("strips_*.partial.parquet"):
        have |= set(pd.read_parquet(p, columns=["source"]).source.astype(str))
    clash = sorted(s for s in have if (s == "stream") != (source == "stream"))
    if clash:
        raise SystemExit(f"{out} holds strips of {clash[:3]}: {source} strips go in "
                         f"another directory (training strips from the stream and "
                         f"window strips are kept apart)")


# ---- Streaming a half -----------------------------------------------------------

def resigning_fetch(sign, session=None, tries: int = TRIES, timeout: float = 120,
                    sleep=time.sleep, log=print):
    """(fetch, size) for `st.RemoteFile` over the address `sign()` gives,
    which expires (the CDN signs it for an hour; a half streams for 20 min
    or more). On a 401 / 403 the address is signed again; a 5xx, a 429, a
    connection error, a timeout or a short read is tried again after
    BACKOFF_S, 2 BACKOFF_S, ... s; `tries` tries in all, then the last
    error is raised. Any other 4xx is raised at once. The revision is
    pinned, so a new signature serves the same bytes. Errors are logged and
    raised without the address (a credential while it lasts). `fetch.stats`
    counts the signatures and the retries."""
    import requests

    s = requests.Session() if session is None else session
    address = [None]
    stats = {"signs": 0, "retries": 0}
    transient = (requests.ConnectionError, requests.Timeout,
                 requests.exceptions.ChunkedEncodingError)

    def what(err) -> str:
        code = getattr(getattr(err, "response", None), "status_code", None)
        return f"HTTP {code}" if code else type(err).__name__

    def plain(err, n):
        return type(err)(f"{what(err)} after {n} tr{'y' if n == 1 else 'ies'}",
                         response=getattr(err, "response", None))

    def call(do):
        for i in range(tries):
            try:
                if address[0] is None:
                    address[0] = sign()
                    stats["signs"] += 1
                r = do(address[0])
                if r.status_code in (401, 403):          # the signature expired
                    address[0] = None
                r.raise_for_status()
                return r
            except requests.HTTPError as e:
                code = None if e.response is None else e.response.status_code
                if code is not None and 400 <= code < 500 and code not in (401, 403, 429):
                    raise plain(e, i + 1) from None
                err = e
            except transient as e:
                err = e
            if i == tries - 1:
                raise plain(err, tries) from None
            stats["retries"] += 1
            wait = 0.0 if address[0] is None else BACKOFF_S * 2 ** i
            if log:
                log(f"    {what(err)}: try {i + 2} of {tries}"
                    f"{' with a new signature' if address[0] is None else ''}"
                    f" in {wait:.0f} s", flush=True)
            if wait:
                sleep(wait)
        raise AssertionError("unreachable")

    def fetch(a: int, b: int) -> bytes:
        def do(addr):
            r = s.get(addr, headers={"Range": f"bytes={a}-{b}"}, timeout=timeout)
            if r.ok and len(r.content) != b - a + 1:
                raise requests.ConnectionError(f"bytes {a}-{b}: {len(r.content)} "
                                               f"came back")
            return r
        return call(do).content

    head = call(lambda addr: s.head(addr, allow_redirects=True, timeout=60))
    fetch.stats = stats
    return fetch, int(head.headers["Content-Length"])


def open_half(match: str, half: int, tries: int = TRIES):
    """A half's panorama (PyAV container), as `st.open_video` but read
    through `resigning_fetch`."""
    import av

    path = st.video_path(match, half)
    fetch, size = resigning_fetch(lambda: st.signed_url(path), tries=tries)
    return av.open(st.RemoteFile(fetch=fetch, size=size))


# ---- Cutting --------------------------------------------------------------------

def stream_frames(container, first: int, last: int):
    """(absolute frame, fetch) over frames first..last of a PyAV
    container, decoded in order (seeking first when first > 0); `fetch()`
    gives the BGR image, converted only when asked for."""
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    tb = float(stream.time_base)
    t0 = float(stream.start_time or 0) * tb
    if first > 0:
        container.seek(int((first / st.FPS + t0) / tb), stream=stream,
                       backward=True, any_frame=False)
    for frame in container.decode(stream):
        k = int(round((float(frame.pts) * tb - t0) * st.FPS))
        if k < first:
            continue
        if k > last:
            return
        yield k, (lambda fr=frame: fr.to_ndarray(format="bgr24"))


def window_frames(path, start: int, first: int, last: int):
    """The same over a local window mp4 read with OpenCV, as the detectors
    read it (absolute frame = start + local)."""
    cap = cv2.VideoCapture(str(path))
    k = start
    try:
        while k <= last and cap.grab():
            if k >= first:
                yield k, (lambda: cap.retrieve()[1])
            k += 1
    finally:
        cap.release()


def strip_name(r: dict, match: str, half: int) -> str:
    return f"{r['kind']}_{match}_{half}_{r['frame']:06d}_{r['x0']}_{r['y0']}.jpg"


def assemble(crops, bg) -> np.ndarray:
    """[t-2 | t | t+2 | B] from three BGR crops (None at an edge: t) and a
    grey background crop (None: grey t)."""
    prev, cur, nxt = crops
    b = grey(cur) if bg is None else bg
    return np.hstack([cur if prev is None else prev, cur,
                      cur if nxt is None else nxt,
                      cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)])


def cut(targets: list[dict], frames, first: int, last: int, box, out: Path,
        match: str, half: int, write=None, log_every: int = 5000, flush=None,
        flush_every: int = FLUSH_EVERY) -> list[dict]:
    """Decode `frames` in order and write a strip per target, each once its
    background is ready; returns the targets written, with file, n_bg,
    has_prev, has_next. `box` is the pitch box the background is kept in
    (every crop lies inside it). `flush(done)` is called with the rows so
    far every `flush_every` strips."""
    if write is None:
        def write(path, strip):
            cv2.imwrite(str(path), strip, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    targets = sorted(targets, key=lambda r: r["frame"])
    bx0, by0 = box[0], box[1]
    cache = BackgroundCache(first, last, crop=box)
    want: dict[int, list] = {}
    for i, r in enumerate(targets):
        for slot, f in enumerate((r["frame"] - OFFSET, r["frame"], r["frame"] + OFFSET)):
            want.setdefault(f, []).append((i, slot))
    held: dict[int, list] = {}
    done, nxt, peak = [], 0, 0
    t0 = time.time()

    def finish(i):
        r = targets[i]
        crops = held.pop(i, [None, None, None])
        if crops[1] is None:                    # frame t was never decoded
            return
        bg, n = cache.get(r["frame"])
        if bg is not None:
            y, x = r["y0"] - by0, r["x0"] - bx0
            bg = bg[y:y + SIZE, x:x + SIZE]
        name = strip_name(r, match, half)
        write(out / name, assemble(crops, bg))
        done.append(dict(r, file=name, n_bg=int(n), has_prev=crops[0] is not None,
                         has_next=crops[2] is not None))
        if flush is not None and len(done) % flush_every == 0:
            flush(done)

    k = first - 1
    for k, fetch in frames:
        image = fetch() if (k % BG_STEP == 0 or k in want) else None
        cache.add(k, image)
        for i, slot in want.pop(k, ()):
            r = targets[i]
            held.setdefault(i, [None, None, None])[slot] = \
                image[r["y0"]:r["y0"] + SIZE, r["x0"]:r["x0"] + SIZE].copy()
        while nxt < len(targets) and k >= min(targets[nxt]["frame"] + OFFSET, last) \
                and cache.ready(targets[nxt]["frame"]):
            finish(nxt)
            nxt += 1
        if nxt == len(targets):
            break
        need = bg_sample_frames(targets[nxt]["frame"], first, last)
        cache.drop_before(need[0] if need else targets[nxt]["frame"])
        peak = max(peak, len(cache.samples))
        if log_every and k % log_every == 0 and k > first:
            print(f"    frame {k}: {len(done)} strips, {len(targets) - nxt} queued, "
                  f"{len(cache.samples)} background samples, "
                  f"{(k - first) / (time.time() - t0):.0f} frames/s", flush=True)
    cache.finish()
    while nxt < len(targets):
        finish(nxt)
        nxt += 1
    cut.peak_samples = peak
    return done


def partial_path(out: Path, match: str, half: int, source: str, first: int,
                 last: int) -> Path:
    """Where a run keeps its finished rows until merged: by half, source and
    frame range (the range is the background's, so it is in the strips)."""
    end = "end" if last >= 10 ** 9 else str(last)
    return out / f"strips_{match}_{half}_{source}_{first}-{end}.partial.parquet"


def _save(table: pd.DataFrame, path: Path):
    """Write a parquet whole or not at all: a crash leaves the old file."""
    tmp = path.with_name(path.name + ".tmp")
    table.to_parquet(tmp)
    os.replace(tmp, path)


@contextlib.contextmanager
def _locked(out: Path):
    """strips.parquet to one writer at a time (halves run side by side)."""
    try:
        import fcntl
    except ImportError:                     # not POSIX: no lock
        yield
        return
    with open(out / "strips.parquet.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def resume_cut(out: Path, targets: list[dict], frames_from, first: int, last: int,
               box, match: str, half: int, extra: dict, write=None,
               flush_every: int = FLUSH_EVERY, log_every: int = 5000):
    """`cut` that a crash costs minutes, not the half: the finished rows
    (with `extra`: match, half, width, height, source, allow_any) are saved
    to the run's `partial_path` every `flush_every` strips. Run again, it
    keeps the saved rows whose JPEG exists and cuts the other targets only,
    decoding from the first frame they need (`frames_from(k)`: the frames
    from k on; the background range stays [first, last], so the strips are
    those of an uninterrupted run). Returns (rows in the order of an
    uninterrupted run, number of rows kept from the saved ones)."""
    part = partial_path(out, match, half, extra["source"], first, last)
    targets = sorted(targets, key=lambda r: r["frame"])
    order = {strip_name(r, match, half): i for i, r in enumerate(targets)}
    kept = []
    if part.exists():
        kept = [r for r in pd.read_parquet(part).to_dict("records")
                if r["file"] in order and (out / r["file"]).exists()]
    have = {r["file"] for r in kept}
    todo = [r for r in targets if strip_name(r, match, half) not in have]
    rows = []
    cut.peak_samples = 0
    if todo:
        k0 = max(first, min(min(bg_sample_frames(r["frame"], first, last)[:1]
                                + [r["frame"] - OFFSET]) for r in todo))
        if kept:
            print(f"  resuming: {len(kept)} strips saved, {len(todo)} to cut, "
                  f"decoding from frame {k0}", flush=True)

        def flush(done):
            _save(pd.DataFrame(kept + [dict(r, **extra) for r in done]), part)

        rows = cut(todo, frames_from(k0), first, last, box, out, match, half,
                   write=write, log_every=log_every, flush=flush,
                   flush_every=flush_every)
    rows = kept + [dict(r, **extra) for r in rows]
    rows.sort(key=lambda r: order[r["file"]])
    return rows, len(kept)


def merge(out: Path, rows: list[dict], match: str, half: int, first: int,
          last: int, kinds=KINDS, source: str | None = None) -> pd.DataFrame:
    """strips.parquet with this run's rows replacing the earlier ones of
    the same half, frame range, kinds and source (any source when None);
    the other kinds' and sources' rows stay."""
    table = pd.DataFrame(rows)
    path = out / "strips.parquet"
    with _locked(out):
        if path.exists():
            old = pd.read_parquet(path)
            mine = ((old.match.astype(str) == match) & (old.half == half)
                    & old.frame.between(first, last) & old.kind.isin(list(kinds)))
            if source is not None and "source" in old:
                mine &= old.source.astype(str) == source
            table = pd.concat([old[~mine], table], ignore_index=True)
        _save(table, path)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match")
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--window", help="a local window instead of the half's stream")
    ap.add_argument("--out")
    ap.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    help="absolute frames to decode (and the background's range)")
    ap.add_argument("--like", help="recut this strips dir's samples (same frames "
                    "and places) instead of drawing new ones")
    ap.add_argument("--allow-any", action="store_true",
                    help="--window of a match not trained on (smoke / mismatch checks)")
    ap.add_argument("--flush-every", type=int, default=FLUSH_EVERY,
                    help="strips between saves of the finished rows (resumed after "
                    "a crash by running the same command)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        _check()
        return
    if not args.out or not (args.window or args.match):
        ap.error("--out and --match/--half or --window are needed")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.window:
        info = json.loads((DATA_DIR / "soccertrack_v2" / f"{args.window}.json").read_text())
        match, half = str(info["match"]), int(info["half"])
        start = int(info["start_frame"])
        first, last = start, start + int(info["frames"]) - 1
        W, H = int(info["width"]), int(info["height"])
        source = args.window
    else:
        match, half, source = str(args.match), args.half, "stream"
        first, last, W, H = 0, 10 ** 9, None, None
    check_allowed(match, half, bool(args.window), args.allow_any)
    check_source(out, source)
    if args.frames:
        first, last = max(first, args.frames[0]), min(last, args.frames[1])
    container = None
    if not args.window:
        container = open_half(match, half)
        cc = container.streams.video[0].codec_context
        W, H = cc.width, cc.height
    poly = np.asarray(outline(match, half, W, H), np.float32)
    box = pitch_box((H, W), poly)
    if box[2] - box[0] < SIZE or box[3] - box[1] < SIZE:
        raise SystemExit(f"pitch box {box} smaller than a strip")
    if args.like:
        like = pd.read_parquet(Path(args.like) / "strips.parquet")
        like = like[(like.match.astype(str) == match) & (like.half == half)
                    & (like.kind.isin(args.kinds))]
        cols = [c for c in like.columns if c not in (
            "file", "match", "half", "n_bg", "has_prev", "has_next", "width",
            "height", "source", "allow_any")]
        targets = like[cols].to_dict("records")
    else:
        h = Half(match, half, window=args.window)
        targets = samples(h, args.kinds, box, poly, args.seed)
    n_all = len(targets)
    targets = [r for r in targets if first <= r["frame"] <= last
               and not excluded(r["frame"], match, half)]
    unique = {}
    for r in targets:                     # same kind, frame and place: one strip
        unique.setdefault(strip_name(r, match, half), r)
    targets = list(unique.values())
    print(f"  {match} half {half} ({source}), frames {first}-"
          f"{'end' if last >= 10 ** 9 else last}: {len(targets)} samples "
          f"of {n_all}", flush=True)
    t0 = time.time()
    extra = dict(match=match, half=half, width=W, height=H, source=source,
                 allow_any=bool(args.allow_any))

    def frames_from(k0):
        if args.window:
            return window_frames(DATA_DIR / "soccertrack_v2" / f"{args.window}.mp4",
                                 start, k0, last)
        return stream_frames(container, k0, last)

    try:
        rows, resumed = resume_cut(out, targets, frames_from, first, last, box, match,
                                   half, extra, flush_every=args.flush_every)
    finally:
        if container is not None:
            container.close()
    merge(out, rows, match, half, first, last, args.kinds, source)
    partial_path(out, match, half, source, first, last).unlink(missing_ok=True)
    t = pd.DataFrame(rows)
    if len(t):
        print(f"  {len(t)} strips in {time.time() - t0:.0f} s"
              f"{f' ({resumed} from the interrupted run)' if resumed else ''}: "
              f"{t.kind.value_counts().to_dict()}; background < {BG_MIN} samples "
              f"on {(t.n_bg < BG_MIN).mean():.1%}, no t-2 / t+2 on "
              f"{(~t.has_prev | ~t.has_next).mean():.1%}; at most "
              f"{cut.peak_samples} background samples held", flush=True)
    else:
        print("  no strips", flush=True)


# ---- Self-check -----------------------------------------------------------------

def synthetic_cal(width: int = 2400, height: int = 700) -> dict:
    """A camera model of the release's form (fisheye K, D; Knew; Hinv)
    whose pitch corners land at chosen pixels: the far touchline at
    v = 120, the near one at v = 660, the near side twice the scale."""
    K = np.array([[900.0, 0, width / 2], [0, 900.0, height / 2], [0, 0, 1]])
    D = np.array([[-0.02], [0.005], [0.0], [0.0]])
    Knew = K.copy()
    img = np.array([[700, 120], [1700, 120], [2340, 660], [60, 660]], np.float64)
    und = cv2.fisheye.undistortPoints(img.reshape(-1, 1, 2), K, D, P=Knew).reshape(-1, 2)
    pitch = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], np.float32)
    Hm = cv2.getPerspectiveTransform(pitch, und.astype(np.float32))
    return {"K": K, "D": D, "Knew": Knew, "Hinv": np.linalg.inv(Hm)}


def synthetic_half(seed: int = 0):
    """A synthetic half (frames 1000-1599): seven players walking slowly,
    a pass, an interception, a dribble, a high pass; the ball track at the
    actor on every action and in a straight line between."""
    cal = synthetic_cal()
    frames = np.arange(1000, 1600)
    start = {"p1": (-10, 0), "p2": (10, 5), "p3": (24, 2), "p4": (-5, 20),
             "p5": (30, -20), "p6": (31, -18), "p7": (29, -17)}
    side = {"p1": "left", "p2": "left", "p3": "right", "p4": "right",
            "p5": "right", "p6": "left", "p7": "left"}
    rows = []
    for f in frames:
        for k, (pid, (x, y)) in enumerate(start.items()):
            rows.append({"frame": f, "track_id": k, "player_id": pid,
                         "role": "player", "jersey": k, "side": side[pid],
                         "x": x + 0.004 * (f - 1000), "y": y, "bx": 0, "by": 0,
                         "bw": 0, "bh": 0})
    gsr = pd.DataFrame(rows)
    people = People(gsr)
    acts = [(1100, "PASS", "p1"), (1150, "PASS", "p2"), (1190, "DRIVE", "p3"),
            (1240, "HIGH PASS", "p3"), (1290, "PASS", "p4"), (1330, "OUT", "p4"),
            (1400, "THROW IN", "p1"), (1440, "PASS", "p2")]
    ev = pd.DataFrame([{"half": 1, "frame": f, "label": lab, "kind": None,
                        "side": side[p], "player_id": p, "visibility": None,
                        "event_id": 10 + i} for i, (f, lab, p) in enumerate(acts)])
    # Ball track: at the actor at each action, linear between.
    kf = [(f, people.of(f, p)) for f, _, p in acts]
    xs = np.interp(frames, [f for f, _ in kf], [p[0] for _, p in kf])
    ys = np.interp(frames, [f for f, _ in kf], [p[1] for _, p in kf])
    ball = pd.DataFrame({"x": xs, "y": ys, "status": 1}, index=frames)
    ball.loc[1330:1399, "status"] = 0                 # out of play
    return Half("118575", 1, cal=cal, ev=ev, people=people, ball=ball), gsr


class _FlakyServer:
    """A local file served as the CDN serves a half, for `resigning_fetch`
    (as `session`, with `sign`): every address signed so far stops working
    (403) at GET number `expire_at`, GET number `timeout_at` times out, and
    GETs numbered in `fail` get that status instead of the bytes."""

    def __init__(self, path, expire_at=None, timeout_at=None, fail=None):
        self.data = Path(path).read_bytes()
        self.expire_at, self.timeout_at = expire_at, timeout_at
        self.fail = dict(fail or {})
        self.valid, self.signs, self.n = set(), 0, 0

    def sign(self) -> str:
        self.signs += 1
        address = f"https://cdn.invalid/half.mp4?Signature={self.signs}"
        self.valid.add(address)
        return address

    @staticmethod
    def _response(address, status, content=b"", length=None):
        import requests

        r = requests.Response()
        r.status_code, r._content, r.url = status, content, address
        r.headers["Content-Length"] = str(len(content) if length is None else length)
        return r

    def head(self, address, **kw):
        return self._response(address, 200 if address in self.valid else 403,
                              length=len(self.data))

    def get(self, address, headers=None, timeout=None):
        import requests

        self.n += 1
        if self.n == self.expire_at:
            self.valid.clear()
        if self.n == self.timeout_at:
            raise requests.Timeout("read timed out (injected)")
        if self.n in self.fail:
            return self._response(address, self.fail[self.n])
        if address not in self.valid:
            return self._response(address, 403)
        a, b = (int(x) for x in headers["Range"].split("=")[1].split("-"))
        return self._response(address, 206, self.data[a:b + 1])


def _check():
    """Samples, placement and cutting on a synthetic half: the passes give
    the right flight / lofted / dribble / action samples at the right
    frames, free strips keep 12 m from the ball, crops sit inside the pitch
    box near their bag, and the decode writes [t-2 | t | t+2 | B] with B the
    median of the 50-frame samples (the moving ball gone). Then the run's
    safety: a video read through expiring signatures and a timeout decodes
    to the same frames; a run that crashes and is run again gives the same
    strips and strips.parquet; a rebuild of some kinds keeps the others'
    rows; stream and window strips do not share a directory."""
    import contextlib
    import io
    import tempfile

    t_start = time.time()
    assert abs(t_hat(0.5) - 0.6) < 1e-12 and t_hat(0) == 0 and t_hat(1) == 1
    lo, hi = bag_window(t_hat(0.5), 20.0)
    assert abs(lo - 0.38) < 1e-9 and abs(hi - 0.88) < 1e-9
    lo, hi = bag_window(t_hat(0.25), 20.0, narrow=True)
    assert abs(lo - max(t_hat(0.25) - 0.15, 0.1)) < 1e-9
    assert len(LANDMARKS) == 29 and len(set(LANDMARKS)) == 29
    assert excluded(13498, "118576", 2) and not excluded(13497, "118576", 2)
    assert excluded(21001, "128058", 2) and not excluded(21002, "128058", 2)
    assert not excluded(15000, "118576", 1)
    for m, hf, w, a in (("128057", 1, True, True), ("118578", 1, False, False),
                        ("118578", 1, True, False), ("118575", 3, False, False)):
        try:
            check_allowed(m, hf, w, a)
            raise AssertionError(f"{m} {hf} allowed")
        except SystemExit:
            pass
    check_allowed("118575", 2, False, False)
    with contextlib.redirect_stdout(io.StringIO()):      # its warning
        check_allowed("118578", 1, True, True)

    h, _ = synthetic_half()
    W, H = 2400, 700
    poly = np.array([[700, 120], [1700, 120], [2340, 660], [60, 660]], np.float32)
    box = pitch_box((H, W), poly)
    rows = samples(h, KINDS, box, poly, seed=0)
    t = pd.DataFrame(rows)
    by = t.groupby("kind").size().to_dict()
    # Flight: the 1100 pass is completed (p1 -> p2), the 1150 one
    # intercepted (p2 -> p3, opponent's DRIVE 1.6 s later); 3 frames each.
    fl = t[t.kind == "flight"]
    assert by["flight"] == 6 and sorted(set(fl.category)) == ["completed", "intercepted"]
    for r in fl.itertuples():
        k0, k1 = (1100, 1150) if r.category == "completed" else (1150, 1190)
        assert k0 + 0.17 * (k1 - k0) - 0.5 <= r.frame <= k0 + 0.83 * (k1 - k0) + 0.5
        assert abs(r.s - (r.frame - k0) / (k1 - k0)) < 1e-9
        assert abs(r.t_hat - t_hat(r.s)) < 1e-12 and r.L >= MIN_PASS_M
    # The 1240 HIGH PASS to p4's PASS 2 s later: lofted at 0.35, 0.65.
    assert sorted(t[t.kind == "lofted"].frame) == [1258, 1272]
    # The DRIVE at 1190 is closed by the same player at 1240: dribble at 1215.
    assert list(t[t.kind == "dribble"].frame) == [1215]
    # Actions: PASS, PASS, DRIVE, HIGH PASS, PASS, THROW IN, PASS (not OUT).
    assert sorted(t[t.kind == "action"].frame) == [1100, 1150, 1190, 1240, 1290,
                                                   1400, 1440]
    # Free: every second of the 7 frames in play with the actor at the
    # ball (the OUT at 1330 is out of play), two each, 12 m from the ball.
    fr = t[t.kind == "free"]
    assert set(fr.frame) <= {1100, 1190, 1290, 1440} and len(fr) >= 3, fr
    assert set(fr.category) <= set(FREE_MIX)
    for r in fr.itertuples():
        T = h.track(r.frame)[0]
        A = h.people.of(r.frame, r.player_a)
        P = crop_ground(h.cal, r.x0, r.y0, step=1)
        assert min(np.hypot(*(P - A).T).min(), np.hypot(*(P - T).T).min()) >= FREE_MIN_M
    # Crops inside the pitch box; a flight crop within the jitter of its
    # bag's centre (where the box does not push it).
    assert ((t.x0 >= box[0]) & (t.x0 + SIZE <= box[2]) & (t.y0 >= box[1])
            & (t.y0 + SIZE <= box[3])).all()
    for r in fl.itertuples():
        b = flight_bbox((r.a_x, r.a_y), (r.b_x, r.b_y), *bag_window(r.t_hat, r.L), h.cal)
        assert abs(r.x0 + SIZE / 2 - (b[0] + b[2]) / 2) <= JITTER + 1
    # Draws are per sample: another frame range or kind set changes nothing.
    again = samples(h, ("flight",), box, poly, seed=0)
    assert [(r["frame"], r["x0"], r["y0"]) for r in again] == \
        [(r.frame, r.x0, r.y0) for r in fl.itertuples()]
    other = samples(h, ("flight",), box, poly, seed=1)
    assert [(r["x0"], r["y0"]) for r in other] != [(r.x0, r.y0) for r in fl.itertuples()]

    # Cutting: a textured pitch, a bright ball moving 6 px a frame, still
    # players; range 1000-1599, so the first samples have t-2 missing and
    # a short background.
    rng = np.random.default_rng(0)
    base = rng.integers(40, 160, (H, W, 3)).astype(np.uint8)
    cv2.rectangle(base, (1150, 300), (1160, 340), (20, 20, 220), -1)

    def frame(k):
        im = base.copy()
        cv2.circle(im, (300 + 6 * (k - 1000) % 1800, 400), 3, (255, 255, 255), -1)
        return im

    calls = {"n": 0}

    def frames(first, last):
        for k in range(first, last + 1):
            def fetch(k=k):
                calls["n"] += 1
                return frame(k)
            yield k, fetch

    targets = [dict(r) for r in rows] + [_row("action", 1000, 400, 200),
                                         _row("action", 1599, 400, 200)]
    got = {}
    with tempfile.TemporaryDirectory() as tmp:
        done = cut(targets, frames(1000, 1599), 1000, 1599, box, Path(tmp),
                   "118575", 1, write=lambda p, s: got.__setitem__(p.name, s),
                   log_every=0)
        # And once more through JPEG, to read a strip back.
        done2 = cut(targets[:1], frames(1000, 1599), 1000, 1599, box, Path(tmp),
                    "118575", 1, log_every=0)
        back = cv2.imread(str(Path(tmp) / done2[0]["file"]))
    assert len(done) == len(targets) and back.shape == (SIZE, 4 * SIZE, 3)
    assert calls["n"] < 2 * 600 // 3, calls["n"]      # only frames that are used
    by_name = {r["file"]: r for r in done}
    for name, strip in got.items():
        r = by_name[name]
        f, x0, y0 = r["frame"], r["x0"], r["y0"]
        assert strip.shape == (SIZE, 4 * SIZE, 3)
        for slot, g in enumerate((f - 2, f, f + 2)):
            ref = frame(g if 1000 <= g <= 1599 else f)[y0:y0 + SIZE, x0:x0 + SIZE]
            assert (strip[:, slot * SIZE:(slot + 1) * SIZE] == ref).all(), (name, slot)
        assert r["has_prev"] == (f - 2 >= 1000) and r["has_next"] == (f + 2 <= 1599)
        need = bg_sample_frames(f, 1000, 1599)
        assert r["n_bg"] == len(need)
        bg = strip[:, 3 * SIZE:, 0]
        if len(need) >= BG_MIN:
            stack = np.stack([grey(frame(g))[y0:y0 + SIZE, x0:x0 + SIZE] for g in need])
            ref = np.floor(np.median(stack, axis=0) + 0.5).astype(np.uint8)
            assert (bg == ref).all(), name
            assert (bg == grey(base)[y0:y0 + SIZE, x0:x0 + SIZE]).all()   # no ball
        else:
            assert (bg == grey(frame(f))[y0:y0 + SIZE, x0:x0 + SIZE]).all()
    assert cut.peak_samples <= 2 * 350 // BG_STEP + 3, cut.peak_samples
    # Read back (luminance: JPEG halves the colour of this noise texture).
    assert np.abs(grey(back).astype(int) - grey(got[done2[0]["file"]])).mean() < 2

    # Crash and resume: a run that dies after some saves and is run again
    # gives the JPEGs and strips.parquet of an uninterrupted run, cutting
    # only what was not saved and decoding from the first frame it needs.
    extra = dict(match="118575", half=1, width=W, height=H, source="stream",
                 allow_any=False)
    late = targets + [_row("action", 1900, 400, 200), _row("free", 2000, 700, 250),
                      _row("action", 2390, 400, 200)]

    def run(d, crash_at=None, flush_every=4):
        log = {"decoded": [], "written": 0}

        def frames_from(k0):
            for k, fetch in frames(k0, 2399):
                if k == crash_at:
                    raise RuntimeError("crash (injected)")
                log["decoded"].append(k)
                yield k, fetch

        def write(path, strip):
            log["written"] += 1
            cv2.imwrite(str(path), strip, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])

        with contextlib.redirect_stdout(io.StringIO()):
            rows_, kept = resume_cut(d, late, frames_from, 1000, 2399, box, "118575",
                                     1, extra, write=write, flush_every=flush_every,
                                     log_every=0)
        merge(d, rows_, "118575", 1, 1000, 2399, KINDS, "stream")
        partial_path(d, "118575", 1, "stream", 1000, 2399).unlink(missing_ok=True)
        return kept, log

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        a, b = Path(a), Path(b)
        kept, log_a = run(a)
        assert kept == 0 and log_a["decoded"][0] == 1000
        part = partial_path(b, "118575", 1, "stream", 1000, 2399)
        try:
            run(b, crash_at=2000)
            raise AssertionError("no crash")
        except RuntimeError:
            pass
        saved = pd.read_parquet(part)
        assert not (b / "strips.parquet").exists() and len(saved) % 4 == 0
        assert 0 < len(saved) < len(late), len(saved)
        kept, log_b = run(b)
        assert kept == len(saved) and log_b["written"] == len(late) - kept
        assert log_b["decoded"][0] > 1000 and not part.exists()
        ta = pd.read_parquet(a / "strips.parquet")
        tb = pd.read_parquet(b / "strips.parquet")
        pd.testing.assert_frame_equal(ta, tb)
        assert len(ta) == len(late)
        files = sorted(p.name for p in a.glob("*.jpg"))
        assert files == sorted(p.name for p in b.glob("*.jpg")) == sorted(ta.file)
        assert all((a / f).read_bytes() == (b / f).read_bytes() for f in files)
        n_resumed, k_resumed = kept, log_b["decoded"][0]

        # Rebuilding some kinds keeps the other kinds' rows, and another
        # source's rows, of the same half and range.
        free = ta[ta.kind == "free"].to_dict("records")
        merge(a, free[:1], "118575", 1, 1000, 2399, ("free",), "stream")
        t2 = pd.read_parquet(a / "strips.parquet")
        assert (t2.kind != "free").sum() == (ta.kind != "free").sum()
        assert (t2.kind == "free").sum() == 1
        merge(a, [dict(free[0], source="st2_w")], "118575", 1, 1000, 2399, ("free",),
              "st2_w")
        merge(a, free[1:2], "118575", 1, 1000, 2399, ("free",), "stream")
        t3 = pd.read_parquet(a / "strips.parquet")
        assert t3.groupby("source").size().to_dict() == {"st2_w": 1, "stream": len(t2)}
        merge(a, free[:1], "118575", 1, 1000, 2399)      # all kinds and sources
        assert len(pd.read_parquet(a / "strips.parquet")) == 1

        # One source per directory: window strips are refused where stream
        # strips are (in strips.parquet or an interrupted run's rows), and
        # the other way round.
        c = b / "w"
        c.mkdir()
        for d, src in ((c, "stream"), (c, "st2_w"), (b, "stream")):
            check_source(d, src)
        for d, src in ((b, "st2_118575_1st_f015000"),):
            try:
                check_source(d, src)
                raise AssertionError(f"{src} into a stream directory")
            except SystemExit:
                pass
        _save(saved.assign(source="st2_w"), partial_path(c, "118575", 1, "st2_w", 0, 99))
        try:
            check_source(c, "stream")
            raise AssertionError("stream into a window directory")
        except SystemExit:
            pass
        (b / "strips.parquet").unlink()
        _save(saved, part)
        try:
            check_source(b, "st2_w")
            raise AssertionError("window next to an interrupted stream run")
        except SystemExit:
            pass

    # Streaming: a video served with all signatures expiring mid-decode
    # (403) and a timeout decodes to the frames of the local file; a 404 is
    # raised at once, a 503 that lasts after TRIES tries with doubling waits.
    import av
    import requests

    with tempfile.TemporaryDirectory() as tmp:
        mp4 = Path(tmp) / "half.mp4"
        vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
        for k in range(60):
            vw.write(np.roll(base[:240, :320], 5 * k, axis=1))
        vw.release()
        with av.open(str(mp4)) as cont:
            ref = [f() for _, f in stream_frames(cont, 0, 10 ** 9)]
        server = _FlakyServer(mp4, expire_at=20, timeout_at=30)
        waits, logged = [], []
        fetch, size = resigning_fetch(server.sign, session=server, sleep=waits.append,
                                      log=lambda m, **k: logged.append(m))
        with av.open(st.RemoteFile(fetch=fetch, size=size, block=4096)) as cont:
            got = [(k, f()) for k, f in stream_frames(cont, 0, 10 ** 9)]
        assert [k for k, _ in got] == list(range(60)) and len(ref) == 60
        assert all((g == r).all() for (_, g), r in zip(got, ref))
        assert server.n > 30 and server.signs == 2 and waits == [BACKOFF_S]
        assert fetch.stats == {"signs": 2, "retries": 2}, fetch.stats
        server.fail = {server.n + 1: 404}
        try:
            fetch(0, 99)
            raise AssertionError("a 404 was not raised")
        except requests.HTTPError as e:
            assert e.response.status_code == 404 and server.n == max(server.fail)
            logged.append(str(e))
        server.fail = {server.n + k: 503 for k in range(1, 20)}
        waits.clear()
        n0 = server.n
        try:
            fetch(0, 99)
            raise AssertionError("a lasting 503 was not raised")
        except requests.HTTPError as e:
            assert e.response.status_code == 503 and server.n - n0 == TRIES
            logged.append(str(e))
        assert waits == [BACKOFF_S * 2 ** i for i in range(TRIES - 1)], waits
        assert len(logged) == 2 + TRIES - 1 + 2
        assert not any("cdn.invalid" in m or "Signature" in m for m in logged), logged
    print(f"  build_ball_strips: {by} samples on a synthetic half (flight "
          f"completed + intercepted, lofted, dribble, actions, free 12 m from the "
          f"ball); {len(done)} strips cut with t-2 / t / t+2 / median background, "
          f"edges and short backgrounds as zero motion; a crash resumed "
          f"({n_resumed} strips kept, decoding from {k_resumed}) gives the same "
          f"strips and strips.parquet; a rebuild of one kind keeps the others; one "
          f"source per directory; 60 frames read through an expiring signature "
          f"and a timeout as from the file; {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
