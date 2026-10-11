"""Measure a panorama ball detector on SoccerTrack v2 windows, against the
COCO detector, on the same frames.

The measure that a detector of feet cannot pass is the ball in free
flight: over the middle half of every completed ground pass (PASS, 10 m
or more, 0.8 s or more, received by a team-mate), the top candidate counts
as a hit when it is within 1 m of the line from passer to receiver
(between 10% and 90% along it) and 2 m or more from every player. Per
pass, the share of its four frames hit; reported over passes, with the
paired difference to COCO and a bootstrap 95% interval.

Also: how often the top candidate in free flight is within 1 m of a player
(`feet share`); at PASS / DRIVE frames, how often it is within 1.5 m of
the actor or of the ball track's point (`action hit`), next to the same
for a random point on the pitch; and candidates per frame above `tau` on
random frames (`strays`, with `tau` the value giving one a frame).
`--save` also keeps every frame's candidates, for `ball_tiles.py`.

    python eval_ball_detector.py --heatmap .cache/ball_heatmap.3000.pt \\
        --windows st2_118576_2nd_f015000 ... --save scores.json

Protocol v3 (ball detector v2) adds three measures for every detector,
next to the ones above (which are unchanged, and printed as before under
"windows" and "paired"):

- **FF-auto** (`ff_hit`), the primary one: at each of a pass's four
  frames (fractions s = 0.25, 0.417, 0.583, 0.75 of the pass's time), the
  top-1 candidate at any p (none = miss) is a hit when its ground point
  (u, v + size / 2 through the calibration) is within 4.0 m of the segment
  from the passer at the kick (A) to the receiver at reception (B), its
  projection t on A->B is in [0, 1] with |t - s| <= 0.30, and it is 1.0 m
  or more from every ground-truth person. The 1 m rule above missed most
  balls the blind tiles confirmed, which sit 1.3-3 m off the line; this
  one agrees with the judge (`validate_ff_auto.py`). A pass scores the
  mean of its four frames.
- **Placebo, before the kick**: at kick - 15 and kick - 30 of passes whose
  previous event is 35 frames or more before the kick and is a DRIVE by
  the passer or a completed PASS to him (same side, another player), the
  ball is at or coming to the passer's feet, so a top-1 within 4 m of the
  upcoming segment at t in [0.2, 0.9] and 1 m or more from everyone is a
  false hit -- a detector of "the grass between two players" scores here.
- **Placebo, after reception**: at reception + 15 (inside the window and
  before the next event), the same against the just-completed segment at
  t in [0.1, 0.7] -- a learned pass-corridor prior scores here.

`compare_ff()` gives the paired per-pass difference with a 1-minute block
bootstrap (blocks: window x minute of the kick; 4,000 draws, seed 0):
the 2.5th percentile is the one-sided 0.025 lower bound.

Detectors: COCO (unless `--no-coco`), `--heatmap` v1 checkpoints,
`--v2` checkpoints (`ball_heatmap.load_v2`: a motion one with the motion
scales of its provenance, an RGB-only `--no-motion` one with zero motion,
exactly its RGB net, "rgb_only" in the metadata; a first pass over the
window keeps the background samples (`motion_bg`), then one sequential
decode runs every detector, the motion ones with frames t-2, t+2 from a
ring buffer; `--motion-zero` adds each motion checkpoint with zero motion,
named `<ckpt>:motion-zero`), and `--candidates NAME=PARQUET` (rows of
`detect_ball_heatmap.py --v2`; `{window}` in the path is replaced). The
parquet's sidecar `<parquet>.meta.json` says which frames were processed:
only those are measured, and a measured frame it did not process is
refused (rerun with `--extra-frames measured`) unless `--allow-partial`.
A v2 checkpoint is refused on a match of its training provenance unless
`--allow-seen` (the T diagnostic only).

A frame a detector did not measure (not processed, past the video's end,
`--limit-frames`, a partial file) is left out of FF-auto and the
placebos, never counted as a miss: a pass with such a frame is not scored
("ff per pass" None, counted in "ff incomplete passes") nor paired in
`compare_ff()`, a placebo frame is None. The legacy measures count it as
a miss, as they always did; every run logs each detector's measured-frame
count when it falls short.

`--reuse SCORES.json [...]` measures again from saved candidates, without
the video (several files are merged per window and detector; `--candidates`
may add a detector). On V.json it gives V's registered 0.23 / 0.23 and
every per-window and per-pass value, except a few "candidates per frame":
files written before protocol v3 kept p to 3 decimals, and a p saved as
exactly a threshold does not say which side of it it was (files written
now keep p and the logit to 6). `--sealed OUT` writes everything to OUT
and prints only frame counts and timings: no measure of any kind, the
legacy ones included, reaches the screen before the blind sittings are
judged. `--frames placebo` (or passes, actions, random) measures only
those frames, e.g. to add the placebo frames to V.json's detectors
(then `--reuse V.json NEW.json`); `--limit-frames N` (smoke tests) only
the first N; `--check` runs the self-check.

The scores file (`--save` / `--sealed`), read by `ball_tiles.py`,
`validate_ff_auto.py` and `calibrate_ball_heatmap.py` (`load_scores()`
reads and merges several):

    {"summary": {"windows": [...], "paired": {...},          # legacy
                 "ff-auto": {NAME: {"pooled", "passes", "windows": {W: x},
                                    "matches": {M: x}}},
                 "ff-auto paired": {"A vs B": compare_ff(...)},
                 "placebo": {NAME: {"pre", "pre frames", "pre not measured",
                                    "post", "post frames", "post not measured",
                                    "windows"}}},
     "detectors": {NAME: {"kind": "coco" | "v1" | "v2" | "v2-motion-zero",
                          "path", "calib" (<ckpt>.calib.json if there),
                          "train" (provenance), "ckpt sha256", "rgb_only",
                          "sigma" ({"SIGMA_D", "SIGMA_F"} used),
                          "source": "parquet", "windows" ({W: its sidecar
                          less the frame list})}},
     "protocol": {the FF-auto and placebo constants},
     "results": [one per window: {
        "window", "passes" (count), "actions" (count),
        "random point action hit",                          # legacy
        "match", "half", "start_frame",
        "frames": {
          "passes": [{"i" (pass id in the window), "event" (row of the
                      sorted events csv at the kick), "kick", "reception",
                      "frames" (4 local frames), "s" (their fractions),
                      "from" [x, y], "to" [x, y] (pitch m, centre origin),
                      "minute" (kick // 1500), "block" ("W:minute")}],
          "actions": [{"frame", "label" (PASS | DRIVE), "player_id",
                       "x", "y"}],                          # x, y: actor
          "random": [frames],
          "placebo_pre": [{"frame", "pass" (i), "lead" (15 | 30)}],
          "placebo_post": [{"frame", "pass" (i)}]},
        "people": {"frame": [[x, y], ...]},   # ground-truth persons, m
        "n_bg": {"frame": n},     # background samples: decoded in v2 runs,
                                  # else those a complete decode would have
        "frames to measure",      # the window's measured frames, all kinds
        NAME: {
          "free-flight hit", "per pass", "feet share", "action hit",
          "candidates per frame",                           # legacy
          "candidates": {"frame": [[u, v, p, size], ...]},  # best first;
              # v2 rows are [u, v, p_raw, size, logit] (p_raw >= 0.05,
              # at most 8); v1 [u, v, p, size] (p >= 0.01, at most 8);
              # COCO [u, v, conf, box height]; frames are local
          "frames measured",                                # of those
          "ff-auto", "ff per pass" (None: incomplete), "ff frames" ([4
              hits] per pass, None where not measured), "ff incomplete passes",
          "placebo pre", "placebo post" ({"rate", "hits", "frames",
              "not measured"}),
          "placebo pre hits", "placebo post hits"}]}  # per entry, None
                                                     # where not measured

All frames are local to the window (absolute = start_frame + local).
A frame missing from a detector's "candidates" was not measured; an empty
list was measured and had no candidate (a miss). To recompute geometry
without the video, use `ground()` (needs `st.calibration(match)`) and
`ff_hit()` / `corridor_hit()` with the people of that frame -- e.g. for a
Platt fit, `ff_hit(ground(np.asarray(rows)[k:k + 1], cal)[0], pass, q,
people)` for each
row at a pass frame, 1.5 m from the action's x, y at its PASS / DRIVE
frame, and the rows at the "random" frames for candidates per frame.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import soccertrack_v2 as st
from src.ball_heatmap import BALL_M, build, detect, local_scale
from src.paths import DATA_DIR

FPS = 25
TAUS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7)
N_RANDOM = 50

# Protocol v3: FF-auto and the placebos (frozen; see the module docstring).
FF_W = 4.0                  # m from the segment
FF_DT = 0.30                # |t - s|
FF_R = 1.0                  # m from every person
FF_S = (0.25, 0.25 + 0.5 / 3, 0.25 + 1.0 / 3, 0.75)
PRE_LEADS = (15, 30)        # frames before the kick
PRE_GAP = 35                # previous event at least this long before it
PRE_T = (0.2, 0.9)
POST_LAG = 15               # frames after reception
POST_T = (0.1, 0.7)
BLOCK_FRAMES = 60 * FPS     # bootstrap blocks: a minute of a window
DRAWS = 4000
FULL_BG = 15                # background samples with nothing clipped
V2_TAU, TOP_K = 0.05, 8     # as detect_ball_heatmap.py --v2
TEST_MATCHES = {"128057", "132831"}     # M7, M9: never opened
LEGACY = ("free-flight hit", "feet share", "action hit", "candidates per frame")


def frames_of(window: str, seed: int = 0) -> dict:
    """The frames measured in a window: free-flight frames per pass,
    action frames, random frames, and (protocol v3) the placebo frames.
    Each pass also carries its id, kick and reception frames, the
    fractions s of its four frames and its minute block."""
    lab = pd.read_csv(DATA_DIR / "soccertrack_v2" / f"{window}_events.csv")
    n = len(pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{window}_gt.parquet",
                            columns=["frame"]).frame.unique())
    return frame_sets(lab, n, window, seed)


def frame_sets(lab: pd.DataFrame, n: int, window: str, seed: int = 0) -> dict:
    """`frames_of` from a window's events table and its number of frames."""
    lab = lab.sort_values("frame").reset_index(drop=True)
    passes = []
    for i in range(len(lab) - 1):
        a, b = lab.iloc[i], lab.iloc[i + 1]
        if (a.label != "PASS" or b.side != a.side or b.player_id == a.player_id
                or pd.isna(a.x) or pd.isna(b.x)):
            continue
        dist = np.hypot(b.x - a.x, b.y - a.y)
        if dist < 10 or b.frame - a.frame < 0.8 * FPS:
            continue
        lo, hi = a.frame + 0.25 * (b.frame - a.frame), a.frame + 0.75 * (b.frame - a.frame)
        minute = int(a.frame) // BLOCK_FRAMES
        passes.append({"frames": [int(round(f)) for f in np.linspace(lo, hi, 4)],
                       "from": (a.x, a.y), "to": (b.x, b.y),
                       "i": len(passes), "event": i, "kick": int(a.frame),
                       "reception": int(b.frame), "s": list(FF_S),
                       "minute": minute, "block": f"{window}:{minute}"})
    actions = lab[lab.label.isin(("PASS", "DRIVE"))][["frame", "player_id", "x", "y",
                                                      "label"]]
    rng = np.random.default_rng(seed)
    random = sorted(rng.choice(n, min(N_RANDOM, n), replace=False).tolist())
    pre, post = [], []
    for p in passes:
        k = p["event"]
        a = lab.iloc[k]
        if k > 0:
            prev = lab.iloc[k - 1]
            drive = prev.label == "DRIVE" and prev.player_id == a.player_id
            given = (prev.label == "PASS" and prev.side == a.side
                     and prev.player_id != a.player_id)
            if a.frame - prev.frame >= PRE_GAP and (drive or given):
                pre += [{"frame": p["kick"] - lead, "pass": p["i"], "lead": lead}
                        for lead in PRE_LEADS]
        f = p["reception"] + POST_LAG
        nxt = lab.iloc[k + 2].frame if k + 2 < len(lab) else None
        if f < n and (nxt is None or f < nxt):
            post.append({"frame": int(f), "pass": p["i"]})
    return {"passes": passes, "actions": actions, "random": random,
            "placebo_pre": pre, "placebo_post": post}


FRAME_KINDS = ("passes", "placebo", "actions", "random")


def measured_frames(sets: dict, kinds=FRAME_KINDS) -> list[int]:
    """The local frames a window is measured at, of the given kinds."""
    wanted = set()
    if "random" in kinds:
        wanted |= set(sets["random"])
    if "actions" in kinds:
        wanted |= set(int(f) for f in sets["actions"].frame)
    if "passes" in kinds:
        for p in sets["passes"]:
            wanted |= set(p["frames"])
    if "placebo" in kinds:
        wanted |= {e["frame"] for e in sets["placebo_pre"] + sets["placebo_post"]}
    return sorted(wanted)


def coco_candidates(model, frame, outline, margin=10.0):
    poly = np.asarray(outline, np.float32).reshape(-1, 1, 2)
    imgsz = -(-frame.shape[1] // 32) * 32
    r = model(frame, imgsz=imgsz, conf=0.02, classes=[32], verbose=False)[0]
    rows = []
    for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.cpu().numpy(),
                                   r.boxes.conf.cpu().numpy()):
        u, v = (x0 + x1) / 2, (y0 + y1) / 2
        if cv2.pointPolygonTest(poly, (float(u), float(v)), True) >= -margin:
            rows.append((u, v, float(c), float(y1 - y0)))
    rows.sort(key=lambda t: -t[2])
    return np.array(rows) if rows else np.zeros((0, 4))


def ground(cands, cal):
    """Candidates' ground points in pitch metres (the ball's bottom)."""
    if not len(cands):
        return np.zeros((0, 2))
    size = np.nan_to_num(cands[:, 3], nan=0.0)
    return st.image_to_pitch(np.c_[cands[:, 0], cands[:, 1] + size / 2], cal)


def seg_dist(p, a, b):
    a, b, p = map(np.asarray, (a, b, p))
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
    return float(np.hypot(*(p - (a + t * ab)))), t


# ---- Protocol v3 geometry ---------------------------------------------------------

def top_ground(rows, cal):
    """The top-1 candidate's ground point, or None (no candidate, or at or
    above the horizon)."""
    rows = np.asarray(rows, float)
    if not len(rows):
        return None
    g = ground(rows[:1], cal)[0]
    return g if np.isfinite(g).all() else None


def corridor_hit(g, a, b, t_lo: float, t_hi: float, people,
                 s: float | None = None) -> bool:
    """Ground point `g` within FF_W of segment a->b, its projection t (not
    clipped) in [t_lo, t_hi] (and within FF_DT of `s` when given), and
    FF_R or more from every person (`people` N x 2; None = not known, no
    hit)."""
    if g is None or people is None:
        return False
    a, b, g = (np.asarray(x, float) for x in (a, b, g))
    ab = b - a
    t = float(np.dot(g - a, ab) / max(np.dot(ab, ab), 1e-9))
    d = float(np.hypot(*(g - (a + np.clip(t, 0, 1) * ab))))
    people = np.asarray(people, float).reshape(-1, 2)
    near = float(np.hypot(*(people - g).T).min()) if len(people) else np.inf
    return bool(d <= FF_W and t_lo <= t <= t_hi and near >= FF_R
                and (s is None or abs(t - s) <= FF_DT))


def ff_hit(g, p: dict, q: int, people) -> bool:
    """FF-auto at the q-th (0-based) frame of pass `p` (a `frames_of` pass
    or a saved one) for ground point `g`."""
    return corridor_hit(g, p["from"], p["to"], 0.0, 1.0, people, s=p["s"][q])


def ball_track(info: dict) -> pd.DataFrame:
    """The release's ball track over a window (x, y by local frame), kept
    in the cache after the first read."""
    name = f"{info['match']}_{st._half(info['half'])}_ball.npz"
    path = st.cache_dir() / "ball" / name
    if not path.exists():
        data = st.open_stream(f"ball/{name}").read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    z = np.load(io.BytesIO(path.read_bytes()), allow_pickle=False)
    return pd.DataFrame({"x": z["x"], "y": z["y"]},
                        index=z["frame"].astype(int) - 1 - info["start_frame"])


def window_data(window: str, seed: int = 0) -> dict:
    """Everything measuring a window needs besides the candidates."""
    info = json.loads((DATA_DIR / "soccertrack_v2" / f"{window}.json").read_text())
    if str(info["match"]) in TEST_MATCHES:
        raise ValueError(f"{window}: test match {info['match']} is never opened")
    gt = pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{window}_gt.parquet")
    return {"window": window, "info": info, "cal": st.calibration(info["match"]),
            "sets": frames_of(window, seed),
            "people": {f: g[["x", "y"]].to_numpy(float) for f, g in gt.groupby("frame")},
            "track": ball_track(info)}


# ---- Candidates -------------------------------------------------------------------

def decode(wd: dict, detectors: dict, motion: dict, wanted,
           log=print) -> tuple[dict, dict]:
    """Run the detectors over a window's `wanted` local frames, in one
    sequential decode. `detectors`: name -> fn(image, outline, cal);
    `motion`: name -> fn(prev, cur, nxt, bg, outline, cal), run at t once
    t+2 is decoded, after a first pass that keeps the background samples
    (`motion_bg`; the window is the available range). Returns the
    candidates {name: {frame: rows}} and, with motion, {frame: n_bg}."""
    from src.ball_heatmap import pitch_box
    from src.motion_bg import (BG_HALF_SPAN, BG_STEP, OFFSET, BackgroundCache,
                               RingBuffer, bg_sample_frames)

    info, cal, pitch = wd["info"], wd["cal"], wd["info"]["pitch"]
    path = str(DATA_DIR / "soccertrack_v2" / f"{wd['window']}.mp4")
    first = info["start_frame"]
    last = first + info["frames"] - 1
    wanted = set(wanted)
    cands = {name: {} for name in [*detectors, *motion]}
    n_bg, cache, look = {}, None, 0
    t0 = time.time()
    if motion:
        # Pass 1: grab every frame, keep (cropped, grey) only the samples
        # some measured frame's background needs.
        need = set()
        for t in wanted:
            need |= set(bg_sample_frames(first + t, first, last))
        box = pitch_box((info["height"], info["width"]), pitch)
        cache = BackgroundCache(first, last, crop=box)
        cap = cv2.VideoCapture(path)
        idx, stop = 0, max(need) - first
        while idx <= stop:
            if not cap.grab():
                break
            if first + idx in need:
                ok, image = cap.retrieve()
                if ok:
                    cache.add(first + idx, image)
            idx += 1
        cap.release()
        cache.finish()
        look = OFFSET
        log(f"  {wd['window']}: background pass, {idx} frames, "
            f"{len(cache.samples)} samples [{time.time() - t0:.0f} s]")
    spent = {name: 0.0 for name in cands}

    def run(t, ring):
        cur = ring.get(t)
        for name, fn in detectors.items():
            s = time.time()
            cands[name][t] = fn(cur, pitch, cal)
            spent[name] += time.time() - s
        if motion:
            bg, n = cache.get(first + t)
            n_bg[t] = n
            prev, nxt = ring.get(t - look), ring.get(t + look)
            for name, fn in motion.items():
                s = time.time()
                cands[name][t] = fn(prev, cur, nxt, bg, pitch, cal)
                spent[name] += time.time() - s
            cache.drop_before(first + t - BG_HALF_SPAN - BG_STEP)

    ring = RingBuffer(2 * OFFSET + 1)
    cap = cv2.VideoCapture(path)
    idx, end = 0, max(wanted) + look
    while idx <= end:
        ok, image = cap.read()
        if not ok:
            break
        ring.push(idx, image)
        if idx - look in wanted:
            run(idx - look, ring)
        idx += 1
    cap.release()
    for t in range(max(idx - look, 0), idx):       # t + 2 past the end
        if t in wanted:
            run(t, ring)
    log(f"  {wd['window']}: {len(wanted)} frames measured, {idx} decoded "
        f"[{time.time() - t0:.0f} s; " +
        ", ".join(f"{k} {v:.0f} s" for k, v in spent.items()) + "]")
    return cands, n_bg


def parquet_candidates(path, wanted, top_k: int = TOP_K, allow_partial: bool = False,
                       window: str | None = None, sidecar: dict | None = None) -> dict:
    """{frame: rows} at the `wanted` local frames a dense ball-rows parquet
    (`detect_ball_heatmap.py --v2`) processed, as its sidecar
    (`<path>.meta.json`) lists them: [u, v, p_raw, size, logit] when it has
    a logit column (p_raw = sigmoid(logit) unless saved), else [u, v,
    confidence, size]; best first, at most `top_k`. A processed frame
    without rows is a measured frame without candidates; a wanted frame
    not processed raises (`allow_partial`: left out, i.e. not measured),
    as do a missing or stale sidecar, one of another `window`, and rows
    at a frame it does not list. `sidecar`, when given, receives the
    sidecar's contents."""
    from detect_ball_heatmap import read_sidecar

    meta = read_sidecar(path)
    if window is not None and meta.get("window") not in (None, window):
        raise ValueError(f"{path}: the sidecar is of window {meta['window']}, not {window}")
    done = {int(f) for f in meta["frames"]}
    df = pd.read_parquet(path)
    if "cls" in df:
        df = df[df.cls == "ball"]
    if "logit" in df:
        z = df.logit.to_numpy(float)
        p = df.p_raw.to_numpy(float) if "p_raw" in df else 1 / (1 + np.exp(-z))
        cols = np.c_[df.px, df.py, p, df.crop_h, z]
    else:
        cols = np.c_[df.px, df.py, df.confidence, df.crop_h]
    frames = df.frame.to_numpy(int)
    stray = sorted(set(frames.tolist()) - done)
    if stray:
        raise ValueError(f"{path}: rows at {len(stray)} frames its sidecar does not "
                         f"list as processed (e.g. {stray[:5]})")
    want = sorted({int(f) for f in wanted})
    missing = [f for f in want if f not in done]
    if missing and not allow_partial:
        raise ValueError(f"{path}: {len(missing)} of {len(want)} measured frames were "
                         f"not processed (e.g. {missing[:5]}); rerun "
                         f"detect_ball_heatmap.py --v2 with --extra-frames measured, "
                         f"or --allow-partial to leave them unmeasured")
    if sidecar is not None:
        sidecar.update(meta)
    out = {}
    for f in want:
        if f in done:
            rows = cols[frames == f]
            out[f] = rows[np.argsort(-rows[:, 2], kind="stable")][:top_k]
    return out


# ---- Measures ---------------------------------------------------------------------

def _saved(rows) -> list:
    """Candidate rows as saved: pixels and size to 3 decimals, p and the
    logit to 6 (at 3, a p saved as exactly a `TAUS` value cannot say which
    side of it it was, and `--reuse` could not redo the counts)."""
    rows = np.asarray(rows, float)
    a = np.round(rows, 3)
    if a.ndim == 2 and len(a):
        a[:, 2] = np.round(rows[:, 2], 6)
        a[:, 4:] = np.round(rows[:, 4:], 6)
    return a.tolist()


def _rate(hits) -> dict:
    """Hit rate over the measured entries (None: not measured, counted)."""
    h = [x for x in hits if x is not None]
    return {"rate": round(float(np.mean(h)), 3) if h else None,
            "hits": int(sum(h)), "frames": len(h), "not measured": len(hits) - len(h)}


def measure(wd: dict, cands: dict, seed: int = 0, n_bg: dict | None = None) -> dict:
    """Every measure of a window from its candidates {name: {frame: rows}}:
    the registered ones exactly as before, then FF-auto and the placebos;
    with the frame sets, people and n_bg for the scores file."""
    sets, cal, people, track = wd["sets"], wd["cal"], wd["people"], wd["track"]
    info, window = wd["info"], wd["window"]
    out = {"window": window, "passes": len(sets["passes"]),
           "actions": len(sets["actions"])}
    rng = np.random.default_rng(seed)
    pitch = np.asarray(info["pitch"], np.float32)
    for name, c in cands.items():
        hits, feet = [], []
        for p in sets["passes"]:
            ph = []
            for f in p["frames"]:
                if f not in c or not len(c[f]) or f not in people:
                    ph.append(0.0)
                    continue
                g = ground(c[f][:1], cal)[0]
                if not np.isfinite(g).all():
                    ph.append(0.0)
                    continue
                d, t = seg_dist(g, p["from"], p["to"])
                near = np.hypot(*(people[f] - g).T).min()
                feet.append(near <= 1.0)
                ph.append(float(d <= 1.0 and 0.1 <= t <= 0.9 and near >= 2.0))
            hits.append(float(np.mean(ph)))
        act = []
        for a in sets["actions"].itertuples():
            f = int(a.frame)
            if f not in c or not len(c[f]) or f not in track.index:
                act.append(0.0)
                continue
            g = ground(c[f][:1], cal)[0]
            ok = np.isfinite(g).all() and (
                np.hypot(g[0] - a.x, g[1] - a.y) <= 1.5 or
                np.hypot(*(g - track.loc[f, ["x", "y"]].to_numpy(float))) <= 1.5)
            act.append(float(ok))
        per_frame = {t: float(np.mean([(c[f][:, 2] >= t).sum() if len(c.get(f, ())) else 0
                                       for f in sets["random"]])) for t in TAUS}
        # Protocol v3. A frame not in `c` was not measured: None, as for the
        # placebos; a pass with one is incomplete (None), not a miss.
        ff = [[None if f not in c else int(ff_hit(top_ground(c[f], cal), p, q,
                                                  people.get(f)))
               for q, f in enumerate(p["frames"])] for p in sets["passes"]]
        ff_pass = [None if None in h else float(np.mean(h)) for h in ff]
        scored = [x for x in ff_pass if x is not None]
        placebo = {}
        for kind, (lo, hi) in (("pre", PRE_T), ("post", POST_T)):
            placebo[kind] = []
            for e in sets[f"placebo_{kind}"]:
                f, p = e["frame"], sets["passes"][e["pass"]]
                placebo[kind].append(None if f not in c else int(corridor_hit(
                    top_ground(c[f], cal), p["from"], p["to"], lo, hi, people.get(f))))
        out[name] = {"free-flight hit": round(float(np.mean(hits)), 3) if hits else None,
                     "per pass": hits,
                     "candidates": {str(f): _saved(v) for f, v in sorted(c.items())},
                     "feet share": round(float(np.mean(feet)), 3) if feet else None,
                     "action hit": round(float(np.mean(act)), 3) if act else None,
                     "candidates per frame": {str(k): round(v, 2)
                                              for k, v in per_frame.items()},
                     "frames measured": sum(f in c for f in measured_frames(sets)),
                     "ff-auto": round(float(np.mean(scored)), 3) if scored else None,
                     "ff per pass": ff_pass, "ff frames": ff,
                     "ff incomplete passes": len(ff_pass) - len(scored),
                     "placebo pre": _rate(placebo["pre"]),
                     "placebo post": _rate(placebo["post"]),
                     "placebo pre hits": placebo["pre"],
                     "placebo post hits": placebo["post"]}
    # A random point on the pitch, for the action hit.
    lo, hi = pitch.min(axis=0), pitch.max(axis=0)
    rnd = []
    for a in sets["actions"].itertuples():
        g = st.image_to_pitch([rng.uniform(lo, hi)], cal)[0]
        rnd.append(float(np.isfinite(g).all() and np.hypot(g[0] - a.x, g[1] - a.y) <= 1.5))
    out["random point action hit"] = round(float(np.mean(rnd)), 3) if rnd else None
    out.update(match=str(info["match"]), half=int(info["half"]),
               start_frame=int(info["start_frame"]), frames=frames_detail(sets),
               people={str(f): np.round(people[f], 3).tolist()
                       for f in measured_frames(sets) if f in people})
    out["frames to measure"] = len(measured_frames(sets))
    # Background samples per measured frame: as decoded for v2, else as a
    # complete decode of the window gives them.
    from src.motion_bg import bg_sample_frames

    a = int(info["start_frame"])
    z = a + int(info.get("frames", 0)) - 1
    n_bg = n_bg or ({f: len(bg_sample_frames(a + f, a, z)) for f in measured_frames(sets)}
                    if "frames" in info else {})
    if n_bg:
        out["n_bg"] = {str(f): int(n) for f, n in sorted(n_bg.items())}
    return out


def _num(x):
    return None if pd.isna(x) else float(x)


def frames_detail(sets: dict) -> dict:
    """`frames_of` as JSON: what a reader needs to redo any measure's
    geometry without the video."""
    passes = [dict(p, frames=[int(f) for f in p["frames"]],
                   **{"from": [float(v) for v in p["from"]],
                      "to": [float(v) for v in p["to"]]}) for p in sets["passes"]]
    actions = [{"frame": int(a.frame), "label": a.label,
                "player_id": None if pd.isna(a.player_id) else str(a.player_id),
                "x": _num(a.x), "y": _num(a.y)} for a in sets["actions"].itertuples()]
    return {"passes": passes, "actions": actions,
            "random": [int(f) for f in sets["random"]],
            "placebo_pre": sets["placebo_pre"], "placebo_post": sets["placebo_post"]}


def run_window(window: str, detectors: dict, seed: int = 0, motion: dict | None = None,
               limit: int | None = None, extra: dict | None = None, log=print,
               kinds=FRAME_KINDS) -> dict:
    """Decode a window once, run the detectors at its measured frames (of
    `kinds` only, e.g. the placebo frames to add to an older file; the
    first `limit` of them in a smoke test) and measure. `extra`: name ->
    fn(window, frames) -> {frame: rows}, candidates obtained otherwise
    (`--candidates`)."""
    wd = window_data(window, seed)
    wanted = measured_frames(wd["sets"], kinds)[:limit]
    cands, n_bg = {}, {}
    if detectors or motion:
        cands, n_bg = decode(wd, detectors, motion or {}, wanted, log)
    for name, fn in (extra or {}).items():
        rows = fn(window, wanted)
        cands[name] = {f: rows[f] for f in wanted if f in rows}
    log_coverage(window, cands, wanted, log)
    return measure(wd, cands, seed, n_bg)


def log_coverage(window: str, cands: dict, wanted, log=print):
    """Log the detectors measured at fewer than the `wanted` frames, with
    their measured-frame counts (counts only: allowed when sealed)."""
    want = set(wanted)
    short = {k: len(want & set(v)) for k, v in cands.items()}
    short = {k: n for k, n in short.items() if n < len(want)}
    if short:
        log(f"  {window}: not every one of its {len(want)} frames measured: " +
            ", ".join(f"{k} {n}" for k, n in short.items()))


def sealed_counts(r: dict, names) -> str:
    """A window's line of the `--sealed` log: frame counts only -- passes,
    placebo frames, and per detector the frames it measured, with its
    incomplete passes and placebo frames not measured."""
    def gaps(d):
        n = d["ff incomplete passes"]
        g = [f"{n} passes incomplete"] if n else []
        g += [f"{d[f'placebo {k}']['not measured']} placebo {k} frames not measured"
              for k in ("pre", "post") if d[f"placebo {k}"]["not measured"]]
        return f" ({', '.join(g)})" if g else ""

    fr = r["frames"]
    per = ", ".join(f"{n} {r[n]['frames measured']}{gaps(r[n])}" for n in names if n in r)
    return (f"  {r['window']}: {len(fr['passes'])} passes, {len(fr['placebo_pre'])} + "
            f"{len(fr['placebo_post'])} placebo frames, frames measured of "
            f"{r['frames to measure']}: {per}")


def parquet_source(path: str, allow_partial: bool = False):
    """`--candidates` reader: fn(window, frames) over `path`, with
    `{window}` replaced; a path without it serves one window only.
    `fn.sidecars` keeps each window's sidecar, less its frame list."""
    seen = set()

    def fn(window, wanted):
        seen.add(window)
        if "{window}" not in path and len(seen) > 1:
            raise ValueError(f"{path}: one parquet for several windows; "
                             "put {window} in the path")
        side = {}
        out = parquet_candidates(path.replace("{window}", window), wanted,
                                 allow_partial=allow_partial, window=window, sidecar=side)
        fn.sidecars[window] = {k: v for k, v in side.items() if k != "frames"}
        fn.sidecars[window]["processed"] = len(side["frames"])
        return out

    fn.sidecars = {}
    return fn


# ---- Reuse ------------------------------------------------------------------------

def _rows(rows) -> np.ndarray:
    a = np.asarray(rows, float)
    return a if a.ndim == 2 else np.zeros((0, 4))


def load_scores(paths) -> tuple[dict, dict]:
    """Several scores files merged: {window: {"frames", "n_bg", ...,
    NAME: {"candidates": {frame: rows}}}} with a detector's candidates
    united over the files (a later file wins at a shared frame), and the
    detectors' metadata."""
    merged, meta = {}, {}
    for path in paths:
        blob = json.loads(Path(path).read_text())
        meta.update(blob.get("detectors", {}))
        for r in blob["results"]:
            m = merged.setdefault(r["window"], {"window": r["window"]})
            for k, v in r.items():
                if isinstance(v, dict) and "candidates" in v:
                    m.setdefault(k, {"candidates": {}})["candidates"].update(
                        v["candidates"])
                elif k == "n_bg":
                    m.setdefault("n_bg", {}).update(v)
                elif k not in m:
                    m[k] = v
    return merged, meta


def reuse(paths, windows=None, seed: int = 0, extra=None, log=print) -> tuple[list, dict]:
    """Measure again from saved candidates (no video): the results of
    `load_scores(paths)` per window, in order, and the metadata."""
    merged, meta = load_scores(paths)
    results = []
    for w in windows or list(merged):
        r = merged[w]
        wd = window_data(w, seed)
        cands = {k: {int(f): _rows(rows) for f, rows in v["candidates"].items()}
                 for k, v in r.items() if isinstance(v, dict) and "candidates" in v}
        wanted = measured_frames(wd["sets"])
        for name, fn in (extra or {}).items():
            rows = fn(w, wanted)
            cands[name] = {f: rows[f] for f in wanted if f in rows}
        log_coverage(w, cands, wanted, log)
        n_bg = {int(f): n for f, n in r.get("n_bg", {}).items()}
        results.append(measure(wd, cands, seed, n_bg))
        log(f"  {w}: {len(wanted)} frames to measure (from saved candidates: "
            f"{', '.join(cands)})")
    return results, meta


# ---- Statistics -------------------------------------------------------------------

def compare(results, a: str, b: str = "coco", draws: int = 2000, seed: int = 0):
    """Free-flight hit, a minus b, over all passes; bootstrap 95% interval."""
    da = np.concatenate([r[a]["per pass"] for r in results])
    db = np.concatenate([r[b]["per pass"] for r in results])
    if not len(da):
        return None
    rng = np.random.default_rng(seed)
    boots = [np.mean((da - db)[rng.integers(0, len(da), len(da))])
             for _ in range(draws)]
    return {"passes": int(len(da)), a: round(float(da.mean()), 3),
            b: round(float(db.mean()), 3),
            "difference": round(float((da - db).mean()), 3),
            "95% interval": [round(float(np.percentile(boots, 2.5)), 3),
                             round(float(np.percentile(boots, 97.5)), 3)]}


def compare_ff(results, a: str, b: str = "coco", draws: int = DRAWS, seed: int = 0,
               blocks: bool = True, full_bg: bool = False):
    """FF-auto, a minus b, paired per pass over the windows that have both.

    The bootstrap draws whole 1-minute blocks (window x minute of the kick)
    with replacement and takes the pass-weighted mean, so passes of one
    spell of play move together; `blocks=False` draws single passes. The
    2.5th percentile is the one-sided alpha 0.025 lower bound: a > b when
    it is above 0. `full_bg` (v2 diagnostic) keeps only frames whose
    background had all FULL_BG samples (window edges have fewer). A pass
    with a frame either detector did not measure is left out (counted in
    "passes not measured"), not paired against a miss.
    Returns the means, the difference, the 2.5th / 97.5th percentiles,
    the number of passes and blocks, and the same per match."""
    da, db, keys, match, unmeasured = [], [], [], [], 0
    for r in results:
        if a not in r or b not in r:
            continue
        nb = r.get("n_bg", {})
        for p, fa, fb in zip(r["frames"]["passes"], r[a]["ff frames"], r[b]["ff frames"]):
            if None in fa or None in fb:
                unmeasured += 1
                continue
            if full_bg:
                keep = [nb.get(str(f)) == FULL_BG for f in p["frames"]]
                if not any(keep):
                    continue
                fa = [h for h, k in zip(fa, keep) if k]
                fb = [h for h, k in zip(fb, keep) if k]
            da.append(float(np.mean(fa)))
            db.append(float(np.mean(fb)))
            keys.append(p["block"] if blocks else f"{len(keys):06d}")
            match.append(r["match"])
    if not da:
        return None
    da, db, match = np.array(da), np.array(db), np.array(match)
    d = da - db
    _, bi = np.unique(keys, return_inverse=True)
    S, N = np.bincount(bi, weights=d), np.bincount(bi)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(S), (draws, len(S)))
    boots = S[draw].sum(axis=1) / N[draw].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"passes": int(len(d)), "blocks": int(len(S)), a: round(float(da.mean()), 3),
            b: round(float(db.mean()), 3), "difference": round(float(d.mean()), 3),
            "2.5%": round(float(lo), 3), "97.5%": round(float(hi), 3),
            "lower bound above 0": bool(lo > 0), "passes not measured": unmeasured,
            "matches": {m: {"passes": int((match == m).sum()),
                            a: round(float(da[match == m].mean()), 3),
                            b: round(float(db[match == m].mean()), 3),
                            "difference": round(float(d[match == m].mean()), 3)}
                        for m in sorted(set(match))}}


def ff_summary(results, name: str) -> dict:
    """FF-auto of a detector per window, per match and pooled over passes
    (complete ones: a pass with an unmeasured frame is counted apart)."""
    rs = [r for r in results if name in r]
    allp = [x for r in rs for x in r[name]["ff per pass"] if x is not None]
    by_match = {}
    for r in rs:
        by_match.setdefault(r["match"], []).extend(
            x for x in r[name]["ff per pass"] if x is not None)

    def m(x):
        return round(float(np.mean(x)), 3) if len(x) else None

    return {"pooled": m(allp), "passes": len(allp),
            "incomplete passes": sum(x is None for r in rs for x in r[name]["ff per pass"]),
            "windows": {r["window"]: r[name]["ff-auto"] for r in rs},
            "matches": {k: m(v) for k, v in sorted(by_match.items())}}


def placebo_summary(results, name: str) -> dict:
    rs = [r for r in results if name in r]
    out = {}
    for kind in ("pre", "post"):
        s = _rate([h for r in rs for h in r[name][f"placebo {kind} hits"]])
        out[kind], out[f"{kind} frames"] = s["rate"], s["frames"]
        out[f"{kind} not measured"] = s["not measured"]
    out["windows"] = {r["window"]: {k: r[name][f"placebo {k}"]["rate"]
                                    for k in ("pre", "post")} for r in rs}
    return out


def summarise(results, names, kinds: dict) -> dict:
    """The legacy summary, unchanged ("windows", "paired" when COCO is
    there), then FF-auto, its paired differences (each detector against
    COCO; each v2 against each v1; v2 also on full backgrounds only) and
    the placebos."""
    summary = {"windows": [{k: r[k] for k in ("window", "passes", "actions",
                                              "random point action hit")} | {
        n: {k: r[n][k] for k in LEGACY} for n in names if n in r} for r in results]}
    common = [n for n in names if all(n in r for r in results)]
    if "coco" in common:
        summary["paired"] = {n: compare(results, n) for n in common if n != "coco"}
    summary["ff-auto"] = {n: ff_summary(results, n) for n in names}
    pairs = [(n, "coco") for n in names if n != "coco" and "coco" in names]
    pairs += [(n, m) for n in names if kinds.get(n) == "v2"
              for m in names if kinds.get(m) == "v1"]
    paired = {f"{a} vs {b}": compare_ff(results, a, b) for a, b in pairs}
    for a, b in pairs:
        if kinds.get(a) == "v2" and b == "coco" and all(r.get("n_bg") for r in results):
            paired[f"{a} vs {b} (full background)"] = compare_ff(results, a, b,
                                                                 full_bg=True)
    summary["ff-auto paired"] = paired
    summary["placebo"] = {n: placebo_summary(results, n) for n in names}
    return summary


def kind_of(name: str, results, meta: dict) -> str:
    """coco, v1, v2 or v2-motion-zero: from the metadata, else (old files,
    parquet rows) COCO by name, v2 when the rows carry a logit."""
    kind = meta.get(name, {}).get("kind")
    if kind is not None:
        return kind
    if name == "coco":
        return "coco"
    for r in results:
        for rows in r.get(name, {}).get("candidates", {}).values():
            if len(rows):
                return "v2" if len(rows[0]) == 5 else "v1"
    return "v1"


# ---- Command line -----------------------------------------------------------------

def v2_detectors(paths, matches, allow_seen: bool = False, motion_zero: bool = False,
                 log=print) -> tuple[dict, dict]:
    """`--v2`: {name: fn(prev, cur, nxt, bg, outline, cal)} and their
    metadata. Each checkpoint is refused on a training match among
    `matches` (unless `allow_seen`) and read by `ball_heatmap.load_v2`: a
    motion one runs with its provenance's motion scales, an RGB-only one
    with zero motion -- its RGB net, so `motion_zero` adds no second run
    of it."""
    import hashlib

    from src import ball_heatmap as bh

    motion, meta = {}, {}
    for path in paths:
        for m in matches:
            bh.check_unseen(path, m, allow_seen)
        model, info = bh.load_v2(path)
        prov = bh.read_provenance(path)
        rgb, sd, sf = info["rgb_only"], info["sigma_d"], info["sigma_f"]
        if rgb:
            log(f"  {Path(path).name}: RGB-only, run with zero motion")
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for zero in (False, True) if motion_zero and not rgb else (False,):
            name = Path(path).name + (":motion-zero" if zero else "")
            motion[name] = (lambda mm, z, sd, sf: lambda prev, cur, nxt, bg, pitch, cal:
                            bh.detect_v2(mm, prev, cur, nxt, bg, pitch, tau=V2_TAU,
                                         top_k=TOP_K, cal=cal, motion_zero=z,
                                         sigma_d=sd, sigma_f=sf))(
                model, zero or rgb, sd, sf)
            meta[name] = {"kind": "v2-motion-zero" if zero else "v2",
                          "path": str(path), "calib": bh.load_calibration(path),
                          "train": prov and prov.get("train"), "ckpt sha256": sha,
                          "rgb_only": rgb, "sigma": {"SIGMA_D": sd, "SIGMA_F": sf}}
    return motion, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heatmap", nargs="*", default=[],
                    help="checkpoints of src/ball_heatmap.build")
    ap.add_argument("--v2", nargs="*", default=[],
                    help="checkpoints of src/ball_heatmap.build(motion=True)")
    ap.add_argument("--motion-zero", action="store_true",
                    help="also each --v2 checkpoint with zero motion")
    ap.add_argument("--candidates", nargs="*", default=[], metavar="NAME=PARQUET",
                    help="a detector's rows from a dense parquet ({window} replaced)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="--candidates: leave frames the parquet did not process "
                         "unmeasured instead of refusing")
    ap.add_argument("--windows", nargs="+")
    ap.add_argument("--no-coco", action="store_true")
    ap.add_argument("--reuse", nargs="+", metavar="SCORES.json",
                    help="measure from saved candidates instead of the video")
    ap.add_argument("--allow-seen", action="store_true",
                    help="allow v2 checkpoints on matches they were trained on")
    ap.add_argument("--save")
    ap.add_argument("--sealed", metavar="OUT",
                    help="write everything to OUT, print no measure")
    ap.add_argument("--frames", nargs="+", choices=FRAME_KINDS, default=FRAME_KINDS,
                    help="measure only these frames (merge files with --reuse)")
    ap.add_argument("--limit-frames", type=int, help="smoke tests: first N frames only")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if not args.windows and not args.reuse:
        ap.error("--windows or --reuse is needed")
    if args.reuse and (args.heatmap or args.v2):
        ap.error("--reuse measures saved candidates; it runs no detector")
    if args.sealed and args.save:
        ap.error("--sealed writes to its own file; --save would not be sealed")
    log = print
    t0 = time.time()

    extra, meta = {}, {}
    for spec in args.candidates:
        name, path = spec.split("=", 1)
        extra[name] = parquet_source(path, args.allow_partial)
        meta[name] = {"source": "parquet", "path": path,
                      "windows": extra[name].sidecars}

    if args.reuse:
        results, saved = reuse(args.reuse, args.windows, extra=extra, log=log)
        meta = saved | meta
        names = list(dict.fromkeys(k for r in results for k, v in r.items()
                                   if isinstance(v, dict) and "candidates" in v))
    else:
        import torch

        torch.set_num_threads(args.threads)
        detectors, motion = {}, {}
        if not args.no_coco:
            from ultralytics import YOLO

            yolo = YOLO("yolov8s.pt")
            detectors["coco"] = lambda im, pitch, cal: coco_candidates(yolo, im, pitch)
            meta["coco"] = {"kind": "coco", "path": "yolov8s.pt"}
        for path in args.heatmap:
            m = build()
            m.load_state_dict(torch.load(path))
            m.eval()
            detectors[Path(path).name] = (lambda mm: lambda im, pitch, cal: detect(
                mm, im, pitch, tau=0.01, top_k=8, cal=cal))(m)
            meta[Path(path).name] = {"kind": "v1", "path": str(path)}
        if args.v2:
            matches = [json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
                       ["match"] for w in args.windows]
            motion, v2_meta = v2_detectors(args.v2, matches, args.allow_seen,
                                           args.motion_zero, log)
            meta.update(v2_meta)
        results = [run_window(w, detectors, motion=motion, limit=args.limit_frames,
                              extra=extra, log=log, kinds=args.frames)
                   for w in args.windows]
        names = [*detectors, *motion, *extra]
    kinds = {n: kind_of(n, results, meta) for n in names}
    for n in names:
        meta.setdefault(n, {})["kind"] = kinds[n]
    summary = summarise(results, names, kinds)
    protocol = {"FF_W": FF_W, "FF_DT": FF_DT, "FF_R": FF_R, "FF_S": FF_S,
                "PRE_LEADS": PRE_LEADS, "PRE_GAP": PRE_GAP, "PRE_T": PRE_T,
                "POST_LAG": POST_LAG, "POST_T": POST_T, "BLOCK_FRAMES": BLOCK_FRAMES,
                "DRAWS": DRAWS, "V2_TAU": V2_TAU, "TOP_K": TOP_K}
    blob = {"summary": summary, "detectors": meta, "protocol": protocol,
            "results": results}
    if args.sealed:
        Path(args.sealed).write_text(json.dumps(blob, indent=1))
        for r in results:
            log(sealed_counts(r, names))
        log(f"  sealed: {len(results)} windows, {len(names)} detectors -> {args.sealed} "
            f"[{time.time() - t0:.0f} s]")
        return
    print(json.dumps(summary, indent=1))
    if args.save:
        Path(args.save).write_text(json.dumps(blob, indent=1))


# ---- Self-check -------------------------------------------------------------------

def _check():
    """A synthetic window seen by a distortion-free camera at 10 px a metre
    (pixel (u, v) is pitch (u / 10 - 52.5, v / 10 - 34)): the pass and
    placebo frame rules on an events table, the FF-auto and placebo
    geometry at its edges, `measure` with a detector on the pass line and
    one on the grass ahead of the kick, the block bootstrap, and the
    parquet reader."""
    import tempfile

    K = np.array([[1e6, 0, 525.0], [0, 1e6, 340.0], [0, 0, 1]])
    cal = {"K": K, "D": np.zeros((4, 1)), "Knew": K, "Hinv": np.diag([0.1, 0.1, 1.0])}

    def px(x, y, size=4.0, p=0.9):
        return [(x + 52.5) * 10, (y + 34) * 10 - size / 2, p, size]

    ev = [(0, "DRIVE", "left", 1, -20, 0), (50, "PASS", "left", 1, -20, 0),      # pass 0
          (100, "DRIVE", "left", 2, 0, 0), (200, "PASS", "left", 2, 0, 0),
          (210, "PASS", "left", 3, 0, 10),                                     # too short
          (300, "HIGH PASS", "right", 9, 5, 10),
          (1000, "PASS", "right", 9, 0, 25), (1040, "PASS", "left", 6, 0, 20),  # pass 1
          (1100, "DRIVE", "left", 7, 20, 20),
          (1600, "DRIVE", "left", 4, 10, -10), (1640, "PASS", "left", 4, 10, -10),
          (1700, "DRIVE", "left", 5, 30, -10)]                                 # pass 2
    lab = pd.DataFrame(ev, columns=["frame", "label", "side", "player_id", "x", "y"])
    sets = frame_sets(lab.sample(frac=1, random_state=1), 2000, "syn")
    P = sets["passes"]
    assert [(p["kick"], p["reception"], p["i"]) for p in P] == [
        (50, 100, 0), (1040, 1100, 1), (1640, 1700, 2)]
    assert P[0]["frames"] == [62, 71, 79, 88] and P[2]["block"] == "syn:1"
    # Before the kick: pass 0 follows the passer's own DRIVE, pass 2 too;
    # pass 1 follows the opponents' pass (not given to him).
    assert [(e["frame"], e["pass"]) for e in sets["placebo_pre"]] == [
        (35, 0), (20, 0), (1625, 2), (1610, 2)]
    # After reception: 115 is before the next event (200) and 1715 inside
    # the window; pass 1's 1115 is too.
    assert [(e["frame"], e["pass"]) for e in sets["placebo_post"]] == [
        (115, 0), (1115, 1), (1715, 2)]
    assert set(measured_frames(sets)) >= {35, 62, 115, 1715}
    # Geometry at the edges: 4 m, |t - s| 0.30, 1 m from people, t in [0, 1].
    A, B, far = (-20, 0), (0, 0), np.array([[50.0, 30.0]])
    p0 = P[0]
    assert ff_hit(np.array([-15.0, 3.9]), p0, 0, far)
    assert not ff_hit(np.array([-15.0, 4.1]), p0, 0, far)
    assert ff_hit(np.array([-20 + 20 * 0.54, 0.0]), p0, 0, far)          # t - s = 0.29
    assert not ff_hit(np.array([-20 + 20 * 0.56, 0.0]), p0, 0, far)      # 0.31
    assert not ff_hit(np.array([-15.0, 0.0]), p0, 0, np.array([[-15.0, 0.9]]))
    assert not ff_hit(np.array([-21.0, 0.0]), p0, 0, far)                # t < 0
    assert not ff_hit(np.array([-15.0, 0.0]), p0, 0, None)               # no people
    assert corridor_hit(np.array([-4.0, 1.0]), A, B, *PRE_T, far)        # t 0.8
    assert not corridor_hit(np.array([-4.0, 1.0]), A, B, *POST_T, far)
    # Two detectors: "line" puts the ball on the pass line at s (and at the
    # passer before the kick, at the receiver after); "ahead" has nothing
    # in flight and the grass halfway along the coming pass before it.
    people = {f: np.array([[-30.0, -30.0], [40.0, 30.0]]) for f in range(2000)}
    line, ahead = {}, {}
    for p in P:
        a, b = np.array(p["from"]), np.array(p["to"])
        for q, f in enumerate(p["frames"]):
            line[f] = np.array([px(*(a + p["s"][q] * (b - a)))])
            ahead[f] = np.zeros((0, 4))
    for e in sets["placebo_pre"]:
        p = P[e["pass"]]
        line[e["frame"]] = np.array([px(*p["from"])])
        mid = (np.array(p["from"]) + p["to"]) / 2
        ahead[e["frame"]] = np.array([px(*mid), px(0, 0, p=0.1)])
    for e in sets["placebo_post"][:2]:
        line[e["frame"]] = np.array([px(*P[e["pass"]]["to"])])
        ahead[e["frame"]] = np.array([px(*P[e["pass"]]["to"])])
    for f in sets["random"] + list(lab.frame):
        line.setdefault(f, np.zeros((0, 4)))
        ahead.setdefault(f, np.zeros((0, 4)))
    track = pd.DataFrame({"x": 0.0, "y": 0.0}, index=range(2000))
    wd = {"window": "syn", "cal": cal, "sets": sets, "people": people, "track": track,
          "info": {"match": "000000", "half": 1, "start_frame": 0,
                   "pitch": [[0, 0], [1050, 0], [1050, 680], [0, 680]]}}
    r = measure(wd, {"line": line, "ahead": ahead}, n_bg={62: 15, 71: 9})
    assert r["line"]["ff per pass"] == [1.0, 1.0, 1.0] and r["line"]["ff-auto"] == 1.0
    assert r["line"]["free-flight hit"] == 1.0                 # on the line: 1 m rule too
    assert r["ahead"]["ff per pass"] == [0.0, 0.0, 0.0]
    assert r["line"]["placebo pre"] == {"rate": 0.0, "hits": 0, "frames": 4,
                                        "not measured": 0}
    assert r["ahead"]["placebo pre"] == {"rate": 1.0, "hits": 4, "frames": 4,
                                         "not measured": 0}
    assert r["ahead"]["placebo post hits"] == [0, 0, None]     # 1715 not measured
    assert r["ahead"]["placebo post"]["not measured"] == 1
    assert r["passes"] == 3 and r["frames"]["passes"][1]["from"] == [0.0, 20.0]
    assert r["n_bg"] == {"62": 15, "71": 9}
    json.dumps(r)
    # The block bootstrap: equal detectors give 0 throughout; every pass
    # better gives a lower bound above 0; full_bg keeps pass 0's frame 62.
    c = compare_ff([r], "line", "ahead")
    assert c["difference"] == 1.0 and c["2.5%"] == 1.0 and c["blocks"] == 2
    assert c["lower bound above 0"] and c["matches"]["000000"]["passes"] == 3
    z = compare_ff([r], "line", "line")
    assert z["difference"] == 0 and z["2.5%"] == z["97.5%"] == 0
    assert compare_ff([r], "line", "ahead", full_bg=True)["passes"] == 1
    s = summarise([r], ["line", "ahead"], {})
    assert s["ff-auto"]["line"]["pooled"] == 1.0 and "paired" not in s
    assert s["placebo"]["ahead"]["pre"] == 1.0
    assert r["line"]["ff incomplete passes"] == 0 and c["passes not measured"] == 0
    # A frame not measured is left out, never a miss: "ahead" without pass
    # 0's 2nd frame and a placebo frame. Pass 0 is incomplete (not scored,
    # not paired), the placebo frame None; the legacy measures as before.
    gap = {f: v for f, v in ahead.items() if f not in (P[0]["frames"][1], 35)}
    logs, n = [], len(measured_frames(sets))
    log_coverage("syn", {"line": line, "ahead": ahead, "gap": gap}, measured_frames(sets),
                 logs.append)                                  # all lack 1715
    assert len(logs) == 1 and f"line {n - 1}, ahead {n - 1}, gap {n - 3}" in logs[0], logs
    log_coverage("syn", {"line": line}, [62, 71], logs.append)
    assert len(logs) == 1
    r2 = measure(wd, {"line": line, "gap": gap})
    g = r2["gap"]
    assert g["ff frames"][0] == [0, None, 0, 0] and g["ff per pass"] == [None, 0.0, 0.0]
    assert g["ff incomplete passes"] == 1 and g["ff-auto"] == 0.0
    assert g["frames measured"] == r2["frames to measure"] - 3 == n - 3
    assert g["placebo pre hits"] == [None, 1, 1, 1] and g["placebo pre"]["frames"] == 3
    assert g["placebo pre"]["not measured"] == 1
    sp = summarise([r2], ["line", "gap"], {})["placebo"]["gap"]
    assert sp["pre frames"] == 3 and sp["pre not measured"] == 1
    # The sealed log: counts only, with each detector's gaps.
    line_ = sealed_counts(r2, ["line", "gap", "none"])
    assert line_.endswith(
        f"frames measured of {n}: line {n - 1} (1 placebo post frames not measured), "
        f"gap {n - 3} (1 passes incomplete, 1 placebo pre frames not measured, "
        f"1 placebo post frames not measured)"), line_
    assert "." not in line_                                    # no rate, no mean
    assert g["free-flight hit"] == r["ahead"]["free-flight hit"]
    cg = compare_ff([r2], "line", "gap")
    assert cg["passes"] == 2 and cg["passes not measured"] == 1 and cg["difference"] == 1.0
    fs = ff_summary([r2], "gap")
    assert fs["passes"] == 2 and fs["incomplete passes"] == 1
    # Blocks that move together widen the interval against single passes.
    rng = np.random.default_rng(1)
    eff = np.repeat(rng.normal(0, 0.4, 20), 10)
    fake = {"window": "w", "match": "m", "frames": {"passes": [
        {"frames": [0, 1, 2, 3], "block": f"w:{k // 10}"} for k in range(200)]},
        "a": {"ff frames": [[float(e > 0)] * 4 for e in eff]},
        "b": {"ff frames": [[0.0] * 4 for _ in eff]}}
    wb = compare_ff([fake], "a", "b")
    wp = compare_ff([fake], "a", "b", blocks=False)
    assert wb["97.5%"] - wb["2.5%"] > 1.5 * (wp["97.5%"] - wp["2.5%"])
    # A dense parquet: best first, at most top_k, p_raw from the logit, an
    # empty processed frame; frames it did not process (its sidecar's list)
    # refused, or left unmeasured with allow_partial; no sidecar refused.
    from detect_ball_heatmap import sidecar_path, write_sidecar

    with tempfile.TemporaryDirectory() as tmp:
        pq = Path(tmp) / "syn_w.parquet"
        pd.DataFrame({"frame": [4, 4, 4, 6], "px": [1.0, 2, 3, 4], "py": [5.0, 6, 7, 8],
                      "confidence": [0.3, 0.9, 0.5, 0.2], "crop_h": [4.0, 4, 4, 4],
                      "logit": [-1.0, 2.0, 0.0, -2.0], "cls": "ball"}).to_parquet(pq)
        write_sidecar(pq, [4, 5, 6], window="syn_w", stride=2)
        src = parquet_source(str(Path(tmp) / "{window}.parquet"))
        got = src("syn_w", [4, 5, 6])
        assert got[4][:, 0].tolist() == [2.0, 3.0, 1.0] and got[4].shape[1] == 5
        assert abs(got[4][0, 2] - 1 / (1 + np.exp(-2))) < 1e-12 and got[5].shape == (0, 5)
        side = src.sidecars["syn_w"]
        assert side["processed"] == 3 and "frames" not in side
        assert len(parquet_candidates(pq, [4], top_k=2)[4]) == 2
        write_sidecar(pq, [4, 6], window="syn_w")
        part = parquet_source(str(pq), allow_partial=True)("syn_w", [4, 5, 6])
        assert sorted(part) == [4, 6]
        sidecar_path(pq).unlink()
        for bad in (lambda: parquet_source(str(pq))("syn_w", [4, 5, 6]),
                    lambda: parquet_candidates(pq, [4], allow_partial=True)):
            try:
                bad()
                raise AssertionError("an unprocessed frame or no sidecar passed")
            except ValueError:
                pass
        _check_v2_detectors(Path(tmp))
    assert _saved(np.array([[1.23456, 2.0, 0.0496123, 3.0, 1.2345678]])) == [
        [1.235, 2.0, 0.049612, 3.0, 1.234568]]
    print("  eval_ball_detector: pass and placebo frames, FF-auto / placebo edges, "
          "measure, unmeasured frames left out and counted, sealed log, block "
          "bootstrap, parquet reader and its coverage, v2 loading (scales, RGB-only): ok")


def _check_v2_detectors(tmp: Path):
    """`v2_detectors` on two small checkpoints: the motion one with its
    provenance's scales and a motion-zero twin, the RGB-only one once, with
    zero motion; a training match refused."""
    import torch

    from src import ball_heatmap as bh

    torch.manual_seed(0)
    net = bh.build(motion=True)
    with torch.no_grad():
        net.conv1_motion.weight.normal_(0, 0.05)
        net.head[-1].bias.fill_(0.0)
    rgb = bh.build()
    rgb.load_state_dict({k: v for k, v in net.state_dict().items()
                         if k != "conv1_motion.weight"})
    mk, rk = tmp / "m.pt", tmp / "rgb.pt"
    torch.save(net.state_dict(), mk)
    torch.save(rgb.state_dict(), rk)
    sigma = {"SIGMA_D": 0.07, "SIGMA_F": 0.03}
    bh.write_provenance(mk, {"train": {"118575": [1]}, "motion": True, "sigma": sigma})
    bh.write_provenance(rk, {"train": {"118575": [1]}, "motion": False, "sigma": sigma})
    motion, meta = v2_detectors([mk, rk], ["118578"], motion_zero=True, log=lambda *a: None)
    assert list(motion) == ["m.pt", "m.pt:motion-zero", "rgb.pt"]
    assert meta["m.pt"]["sigma"] == sigma and not meta["m.pt"]["rgb_only"]
    assert meta["rgb.pt"]["rgb_only"] and meta["rgb.pt"]["kind"] == "v2"
    assert meta["m.pt:motion-zero"]["kind"] == "v2-motion-zero"
    rng = np.random.default_rng(0)
    fr = [rng.integers(0, 255, (64, 96, 3), dtype=np.uint8) for _ in range(3)]
    args = (fr[0], fr[1], fr[2], cv2.cvtColor(fr[1], cv2.COLOR_BGR2GRAY),
            [[8, 8], [88, 8], [88, 56], [8, 56]])
    net.eval()
    want = bh.detect_v2(net, *args, tau=V2_TAU, top_k=TOP_K, sigma_d=0.07, sigma_f=0.03)
    assert len(want) and np.array_equal(motion["m.pt"](*args, None), want, equal_nan=True)
    zero = motion["m.pt:motion-zero"](*args, None)
    assert np.array_equal(motion["rgb.pt"](*args, None), zero, equal_nan=True)
    assert not np.array_equal(zero, want)
    try:
        v2_detectors([mk], ["118575"])
        raise AssertionError("a training match was accepted")
    except ValueError:
        pass


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _check()
    else:
        main()
