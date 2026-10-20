"""The shot gate of the goals protocol: is the ball near the goals seen at all?

Implements the SHOT GATE section of the registered protocol
(research/session_scratchpad/shots_goals/protocol.json, checked against its
sha256 before anything is read): on every BAS SHOT and GOAL of 128057 half 1
(10) and 132831 half 2 (16), 26 attempts, footage v3 never trained on, is the
ball "reachable"?

Method, per attempt (BAS label frame L, 0-indexed half frame, 25 fps):
  1. Clip: fetch_soccertrack_v2.fetch cuts a clip of the half around L into
     GATE_DATA/soccertrack_v2/st2_<match>_<1st|2nd>_f<first>.* (a data
     directory of its own, gitignored under /data, so no clip can overwrite
     a processed window of the same name). v3's motion input needs the
     background B_t of every detected frame t: the median of the frames that
     are multiples of 50 within 350 of t's centre (src/motion_bg.py), taken
     inside the video it is given. The clip starts at bg_centre(L - 25) - 350
     (0 at the half's start) and is 801 frames long (2 x 350 + 2 x 50 + 1),
     so every detected frame has the same 15-sample background a 3-min
     window gives it (fewer only at a half's edges), and t - 2 / t + 2 are
     inside. The fixed length makes the clip's name (its first frame) name
     one clip; two attempts with the same first frame share it.
  2. Detection: detect_ball_heatmap.py --v2 with .cache/ball_v3b.1500.pt at
     stride 1 on local frames [L - 25, L + 55] (-1 s to +2.2 s, both ends
     included: --range A B excludes B) into gate_v3/<match>_<half>_<L>.parquet
     (+ .meta.json sidecar, its processed frames). Resumable: an attempt is
     done (skipped) only when its sidecar matches its parquet, names the
     planned clip, start frame and range at stride 1, records the sha256 of
     the weights file as it is now, and lists every requested frame inside
     the half (only frames past the half's end, or before its frame 0, may
     be absent). A clip is
     used only with CLIP_FRAMES frames, fewer only when it ends at the
     half's end (the half video's frame count, read from its header). After
     every attempt -- done, skipped or failed -- its clip's mp4 is deleted
     unless an attempt still to come in this pass, and not done, needs it;
     the clips of attempts done before the pass are deleted at its end
     (disk; --keep-clips keeps them). M7 / M9 are sealed (src/unseal.py):
     VEO_UNSEAL must be set, as for every opening of them, and no
     --allow-seen (v3 never saw them).
  3. Candidates: rows with confidence (v3's calibrated p_cal) >= 0.10; the
     painted-spot marks of that match dropped (src/ball_spots.drop_marks,
     4 px), the marks being those found on the match half's processed window
     (spot_marks in .cache/v3m_balls_<window>.parquet.meta.json: M7
     st2_128057_1st_f030000, M9 st2_132831_2nd_f030000; same fixed view);
     feet = the bottom of the candidate box (px, py + crop_h / 2, a missing
     size taken as 0, as src/ball_spots.py), to pitch metres (centre origin)
     with the release calibration (st.calibration; not M1, so no override);
     points that do not project (above the horizon) are dropped.
  4. Reachable (as registered): >= 2 frames with a candidate within 4 m of
     the GSR shooter in [-1 s, +0.5 s] AND >= 2 frames with a candidate
     within 4 m of the segment from the shooter to the centre of the goal his
     team attacks in [0 s, +2 s]. The attacked goal: opposite to the end the
     shooter's team defends, from each GSR side's mean x over the half.
  5. Decision over the 26: GO >= 13; GO, capped 9-12 (expected recall capped
     at the reachable share in every report); STOP < 9 ("shots not
     available: ball reachable k/26", Wilson 95%). The decision is written
     only when all 26 attempts are scored with no frame missing inside the
     half AND the 3 reproducibility attempts (below) are all scored and all
     not reachable; otherwise it is "none: incomplete" or "withheld:
     re-check the gate", with no would-be decision.

Reproducibility (protocol: "the 3 attempts already measured read-only (M7
f1370 and f3829 of 1st f030000 window time, M9 f2296) must come out not
reachable"): the same scoring on the cached dense v3 rows of the processed
windows (.cache/v3_balls_<window>.parquet, stride 1 over the whole window,
the same checkpoint, calibration, tau and top-k as the clip runs; marks
dropped as above), window frames + the window's start_frame
(data/soccertrack_v2/<window>.json) = half frames. Nothing is detected.

Choices where the protocol is silent (simplest reading):
  - Time windows in frames from L, both ends included: [-1 s, +0.5 s] =
    offsets -25..+12 (12.5 is not a frame), [0 s, +2 s] = 0..50, the
    detected span [-1 s, +2.2 s] = -25..+55.
  - "The GSR shooter": the GSR record with the BAS action's player_id on
    each frame (the person, where he is on that frame), in both conditions:
    the segment of a frame runs from his position on that frame. A frame
    where he has no record cannot count. (The fixed reading -- his position
    on the label frame for every frame -- is computed beside it as
    "sensitivity: label-frame shooter"; it is not the result.) Duplicate
    (frame, player_id) records: the first is kept (as xg_gt.py).
  - "The shooter's team": the GSR side of his records over the half (their
    most common value). "Its mean x over the half": all GSR rows of each
    side over the half (players and goalkeepers; GSR gives no side to
    anyone else), and the two sides' means are compared -- the lower mean x
    defends the left end (x = -52.5) and attacks the goal at (+52.5, 0), the
    other the reverse -- the reading xg_gt.py and formation.py use (the sign
    of each mean alone can put both sides on one end). The BAS side is
    reported beside it.
  - An attempt with no player_id, or a shooter with no GSR record in the
    half, has no shooter and is not reachable (counted, reason given). Only
    frames past the half's end (its video's frame count) may be missing
    from an attempt's detection; it is scored on the frames it has and the
    missing ones are listed. A frame missing inside the half leaves the
    attempt not done, and the score incomplete (no decision).
  - Two BAS actions on one frame share a parquet; each is its own attempt.
  - Test hygiene (protocol GOALS: all frozen "before any test half is
    scanned"; protocol_notes.json: the goals rule is frozen before the gate
    reads anything of 132831 half 2): 132831 half 2 is a GOALS test half, so
    its attempts (BAS frames, GSR, video, detections, its window's json and
    M9's reproducibility attempt) are refused until frozen_goals.json and
    frozen_goals.json.sha256 both exist here and agree, and the json records
    the registered protocol's sha256 ("protocol sha256") and the frozen
    "T", "N" and "team rule". There is no override.
  - Wilson interval at 95% (z = 1.96) on k/26, reported whatever the
    decision.

    PYTHONPATH=/home/user/Veo python research/session_scratchpad/shots_goals/ball_gate.py --check
    VEO_UNSEAL=... python .../ball_gate.py repro [--matches 128057]  -> ball_gate_repro.json
    VEO_UNSEAL=... python .../ball_gate.py detect [--dry-run] [--threads 2]
    VEO_UNSEAL=... python .../ball_gate.py score                      -> ball_gate.json
    VEO_UNSEAL=... sh research/session_scratchpad/shots_goals/gate_driver.sh   (detect + score)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/user/Veo")
sys.path.insert(0, str(ROOT))
from src import soccertrack_v2 as st  # noqa: E402
from src.motion_bg import BG_HALF_SPAN, BG_STEP, bg_centre  # noqa: E402

HERE = ROOT / "research/session_scratchpad/shots_goals"
PROTOCOL = HERE / "protocol.json"
PROTOCOL_SHA256 = "1d4a3188f76102baff651128960461b7bcc2ff2efe6d495e2d49223d0751debd"
FROZEN_GOALS = HERE / "frozen_goals.json"
FROZEN_KEYS = ("T", "N", "team rule")         # what frozen_goals.json must record
OUT = HERE / "ball_gate.json"
REPRO_OUT = HERE / "ball_gate_repro.json"
DET_DIR = HERE / "gate_v3"
GATE_DATA = ROOT / "data/shot_gate"            # VEO_DATA_DIR of the clips
CLIPS = GATE_DATA / "soccertrack_v2"           # the clips' mp4 + json
WEIGHTS = ".cache/ball_v3b.1500.pt"
FPS = st.FPS
LABELS = ("SHOT", "GOAL")
HALF_NAMES = {1: "1st", 2: "2nd"}

# The registered numbers.
GATE_HALVES = {("128057", 1): 10, ("132831", 2): 16}     # attempts per half
N_ATTEMPTS = 26
SPAN_S = (-1.0, 2.2)          # detected
NEAR_S = (-1.0, 0.5)          # candidate near the shooter
PATH_S = (0.0, 2.0)           # candidate near the shot segment
CONF = 0.10
STATIC_SHARE = 0.25   # re-check (protocol_notes amendment): a point static over the attempt
STATIC_PX = 4.0
RADIUS_M = 4.0
MIN_FRAMES = 2
GO, CAPPED = 13, 9
REPRO = [("st2_128057_1st_f030000", 1370), ("st2_128057_1st_f030000", 3829),
         ("st2_132831_2nd_f030000", 2296)]
MARK_WINDOWS = {("128057", 1): "st2_128057_1st_f030000",
                ("132831", 2): "st2_132831_2nd_f030000"}
CLIP_FRAMES = 2 * BG_HALF_SPAN + 2 * BG_STEP + 1
GOAL_X = 52.5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol() -> dict:
    """The registered protocol, refused if it no longer matches its hash."""
    want = (HERE / "protocol.json.sha256").read_text().split()[0]
    got = sha256(PROTOCOL)
    if got != want or got != PROTOCOL_SHA256:
        raise SystemExit(f"protocol.json sha256 {got} != registered {want} / {PROTOCOL_SHA256}")
    return json.loads(PROTOCOL.read_text())


def offsets(window_s) -> tuple[int, int]:
    """Frame offsets from L of a [lo s, hi s] window, both ends included."""
    lo, hi = window_s
    return int(np.ceil(lo * FPS - 1e-9)), int(np.floor(hi * FPS + 1e-9))


# ---- Hygiene -----------------------------------------------------------------

def frozen_problem(path: Path = FROZEN_GOALS) -> str | None:
    """Why the goals rule does not count as frozen; None when it does:
    `path` (frozen_goals.json) and `path`.sha256 both exist and agree, and
    the json records the registered protocol's sha256 and the frozen T, N
    and team rule. Read once, so the hash and the content are one file."""
    side = Path(f"{path}.sha256")
    if not path.is_file():
        return f"no {path.name}"
    if not side.is_file():
        return f"no {side.name}"
    try:
        blob = path.read_bytes()
        want = side.read_text().split()
    except OSError as e:
        return f"unreadable: {e}"
    if not want or want[0] != hashlib.sha256(blob).hexdigest():
        return f"{path.name} does not match {side.name}"
    try:
        rule = json.loads(blob)
    except ValueError:
        return f"{path.name} is not JSON"
    if not isinstance(rule, dict):
        return f"{path.name} is not a JSON object"
    if rule.get("protocol sha256") != PROTOCOL_SHA256:
        return f"{path.name} does not record \"protocol sha256\": {PROTOCOL_SHA256}"
    absent = [k for k in FROZEN_KEYS if rule.get(k) is None]
    if absent:
        return f"{path.name} does not record {absent}"
    return None


def goals_frozen() -> bool:
    return frozen_problem() is None


def allowed(match: str, half: int, test_halves: set, what: str) -> bool:
    """Whether the gate may read this match half: a GOALS test half only once
    the goals rule is frozen (no override), and M7 / M9 only through
    src/unseal.py (the one way to open them; logged there)."""
    from src.unseal import unsealed

    if f"{match}_{HALF_NAMES[half]}" in test_halves:
        why = frozen_problem()
        if why is not None:
            print(f"  {match} half {half}: a GOALS test half, refused until the goals rule "
                  f"is frozen ({why})", flush=True)
            return False
    if not unsealed(match, what):
        print(f"  {match} half {half}: sealed (set VEO_UNSEAL, src/unseal.py)", flush=True)
        return False
    return True


# ---- Attempts, shooter, attacked goal ------------------------------------------

def attempts_of(match: str, half: int) -> pd.DataFrame:
    ev = st.events(match)
    a = ev[(ev.half == half) & ev.label.isin(LABELS)].sort_values("frame", kind="stable")
    return a.reset_index(drop=True)


def half_gsr(match: str, half: int) -> pd.DataFrame:
    """The half's GSR, one record per (frame, player_id) where keyed."""
    g = st.gsr_table(match, half)
    keyed = g[g.player_id.notna()].drop_duplicates(["frame", "player_id"])
    return pd.concat([keyed, g[g.player_id.isna()]], ignore_index=True)


def defended_ends(gsr: pd.DataFrame) -> tuple[dict, dict]:
    """Each GSR side's mean x over the half and the end it defends: the
    lower mean x defends the left end (x < 0)."""
    mean_x = gsr[gsr.side.notna()].groupby("side").x.mean().to_dict()
    if len(mean_x) != 2:
        raise ValueError(f"expected two GSR sides, got {sorted(mean_x)}")
    low = min(mean_x, key=mean_x.get)
    return mean_x, {s: ("left" if s == low else "right") for s in mean_x}


def shooter_of(gsr: pd.DataFrame, player_id, ends: dict) -> dict | None:
    """The shooter's GSR track (frame -> x, y), side and attacked goal
    centre; None without a player_id or a GSR record."""
    if player_id is None or (isinstance(player_id, float) and np.isnan(player_id)):
        return None
    rec = gsr[gsr.player_id == str(player_id)]
    rec = rec[rec.x.notna() & rec.y.notna()]
    if rec.empty:
        return None
    side = rec.side.mode().iloc[0] if rec.side.notna().any() else None
    if side not in ends:
        return None
    goal = (GOAL_X, 0.0) if ends[side] == "left" else (-GOAL_X, 0.0)
    track = rec.set_index("frame")[["x", "y"]].astype(float)
    return {"track": track, "side": side, "goal": goal}


# ---- Candidates --------------------------------------------------------------

def mark_points(match: str, half: int) -> list:
    """The painted-spot marks found on the match half's processed window."""
    w = MARK_WINDOWS[(match, half)]
    meta = json.loads((ROOT / ".cache" / f"v3m_balls_{w}.parquet.meta.json").read_text())
    return [tuple(m) for m in meta["spot_marks"]["marks"]]


def drop_static(c: pd.DataFrame, n_frames: int) -> pd.DataFrame:
    """Re-check reading: drop candidates within STATIC_PX of a point that holds
    a candidate on >= STATIC_SHARE of the attempt's detected frames (a ball at
    rest, fixed clutter)."""
    if c.empty or n_frames == 0:
        return c
    xy = c[["px", "py"]].to_numpy(float)
    fr = c.frame.to_numpy()
    keep = np.ones(len(c), bool)
    for i in range(len(c)):
        near = (np.abs(xy[:, 0] - xy[i, 0]) <= STATIC_PX) & (np.abs(xy[:, 1] - xy[i, 1]) <= STATIC_PX)
        if len(np.unique(fr[near])) >= STATIC_SHARE * n_frames:
            keep[i] = False
    return c[keep]


def candidates(rows: pd.DataFrame, marks, cal, static_frames: int | None = None) -> pd.DataFrame:
    """Rows (frame = half frame, px, py, crop_h, confidence) to candidate
    feet in metres: confidence >= CONF, the marks dropped (and, for the
    re-check, static points over `static_frames` detected frames), the bottom
    of the box projected; those that do not project dropped."""
    from src.ball_spots import drop_marks

    c = rows[rows.confidence >= CONF]
    c = drop_marks(c, marks)
    if static_frames is not None:
        c = drop_static(c, static_frames)
    if c.empty:
        return pd.DataFrame({"frame": [], "x": [], "y": []})
    size = np.nan_to_num(c.crop_h.to_numpy(float), nan=0.0)
    xy = st.image_to_pitch(np.c_[c.px.to_numpy(float), c.py.to_numpy(float) + size / 2], cal)
    out = pd.DataFrame({"frame": c.frame.to_numpy(int), "x": xy[:, 0], "y": xy[:, 1]})
    return out[np.isfinite(out.x) & np.isfinite(out.y)].reset_index(drop=True)


# ---- Reachable -----------------------------------------------------------------

def seg_distance(p: np.ndarray, a, b) -> np.ndarray:
    """Distance of points p (N, 2) to the segment a-b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L2 = float(d @ d)
    t = np.zeros(len(p)) if L2 == 0 else np.clip(((p - a) @ d) / L2, 0.0, 1.0)
    return np.hypot(*(p - (a + t[:, None] * d)).T)


def reachable(cand: pd.DataFrame, label: int, shooter: dict | None,
              fixed: bool = False) -> dict:
    """The registered test on one attempt. `cand`: frame (half), x, y.
    `fixed`: the shooter's label-frame position on every frame (the
    sensitivity reading)."""
    if shooter is None:
        return {"reachable": False, "near frames": 0, "path frames": 0,
                "reason": "no shooter"}
    track, goal = shooter["track"], shooter["goal"]
    if fixed and label not in track.index:
        return {"reachable": False, "near frames": 0, "path frames": 0,
                "reason": "no shooter record on the label frame"}
    by = {f: g[["x", "y"]].to_numpy(float) for f, g in cand.groupby("frame")}

    def where(f):
        k = label if fixed else f
        return track.loc[k].to_numpy(float) if k in track.index else None

    near, path, d_near, d_path, no_rec = [], [], [], [], set()
    lo, hi = offsets(NEAR_S)
    for f in range(label + lo, label + hi + 1):
        s = where(f)
        if s is None:
            no_rec.add(f)
            continue
        if f in by:
            d = float(np.hypot(*(by[f] - s).T).min())
            d_near.append(d)
            if d <= RADIUS_M:
                near.append(f - label)
    lo, hi = offsets(PATH_S)
    for f in range(label + lo, label + hi + 1):
        s = where(f)
        if s is None:
            no_rec.add(f)
            continue
        if f in by:
            d = float(seg_distance(by[f], s, goal).min())
            d_path.append(d)
            if d <= RADIUS_M:
                path.append(f - label)
    ok = len(near) >= MIN_FRAMES and len(path) >= MIN_FRAMES
    return {"reachable": bool(ok), "near frames": len(near), "path frames": len(path),
            "near offsets": near, "path offsets": path,
            "min m to shooter": min(d_near) if d_near else None,
            "min m to segment": min(d_path) if d_path else None,
            "frames without a shooter record": len(no_rec)}


def score_one(rows: pd.DataFrame, processed: set, label: int, shooter, marks, cal) -> dict:
    """Both readings of one attempt, with the detected-frame accounting."""
    lo, hi = offsets(SPAN_S)
    want = set(range(label + lo, label + hi + 1))
    cand = candidates(rows[rows.frame.isin(want)], marks, cal)
    out = {"frames detected": len(want & processed),
           "frames missing": sorted(f - label for f in want - processed),
           "candidates >= conf, marks dropped, projected": int(len(cand)),
           **reachable(cand, label, shooter)}
    alt = reachable(cand, label, shooter, fixed=True)
    out["sensitivity: label-frame shooter"] = {k: alt[k] for k in ("reachable", "near frames", "path frames")}
    cs = candidates(rows[rows.frame.isin(want)], marks, cal, static_frames=len(want & processed))
    rc = reachable(cs, label, shooter)
    out["re-check: static points dropped"] = {"candidates": int(len(cs)),
                                              **{k: rc[k] for k in ("reachable", "near frames", "path frames")}}
    if shooter is not None:
        out["shooter side (gsr)"] = shooter["side"]
        out["attacked goal x"] = shooter["goal"][0]
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list:
    c = (k + z * z / 2) / (n + z * z)
    h = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return [float(c - h), float(c + h)]


def decide(k: int, n: int = N_ATTEMPTS) -> dict:
    ci = wilson(k, n)
    if k >= GO:
        d, note = "GO", f">= {GO}/{n} reachable"
    elif k >= CAPPED:
        d, note = "GO, capped", (f"shots may proceed, with expected recall capped at the "
                                 f"reachable share {k}/{n} = {k / n:.3f} in every report")
    else:
        d, note = "STOP", (f"shots not available: ball reachable {k}/{n} "
                           f"(Wilson 95% [{ci[0]:.2f}, {ci[1]:.2f}])")
    return {"decision": d, "note": note, "reachable": k, "of": n, "share": k / n,
            "wilson95": ci}


# ---- Clips and detection -------------------------------------------------------

def clip_plan(match: str, half: int, label: int) -> dict:
    """The clip and the local detection range of an attempt."""
    lo, hi = offsets(SPAN_S)
    first = max(0, bg_centre(label + lo) - BG_HALF_SPAN)
    return {"clip": f"st2_{match}_{HALF_NAMES[half]}_f{first:06d}", "first": first,
            "count": CLIP_FRAMES, "range": [label + lo - first, label + hi + 1 - first]}


def det_path(match: str, half: int, label: int) -> Path:
    return DET_DIR / f"{match}_{HALF_NAMES[half]}_{label:06d}.parquet"


_HALF_FRAMES: dict = {}
_SHA: dict = {}


def half_frames(match: str, half: int) -> int | None:
    """The frame count of a half's panorama, from its video header; None
    when it cannot be read (then no frame counts as past the half's end).
    Asked only of a half the gate may read, and only when a clip or a
    detection stops short."""
    key = (str(match), int(half))
    if key not in _HALF_FRAMES:
        n = None
        try:
            c = st.open_video(*key)
            try:
                n = int(c.streams.video[0].frames) or None
            finally:
                c.close()
        except Exception as e:  # noqa: BLE001 -- unknown length: fail closed
            print(f"  {match} half {half}: video frame count unreadable ({e})", flush=True)
        _HALF_FRAMES[key] = n
    return _HALF_FRAMES[key]


def inside_half(match: str, half: int, frames) -> list[int]:
    """The half frames of `frames` inside the half: from its frame 0 to
    before its end (to no end when the half's length is unknown). Frames
    below 0 (an attempt within 1 s of the half's start) do not exist."""
    frames = sorted(f for f in (int(f) for f in frames) if f >= 0)
    if not frames:
        return []
    n = half_frames(match, half)
    return [f for f in frames if n is None or f < n]


def weights_sha256() -> str:
    """The sha256 of the weights file as it is now (once per run)."""
    if WEIGHTS not in _SHA:
        _SHA[WEIGHTS] = sha256(ROOT / WEIGHTS)
    return _SHA[WEIGHTS]


def det_done(path: Path, t: dict) -> bool:
    """Whether attempt `t`'s detection at `path` is done: the sidecar
    matches its parquet, its window is the planned clip, its start frame
    the planned first frame, its range the planned one at stride 1, its
    checkpoint sha256 that of the weights file now, and it lists every
    requested frame, but those past the half's end."""
    from detect_ball_heatmap import read_sidecar

    try:
        meta = read_sidecar(path)
    except (ValueError, OSError):        # no sidecar, a stale one, or no parquet
        return False
    a, b = t["range"]
    if (meta.get("window") != t["clip"] or meta.get("start_frame") != t["first"]
            or meta.get("range") != [a, b] or meta.get("stride") != 1
            or meta.get("ckpt sha256") != weights_sha256()):
        return False
    try:
        got = {int(f) for f in meta.get("frames", [])}
    except (TypeError, ValueError):
        return False
    absent = set(range(a, b)) - got
    return not inside_half(t["match"], t["half"], {t["first"] + f for f in absent})


def clip_ready(t: dict, root: Path | None = None) -> bool:
    """Whether attempt `t`'s planned clip is on disk: its json and mp4, the
    match, half and first frame planned, and CLIP_FRAMES frames -- fewer
    only when the clip runs to the half's end."""
    root = CLIPS if root is None else root
    j = root / f"{t['clip']}.json"
    if not j.exists() or not j.with_suffix(".mp4").exists():
        return False
    try:
        info = json.loads(j.read_text())
        first, n = int(info["start_frame"]), int(info["frames"])
        same = str(info["match"]) == t["match"] and int(info["half"]) == t["half"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not same or first != t["first"] or not 1 <= n <= CLIP_FRAMES:
        return False
    if n == CLIP_FRAMES:
        return True
    end = half_frames(t["match"], t["half"])
    return end is not None and first + n >= end


def fetch_clip(match: str, half: int, plan: dict):
    from fetch_soccertrack_v2 import fetch

    path = fetch(match, half, plan["first"] / (60 * FPS), plan["count"] / (60 * FPS), CLIPS)
    if path.stem != plan["clip"]:
        raise RuntimeError(f"fetch named the clip {path.stem}, expected {plan['clip']}")
    if not clip_ready(plan):
        raise RuntimeError(f"clip {plan['clip']}: not {CLIP_FRAMES} frames and not at the "
                           f"half's end")


def drop_clip(t: dict):
    (CLIPS / f"{t['clip']}.mp4").unlink(missing_ok=True)


def still_needed(clip: str, attempts) -> bool:
    """Whether an attempt of `attempts` that is not done uses `clip`."""
    return any(u["clip"] == clip and not det_done(det_path(u["match"], u["half"], u["frame"]), u)
               for u in attempts)


def detect_one(plan: dict, out: Path, threads: int):
    env = dict(os.environ, VEO_DATA_DIR=str(GATE_DATA), PYTHONPATH=str(ROOT))
    out.parent.mkdir(parents=True, exist_ok=True)
    a, b = plan["range"]
    subprocess.run([sys.executable, "detect_ball_heatmap.py", "--v2", "--window", plan["clip"],
                    "--weights", WEIGHTS, "--stride", "1", "--threads", str(threads),
                    "--range", str(a), str(b), "--out", str(out)], cwd=ROOT, env=env, check=True)


def gate_attempts(args, proto) -> list[dict]:
    """The 26 attempts of the halves the gate may read now."""
    test = set(proto["GOALS"]["test halves"])
    out = []
    for (match, half), n in GATE_HALVES.items():
        if not allowed(match, half, test, f"shot gate {args.cmd}"):
            continue
        a = attempts_of(match, half)
        if len(a) != n:
            raise SystemExit(f"{match} half {half}: {len(a)} BAS SHOT/GOAL, protocol says {n}")
        for r in a.itertuples():
            out.append({"match": match, "half": half, "frame": int(r.frame), "label": r.label,
                        "bas side": r.side, "player_id": r.player_id,
                        **clip_plan(match, half, int(r.frame))})
    return out


def cmd_detect(args, proto):
    todo = gate_attempts(args, proto)
    pending = [t for t in todo if not det_done(det_path(t["match"], t["half"], t["frame"]), t)]
    print(f"  {len(todo)} attempts readable now, {len(todo) - len(pending)} done, "
          f"{len(pending)} to run", flush=True)
    if args.dry_run:
        for t in todo:
            print(f"    {t['match']} {t['half']} L={t['frame']} {t['label']}: clip {t['clip']} "
                  f"({t['count']} frames), range {t['range']}", flush=True)
        return
    failed = []
    for i, t in enumerate(pending):
        out = det_path(t["match"], t["half"], t["frame"])
        t0 = time.time()
        try:
            if det_done(out, t):         # a shared frame already run
                continue
            if not clip_ready(t):
                fetch_clip(t["match"], t["half"], t)
            detect_one(t, out, args.threads)
            if not det_done(out, t):
                raise RuntimeError("the sidecar is not a complete run of the planned clip, "
                                   "range and checkpoint")
            print(f"  {out.name} done [{time.time() - t0:.0f} s]", flush=True)
        except Exception as e:  # noqa: BLE001 -- go on with the others, fail at the end
            failed.append(out.name)
            print(f"  FAILED {out.name}: {e}", flush=True)
        finally:
            # Decided after every attempt (done, skipped or failed), over the
            # attempts still to come in this pass that are not done.
            if not args.keep_clips and not still_needed(t["clip"], pending[i + 1:]):
                drop_clip(t)
    if not args.keep_clips:              # clips left by attempts done before this pass
        for t in todo:
            drop_clip(t)
    if failed:
        raise SystemExit(f"{len(failed)} attempts failed: {failed}; rerun to resume")
    if len(todo) < N_ATTEMPTS:
        raise SystemExit(f"only {len(todo)} of {N_ATTEMPTS} attempts readable now")
    print("  all attempts detected", flush=True)


# ---- Scoring -------------------------------------------------------------------

def rows_of(path: Path) -> tuple[pd.DataFrame, set]:
    """A detection parquet's rows and processed frames, in half frames."""
    from detect_ball_heatmap import read_sidecar

    meta = read_sidecar(path)
    s = int(meta["start_frame"])
    df = pd.read_parquet(path)
    df["frame"] = df.frame.astype(int) + s
    return df, {int(f) + s for f in meta["frames"]}


def window_half(w: str) -> tuple[str, int]:
    """(match, half) of a window named st2_<match>_<1st|2nd>_f<first>."""
    prefix, match, h, first = w.split("_")
    names = {v: k for k, v in HALF_NAMES.items()}
    if prefix != "st2" or h not in names or not first.startswith("f"):
        raise ValueError(f"not a window name: {w}")
    return match, names[h]


class Half:
    """Per match half: GSR, defended ends, marks, calibration (loaded once)."""

    def __init__(self, match: str, half: int):
        self.gsr = half_gsr(match, half)
        self.mean_x, self.ends = defended_ends(self.gsr)
        self.marks = mark_points(match, half)
        self.cal = st.calibration(match)

    def shooter(self, player_id):
        return shooter_of(self.gsr, player_id, self.ends)


def cmd_repro(args, proto) -> dict:
    test = set(proto["GOALS"]["test halves"])
    halves, rows = {}, []
    for w, local in REPRO:
        # The match half from the window's name: nothing of a refused half
        # (not even its window's json) is read.
        match, half = window_half(w)
        if args.matches and match not in args.matches:
            rows.append({"window": w, "window frame": local, "status": "not run (--matches)"})
            continue
        if (match, half) not in halves:
            if not allowed(match, half, test, "shot gate reproducibility"):
                halves[(match, half)] = None
            else:
                halves[(match, half)] = Half(match, half)
        h = halves[(match, half)]
        if h is None:
            rows.append({"window": w, "window frame": local, "status": "not run (refused)"})
            continue
        info = json.loads((ROOT / "data/soccertrack_v2" / f"{w}.json").read_text())
        if (str(info["match"]), int(info["half"])) != (match, half):
            raise SystemExit(f"{w}.json is {info['match']} half {info['half']}, not its name")
        label = int(info["start_frame"]) + local
        a = attempts_of(match, half)
        hit = a[a.frame == label]
        if hit.empty:
            raise SystemExit(f"{w} frame {local} (half {label}): no BAS SHOT/GOAL there")
        det, processed = rows_of(ROOT / ".cache" / f"v3_balls_{w}.parquet")
        for r in hit.itertuples():
            res = score_one(det, processed, label, h.shooter(r.player_id), h.marks, h.cal)
            rows.append({"window": w, "window frame": local, "match": match, "half": half,
                         "half frame": label, "label": r.label, "bas side": r.side,
                         "player_id": r.player_id, "status": "scored", **res})
    scored = [r for r in rows if r["status"] == "scored"]
    ok = len(scored) == len(REPRO) and not any(r["reachable"] for r in scored)
    ok_static = len(scored) == len(REPRO) and not any(
        r["re-check: static points dropped"]["reachable"] for r in scored)
    out = {"protocol sha256": sha256(PROTOCOL),
           "rule": "the 3 attempts already measured must come out not reachable",
           "source": "cached dense v3 rows (.cache/v3_balls_<window>.parquet), marks dropped",
           "scored": len(scored), "of": len(REPRO),
           "not reachable": sum(not r["reachable"] for r in scored),
           "reproduced": ok if len(scored) == len(REPRO) else None,
           "re-check reproduced (static points dropped)": ok_static if len(scored) == len(REPRO) else None,
           "goals frozen": goals_frozen(),
           "attempts": rows}
    if args.write:
        REPRO_OUT.write_text(json.dumps(out, indent=1, default=_jsonable))
    for r in rows:
        if r["status"] != "scored":
            print(f"  {r['window']} f{r['window frame']}: {r['status']}", flush=True)
            continue
        print(f"  {r['window']} f{r['window frame']} (half {r['half frame']}, {r['label']}): "
              f"{'REACHABLE' if r['reachable'] else 'not reachable'} -- near {r['near frames']}, "
              f"path {r['path frames']} frames; min to shooter {_m(r.get('min m to shooter'))}, "
              f"to segment {_m(r.get('min m to segment'))}; "
              f"label-frame shooter: {r['sensitivity: label-frame shooter']}", flush=True)
    return out


def cmd_score(args, proto):
    repro = cmd_repro(argparse.Namespace(**{**vars(args), "matches": None, "write": False}), proto)
    todo = gate_attempts(args, proto)
    halves, rows, missing, short = {}, [], [], []
    for t in todo:
        path = det_path(t["match"], t["half"], t["frame"])
        if not det_done(path, t):
            missing.append(path.name)
            continue
        key = (t["match"], t["half"])
        if key not in halves:
            halves[key] = Half(*key)
        h = halves[key]
        det, processed = rows_of(path)
        res = score_one(det, processed, t["frame"], h.shooter(t["player_id"]), h.marks, h.cal)
        inside = inside_half(t["match"], t["half"], [t["frame"] + o for o in res["frames missing"]])
        res["frames missing inside the half"] = [f - t["frame"] for f in inside]
        if inside:                       # det_done refuses these; kept as a second check
            short.append(path.name)
        rows.append({k: t[k] for k in ("match", "half", "frame", "label", "bas side", "player_id",
                                       "clip")} | res)
    k = sum(r["reachable"] for r in rows)
    k_static = sum(r["re-check: static points dropped"]["reachable"] for r in rows)
    complete = len(rows) == N_ATTEMPTS and not missing and not short
    # gate_recheck.json: the planning measurement of M9 f2296 was wrong (a ball
    # in flight); the registered reading decides, and the reproducibility
    # condition is the reference (not, not, reachable) under both readings.
    REF = [False, False, True]
    sc = [r for r in repro["attempts"] if r["status"] == "scored"]
    repro_ok = (len(sc) == len(REPRO)
                and [r["reachable"] for r in sc] == REF
                and [r["re-check: static points dropped"]["reachable"] for r in sc] == REF)
    out = {"protocol sha256": sha256(PROTOCOL), "section": "SHOT GATE",
           "readings": "see ball_gate.py docstring", "goals frozen": goals_frozen(),
           "attempts scored": len(rows), "of": N_ATTEMPTS, "missing": missing,
           "frames missing inside the half": short,
           "reachable": k,
           "reachable, second reading (static points dropped; reported)": k_static,
           "second reading's decision (reported)": decide(k_static)["decision"],
           "sensitivity: label-frame shooter reachable": sum(
               r["sensitivity: label-frame shooter"]["reachable"] for r in rows),
           "no shooter": sum(r.get("reason") == "no shooter" for r in rows),
           "per half": {f"{m}_{HALF_NAMES[hf]}": {
               "attempts": sum(1 for r in rows if (r["match"], r["half"]) == (m, hf)),
               "reachable": sum(r["reachable"] for r in rows if (r["match"], r["half"]) == (m, hf)),
               "reachable, second reading": sum(r["re-check: static points dropped"]["reachable"]
                                                for r in rows if (r["match"], r["half"]) == (m, hf)),
               "mean x by gsr side": halves[(m, hf)].mean_x if (m, hf) in halves else None}
               for m, hf in GATE_HALVES},
           "reproducibility": {k2: repro[k2] for k2 in ("scored", "of", "not reachable", "reproduced",
                                                         "re-check reproduced (static points dropped)")},
           "reproducibility attempts": repro["attempts"]}
    # The decision is written only when all 26 attempts are scored with no
    # frame missing inside the half and all 3 reproducibility attempts are
    # scored and not reachable; never a would-be decision.
    if not complete:
        out["decision"] = {"decision": "none: incomplete",
                           "note": f"{len(rows)}/{N_ATTEMPTS} attempts scored, {len(missing)} not "
                                   f"done, {len(short)} with frames missing inside the half; "
                                   f"rerun detect"}
    elif not repro_ok:
        out["decision"] = {"decision": "withheld: re-check the gate",
                           "note": (f"reproducibility scored {repro['scored']}/{len(REPRO)}"
                                    if repro["scored"] != len(REPRO) else
                                    "the reproducibility attempts did not match the reference "
                                    "(not, not, reachable) under both readings (gate_recheck.json)") + " (protocol: re-check before deciding)"}
    else:
        out["decision"] = decide(k)
    out["attempts"] = rows
    OUT.write_text(json.dumps(out, indent=1, default=_jsonable))
    print(f"  reachable {k}/{len(rows)} (label-frame shooter: "
          f"{out['sensitivity: label-frame shooter reachable']}); reproducibility "
          f"{repro['not reachable']}/{repro['scored']} not reachable -> "
          f"{out['decision']['decision']} -> {OUT}", flush=True)


def _m(v):
    return "-" if v is None else f"{v:.1f} m"


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ---- Self-check ----------------------------------------------------------------

def _check():
    assert offsets(NEAR_S) == (-25, 12) and offsets(PATH_S) == (0, 50) and offsets(SPAN_S) == (-25, 55)
    p = clip_plan("1", 1, 31370)
    assert p["first"] == 31000 and p["range"] == [345, 426] and p["count"] == 801
    for L in (25, 330, 399, 31370, 33829, 40012, 40037, 40038):
        p = clip_plan("1", 1, L)
        need = [f for t in range(L - 25, L + 56)
                for f in range(bg_centre(t) - BG_HALF_SPAN, bg_centre(t) + BG_HALF_SPAN + 1, BG_STEP)
                if f >= 0]
        assert p["first"] <= min(need) and max(need) <= p["first"] + p["count"] - 1, L
        assert p["first"] <= max(0, L - 27) and L + 57 <= p["first"] + p["count"] - 1
    assert np.allclose(seg_distance(np.array([[0, 1], [-1, 0], [3, 0]]), (0, 0), (2, 0)), [1, 1, 1])
    # A shot at L = 1000 from (40, 0) towards the goal at (52.5, 0).
    track = pd.DataFrame({"x": 40.0, "y": 0.0}, index=pd.Index(range(900, 1100), name="frame"))
    sh = {"track": track, "side": "left", "goal": (52.5, 0.0)}
    ball = pd.DataFrame({"frame": [990, 991, 1020, 1021], "x": [41, 42, 50, 51], "y": [0, 0, 3, 3]})
    assert reachable(ball, 1000, sh)["reachable"]
    assert not reachable(ball[ball.frame != 991], 1000, sh)["reachable"]          # one near frame
    assert not reachable(ball.assign(y=[0, 0, 5, 5]), 1000, sh)["reachable"]      # off the line
    off = ball.assign(frame=[990, 991, 1051, 1052])                               # after +2 s
    assert not reachable(off, 1000, sh)["reachable"]
    near_only = ball.assign(frame=[975, 1012, 1000, 1050])        # the windows' ends count
    assert reachable(near_only, 1000, sh)["reachable"]
    assert not reachable(ball.assign(frame=[974, 1013, 1000, 1050]), 1000, sh)["reachable"]
    assert not reachable(ball, 1000, None)["reachable"]
    gone = {**sh, "track": track.drop(index=[990, 991])}           # no record: frame cannot count
    assert not reachable(ball, 1000, gone)["reachable"]
    assert reachable(ball, 1000, gone, fixed=True)["reachable"]
    g = pd.DataFrame({"side": ["a", "a", "b", "b"], "x": [-10.0, 5.0, -2.0, -1.0]})
    assert defended_ends(g)[1] == {"a": "left", "b": "right"}
    # Both half means below 0 (the sign reading would put both sides on the
    # left end): the lower mean defends x < 0 and attacks (+52.5, 0).
    g = pd.DataFrame({"frame": [1, 1, 2, 2], "player_id": ["p", "q", "p", "q"],
                      "side": ["right", "left", "right", "left"], "x": [-20.0, -3.0, -18.0, -1.0],
                      "y": [0.0, 0.0, 1.0, 1.0]})
    _, ends = defended_ends(g)
    assert ends == {"right": "left", "left": "right"}
    assert shooter_of(g, "p", ends)["goal"] == (52.5, 0.0)
    assert shooter_of(g, "q", ends)["goal"] == (-52.5, 0.0)
    assert shooter_of(g, None, ends) is None and shooter_of(g, "z", ends) is None
    _check_files()
    assert decide(13)["decision"] == "GO" and decide(12)["decision"] == "GO, capped"
    assert decide(9)["decision"] == "GO, capped" and decide(8)["decision"] == "STOP"
    lo, hi = wilson(8, 26)
    assert 0.16 < lo < 0.17 and 0.49 < hi < 0.50
    print("  ball_gate: windows, clip plan covers the background, segment distance, "
          "reachable rule, defended ends and attacked goal, goals freeze, done detection, "
          "clip ready, decision, Wilson: ok")


def _check_files():
    """frozen_problem, det_done and clip_ready on files in a temporary
    directory (nothing is written here; the half's length is faked)."""
    from detect_ball_heatmap import write_sidecar

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # The goals freeze (a stand-in named rule.json: no frozen_goals.json is written).
        f = d / "rule.json"
        assert frozen_problem(f) is not None                              # absent
        good = {"protocol sha256": PROTOCOL_SHA256, "T": 0.85, "N": 2, "team rule": "kick_spot"}
        for rule, ok in ((good, True), ({**good, "protocol sha256": "0" * 64}, False),
                         ({k: v for k, v in good.items() if k != "N"}, False),
                         ({**good, "team rule": None}, False), ([good], False)):
            f.write_text(json.dumps(rule))
            assert frozen_problem(f) is not None                          # no .sha256 yet
            Path(f"{f}.sha256").write_text(f"{sha256(f)}  {f.name}\n")
            assert (frozen_problem(f) is None) == ok, rule
            f.write_text(json.dumps(rule) + " ")                           # changed after hashing
            assert frozen_problem(f) is not None
            Path(f"{f}.sha256").unlink()
        # A done detection: attempt at L = 1000 of fake match "x" half 1.
        t = {"match": "x", "half": 1, "frame": 1000, **clip_plan("x", 1, 1000)}
        a, b = t["range"]
        out = d / "x.parquet"
        pd.DataFrame({"frame": [a], "confidence": [0.5]}).to_parquet(out)
        meta = {"window": t["clip"], "start_frame": t["first"], "range": [a, b], "stride": 1,
                "ckpt sha256": weights_sha256()}
        _HALF_FRAMES[("x", 1)] = t["first"] + b + 100                   # the half ends later
        write_sidecar(out, range(a, b), **meta)
        assert det_done(out, t)
        write_sidecar(out, range(a, b - 1), **meta)                       # one frame inside missing
        assert not det_done(out, t)
        _HALF_FRAMES[("x", 1)] = t["first"] + b - 1                       # ... past the half's end
        assert det_done(out, t)
        _HALF_FRAMES[("x", 1)] = None                                     # length unknown
        assert not det_done(out, t)
        for k, v in (("window", "other"), ("start_frame", t["first"] + 1), ("range", [a, b + 1]),
                     ("stride", 2), ("ckpt sha256", "0" * 64)):
            write_sidecar(out, range(a, b), **{**meta, k: v})
            assert not det_done(out, t), k
        write_sidecar(out, range(a, b), **meta)
        pd.DataFrame({"frame": [a + 1], "confidence": [0.5]}).to_parquet(out)  # rewritten after
        assert not det_done(out, t)
        # Within 1 s of the half's start: the frames before 0 do not exist.
        t0 = {"match": "x", "half": 1, "frame": 10, **clip_plan("x", 1, 10)}
        assert t0["first"] == 0 and t0["range"][0] < 0
        _HALF_FRAMES[("x", 1)] = 10 ** 5
        write_sidecar(out, range(0, t0["range"][1]), **{**meta, "window": t0["clip"],
                                                        "start_frame": 0, "range": t0["range"]})
        assert det_done(out, t0)
        write_sidecar(out, range(1, t0["range"][1]), **{**meta, "window": t0["clip"],
                                                        "start_frame": 0, "range": t0["range"]})
        assert not det_done(out, t0)
        # A ready clip.
        j = d / f"{t['clip']}.json"
        info = {"match": "x", "half": 1, "start_frame": t["first"], "frames": CLIP_FRAMES}
        j.write_text(json.dumps(info))
        assert not clip_ready(t, d)                                       # no mp4
        j.with_suffix(".mp4").write_bytes(b"")
        assert clip_ready(t, d)
        for k, v in (("start_frame", t["first"] + 1), ("half", 2), ("frames", CLIP_FRAMES + 1)):
            j.write_text(json.dumps({**info, k: v}))
            assert not clip_ready(t, d), k
        j.write_text(json.dumps({**info, "frames": CLIP_FRAMES - 100}))
        _HALF_FRAMES[("x", 1)] = t["first"] + CLIP_FRAMES                 # short, half goes on
        assert not clip_ready(t, d)
        _HALF_FRAMES[("x", 1)] = t["first"] + CLIP_FRAMES - 100           # short at the half's end
        assert clip_ready(t, d)
        _HALF_FRAMES[("x", 1)] = None
        assert not clip_ready(t, d)
        del _HALF_FRAMES[("x", 1)]
    assert window_half("st2_132831_2nd_f030000") == ("132831", 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("repro", "detect", "score"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--matches", nargs="*", help="repro: only these matches")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="detect: list the plan only")
    ap.add_argument("--keep-clips", action="store_true")
    args = ap.parse_args()
    if args.check:
        _check()
        return
    if not args.cmd:
        ap.error("repro, detect or score (or --check)")
    proto = protocol()
    if args.cmd == "repro":
        args.write = True
        cmd_repro(args, proto)
    elif args.cmd == "detect":
        cmd_detect(args, proto)
    else:
        cmd_score(args, proto)


if __name__ == "__main__":
    main()
