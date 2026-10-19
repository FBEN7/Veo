"""Goals protocol (protocol.json, GOALS "pipeline"): the 1-fps pass over a whole
half. yolov8s players on one frame a second, feet to pitch metres, kit
features, and two teams clustered on the half's own detections ->
.cache/formation_scan/<match>_<half>.parquet (frame, x, y, team, conf), the
input of the formation rule (formation.py) on detections.

1. Guard: protocol.json must match its sha256; a test half (protocol "test
   halves") is refused unless frozen_goals.json exists next to protocol.json
   (and matches frozen_goals.json.sha256 when that file exists).
2. Camera: for 117092 (M1) calib_fix/calib_override.py is imported first (the
   refitted model); every other match keeps the release's. The pitch outline
   in pixels is the 105 x 68 boundary projected through that camera
   (crop_check.calibrated_outline).
3. Frames: st.open_video + st.read_frames(step=25): frames 0, 25, 50, ... of
   the half video, decoded in a reader thread while the detector runs.
4. Detector: the pipeline's call (crop_check.detect: yolov8s, persons and
   balls, the frame's own scale) on the crop of crop_check.crop_box(outline)
   if crop_check.json says the crop check passed, else on the whole frame at
   its own width. It runs at conf 0.02 (the lowest value of auto_tune's grid)
   so that the confidence can be chosen afterwards the way the pipeline
   chooses it; greedy NMS keeps exactly the boxes >= c that a run at c keeps.
5. Per person box: feet (bottom centre) -> metres (st.image_to_pitch), near
   grass (_near_pitch at 12 px, as detect_track_hybrid), signed pixel
   distance of the feet to the outline, kit feature (_torso_feature on the
   full frame: cos hue, sin hue, saturation, value).
6. Chunks: every --chunk samples, <name>.parts/chunk_<first>_<last>.parquet
   and then its .json (sample frames and timings), each written to a temp
   file and renamed; a chunk without its json does not count. A restart
   continues after the last sample done; key.json pins the settings and a
   different setting is refused.
7. Assembly, over the samples done:
   - conf_player: auto_tune._pick_player_conf (the pipeline's per-video rule)
     on the near-grass person confidences of 12 samples spread evenly over
     them (run_pipeline profiles a fixed camera on 12 frames);
   - players: conf >= conf_player, box >= 16 px tall (MIN_DETECTION_HEIGHT_PX),
     feet near grass and within 10 px of the outline (the pitch margin of
     the processed dev windows);
   - teams: team_assignment_v2.cluster_teams on the kit vectors ("hsv4",
     FIXED_VIEW_KIT_FEATURE), each detection an item in place of a track,
     fitted on the detections whose feet are >= 0.15 box heights inside
     the outline (FIXED_VIEW_CORE_DEPTH); the rest, and boxes without a
     kit feature, are 'other';
   -> <name>.parquet when the half is complete, else <name>.partial.parquet
   (a --minutes run), and <name>.json (settings, counts, throughput).

Readings where the protocol is silent (the simplest one each time):
- One sample a second is a half-video frame that is a multiple of 25
  (formation.SAMPLE), as on the ground truth.
- The outline comes from the camera model, not from the release's pitch
  lines: those are GSR annotations, ground truth that may not be read on the
  test halves, so every half is treated alike. On four dev windows the
  projected outline's bounding box is within 20 px of the GSR one (inside
  the crop's 64-px side margin).
- Per detection instead of per track: there is no tracker at 1 fps, so the
  core depth is a detection's own (signed distance / its box height) rather
  than a track's median, and clustering items are detections.
- The 12 profiling samples are the samples at indices round(linspace(0,
  n - 1, 12)) of the samples done; conf_player is chosen on the crop's
  detections (the full-frame rule saw the whole frame; crowds stay off the
  crop and are cut by the grass test either way).
- Throughput is wall-clock seconds per sample of the chunks written
  (decode + detection + features, the reader thread overlapping); the
  projected cost of a half is that times its number of samples.

Smoke test (2026-10-19, dev half 128057 2nd, --minutes 2 then --minutes 3
to check the resume, 2 torch threads): 180 samples, crop (32, 28, 4000,
796), conf_player 0.5, per sample a median of 9 team_A, 9 team_B and 6
other; 1.18 s per sample (1.14 s detector, decoding hidden by the reader
thread) -> 2814 samples, about 0.92 h for the full half.

    PYTHONPATH=/home/user/Veo python research/session_scratchpad/shots_goals/formation_scan.py \\
        128057 2 [--minutes 3] [--chunk 60] [--threads 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("/home/user/Veo")
HERE = ROOT / "research/session_scratchpad/shots_goals"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from src import soccertrack_v2 as st  # noqa: E402
from src.constants import MIN_DETECTION_HEIGHT_PX, PERSON_CLASS_ID  # noqa: E402
from src.grass import _grass_mask, _near_pitch  # noqa: E402
from src.team_assignment_v2 import (FIXED_VIEW_CORE_DEPTH,  # noqa: E402
                                    FIXED_VIEW_KIT_FEATURE, _torso_feature,
                                    cluster_teams)
import crop_check as cc  # noqa: E402

PROTOCOL = HERE / "protocol.json"
FROZEN = HERE / "frozen_goals.json"
CROP_CHECK = HERE / "crop_check.json"
CALIB_FIX = ROOT / "research/session_scratchpad/ball/calib_fix"
CACHE = ROOT / ".cache/formation_scan"
FPS = 25
SAMPLE = 25
CONF_FLOOR = 0.02           # min(auto_tune.CONF_GRID)
PROFILE_SAMPLES = 12
NEAR_PX = 12
PITCH_MARGIN_PX = 10.0
HALF = {1: "1st", 2: "2nd"}
KIT = ["k0", "k1", "k2", "k3"]
VERSION = 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol() -> dict:
    want = (HERE / "protocol.json.sha256").read_text().split()[0]
    got = sha256(PROTOCOL)
    assert got == want, f"protocol.json changed: {got} != {want}"
    return json.loads(PROTOCOL.read_text())


def guard(name: str, protocol: dict):
    """A test half only once the goal rule is frozen."""
    if name not in protocol["GOALS"]["test halves"]:
        return
    if not FROZEN.exists():
        raise SystemExit(f"{name} is a test half and {FROZEN.name} does not "
                         "exist yet: refused")
    sidecar = FROZEN.with_name(FROZEN.name + ".sha256")
    if sidecar.exists() and sidecar.read_text().split()[0] != sha256(FROZEN):
        raise SystemExit(f"{FROZEN.name} does not match its sha256: refused")


def camera(match: str) -> dict:
    """M1: the refitted model (calib_override rebinds st.calibration);
    others: the release's."""
    if match == "117092":
        sys.path.insert(0, str(CALIB_FIX))
        import calib_override  # noqa: F401
    return st.calibration(match)


def crop_passed() -> bool:
    if not CROP_CHECK.exists():
        raise SystemExit(f"{CROP_CHECK.name} missing: run crop_check.py first")
    return bool(json.loads(CROP_CHECK.read_text())["pass"])


def video_geometry(container) -> tuple[int, int, int]:
    """(width, height, frames) of a half video."""
    s = container.streams.video[0]
    n = s.frames or int(round(float(s.duration * s.time_base) * FPS))
    return s.codec_context.width, s.codec_context.height, int(n)


# ---- chunks ------------------------------------------------------------------------

def check_key(parts: Path, key: dict):
    parts.mkdir(parents=True, exist_ok=True)
    path = parts / "key.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != key:
            raise SystemExit(f"{parts} was built with other settings: {old}; "
                             "delete it to rebuild")
    else:
        path.write_text(json.dumps(key, indent=1))


def chunks(parts: Path) -> list[tuple[Path, dict]]:
    """Committed chunks (parquet with its json), in frame order."""
    out = []
    for meta in sorted(parts.glob("chunk_*.json")):
        pq = meta.with_suffix(".parquet")
        if pq.exists():
            out.append((pq, json.loads(meta.read_text())))
    return out


def write_chunk(parts: Path, rows: list[pd.DataFrame], frames: list[int], timing: dict):
    name = f"chunk_{frames[0]:06d}_{frames[-1]:06d}"
    df = pd.concat(rows, ignore_index=True)
    tmp = parts / f"{name}.tmp.parquet"
    df.to_parquet(tmp)
    tmp.rename(parts / f"{name}.parquet")
    tmp = parts / f"{name}.tmp.json"
    tmp.write_text(json.dumps({"frames": frames, **timing}))
    tmp.rename(parts / f"{name}.json")
    print(f"  chunk {frames[0]}-{frames[-1]}: {len(frames)} samples, {len(df)} boxes, "
          f"{timing['wall s'] / len(frames):.2f} s/sample", flush=True)


# ---- one sample ---------------------------------------------------------------------

def prefetch(gen, depth: int = 3):
    """Items of `gen` produced in a thread, `depth` ahead; (item, seconds
    waited for it)."""
    q: queue.Queue = queue.Queue(depth)
    end = object()

    def work():
        try:
            for item in gen:
                q.put(item)
            q.put(end)
        except BaseException as e:  # handed to the consumer
            q.put(e)

    threading.Thread(target=work, daemon=True).start()
    while True:
        t = time.time()
        item = q.get()
        if item is end:
            return
        if isinstance(item, BaseException):
            raise item
        yield item, time.time() - t


def sample_rows(frame_idx: int, img: np.ndarray, model, box, cal: dict,
                poly: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Person boxes of one sample with what the assembly needs; and the
    detector's seconds."""
    t = time.time()
    d = cc.detect(model, img, box, CONF_FLOOR)
    t_det = time.time() - t
    d = d[d.cls == PERSON_CLASS_ID].drop(columns="cls").reset_index(drop=True)
    px, py = ((d.x1 + d.x2) / 2).to_numpy(), d.y2.to_numpy()
    h = (d.y2 - d.y1).to_numpy()
    mask = _grass_mask(img)
    d.insert(0, "frame", frame_idx)
    d["h"] = h
    d["near"] = [_near_pitch(mask, x, y, NEAR_PX) for x, y in zip(px, py)]
    d["depth_px"] = [cv2.pointPolygonTest(poly, (float(x), float(y)), True)
                     for x, y in zip(px, py)]
    xy = st.image_to_pitch(np.c_[px, py], cal) if len(d) else np.zeros((0, 2))
    d["x"], d["y"] = xy[:, 0], xy[:, 1]
    kit = np.full((len(d), 4), np.nan)
    for i, (x, y, hh) in enumerate(zip(px, py, h)):
        f = _torso_feature(img, float(x), float(y), float(hh))
        if f is not None:
            kit[i] = f
    for j, c in enumerate(KIT):
        d[c] = kit[:, j]
    return d, t_det


# ---- the pass -----------------------------------------------------------------------

def scan(match: str, half: int, minutes: float | None, chunk: int,
         threads: int) -> dict:
    protocol = load_protocol()
    name = f"{match}_{HALF[half]}"
    guard(name, protocol)
    use_crop = crop_passed()
    cal = camera(match)
    container = st.open_video(match, half)
    w, h, n_frames = video_geometry(container)
    outline = cc.calibrated_outline(cal)
    poly = outline.astype(np.float32).reshape(-1, 1, 2)
    box = cc.crop_box(outline, w, h) if use_crop else None
    expected = list(range(0, n_frames, SAMPLE))
    limit = expected if minutes is None else [
        f for f in expected if f < minutes * 60 * FPS]
    parts = CACHE / f"{name}.parts"
    check_key(parts, {"match": match, "half": half, "model": cc.MODEL.name,
                      "crop": list(box) if box else None, "conf floor": CONF_FLOOR,
                      "step": SAMPLE, "frames": n_frames, "size": [w, h],
                      "calibration": "calib_override" if match == "117092" else "release",
                      "outline": [[round(float(a), 1), round(float(b), 1)]
                                  for a, b in outline],
                      "version": VERSION})
    done = [f for _, m in chunks(parts) for f in m["frames"]]
    start = max(done) + SAMPLE if done else 0
    todo = [f for f in limit if f >= start]
    print(f"{name}: {n_frames} frames, {len(expected)} samples, {len(done)} done, "
          f"{len(todo)} to do; crop {box}", flush=True)
    if todo:
        run_samples(container, todo, chunk, threads, box, cal, poly, parts)
    container.close()
    return assemble(name, parts, expected, limit, box, outline)


def run_samples(container, todo: list[int], chunk: int, threads: int, box,
                cal: dict, poly: np.ndarray, parts: Path):
    """Detect on the samples `todo` (consecutive), a chunk every `chunk`
    samples. Samples the decoder never delivers (the video ending early, a
    gap) are skipped and counted at assembly."""
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(threads)
    model = YOLO(str(cc.MODEL))
    rows, frames = [], []
    timing = {"wall s": 0.0, "detect s": 0.0, "decode wait s": 0.0}
    t0 = time.time()
    for (k, img), waited in prefetch(st.read_frames(container, todo[0], len(todo), SAMPLE)):
        r, t_det = sample_rows(k, img, model, box, cal, poly)
        rows.append(r)
        frames.append(k)
        timing["detect s"] += t_det
        timing["decode wait s"] += waited
        if len(frames) >= chunk:
            timing["wall s"] = time.time() - t0
            write_chunk(parts, rows, frames, timing)
            rows, frames = [], []
            timing = {"wall s": 0.0, "detect s": 0.0, "decode wait s": 0.0}
            t0 = time.time()
    if frames:
        timing["wall s"] = time.time() - t0
        write_chunk(parts, rows, frames, timing)


# ---- assembly -------------------------------------------------------------------------

def pick_conf(det: pd.DataFrame, samples: list[int]) -> float:
    """The pipeline's per-video player confidence on 12 spread samples."""
    from src.auto_tune import _pick_player_conf

    idx = np.unique(np.round(np.linspace(0, len(samples) - 1, PROFILE_SAMPLES)).astype(int))
    near = det[det.near]
    conf = [near.conf[near.frame == samples[i]].to_numpy() for i in idx]
    return float(_pick_player_conf(conf)[0])


def teams(p: pd.DataFrame) -> pd.Series:
    """team_A / team_B / other per player detection (index of `p`)."""
    kit = p[KIT].to_numpy(dtype=float)
    has = np.isfinite(kit).all(axis=1)
    feats = {int(i): kit[i] for i in np.flatnonzero(has)}
    depth = (p.depth_px / p.h).to_numpy()
    core = {int(i) for i in np.flatnonzero(has & (depth >= FIXED_VIEW_CORE_DEPTH))}
    team = cluster_teams(feats, kit_feature=FIXED_VIEW_KIT_FEATURE, fit_tracks=core)
    return pd.Series([team.get(i, "other") for i in range(len(p))], index=p.index)


def assemble(name: str, parts: Path, expected: list[int], limit: list[int],
             box, outline: np.ndarray) -> dict:
    cs = chunks(parts)
    if not cs:
        raise SystemExit(f"{name}: no sample done")
    det = pd.concat([pd.read_parquet(pq) for pq, _ in cs], ignore_index=True)
    samples = sorted(f for _, m in cs for f in m["frames"])
    conf_player = pick_conf(det, samples)
    p = det[(det.conf >= conf_player) & (det.h >= MIN_DETECTION_HEIGHT_PX)
            & det.near & (det.depth_px >= -PITCH_MARGIN_PX)].reset_index(drop=True)
    p["team"] = teams(p)
    complete = limit == expected
    out = CACHE / (f"{name}.parquet" if complete else f"{name}.partial.parquet")
    p[["frame", "x", "y", "team", "conf"]].to_parquet(out)

    per = p.groupby(["frame", "team"]).size().unstack(fill_value=0).reindex(samples, fill_value=0)
    timing = {k: sum(m[k] for _, m in cs) for k in ("wall s", "detect s", "decode wait s")}
    s_per = timing["wall s"] / len(samples)
    meta = {
        "half": name, "output": str(out.relative_to(ROOT)), "complete": complete,
        "protocol sha256": sha256(PROTOCOL),
        "samples": {"done": len(samples), "expected in half": len(expected),
                    "first": samples[0], "last": samples[-1],
                    "missing": len(set(f for f in limit if f <= samples[-1])
                                   - set(samples))},
        "crop": list(box) if box else None,
        "outline bbox px": [round(float(v), 1) for v in (*outline.min(0), *outline.max(0))],
        "conf floor": CONF_FLOOR, "conf_player": conf_player,
        "boxes per sample at the floor, max": int(det.groupby("frame").size().max()),
        "player rows": len(p),
        "per sample, median": {t: float(per[t].median()) for t in per.columns},
        "team rows": p.team.value_counts().to_dict(),
        "team-labelled rows with x, y inside the pitch": round(float(
            ((p.x.abs() <= 52.5) & (p.y.abs() <= 34))[p.team != "other"].mean()), 4),
        "throughput": {"s per sample": round(s_per, 3),
                       "detect s per sample": round(timing["detect s"] / len(samples), 3),
                       "decode wait s per sample": round(timing["decode wait s"] / len(samples), 3),
                       "projected full half h": round(s_per * len(expected) / 3600, 2)},
    }
    (CACHE / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("match")
    ap.add_argument("half", type=int, choices=(1, 2))
    ap.add_argument("--minutes", type=float, default=None,
                    help="stop after the half's first M minutes (smoke test)")
    ap.add_argument("--chunk", type=int, default=60, help="samples per chunk")
    ap.add_argument("--threads", type=int, default=2, help="torch threads")
    args = ap.parse_args()
    scan(args.match, args.half, args.minutes, args.chunk, args.threads)


if __name__ == "__main__":
    main()
