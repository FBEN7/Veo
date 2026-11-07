"""Goals protocol (protocol.json, GOALS "pipeline"): the crop check. The 1-fps
formation pass runs yolov8s on the pitch-outline crop only if, on 200 frames
of M6 2nd f015000 (st2_118578_2nd_f015000), the crop finds the full-frame
player boxes (recall >= 0.99) where they were (median box shift <= 1 px).

Two crops are checked (--outline), each the same rule (crop_box: the
outline's bounding box + margins on the 32-px grid) on a different outline:
- 'window': the window json 'pitch' polygon (GSR pitch lines; the first
  check, below);
- 'calib': the crop the scan actually uses (formation_scan.py): the 105 x 68
  boundary projected through the scan's camera model (calibrated_outline;
  the release calibration for 118578), scan_crop() -- the crop the formation
  pass requires to pass (results.calib.pass), else it runs on full frames.
Both are written into crop_check.json under "results" (a run of one
outline keeps the other's stored result), with the crop rule and the
outline agreement on the dev windows.

How the pipeline runs the detector (score_soccernet.run_pipeline with
fixed_camera=True -> src/detect_track_hybrid.run): yolov8s at the frame's own
width (profile.json imgsz 4096, so scale 1 and the frame letterboxed only
vertically, 1080 -> 1088, 4 grey rows on top), `model(frame, conf=min(
conf_player, conf_ball), imgsz, classes=[person, sports ball])`, persons kept
at conf >= conf_player (0.4 on this window), every 2nd frame; tracks.parquet
holds the boxes ByteTrack passed on (feet px, py; crop_w, crop_h) with feet
near grass and >= 16 px tall.

The crop: the bounding box of the pitch outline (the window json 'pitch'
polygon, or the calibrated outline), widened by HEAD_MARGIN_PX above (the heads of people
standing on the far touchline: the tallest box with feet above y = 200 here
is 65 px) and SIDE_MARGIN_PX left, right and below (the pipeline keeps feet
up to 10 px outside the outline, plus half a box and some context), then
snapped outward to YOLOv8's 32-px stride so that it is run exactly as the
pipeline runs the full frame: scale 1, no letterbox padding, and the same
32-px grid (x0 = 0 and y0 + 4 = 0 modulo 32, the 4 being the full frame's top
padding), imgsz = crop width. Boxes are shifted back by the crop's corner.

Measured on frames 0, 22, 44, ... (200 of the window's 4500, all of them
frames the cached run detected on), read in order with cv2.VideoCapture as
the pipeline reads them (decoder and cv2 limited to --threads):
- reference: the cached players of the frame whose feet lie inside the
  crop's own outline or within its 10-px margin (pitch.json margin_px: the
  pipeline's pitch filter, and the scan's PITCH_MARGIN_PX; boxes outside
  cannot be in the crop and are dropped anyway); their boxes are
  x = px -+ crop_w / 2, y = py - crop_h .. py. For 'calib' the recall on
  the window-outline reference is reported beside it (not the criterion);
- crop detections: persons at conf >= conf_player, the cached run's call,
  all of them matchable (no height or outline cut, so a reference cannot be
  lost to a cut on its partner); unmatched ones are counted only when their
  feet lie within the outline + margin;
- matching: one to one per frame, Hungarian on IoU, pairs with IoU >= 0.5;
- recall = matched references / references;
- box shift (the protocol's): the distance in pixels between the matched
  boxes' bottom centres -- the box's position in this pipeline (px, py are
  what the tracks keep and what is mapped to pitch metres). The largest
  corner-coordinate difference and the IoU are reported beside it.
A control runs the full frame through the same call on every 10th sampled
frame and is scored the same way: what the replication itself reproduces of
the cache (a different ultralytics or decoder would show here, not as a crop
effect).

Outline agreement (every run, no detector): on the dev windows (test-half
and M1 windows skipped by name, before anything is read) the bounding box of
the outline projected through the release calibration against that of the
window json 'pitch' polygon, side by side (projected - window, px).

Results (2026-10-19, 200 frames, 2 torch threads):
- window: crop (32, 28, 4000, 796), 3968 x 768; recall 0.9956 (4317/4336),
  median box shift 0.012 px (p90 0.057): PASS. The 19 misses are far-side
  players 25-49 px tall, >= 86 px from any crop side. Control: 440/440 at
  0.0 px (the cache is reproduced exactly). 1.18 s per crop against 2.03 s
  per full frame (310 s).
- calib: CALIB_RESULT
- outline agreement: OUTLINE_RESULT

    PYTHONPATH=/home/user/Veo nice -n 10 python research/session_scratchpad/shots_goals/crop_check.py \\
        [--outline window|calib|both] [--frames 200] [--threads 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path("/home/user/Veo")
HERE = ROOT / "research/session_scratchpad/shots_goals"
sys.path.insert(0, str(ROOT))
from src import soccertrack_v2 as st  # noqa: E402
from src.constants import BALL_CLASS_ID, PERSON_CLASS_ID  # noqa: E402

PROTOCOL = HERE / "protocol.json"
OUT = HERE / "crop_check.json"
WINDOW = "st2_118578_2nd_f015000"
MODEL = ROOT / "yolov8s.pt"
STRIDE = 32                 # YOLOv8's largest stride
HEAD_MARGIN_PX = 96
SIDE_MARGIN_PX = 64
IOU_MATCH = 0.5
RECALL_MIN, SHIFT_MAX_PX = 0.99, 1.0     # protocol GOALS "pipeline"
CONTROL_EVERY = 10
OUTLINES = ("window", "calib")
# The scan's crop (formation_scan.py, scan_crop): part of its scan key, and
# what results.calib must have been measured with.
CROP_RULE = {"outline": "105 x 68 boundary projected through the scan's camera "
                        "model (crop_check.calibrated_outline)",
             "head margin px": HEAD_MARGIN_PX, "side margin px": SIDE_MARGIN_PX,
             "stride px": STRIDE, "grid": "full frame's (top letterbox pad)"}


def load_protocol() -> dict:
    blob = PROTOCOL.read_bytes()
    want = (HERE / "protocol.json.sha256").read_text().split()[0]
    got = hashlib.sha256(blob).hexdigest()
    assert got == want, f"protocol.json changed: {got} != {want}"
    return json.loads(blob)


# ---- the crop ------------------------------------------------------------------

def full_frame_top_pad(height: int) -> int:
    """Grey rows ultralytics' LetterBox puts above a frame run at its own
    width (scale 1, auto): half the padding to the next multiple of 32."""
    dh = (-height) % STRIDE / 2
    return int(round(dh - 0.1))


def crop_box(outline, width: int, height: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the crop: the outline's bounding box plus the
    margins, snapped outward to the full frame's 32-px grid, sides multiples
    of 32 (shrunk by whole strides where the frame ends first)."""
    if width % STRIDE:
        raise ValueError(f"frame width {width} is not a multiple of {STRIDE}")
    top = full_frame_top_pad(height)
    p = np.asarray(outline, dtype=float).reshape(-1, 2)
    x0 = max(0, int(np.floor((p[:, 0].min() - SIDE_MARGIN_PX) / STRIDE)) * STRIDE)
    x1 = min(width, int(np.ceil((p[:, 0].max() + SIDE_MARGIN_PX) / STRIDE)) * STRIDE)
    y0 = int(np.floor((p[:, 1].min() - HEAD_MARGIN_PX + top) / STRIDE)) * STRIDE - top
    y0 = max(y0, (-top) % STRIDE)
    y1 = y0 + int(np.ceil((p[:, 1].max() + SIDE_MARGIN_PX - y0) / STRIDE)) * STRIDE
    while y1 > height:
        y1 -= STRIDE
    if y1 - y0 > x1 - x0:
        raise ValueError("crop taller than wide: imgsz = width would rescale it")
    return x0, y0, x1, y1


def calibrated_outline(cal: dict, per_side: int = 41) -> np.ndarray:
    """The 105 x 68 pitch boundary in pixels through a camera model (the
    inverse of st.image_to_pitch): centre-origin metres -> corner origin ->
    homography -> undistorted pixels -> fisheye distortion."""
    s = np.linspace(-1, 1, per_side)
    one = np.ones_like(s)
    xy = np.r_[np.c_[52.5 * s, -34 * one], np.c_[52.5 * one, 34 * s],
               np.c_[-52.5 * s, 34 * one], np.c_[-52.5 * one, -34 * s]]
    q = np.c_[xy + (52.5, 34.0), np.ones(len(xy))] @ np.linalg.inv(cal["Hinv"]).T
    und = q[:, :2] / q[:, 2:]
    kn = cal["Knew"]
    norm = (und - kn[:2, 2]) / np.diag(kn)[:2]
    uv = cv2.fisheye.distortPoints(np.ascontiguousarray(norm.reshape(-1, 1, 2)),
                                   cal["K"], cal["D"])
    return uv.reshape(-1, 2)


def scan_crop(cal: dict, width: int, height: int
              ) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """The crop the formation pass uses (CROP_RULE): crop_box of the outline
    projected through `cal`; and that outline."""
    outline = calibrated_outline(cal)
    return crop_box(outline, width, height), outline


# ---- detection -------------------------------------------------------------------

def detect(model, frame: np.ndarray, box, conf: float) -> pd.DataFrame:
    """The pipeline's call on frame[box] (box None: the whole frame at its
    own width): persons and balls at `conf`, boxes in full-frame pixels."""
    if box is None:
        x0, y0, img = 0, 0, frame
        imgsz = -(-frame.shape[1] // STRIDE) * STRIDE
    else:
        x0, y0, x1, y1 = box
        img = np.ascontiguousarray(frame[y0:y1, x0:x1])
        imgsz = x1 - x0
    r = model(img, verbose=False, conf=conf, imgsz=imgsz,
              classes=[PERSON_CLASS_ID, BALL_CLASS_ID])[0]
    xyxy = r.boxes.xyxy.cpu().numpy().astype(float).reshape(-1, 4)
    out = pd.DataFrame(xyxy + (x0, y0, x0, y0), columns=["x1", "y1", "x2", "y2"])
    out["conf"] = r.boxes.conf.cpu().numpy().astype(float)
    out["cls"] = r.boxes.cls.cpu().numpy().astype(int)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (n, 4) and (m, 4) xyxy boxes."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda z: np.prod(z[:, 2:] - z[:, :2], axis=1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def match(ref: np.ndarray, det: np.ndarray) -> list[tuple[int, int, float]]:
    """One-to-one (ref, det, IoU) pairs with IoU >= IOU_MATCH, Hungarian."""
    if not len(ref) or not len(det):
        return []
    m = iou(ref, det)
    rows, cols = linear_sum_assignment(-m)
    return [(int(r), int(c), float(m[r, c])) for r, c in zip(rows, cols)
            if m[r, c] >= IOU_MATCH]


def feet(b: np.ndarray) -> np.ndarray:
    return np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]


def reference(tracks: pd.DataFrame, frame: int, outline, margin: float) -> np.ndarray:
    """Cached player boxes of a frame with feet within `margin` px of the
    outline (xyxy)."""
    p = tracks[tracks.frame == frame]
    poly = np.asarray(outline, dtype=np.float32).reshape(-1, 1, 2)
    inside = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -margin
                       for x, y in zip(p.px, p.py)], dtype=bool)
    p = p[inside] if len(p) else p
    return np.c_[p.px - p.crop_w / 2, p.py - p.crop_h, p.px + p.crop_w / 2, p.py
                 ].astype(float).reshape(-1, 4)


def score(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, list]]) -> dict:
    """Recall and box shifts over frames of (reference, detections, whether
    each detection is inside the outline + margin, pairs)."""
    n_ref = sum(len(r[0]) for r in rows)
    shift, corner, ious, extra = [], [], [], 0
    for ref, det, inside, pairs in rows:
        extra += int(inside.sum()) - sum(bool(inside[j]) for _, j, _ in pairs)
        for i, j, o in pairs:
            shift.append(float(np.hypot(*(feet(ref[i:i + 1]) - feet(det[j:j + 1]))[0])))
            corner.append(float(np.abs(ref[i] - det[j]).max()))
            ious.append(o)
    n = len(shift)
    q = lambda v, k: round(float(np.quantile(v, k)), 3) if v else None  # noqa: E731
    return {"frames": len(rows), "reference boxes": n_ref,
            "matched": n, "recall": round(n / n_ref, 4) if n_ref else None,
            "median box shift px": q(shift, 0.5), "p90 box shift px": q(shift, 0.9),
            "max box shift px": round(max(shift), 3) if shift else None,
            "median largest corner shift px": q(corner, 0.5),
            "median IoU": q(ious, 0.5),
            "unmatched detections (inside outline + margin)": extra}


def near_border(b: np.ndarray, box) -> np.ndarray:
    """Distance in px from each box to the crop's nearest side."""
    x0, y0, x1, y1 = box
    return np.min(np.c_[b[:, 0] - x0, b[:, 1] - y0, x1 - b[:, 2], y1 - b[:, 3]], axis=1)


# ---- the run ---------------------------------------------------------------------

def frames_wanted(n_frames: int, n: int) -> list[int]:
    """n frames 0, step, 2 step, ... with an even step (the cache's stride 2)."""
    step = (n_frames // n) // 2 * 2
    return list(range(0, step * n, step))


def run(kind: str, n_frames: int, threads: int) -> dict:
    """The check of one crop: 'window' (the window json outline) or 'calib'
    (the scan's crop, scan_crop on the release calibration)."""
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    out_dir = ROOT / f"output_{WINDOW}"
    info = json.loads((ROOT / f"data/soccertrack_v2/{WINDOW}.json").read_text())
    prof = json.loads((out_dir / "profile.json").read_text())
    margin = float(json.loads((out_dir / "pitch.json").read_text())["margin_px"])
    tracks = pd.read_parquet(out_dir / "tracks.parquet",
                             columns=["frame", "cls", "px", "py", "crop_h", "crop_w"])
    tracks = tracks[tracks.cls == "player"]
    conf_player = float(prof["conf_player"])
    conf_call = min(conf_player, float(prof["conf_ball"]))
    w, h = int(info["width"]), int(info["height"])
    window_outline = np.asarray(info["pitch"], dtype=float)
    if kind == "window":
        outline, what = window_outline, "window json 'pitch' polygon"
        box = crop_box(outline, w, h)
    else:
        if str(info["match"]) == "117092":
            raise SystemExit("the scan's M1 camera is calib_override's, not checked here")
        box, outline = scan_crop(st.calibration(info["match"]), w, h)
        what = (f"the scan's (formation_scan.py): the 105 x 68 boundary projected "
                f"through the release calibration of {info['match']}")
    want = frames_wanted(int(info["frames"]), n_frames)
    print(f"{WINDOW} [{kind}]: crop {box} ({box[2] - box[0]}x{box[3] - box[1]}), "
          f"{len(want)} frames, conf {conf_player}", flush=True)

    model = YOLO(str(MODEL))
    cap = cv2.VideoCapture(info["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, threads])
    crop_rows, full_rows, win_rows, missed = [], [], [], []
    t_crop, t_full, idx = 0.0, 0.0, 0
    for k, f in enumerate(want):
        while idx < f:
            cap.grab()
            idx += 1
        ok, frame = cap.read()
        idx += 1
        if not ok:
            raise SystemExit(f"cannot read frame {f}")
        ref = reference(tracks, f, outline, margin)
        t = time.time()
        d = detect(model, frame, box, conf_call)
        t_crop += time.time() - t
        det, inside = players(d, conf_player, outline, margin)
        pairs = match(ref, det)
        crop_rows.append((ref, det, inside, pairs))
        if kind != "window":
            ref_w = reference(tracks, f, window_outline, margin)
            det_w, inside_w = players(d, conf_player, window_outline, margin)
            win_rows.append((ref_w, det_w, inside_w, match(ref_w, det_w)))
        hit = {i for i, _, _ in pairs}
        for i in sorted(set(range(len(ref))) - hit):
            missed.append({"frame": f, "feet": [round(float(v), 1) for v in feet(ref[i:i + 1])[0]],
                           "height": round(float(ref[i, 3] - ref[i, 1]), 1),
                           "px to crop side": round(float(near_border(ref[i:i + 1], box)[0]), 1)})
        if k % CONTROL_EVERY == 0:
            t = time.time()
            dfull = detect(model, frame, None, conf_call)
            t_full += time.time() - t
            fd, fin = players(dfull, conf_player, outline, margin)
            full_rows.append((ref, fd, fin, match(ref, fd)))
        if k % 20 == 0:
            print(f"  frame {f}: {len(ref)} ref, {len(pairs)} matched "
                  f"({t_crop / (k + 1):.2f} s/frame crop)", flush=True)
    cap.release()

    crop = score(crop_rows)
    passed = bool(crop["recall"] is not None and crop["recall"] >= RECALL_MIN
                  and crop["median box shift px"] <= SHIFT_MAX_PX)
    out = {
        "window": WINDOW, "outline kind": kind, "frames": want, "n frames": len(want),
        "detector": {"model": MODEL.name, "full imgsz": int(prof["imgsz"]),
                     "crop imgsz": box[2] - box[0], "conf call": conf_call,
                     "conf_player": conf_player, "classes": [PERSON_CLASS_ID, BALL_CLASS_ID],
                     "torch threads": threads, "cv2 threads": threads},
        "crop": {"box xyxy": list(box), "size": [box[2] - box[0], box[3] - box[1]],
                 "head margin px": HEAD_MARGIN_PX, "side margin px": SIDE_MARGIN_PX,
                 "outline": what,
                 "outline bbox px": [round(float(v), 1) for v in
                                     (*outline.min(0), *outline.max(0))],
                 "full-frame top pad": full_frame_top_pad(h),
                 "area vs letterboxed full frame": round(
                     (box[2] - box[0]) * (box[3] - box[1])
                     / (w * (h + 2 * full_frame_top_pad(h))), 3)},
        "reference": f"output_{WINDOW}/tracks.parquet players with feet within "
                     f"{margin:g} px of the crop's outline",
        "crop vs cache": crop,
        "control: full frame vs cache": score(full_rows),
        "seconds per frame": {"crop": round(t_crop / len(want), 3),
                              "full frame": round(t_full / max(1, len(full_rows)), 3)},
        "missed reference boxes": missed,
        "criterion": {"recall >=": RECALL_MIN, "median box shift px <=": SHIFT_MAX_PX},
        "pass": passed,
    }
    if kind != "window":
        out["crop rule"] = CROP_RULE
        out["crop vs cache, window-outline reference (reported, not the criterion)"] = \
            score(win_rows)
    return out


def outline_agreement(test_halves: list[str]) -> dict:
    """Bounding box of the outline projected through the release calibration
    against that of the window json 'pitch' polygon, on the dev windows:
    (x0, y0, x1, y1) differences, projected - window, px. Windows of a test
    half or of M1 are skipped by name, before anything is read."""
    rows, per_match = {}, {}
    for p in sorted((ROOT / "data/soccertrack_v2").glob("st2_*_f*.json")):
        half = "_".join(p.stem.split("_")[1:3])
        if half in test_halves or half.startswith("117092"):
            continue
        info = json.loads(p.read_text())
        g = np.asarray(info["pitch"], dtype=float)
        c = calibrated_outline(st.calibration(info["match"]))
        d = np.r_[c.min(0), c.max(0)] - np.r_[g.min(0), g.max(0)]
        worst = float(np.abs(d).max())
        rows[p.stem] = {"projected - window bbox px": [round(float(v), 1) for v in d],
                        "largest px": round(worst, 1)}
        per_match[str(info["match"])] = max(per_match.get(str(info["match"]), 0.0), worst)
    worst = list(per_match.values())
    return {"what": "bounding box (x0, y0, x1, y1) of the 105 x 68 boundary projected "
                    "through the release calibration minus that of the window json "
                    "'pitch' polygon (GSR pitch lines)",
            "skipped": "windows of the test halves and of M1 (not read)",
            "windows": len(rows), "matches": len(per_match),
            "largest px per match": {k: round(v, 1) for k, v in sorted(per_match.items())},
            "largest px, median over matches": round(float(np.median(worst)), 1) if worst else None,
            "largest px, max": round(max(worst), 1) if worst else None,
            "per window": rows}


def stored_results() -> dict:
    """The results already in crop_check.json, by outline kind (the first
    run's file held the window outline's result alone)."""
    if not OUT.exists():
        return {}
    old = json.loads(OUT.read_text())
    return old["results"] if "results" in old else {"window": old}


def players(d: pd.DataFrame, conf: float, outline, margin: float
            ) -> tuple[np.ndarray, np.ndarray]:
    """Person boxes at conf >= `conf` (xyxy), all of them matchable, and
    whether each one's feet lie within `margin` px of the outline (only
    those count as unmatched extras)."""
    p = d[(d.cls == PERSON_CLASS_ID) & (d.conf >= conf)]
    poly = np.asarray(outline, dtype=np.float32).reshape(-1, 1, 2)
    inside = np.array([cv2.pointPolygonTest(poly, (float((a + b) / 2), float(y)), True)
                       >= -margin for a, b, y in zip(p.x1, p.x2, p.y2)], dtype=bool)
    return p[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float).reshape(-1, 4), inside


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outline", choices=(*OUTLINES, "both"), default="both",
                    help="which crop to check: the window json outline's, the "
                         "scan's (calib), or both; the other's stored result is kept")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    proto = load_protocol()
    results = stored_results()
    for kind in (OUTLINES if args.outline == "both" else (args.outline,)):
        t = time.time()
        res = run(kind, args.frames, args.threads)
        res["wall s"] = round(time.time() - t, 1)
        others = [r["frames"] for k, r in results.items() if k != kind]
        if any(f != res["frames"] for f in others):
            raise SystemExit("the stored result was measured on other frames: "
                             "rerun with --outline both")
        results[kind] = res
        c = res["crop vs cache"]
        print(f"{kind} crop {res['crop']['box xyxy']}: recall {c['recall']} "
              f"({c['matched']}/{c['reference boxes']}), median shift "
              f"{c['median box shift px']} px -> pass {res['pass']}; control "
              f"{res['control: full frame vs cache']['recall']}, "
              f"{res['control: full frame vs cache']['median box shift px']} px", flush=True)
    agree = outline_agreement(proto["GOALS"]["test halves"])
    out = {"protocol sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
           "window": WINDOW, "scan crop rule": CROP_RULE,
           "criterion": {"recall >=": RECALL_MIN, "median box shift px <=": SHIFT_MAX_PX},
           "pass": {k: results[k]["pass"] for k in OUTLINES if k in results},
           "results": {k: results[k] for k in OUTLINES if k in results},
           "outline agreement (dev windows)": agree}
    tmp = OUT.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.rename(OUT)
    print(f"outline agreement: {agree['windows']} windows of {agree['matches']} matches, "
          f"largest bbox difference {agree['largest px, max']} px (median over matches "
          f"{agree['largest px, median over matches']}) -> {OUT}")


if __name__ == "__main__":
    main()
