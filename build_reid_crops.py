"""Crops of each ground-truth person in a SoccerTrack v2 window, to train the
appearance embedding (`train_player_reid.py --crops`) on fixed panoramas.

The crops are cut the way the pipeline cuts them at inference
(`player_identity.crop`: from a detection's foot point and height), from
our own detections matched to a ground-truth person by their feet
(`eval_soccertrack_v2.match_feet`) -- the released boxes are all 42-43 px
tall and much wider than the players, so they are not used as crops. A
match is kept as a label only when it is close -- within r = min(43 px,
0.6 x the detection's height) of the person's feet -- and no other
person's feet are within 2r of the detection's, so a far-side detection
between two players, or a referee standing in for a missed player, does
not carry a wrong name.

Detections come from the window's pipeline run (`output_<window>/
tracks.parquet`) when there is one, else the detector is run on the
sampled frames only. Every `EVERY`-th frame is used, and up to
`PER_PERSON` crops per person spread over the window.

Writes `<out>/<window>.pkl`: {"window", "match", "half", "people":
{player_id: {"role", "side", "crops", "frame", "h", "dist"}}}, keyed by
the dataset's persistent `player_id`, which is the same person in every
match -- so whoever is in a tuning or report window can be kept out of
training (`train_player_reid.py --exclude`).

    python build_reid_crops.py st2_118576_2nd_f015000 ... --out DIR
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from eval_soccertrack_v2 import boxes, match_feet, truth, FOOT_PX
from src.paths import CACHE_DIR, DATA_DIR
from src.player_identity import crop

EVERY = 10
PER_PERSON = 30
MIN_H = 16.0


def detections(window: str, frames: list[int], model_name: str) -> pd.DataFrame:
    """Player detections on `frames`: from the pipeline run when there is
    one, else from the detector on those frames only."""
    run = Path(f"output_{window}") / "tracks.parquet"
    if run.exists():
        t = pd.read_parquet(run)
        return t[(t.cls == "player") & t.frame.isin(frames)].copy()
    from ultralytics import YOLO

    # As the pipeline detects (score_soccernet.run_pipeline, fixed_camera):
    # full width, its default player confidence.

    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{window}.mp4"))
    imgsz = -(-int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 32) * 32
    want, rows, idx = set(frames), [], 0
    while idx <= max(frames):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            r = model(frame, imgsz=imgsz, conf=0.25, classes=[0],
                      verbose=False)[0]
            for x0, y0, x1, y1 in r.boxes.xyxy.cpu().numpy():
                if y1 - y0 >= MIN_H:
                    rows.append(dict(frame=idx, cls="player",
                                     px=float((x0 + x1) / 2), py=float(y1),
                                     crop_h=float(y1 - y0),
                                     crop_w=float(x1 - x0)))
        idx += 1
    cap.release()
    return pd.DataFrame(rows)


def build(window: str, model_name: str = "yolov8s.pt") -> dict:
    root = DATA_DIR / "soccertrack_v2"
    raw = pd.read_parquet(root / f"{window}_gt.parquet")
    who = raw.groupby("track_id").agg(
        player_id=("player_id", lambda s: s.dropna().mode().iloc[0]
                   if s.notna().any() else None),
        role=("role", lambda s: s.mode().iloc[0]),
        side=("side", lambda s: s.mode().iloc[0]))
    gt = truth(root / f"{window}_gt.parquet")
    frames = sorted(f for f in gt.frame.unique() if f % EVERY == 0)
    gt = gt[gt.frame.isin(frames)]
    det = boxes(detections(window, frames, model_name))
    before = len(det)
    det["person"] = match_feet(det, gt)
    det = det.dropna(subset=["person"])
    matched = len(det)
    # A label only when the match is close and no one else is near.
    feet = {f: g for f, g in gt.groupby("frame")}
    dist, keep = [], []
    for r in det.itertuples():
        g = feet[r.frame]
        d = np.hypot((g.x0 + g.x1) / 2 - (r.x0 + r.x1) / 2, g.y1 - r.y1)
        own = float(d[(g.person == r.person).to_numpy()][0])
        others = d[(g.person != r.person).to_numpy()]
        radius = min(FOOT_PX, 0.6 * r.crop_h)
        dist.append(own)
        keep.append(own <= radius and bool((others >= 2 * radius).all()))
    det = det.assign(dist=dist)[np.array(keep, dtype=bool)]
    print(f"  {window}: {before} detections on {len(frames)} frames, "
          f"{matched} matched, {len(det)} kept as labels", flush=True)
    wanted: dict[int, list] = {}
    for person, rows in det.groupby("person"):
        rows = rows.sort_values("frame")
        pick = rows.iloc[np.linspace(0, len(rows) - 1,
                                     min(PER_PERSON, len(rows))).astype(int)]
        for r in pick.itertuples():
            wanted.setdefault(int(r.frame), []).append(
                (int(person), r.px, r.py, r.crop_h, r.dist))
    out = {int(p): {"crops": [], "frame": [], "h": [], "dist": []}
           for p in det.person.unique()}
    cap = cv2.VideoCapture(str(root / f"{window}.mp4"))
    idx, last = 0, max(wanted) if wanted else -1
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for person, px, py, h, d in wanted.get(idx, ()):
            c = crop(frame, px, py, h)
            if c is not None:
                o = out[person]
                o["crops"].append(c.copy())
                o["frame"].append(idx)
                o["h"].append(float(h))
                o["dist"].append(float(d))
        idx += 1
    cap.release()
    people = {}
    for person, o in out.items():
        w = who.loc[person]
        key = str(w.player_id) if w.player_id is not None else f"t{person}"
        people[key] = {"role": w.role, "side": w.side, **o}
    _, match, half, _ = window.split("_")
    return {"window": window, "match": match, "half": half, "people": people}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows", nargs="+")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--out", default=str(Path(CACHE_DIR) / "soccertrack_v2"
                                         / "reid"))
    args = ap.parse_args()
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    for w in args.windows:
        t0 = time.time()
        blob = build(w, args.model)
        (dest / f"{w}.pkl").write_bytes(pickle.dumps(blob))
        n = [len(p["crops"]) for p in blob["people"].values()]
        print(f"  {w}: {len(n)} people, {sum(n)} crops "
              f"(min {min(n) if n else 0} per person)  "
              f"[{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
