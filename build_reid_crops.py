"""Crops of each ground-truth person in a SoccerTrack v2 window, to train the
appearance embedding (`train_player_reid.py --crops`) on fixed panoramas.

The crops are cut the way the pipeline cuts them at inference
(`player_identity.crop`: from a detection's foot point and height), from
our own detections matched to a ground-truth person by their feet
(`eval_soccertrack_v2.match_feet`) -- the released boxes are all 42-43 px
tall and much wider than the players, so they are not used as crops. A
detection is kept only when no other person's feet are within
`CROWD_PX` of the matched person's, so two players crossing do not swap
labels.

Detections come from the window's pipeline run (`output_<window>/
tracks.parquet`) when there is one, else the detector is run on the
sampled frames only. Every `EVERY`-th frame is used, and up to
`PER_PERSON` crops per person spread over the window.

Writes `<cache>/soccertrack_v2/reid/<window>.pkl`: {person: (side,
[crops])}, persons being the window's ground-truth track ids.

    python build_reid_crops.py st2_118576_2nd_f015000 ...
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
CROWD_PX = FOOT_PX
MIN_H = 16.0


def detections(window: str, frames: list[int], model_name: str) -> pd.DataFrame:
    """Player detections on `frames`: from the pipeline run when there is
    one, else from the detector on those frames only."""
    run = Path(f"output_{window}") / "tracks.parquet"
    if run.exists():
        t = pd.read_parquet(run)
        return t[(t.cls == "player") & t.frame.isin(frames)].copy()
    from ultralytics import YOLO

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
    gt = truth(root / f"{window}_gt.parquet")
    frames = sorted(f for f in gt.frame.unique() if f % EVERY == 0)
    gt = gt[gt.frame.isin(frames)]
    det = boxes(detections(window, frames, model_name))
    det["person"] = match_feet(det, gt)
    det = det.dropna(subset=["person"])
    # Keep a detection only when no other person stands within CROWD_PX of
    # the person it was matched to.
    feet = {f: g for f, g in gt.groupby("frame")}
    clear = []
    for r in det.itertuples():
        g = feet[r.frame]
        own = g[g.person == r.person]
        fx, fy = float((own.x0 + own.x1).iloc[0] / 2), float(own.y1.iloc[0])
        d = np.hypot((g.x0 + g.x1) / 2 - fx, g.y1 - fy)
        clear.append(bool((d[g.person != r.person] > CROWD_PX).all()))
    det = det[np.array(clear, dtype=bool)]
    side = gt.groupby("person").side.agg(lambda s: s.mode().iloc[0])
    wanted: dict[int, list] = {}
    for person, rows in det.groupby("person"):
        rows = rows.sort_values("frame")
        pick = rows.iloc[np.linspace(0, len(rows) - 1,
                                     min(PER_PERSON, len(rows))).astype(int)]
        for r in pick.itertuples():
            wanted.setdefault(int(r.frame), []).append(
                (int(person), r.px, r.py, r.crop_h))
    out = {int(p): (side.get(p), []) for p in det.person.unique()}
    cap = cv2.VideoCapture(str(root / f"{window}.mp4"))
    idx, last = 0, max(wanted) if wanted else -1
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for person, px, py, h in wanted.get(idx, ()):
            c = crop(frame, px, py, h)
            if c is not None:
                out[person][1].append(c.copy())
        idx += 1
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows", nargs="+")
    ap.add_argument("--model", default="yolov8s.pt")
    args = ap.parse_args()
    dest = Path(CACHE_DIR) / "soccertrack_v2" / "reid"
    dest.mkdir(parents=True, exist_ok=True)
    for w in args.windows:
        t0 = time.time()
        people = build(w, args.model)
        (dest / f"{w}.pkl").write_bytes(pickle.dumps(people))
        n = [len(c) for _, c in people.values()]
        print(f"  {w}: {len(people)} people, {sum(n)} crops "
              f"(min {min(n) if n else 0} per person)  "
              f"[{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
