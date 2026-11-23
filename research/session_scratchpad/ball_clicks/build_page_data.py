"""Frames to click for the ball in flight after shots (owner, 2026-10-27).

About 100 labelled shots of the 8 matches whose clips can be fetched from
Kaggle (M7 / M9 left out: sealed), missed shots of the v3 spotter first,
at most 13 a match. Per shot, 6 frames from just before the kick to 1.4 s
after it, each an 800 x 450 crop centred between the shooter and the goal
he shoots at; the 6 crops of a shot are stacked into one JPEG. Writes
out/<shot>.jpg and manifest.json (crop offsets, to map clicks back to
the full frame).

    python build_page_data.py [PART PARTS]     # -> out/, manifest_<PART>.json
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import soccertrack_v2 as st  # noqa: E402

CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
SCRATCH = Path("/tmp/claude-0/-home-user-veo/b69e5b60-e0fe-5920-8f15-285768d7ea57/scratchpad/clicks_dl")
OUT = HERE / "out"
EMPTY = {"veo-st2-train-b3", "veo-st2-train-b6", "veo-st2-train-b8", "veo-st2-m5-b0", "veo-st2-m5-b2"}
OFFSETS = (-4, 4, 10, 16, 24, 34)
CW, CH = 800, 450
PER_MATCH = 13


def to_image(xy, cal, start):
    f = lambda uv: float(np.sum((st.image_to_pitch(np.array([uv]), cal)[0] - xy) ** 2))
    r = minimize(f, start, method="Nelder-Mead", options={"xatol": 0.5, "fatol": 1e-4, "maxiter": 400})
    return r.x


def main():
    shots = pd.read_parquet(ROOT / "research/session_scratchpad/shots_learn/v3/missed.parquet")
    ds_of = json.loads(Path("/tmp/claude-0/-home-user-veo/b69e5b60-e0fe-5920-8f15-285768d7ea57/scratchpad/clip_dataset.json").read_text())
    shots = shots.assign(ds=shots.unit.map(ds_of))
    shots = shots[shots.ds.notna() & ~shots.ds.str.split("/").str[1].isin(EMPTY)]
    rng = np.random.default_rng(0)
    shots = shots.assign(r=rng.random(len(shots))).sort_values(["hit", "r"])     # missed first
    pick = shots.groupby("match").head(PER_MATCH).sort_values(["unit", "frame"])
    OUT.mkdir(exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    manifest = []
    part, parts = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    for i, (w, g) in enumerate(pick.groupby("unit")):
        if i % parts != part:
            continue
        mp4 = SCRATCH / f"{w}.mp4"
        if not mp4.exists():
            subprocess.run(["kaggle", "datasets", "download", g.ds.iloc[0], "-f", f"{w}.mp4", "-p", str(SCRATCH)],
                           check=True, capture_output=True)
        m = w.split("_")[1]
        cal = st.calibration(m)
        ev = pd.read_csv(CLIPS / f"{w}_events.csv")
        gt = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
        cap = cv2.VideoCapture(str(mp4))
        W, H = int(cap.get(3)), int(cap.get(4))
        for s in g.itertuples():
            e = ev[ev.label.isin(["SHOT", "GOAL"])]
            e = e.iloc[(e.frame - s.frame).abs().argsort()[:1]].iloc[0]
            f = int(e.frame)
            q = gt[(gt.frame == f) & (gt.player_id.astype(str) == str(int(e.player_id)))]
            if q.empty:
                continue
            q = q.iloc[0]
            foot = np.array([q.bx + q.bw / 2, q.by + q.bh])
            gx = 52.5 if q.x >= 0 else -52.5
            goal = to_image(np.array([gx, 0.0]), cal, foot)
            c = (foot + goal) / 2 if np.all(np.isfinite(goal)) and np.hypot(*(goal - foot)) < 1500 else foot
            x0 = int(np.clip(c[0] - CW / 2, 0, W - CW))
            y0 = int(np.clip(c[1] - CH / 2, 0, H - CH))
            crops, frames = [], []
            cap.set(cv2.CAP_PROP_POS_FRAMES, f + OFFSETS[0])      # one seek, then read in order
            for i in range(OFFSETS[-1] - OFFSETS[0] + 1):
                ok, img = cap.read()
                if not ok:
                    break
                if OFFSETS[0] + i in OFFSETS:
                    crops.append(img[y0:y0 + CH, x0:x0 + CW])
                    frames.append(f + OFFSETS[0] + i)
            if len(crops) < len(OFFSETS):
                continue
            sid = f"{w}_s{f:05d}"
            cv2.imwrite(str(OUT / f"{sid}.jpg"), np.vstack(crops), [cv2.IMWRITE_JPEG_QUALITY, 82])
            manifest.append({"id": sid, "unit": w, "match": m, "label": e.label, "label_frame": f,
                             "spotter_hit": bool(s.hit), "frames": frames, "x0": x0, "y0": y0,
                             "shooter_foot": [round(float(v), 1) for v in foot],
                             "goal_px": [round(float(v), 1) for v in goal]})
        cap.release()
        mp4.unlink()
        print(w, len(manifest), flush=True)
    (HERE / f"manifest_{part}.json").write_text(json.dumps(manifest, indent=1))
    print("shots", len(manifest))


if __name__ == "__main__":
    main()
