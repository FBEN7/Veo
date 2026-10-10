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

    python eval_ball_detector.py --heatmap .cache/ball_heatmap.3000.pt \\
        --windows st2_118576_2nd_f015000 ... --save scores.json
"""

from __future__ import annotations

import argparse
import io
import json
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


def frames_of(window: str, seed: int = 0) -> dict:
    """The frames measured in a window: free-flight frames per pass,
    action frames, random frames."""
    lab = pd.read_csv(DATA_DIR / "soccertrack_v2" / f"{window}_events.csv")
    lab = lab.sort_values("frame").reset_index(drop=True)
    n = len(pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{window}_gt.parquet",
                            columns=["frame"]).frame.unique())
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
        passes.append({"frames": [int(round(f)) for f in np.linspace(lo, hi, 4)],
                       "from": (a.x, a.y), "to": (b.x, b.y)})
    actions = lab[lab.label.isin(("PASS", "DRIVE"))][["frame", "player_id", "x", "y"]]
    rng = np.random.default_rng(seed)
    return {"passes": passes, "actions": actions,
            "random": sorted(rng.choice(n, min(N_RANDOM, n), replace=False).tolist())}


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


def run_window(window: str, detectors: dict, seed: int = 0) -> dict:
    info = json.loads((DATA_DIR / "soccertrack_v2" / f"{window}.json").read_text())
    cal = st.calibration(info["match"])
    sets = frames_of(window, seed)
    gt = pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{window}_gt.parquet")
    people = {f: g[["x", "y"]].to_numpy(float) for f, g in gt.groupby("frame")}
    z = np.load(io.BytesIO(st.open_stream(
        f"ball/{info['match']}_{st._half(info['half'])}_ball.npz").read()),
        allow_pickle=False)
    track = pd.DataFrame({"x": z["x"], "y": z["y"]},
                         index=z["frame"].astype(int) - 1 - info["start_frame"])
    wanted = set(sets["random"]) | set(sets["actions"].frame)
    for p in sets["passes"]:
        wanted |= set(p["frames"])
    cands = {name: {} for name in detectors}
    cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{window}.mp4"))
    idx = 0
    while idx <= max(wanted):
        ok, image = cap.read()
        if not ok:
            break
        if idx in wanted:
            for name, fn in detectors.items():
                cands[name][idx] = fn(image, info["pitch"], cal)
        idx += 1
    cap.release()
    rng = np.random.default_rng(seed)
    out = {"window": window, "passes": len(sets["passes"]),
           "actions": len(sets["actions"])}
    pitch = np.asarray(info["pitch"], np.float32)
    for name in detectors:
        c = cands[name]
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
        per_frame = {t: float(np.mean([(c[f][:, 2] >= t).sum() if len(c[f]) else 0
                                       for f in sets["random"]])) for t in TAUS}
        out[name] = {"free-flight hit": round(float(np.mean(hits)), 3) if hits else None,
                     "per pass": hits,
                     "candidates": {str(f): np.round(v, 3).tolist()
                                    for f, v in sorted(c.items())},
                     "feet share": round(float(np.mean(feet)), 3) if feet else None,
                     "action hit": round(float(np.mean(act)), 3) if act else None,
                     "candidates per frame": {str(k): round(v, 2)
                                              for k, v in per_frame.items()}}
    # A random point on the pitch, for the action hit.
    lo, hi = pitch.min(axis=0), pitch.max(axis=0)
    rnd = []
    for a in sets["actions"].itertuples():
        g = st.image_to_pitch([rng.uniform(lo, hi)], cal)[0]
        rnd.append(float(np.isfinite(g).all() and np.hypot(g[0] - a.x, g[1] - a.y) <= 1.5))
    out["random point action hit"] = round(float(np.mean(rnd)), 3) if rnd else None
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heatmap", nargs="*", default=[],
                    help="checkpoints of src/ball_heatmap.build")
    ap.add_argument("--windows", nargs="+", required=True)
    ap.add_argument("--no-coco", action="store_true")
    ap.add_argument("--save")
    args = ap.parse_args()
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(4)
    detectors = {}
    if not args.no_coco:
        yolo = YOLO("yolov8s.pt")
        detectors["coco"] = lambda im, pitch, cal: coco_candidates(yolo, im, pitch)
    for path in args.heatmap:
        m = build()
        m.load_state_dict(torch.load(path))
        m.eval()
        detectors[Path(path).name] = (lambda mm: lambda im, pitch, cal: detect(
            mm, im, pitch, tau=0.01, top_k=8, cal=cal))(m)
    results = [run_window(w, detectors) for w in args.windows]
    summary = {"windows": [{k: v for k, v in r.items()
                            if not isinstance(v, dict)} | {
        n: {k: v for k, v in r[n].items() if k not in ("per pass", "candidates")}
        for n in detectors} for r in results]}
    if not args.no_coco:
        summary["paired"] = {n: compare(results, n) for n in detectors
                             if n != "coco"}
    text = json.dumps(summary, indent=1)
    print(text)
    if args.save:
        Path(args.save).write_text(json.dumps({"summary": summary,
                                               "results": results}, indent=1))


if __name__ == "__main__":
    main()
