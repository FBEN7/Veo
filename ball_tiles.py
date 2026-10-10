"""Blind visual check of ball detectors: tiles around detectors' top
candidates and around random pitch points (decoys), shuffled, numbered,
on montages; the key (which tile is which) goes to a separate file, to be
read only once every tile is judged.

Judge each tile: `ball` if the match ball's centre is inside the box the
four ticks mark (24 x 24 px in the frame), else `not ball` (`unsure`
counts as not ball).

    python ball_tiles.py --scores heatmap.json coco.json \\
        --take ball_heatmap.3000.pt=40 coco=40 \\
        --tau 0.2 --decoys 20 --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.paths import DATA_DIR

HALF, ZOOM, BOX = 36, 3, 12
PER_SHEET, COLS = 30, 5


def tile(image, u, v, number):
    H, W = image.shape[:2]
    pad = cv2.copyMakeBorder(image, HALF, HALF, HALF, HALF, cv2.BORDER_CONSTANT)
    u, v = int(round(u)) + HALF, int(round(v)) + HALF
    crop = pad[v - HALF:v + HALF, u - HALF:u + HALF]
    t = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
    c, b = HALF * ZOOM, BOX * ZOOM
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        x, y = c + sx * b, c + sy * b
        cv2.line(t, (x, y), (x + sx * 10, y), (0, 0, 255), 1)
        cv2.line(t, (x, y), (x, y + sy * 10), (0, 0, 255), 1)
    strip = np.full((22, t.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, str(number), (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 1)
    return np.vstack([strip, t])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True,
                    help="eval_ball_detector.py --save files (same windows)")
    ap.add_argument("--take", nargs="+", required=True,
                    help="detector=count, top-1 candidates sampled per detector")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="heatmap candidates below are not taken (COCO: 0.02)")
    ap.add_argument("--decoys", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    merged = {}
    for path in args.scores:
        for r in json.loads(Path(path).read_text())["results"]:
            merged.setdefault(r["window"], {}).update(r)
    results = list(merged.values())
    picks = []
    for spec in args.take:
        name, n = spec.rsplit("=", 1)
        pool = []
        for r in results:
            floor = 0.02 if name == "coco" else args.tau
            for f, rows in r[name]["candidates"].items():
                if rows and rows[0][2] >= floor:
                    pool.append((r["window"], int(f), rows[0][0], rows[0][1], name))
        for i in rng.choice(len(pool), min(int(n), len(pool)), replace=False):
            picks.append(pool[i])
    frames = [(r["window"], int(f)) for r in results
              for f in next(iter(v for k, v in r.items()
                                 if isinstance(v, dict) and "candidates" in v))
              ["candidates"]]
    for i in rng.choice(len(frames), min(args.decoys, len(frames)), replace=False):
        w, f = frames[i]
        info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
        poly = np.asarray(info["pitch"], np.float32)
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        while True:
            u, v = rng.uniform(lo, hi)
            if cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(u), float(v)),
                                    False) > 0:
                break
        picks.append((w, f, float(u), float(v), "decoy"))
    order = rng.permutation(len(picks))
    picks = [picks[i] for i in order]
    tiles = [None] * len(picks)
    for w in sorted({p[0] for p in picks}):
        want = {}
        for k, p in enumerate(picks):
            if p[0] == w:
                want.setdefault(p[1], []).append(k)
        cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{w}.mp4"))
        idx = 0
        while idx <= max(want):
            ok, image = cap.read()
            if not ok:
                break
            for k in want.get(idx, []):
                tiles[k] = tile(image, picks[k][2], picks[k][3], k + 1)
            idx += 1
        cap.release()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blank = np.full_like(next(t for t in tiles if t is not None), 255)
    tiles = [t if t is not None else blank for t in tiles]
    for s in range(0, len(tiles), PER_SHEET):
        sheet = tiles[s:s + PER_SHEET]
        sheet += [blank] * (-len(sheet) % COLS)
        rows = [np.hstack(sheet[i:i + COLS]) for i in range(0, len(sheet), COLS)]
        cv2.imwrite(str(out / f"sheet_{s // PER_SHEET + 1}.jpg"), np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / "key.json").write_text(json.dumps(
        [{"tile": k + 1, "window": p[0], "frame": p[1], "u": p[2], "v": p[3],
          "source": p[4]} for k, p in enumerate(picks)], indent=1))
    print(f"  {len(picks)} tiles on {-(-len(picks) // PER_SHEET)} sheets in {out}")


if __name__ == "__main__":
    main()
