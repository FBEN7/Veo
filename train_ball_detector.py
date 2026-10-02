"""Fine-tune a ball detector on this footage, and test it on a match it never saw.

The COCO detector the pipeline uses proposes the ball on 63 of 138
hand-clicked frames around the labelled events at the clips' working size,
90 at the footage's own width, and never twice in flight between a strike
and the goal line -- which is what reading a shot from the ball's own
flight needs (`src/goal_plane.py`). This trains a one-class detector for
exactly that ball.

## Data

- The clicks from `make_ball_labeller.py` and `make_ball_labeller2.py`: a
  box at each click, sized by the stored candidate it lands on or by the
  clip's typical ball box. Frames marked "not visible" are kept with no box,
  as are other parts of every clicked frame -- the stewards, logos and
  heads the COCO detector takes for the ball are exactly the negatives.
- The Roboflow football-players dataset (CC BY 4.0), its ball class only.

## Scale

The ball is a few pixels across, so nothing is shrunk: training images are
`TILE` x `TILE` tiles cut at the footage's own scale, and the detector runs
on whole 1280 x 720 frames at `imgsz=1280`, the same scale.

## Testing

By match, never by frame: trained on one match's clicks (and the public
data), tested on the other's. Frames from a clip in training are not a test
of the next frame of that clip. Recall is a detection within `HIT_PX` of a
click; a false detection is one further than `MISS_PX` from it, or any on a
"not visible" frame.

    python train_ball_detector.py --roboflow <export dir> \\
        --clicks data/<round 1>.json [data/<round 2>.json] --fold all

Weights and tiles are derived from footage under the data agreement and are
not committed.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np

from src.paths import CACHE_DIR

TILE = 640
# Each positive tile places the ball at a random spot at least this far
# inside the tile, so the detector does not learn "the ball is central".
EDGE_PX = 48
# Negative tiles cut from each clicked frame away from the ball, and from
# each "not visible" frame.
NEGATIVES_PER_FRAME = 1
# A click and a stored candidate this close are the same ball.
CLICK_MATCH_PX = 12.0
HIT_PX = 8.0
MISS_PX = 15.0
CONFS = (0.05, 0.1, 0.25)

MATCHES = {"stoke": ["stoke_1302", "stoke_4207", "stoke_7001"],
           "reading": ["reading_0737", "reading_1155", "reading_2519"]}


def match_of(clip: str) -> str:
    return clip.split("_")[0]


def load_clicks(paths):
    """{(clip, frame): (x, y, size) or None}, later rounds overriding."""
    import pandas as pd

    out, typical = {}, {}
    for path in paths:
        for r in json.loads(Path(path).read_text())["frames"]:
            if "unlabelled" in r:
                continue
            out[(r["clip"], int(r["frame"]))] = (
                None if not r.get("ball") else
                (r["ball"][0] * r["width"], r["ball"][1] * r["height"]))
    sized = {}
    for clip in sorted({c for c, _ in out}):
        balls = pd.read_parquet(Path(f"output_{clip}") / "tracks.parquet")
        balls = balls[balls.cls == "ball"]
        matched = []
        for (c, f), xy in out.items():
            if c != clip or xy is None:
                continue
            here = balls[balls.frame == f]
            gap = np.hypot(here.px.to_numpy() - xy[0],
                           here.py.to_numpy() - xy[1])
            if len(gap) and gap.min() <= CLICK_MATCH_PX:
                size = float(here.crop_h.iloc[int(np.argmin(gap))])
                matched.append(size)
                sized[(c, f)] = (*xy, size)
        typical[clip] = float(np.median(matched)) if matched else float(
            balls.crop_h.median())
        for (c, f), xy in out.items():
            if c == clip and (c, f) not in sized:
                sized[(c, f)] = None if xy is None else (*xy, typical[clip])
    return sized


def tile_origin(w, h, x, y, rng):
    """A tile's top-left corner placing (x, y) at a random inner spot."""
    lo_x = int(np.clip(x - TILE + EDGE_PX, 0, w - TILE))
    hi_x = int(np.clip(x - EDGE_PX, 0, w - TILE))
    lo_y = int(np.clip(y - TILE + EDGE_PX, 0, h - TILE))
    hi_y = int(np.clip(y - EDGE_PX, 0, h - TILE))
    return (int(rng.integers(lo_x, hi_x + 1)),
            int(rng.integers(lo_y, hi_y + 1)))


def away_origin(w, h, x, y, rng):
    """A tile's corner not containing (x, y), or anywhere if None."""
    for _ in range(50):
        ox = int(rng.integers(0, w - TILE + 1))
        oy = int(rng.integers(0, h - TILE + 1))
        if x is None or not (ox - 20 <= x <= ox + TILE + 20
                             and oy - 20 <= y <= oy + TILE + 20):
            return ox, oy
    return None


def write_tile(image, ox, oy, boxes, out_dir: Path, name: str):
    import cv2

    patch = image[oy:oy + TILE, ox:ox + TILE]
    if patch.shape[0] != TILE or patch.shape[1] != TILE:
        pad_b, pad_r = TILE - patch.shape[0], TILE - patch.shape[1]
        patch = cv2.copyMakeBorder(patch, 0, pad_b, 0, pad_r,
                                   cv2.BORDER_CONSTANT)
    cv2.imwrite(str(out_dir / "images" / f"{name}.jpg"), patch,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines = []
    for x, y, size in boxes:
        cx, cy = (x - ox) / TILE, (y - oy) / TILE
        if 0 < cx < 1 and 0 < cy < 1:
            s = size / TILE
            lines.append(f"0 {cx:.6f} {cy:.6f} {s:.6f} {s:.6f}")
    (out_dir / "labels" / f"{name}.txt").write_text("\n".join(lines))


def build(root: Path, roboflow: Path, clicks, train_clips, seed: int = 0):
    """Tiles for one fold's training set; returns the data.yaml path."""
    import cv2

    rng = np.random.default_rng(seed)
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (root / split / sub).mkdir(parents=True)
    counts = {"public balls": 0, "clip balls": 0, "empty tiles": 0}

    images = sorted(glob.glob(str(roboflow / "*" / "images" / "*.jpg")))
    for k, path in enumerate(images):
        image = cv2.imread(path)
        h, w = image.shape[:2]
        label = Path(path).parent.parent / "labels" / (Path(path).stem + ".txt")
        balls = []
        for line in (label.read_text().split("\n") if label.exists() else []):
            parts = line.split()
            if len(parts) == 5 and parts[0] == "0":
                _, cx, cy, bw, bh = map(float, parts)
                balls.append((cx * w, cy * h, max(bw * w, bh * h)))
        split = "val" if k % 10 == 0 else "train"
        for j, (x, y, size) in enumerate(balls):
            ox, oy = tile_origin(w, h, x, y, rng)
            write_tile(image, ox, oy, balls, root / split, f"rf{k}_{j}")
            counts["public balls"] += 1
        where = away_origin(w, h, *(balls[0][:2] if balls else (None, None)),
                            rng)
        if where is not None:
            write_tile(image, *where, [], root / split, f"rf{k}_neg")
            counts["empty tiles"] += 1

    for clip in train_clips:
        info = json.loads((Path(f"output_{clip}") / "clip.json").read_text())
        cap = cv2.VideoCapture(info["path"])
        frames = sorted(f for c, f in clicks if c == clip)
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, image = cap.read()
            if not ok:
                continue
            h, w = image.shape[:2]
            ball = clicks[(clip, f)]
            split = "val" if f % 10 == 5 else "train"
            if ball is not None:
                ox, oy = tile_origin(w, h, ball[0], ball[1], rng)
                write_tile(image, ox, oy, [ball], root / split,
                           f"{clip}_{f}")
                counts["clip balls"] += 1
            for j in range(NEGATIVES_PER_FRAME):
                where = away_origin(w, h, *(ball[:2] if ball else
                                            (None, None)), rng)
                if where is not None:
                    write_tile(image, *where, [], root / split,
                               f"{clip}_{f}_neg{j}")
                    counts["empty tiles"] += 1
        cap.release()

    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root}\ntrain: train/images\nval: val/images\n"
                    f"names:\n  0: ball\n")
    print("  tiles: " + ", ".join(f"{v} {k}" for k, v in counts.items()),
          flush=True)
    return yaml


def detect(model, image, conf: float, imgsz: int = 1280, classes=None):
    """(x, y, score) for every detection on a whole frame."""
    result = model(image, verbose=False, conf=conf, imgsz=imgsz,
                   classes=classes)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    return [((x0 + x1) / 2.0, (y0 + y1) / 2.0, float(s))
            for (x0, y0, x1, y1), s in zip(boxes, scores)]


def evaluate(model, clicks, clips, classes=None, label=""):
    """Recall and false detections on clicked frames, per confidence."""
    import cv2

    lowest = min(CONFS)
    found = []      # (ball visible, [(distance or None, score)])
    for clip in clips:
        info = json.loads((Path(f"output_{clip}") / "clip.json").read_text())
        cap = cv2.VideoCapture(info["path"])
        for f in sorted(f for c, f in clicks if c == clip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, image = cap.read()
            if not ok:
                continue
            ball = clicks[(clip, f)]
            dets = detect(model, image, lowest, classes=classes)
            found.append((ball is not None,
                          [(None if ball is None else
                            float(np.hypot(x - ball[0], y - ball[1])), s)
                           for x, y, s in dets]))
        cap.release()
    visible = sum(v for v, _ in found)
    rows = {}
    for conf in CONFS:
        hits = sum(1 for v, d in found if v and any(
            g is not None and g <= HIT_PX and s >= conf for g, s in d))
        false = sum(1 for _, d in found for g, s in d
                    if s >= conf and (g is None or g > MISS_PX))
        rows[conf] = (hits, false)
        print(f"  {label:>28s}  conf {conf:.2f}: ball found {hits:3d}/"
              f"{visible}, false detections {false:4d} on {len(found)} "
              f"frames", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roboflow", required=True)
    ap.add_argument("--clicks", nargs="+", required=True)
    ap.add_argument("--fold", choices=["stoke", "reading", "all"],
                    default="all",
                    help="the match held out for testing; 'all' runs both")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--baseline", action="store_true",
                    help="also score the COCO detector on the same frames")
    args = ap.parse_args()
    from ultralytics import YOLO

    clicks = load_clicks(args.clicks)
    folds = ["stoke", "reading"] if args.fold == "all" else [args.fold]
    for held_out in folds:
        train_clips = [c for m, cs in MATCHES.items() if m != held_out
                       for c in cs]
        test_clips = MATCHES[held_out]
        root = CACHE_DIR / "ball_detector" / f"without_{held_out}"
        print(f"\n  fold: train on {', '.join(train_clips)}; "
              f"test on {held_out}", flush=True)
        if args.baseline:
            evaluate(YOLO("yolov8m.pt"), clicks, test_clips, classes=[32],
                     label="COCO yolov8m")
        yaml = build(root / "data", Path(args.roboflow), clicks, train_clips)
        model = YOLO(args.model)
        model.train(data=str(yaml), imgsz=TILE, epochs=args.epochs,
                    batch=16, device="cpu", workers=2, project=str(root),
                    name="run", exist_ok=True, verbose=False, plots=False,
                    patience=15, seed=0)
        best = YOLO(str(root / "run" / "weights" / "best.pt"))
        evaluate(best, clicks, test_clips, label=f"fine-tuned {args.model}")


if __name__ == "__main__":
    main()
