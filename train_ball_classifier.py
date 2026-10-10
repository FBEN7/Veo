"""Train the ball classifier, and test it on matches it never saw.

Positives are the hand-labelled balls of the Roboflow football-players
dataset (CC BY 4.0). Negatives are what actually fools the detector: every
`sports ball` candidate yolov8m proposes in the same images, at the lowest
confidence the clips use, that is not within `MATCH_PX` of a labelled ball.

The images come from 15 source matches, and Roboflow's own split shares
matches between train and test, so it is not used. Whole matches are held
out instead (`HELD_OUT`): the clips this classifier is for are different
matches again, and a test on frames from a training match would flatter it.

    python train_ball_classifier.py --data <roboflow export dir>
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from src.ball_classifier import WEIGHTS, crop, network, to_tensor

HELD_OUT = {"08fd33", "4b770a", "2e57b9"}
MATCH_PX = 12.0
CONF = 0.05


def source(path: str) -> str:
    return Path(path).name.split("_")[0]


def mine(data: Path, cache: Path):
    """Crops and labels from every image, cached as an .npz."""
    if cache.exists():
        got = np.load(cache)
        return got["x"], got["y"], got["src"]
    import cv2
    from ultralytics import YOLO

    model = YOLO("yolov8m.pt")
    xs, ys, srcs = [], [], []
    images = sorted(glob.glob(str(data / "*" / "images" / "*.jpg")))
    for k, path in enumerate(images):
        image = cv2.imread(path)
        h, w = image.shape[:2]
        label = Path(path).parent.parent / "labels" / (Path(path).stem + ".txt")
        balls = []
        if label.exists():
            for line in label.read_text().split("\n"):
                parts = line.split()
                if len(parts) == 5 and parts[0] == "0":
                    _, cx, cy, bw, bh = map(float, parts)
                    balls.append((cx * w, cy * h, max(bw * w, bh * h)))
        for cx, cy, size in balls:
            patch = crop(image, cx, cy, size)
            if patch is not None:
                xs.append(patch); ys.append(1); srcs.append(source(path))
        result = model(image, verbose=False, conf=CONF, imgsz=1280,
                       classes=[32])[0]
        for (x0, y0, x1, y1) in result.boxes.xyxy.cpu().numpy():
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            if any(np.hypot(cx - bx, cy - by) <= MATCH_PX
                   for bx, by, _ in balls):
                continue
            patch = crop(image, cx, cy, max(x1 - x0, y1 - y0))
            if patch is not None:
                xs.append(patch); ys.append(0); srcs.append(source(path))
        if k % 100 == 0:
            print(f"  mined {k}/{len(images)} images, {sum(ys)} balls, "
                  f"{len(ys) - sum(ys)} look-alikes", flush=True)
    x, y, src = np.stack(xs), np.array(ys), np.array(srcs)
    np.savez_compressed(cache, x=x, y=y, src=src)
    return x, y, src


# A click and a stored candidate this close, in pixels, are the same ball;
# candidates further than `NEGATIVE_PX` from the click are look-alikes.
CLICK_MATCH_PX = 12.0
NEGATIVE_PX = 15.0


def mine_clip_labels(labels: Path, cache: Path):
    """Crops from hand-clicked balls on the clips, and their look-alikes.

    A click is a ball, sized by the stored candidate it lands on or, where
    the detector missed it, by the clip's median ball box. Stored
    candidates on the same frame away from the click are look-alikes.
    Frames marked "not visible" give nothing: "not sure" is not a negative.
    """
    if cache.exists():
        got = np.load(cache)
        return got["x"], got["y"], got["src"]
    import cv2
    import pandas as pd

    rows = json.loads(labels.read_text())["frames"]
    xs, ys, srcs = [], [], []
    for clip in sorted({r["clip"] for r in rows}):
        out = Path(f"output_{clip}")
        info = json.loads((out / "clip.json").read_text())
        balls = pd.read_parquet(out / "tracks.parquet")
        balls = balls[balls.cls == "ball"]
        typical = float(balls.crop_h.median())
        cap = cv2.VideoCapture(info["path"])
        for r in sorted((r for r in rows if r["clip"] == clip
                         and r.get("ball")), key=lambda r: r["frame"]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
            ok, image = cap.read()
            if not ok:
                continue
            cx, cy = r["ball"][0] * r["width"], r["ball"][1] * r["height"]
            here = balls[balls.frame == r["frame"]]
            gap = np.hypot(here.px.to_numpy() - cx, here.py.to_numpy() - cy)
            size = (float(here.crop_h.iloc[int(np.argmin(gap))])
                    if len(gap) and gap.min() <= CLICK_MATCH_PX else typical)
            patch = crop(image, cx, cy, size)
            if patch is not None:
                xs.append(patch); ys.append(1); srcs.append(clip)
            for row, g in zip(here.itertuples(), gap):
                if g <= NEGATIVE_PX:
                    continue
                patch = crop(image, float(row.px), float(row.py),
                             float(row.crop_h))
                if patch is not None:
                    xs.append(patch); ys.append(0); srcs.append(clip)
        cap.release()
    x, y, src = np.stack(xs), np.array(ys), np.array(srcs)
    np.savez_compressed(cache, x=x, y=y, src=src)
    return x, y, src


def train(x, y, epochs: int = 40, seed: int = 0):
    import torch

    torch.manual_seed(seed)
    model = network()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    # The classes are unbalanced; weight positives to match.
    pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(y))
        for start in range(0, len(order), 128):
            idx = order[start:start + 128]
            batch = x[idx].copy()
            flip = rng.random(len(idx)) < 0.5
            batch[flip] = batch[flip, :, ::-1]
            scale = rng.uniform(0.8, 1.2, (len(idx), 1, 1, 1))
            batch = np.clip(batch * scale, 0, 255).astype(np.uint8)
            logits = model(to_tensor(list(batch))).squeeze(1)
            loss = loss_fn(logits, torch.tensor(y[idx], dtype=torch.float32))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def evaluate(model, x, y):
    import torch
    from sklearn.metrics import roc_auc_score

    with torch.no_grad():
        p = torch.sigmoid(model(to_tensor(list(x))).squeeze(1)).numpy()
    auc = roc_auc_score(y, p)
    at = {}
    for t in (0.3, 0.5, 0.7):
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        at[t] = (tp / max(pred.sum(), 1), tp / max((y == 1).sum(), 1))
    return auc, at, p


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", default=".cache/ball_crops.npz")
    ap.add_argument("--clip-labels",
                    help="ball_labels.json from make_ball_labeller.py: adds "
                         "the clips' own balls, and scores each clip with a "
                         "model that never saw its labels")
    args = ap.parse_args()
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)

    x, y, src = mine(Path(args.data), Path(args.cache))
    if args.clip_labels:
        leave_one_clip_out(x, y, Path(args.clip_labels),
                           Path(args.cache).with_name("clip_crops.npz"))
        return
    test = np.isin(src, list(HELD_OUT))
    print(f"\n  {int(y.sum())} balls and {int((y == 0).sum())} look-alikes "
          f"from {len(set(src))} matches; held out "
          f"{', '.join(sorted(HELD_OUT))}: {int(y[test].sum())} balls, "
          f"{int((y[test] == 0).sum())} look-alikes\n")

    model = train(x[~test], y[~test])
    auc, at, _ = evaluate(model, x[test], y[test])
    print(f"  held-out matches: AUC {auc:.3f}")
    for t, (prec, rec) in at.items():
        print(f"    p >= {t:.1f}: precision {prec:.2f}, recall {rec:.2f}")

    final = train(x, y)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final.state_dict(), WEIGHTS)
    (WEIGHTS.with_suffix(".json")).write_text(json.dumps({
        "held_out": sorted(HELD_OUT), "held_out_auc": round(float(auc), 3),
        "balls": int(y.sum()), "look_alikes": int((y == 0).sum()),
        "data": "Roboflow football-players-detection v1 (CC BY 4.0)"},
        indent=1))
    print(f"\n  trained on all {len(y)} crops -> {WEIGHTS}")


def leave_one_clip_out(pub_x, pub_y, labels: Path, cache: Path):
    """Train without each clip's labels, test and score that clip with it.

    The clips are then scored by models that never saw their answers, so
    the event measurement on them stays a test.
    """
    import torch

    from src.ball_classifier import BallClassifier, score_clip

    cx, cy, csrc = mine_clip_labels(labels, cache)
    print(f"\n  clip labels: {int(cy.sum())} balls and "
          f"{int((cy == 0).sum())} look-alikes on {len(set(csrc))} clips\n")
    print(f"  {'held-out clip':>14s} {'balls':>6s} {'look-alikes':>12s} "
          f"{'AUC public only':>16s} {'AUC + other clips':>18s}")
    public = train(pub_x, pub_y)
    for clip in sorted(set(csrc)):
        test = csrc == clip
        model = train(np.concatenate([pub_x, cx[~test]]),
                      np.concatenate([pub_y, cy[~test]]))
        before = evaluate(public, cx[test], cy[test])[0]
        after = evaluate(model, cx[test], cy[test])[0]
        print(f"  {clip:>14s} {int(cy[test].sum()):6d} "
              f"{int((cy[test] == 0).sum()):12d} {before:16.3f} "
              f"{after:18.3f}", flush=True)
        # Kept, so the clip can be rescored without retraining.
        torch.save(model.state_dict(), cache.with_name(f"loco_{clip}.pt"))
        out = Path(f"output_{clip}")
        info = json.loads((out / "clip.json").read_text())
        score_clip(out, info, verbose=False,
                   classifier=BallClassifier(model=model), overwrite=True)


if __name__ == "__main__":
    main()
