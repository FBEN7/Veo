"""Verified ball labels, step 1b: a tile classifier that replaces the blind
judges once it agrees with them.

Input per candidate tile (window, frame, u, v, sources): a 64 x 64 crop
centred on (u, v) at frame t (RGB) and four grey differences, to t-2 and
t+2 (motion: a ball in play moves, a boot moves with a leg) and to t-50
and t+50 (two seconds: a spare ball or a mark does not move at all), plus
which sources proposed it (v2, v1, COCO, the tracker, the pass line) and
where it is (inside the pitch outline or how far out). Target: the judges'
final label B (the match ball in play centred in the 24 px box) against
anything else.

    python train_tile_classifier.py crops --tiles TILES.parquet --out CROPS.npz
    python train_tile_classifier.py label --crops NEW.npz --judged J.parquet ... --out NEWJ.npz
    python train_tile_classifier.py cv --crops CROPS.npz [MORE.npz ...] --out CV.json \\
        [--by window|match] [--groups G ...]                    # leave one group out
    python train_tile_classifier.py fit --crops CROPS.npz [MORE.npz ...] --out MODEL.pt
    python train_tile_classifier.py apply --model MODEL.pt --crops NEW.npz --out PRED.parquet

`crops` decodes each window once (its frames needed for every tile: t-50,
t-2, t, t+2, t+50; an offset past the window's ends repeats frame t, i.e.
no motion). `cv` trains on all windows but one and scores the held-out
one, for every window: agreement with the judges, precision and recall of
B, and the share of tiles decided at confident thresholds (p >= HI: ball,
p <= LO: not; between: left to a judge). Training is on CPU (2 threads),
seeded; only the summary is printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.paths import DATA_DIR

HALF = 32
OFFSETS = (-50, -2, 0, 2, 50)
SOURCES = ("v2", "v1", "coco", "T", "line")
HI, LO = 0.95, 0.1
ONE_BALL_PX = 24       # tiles within this of a frame's top tile (both axes) may be its ball
EPOCHS, BATCH, LR, SEED = 25, 64, 1e-3, 0
BRIGHT = (0.8, 1.2)    # brightness augmentation range


# ---- Crops ---------------------------------------------------------------------------

def _crop(img, u, v):
    pad = cv2.copyMakeBorder(img, HALF, HALF, HALF, HALF, cv2.BORDER_REFLECT)
    x, y = int(round(u)) + HALF, int(round(v)) + HALF
    return pad[y - HALF:y + HALF, x - HALF:x + HALF]


def extract(tiles: pd.DataFrame, log=print) -> dict:
    """Crops for every tile: rgb (n, 64, 64, 3) uint8 at t and grey (n, 5,
    64, 64) uint8 at the OFFSETS; plus meta (n, k) float32."""
    from src import soccertrack_v2 as st

    n = len(tiles)
    rgb = np.zeros((n, 2 * HALF, 2 * HALF, 3), np.uint8)
    grey = np.zeros((n, len(OFFSETS), 2 * HALF, 2 * HALF), np.uint8)
    have = np.zeros((n, len(OFFSETS)), bool)
    meta = np.zeros((n, len(SOURCES) + 3), np.float32)
    tiles = tiles.reset_index(drop=True)
    for w, g in tiles.groupby("window"):
        info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
        cal = st.calibration(info["match"])
        poly = np.asarray(info["pitch"], np.float32).reshape(-1, 1, 2)
        last = int(info["frames"]) - 1
        need = {}
        for i, r in g.iterrows():
            for k, o in enumerate(OFFSETS):
                f = int(np.clip(r.frame + o, 0, last))
                need.setdefault(f, []).append((i, k))
            src = set(str(r.sources).split("+"))
            meta[i, :len(SOURCES)] = [s in src for s in SOURCES]
            d = cv2.pointPolygonTest(poly, (float(r.u), float(r.v)), True)
            meta[i, len(SOURCES)] = np.clip(d / 100.0, -2, 2)      # px inside (+) / out (-)
            gp = st.image_to_pitch([[float(r.u), float(r.v)]], cal)[0]
            meta[i, len(SOURCES) + 1:] = (np.nan_to_num(gp, nan=0.0) /
                                          np.array([105.0, 68.0])).clip(-2, 2)
        cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{w}.mp4"))
        idx, stop = 0, max(need)
        while idx <= stop:
            ok, img = cap.read()
            if not ok:
                break
            if idx in need:
                gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                for i, k in need[idx]:
                    r = tiles.loc[i]
                    grey[i, k] = _crop(gimg, r.u, r.v)
                    have[i, k] = True
                    if OFFSETS[k] == 0:
                        rgb[i] = _crop(img, r.u, r.v)[:, :, ::-1]
            idx += 1
        cap.release()
        log(f"  {w}: {len(g)} tiles, decoded {idx} frames")
    if not have.all():
        raise SystemExit(f"{(~have).sum()} crops missing (frames past the video's end)")
    return {"rgb": rgb, "grey": grey, "meta": meta}


# ---- Model ---------------------------------------------------------------------------

def inputs(rgb, grey):
    """(n, 7, 64, 64) float32: RGB at t in [0, 1] and the four grey
    differences t - offset, scaled."""
    x = rgb.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    g = grey.astype(np.float32) / 255.0
    t = g[:, OFFSETS.index(0)]
    diffs = np.stack([(t - g[:, k]) * 4.0 for k, o in enumerate(OFFSETS) if o != 0], 1)
    return np.concatenate([x, diffs], 1)


def build(n_meta: int):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            def block(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                     nn.ReLU(), nn.Conv2d(o, o, 3, padding=1),
                                     nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
            self.conv = nn.Sequential(block(7, 32), block(32, 64), block(64, 96),
                                      block(96, 128))           # 64 -> 4
            self.head = nn.Sequential(nn.Linear(128 * 2 * 2 + 128 + n_meta, 128), nn.ReLU(),
                                      nn.Dropout(0.3), nn.Linear(128, 1))

        def forward(self, x, m):
            f = self.conv(x)
            c = f[:, :, 1:3, 1:3].flatten(1)                     # the centre (the box)
            p = f.mean((2, 3))                                   # the context
            return self.head(torch.cat([c, p, m], 1)).squeeze(1)

    return Net()


def augment(x, rng, bright=BRIGHT):
    """Flips (left-right; the pitch is not symmetric top-bottom in view,
    so no vertical flip), shifts of up to 3 px, brightness."""
    if rng.random() < 0.5:
        x = x[:, :, ::-1]
    dx, dy = rng.integers(-3, 4, 2)
    x = np.roll(x, (dy, dx), (1, 2))
    x = x.copy()
    x[:3] = np.clip(x[:3] * rng.uniform(*bright), 0, 1)
    return x


def train(X, M, y, epochs=EPOCHS, seed=SEED, log=None, bright=BRIGHT, threads=2):
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    net = build(M.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pos = max(y.mean(), 1e-3)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(y))
        tot = 0.0
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            xb = torch.from_numpy(np.stack([augment(X[i], rng, bright) for i in b]))
            mb, yb = torch.from_numpy(M[b]), torch.from_numpy(y[b].astype(np.float32))
            loss = lossf(net(xb, mb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        if log:
            log(f"    epoch {ep + 1}: loss {tot / len(y):.4f}")
    return net


def predict(net, X, M) -> np.ndarray:
    import torch

    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), 256):
            out.append(torch.sigmoid(net(torch.from_numpy(X[s:s + 256]),
                                         torch.from_numpy(M[s:s + 256]))).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def scores(y, p, hi=HI, lo=LO) -> dict:
    """Agreement with the judges at 0.5, precision / recall of ball, and at
    the confident thresholds the share decided and its agreement."""
    yb, pb = y.astype(bool), p >= 0.5
    tp, fp, fn = int((yb & pb).sum()), int((~yb & pb).sum()), int((yb & ~pb).sum())
    conf = (p >= hi) | (p <= lo)
    return {"tiles": int(len(y)), "judged ball": int(yb.sum()),
            "agree at 0.5": round(float((yb == pb).mean()), 4),
            "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4),
            "confident share": round(float(conf.mean()), 4),
            "agree when confident": round(float((yb[conf] == (p[conf] >= hi)).mean()), 4)
            if conf.any() else None,
            "confident ball precision": round(float(yb[p >= hi].mean()), 4)
            if (p >= hi).any() else None,
            "confident not-ball NPV": round(float((~yb[p <= lo]).mean()), 4)
            if (p <= lo).any() else None}


# ---- Commands ------------------------------------------------------------------------

def auto_labels(t: pd.DataFrame, hi=HI, lo=LO) -> np.ndarray:
    """"B" where p >= hi and the tile is its frame's top tile (or within
    ONE_BALL_PX of it on both axes: one ball can sit in both boxes);
    "not" where p <= lo, or p < hi away from the top tile; else "judge".
    On M6's leave-one-window-out predictions: ball precision 0.96, 93% of
    tiles decided, agreement with the judges 0.963."""
    key = [t.window.to_numpy(), t.frame.to_numpy()]
    top = t.loc[t.groupby(["window", "frame"]).p.idxmax(), ["window", "frame", "u", "v"]]
    top = top.set_index(["window", "frame"])
    tu = top.u.reindex(pd.MultiIndex.from_arrays(key)).to_numpy()
    tv = top.v.reindex(pd.MultiIndex.from_arrays(key)).to_numpy()
    near = (np.abs(t.u.to_numpy() - tu) <= ONE_BALL_PX) & (np.abs(t.v.to_numpy() - tv) <= ONE_BALL_PX)
    p = t.p.to_numpy()
    pos = (p >= hi) & near
    neg = ~pos & ((p <= lo) | ((p < hi) & ~near))
    return np.where(pos, "B", np.where(neg, "not", "judge"))


def cmd_crops(args):
    tiles = pd.read_parquet(args.tiles)
    tiles = tiles[tiles.kind != "decoy"] if "kind" in tiles else tiles
    data = extract(tiles)
    keep = ["window", "frame", "u", "v", "sources"] + [c for c in ("ball", "label", "match",
                                                                    "chunk", "tile", "batch")
                                                         if c in tiles]
    np.savez_compressed(args.out, **data, table=tiles[keep].to_json(orient="records"))
    print(f"  {len(tiles)} tiles -> {args.out}")


def _load(paths, drop=()):
    """Inputs, meta and table of one or several crop files, concatenated;
    the source flags in `drop` set to 0 (sources not run for every window,
    e.g. the tracker, must not decide the label)."""
    Xs, Ms, ts = [], [], []
    for path in [paths] if isinstance(paths, (str, Path)) else paths:
        z = np.load(path, allow_pickle=False)
        ts.append(pd.DataFrame(json.loads(str(z["table"]))).assign(crops=Path(path).name))
        Xs.append(inputs(z["rgb"], z["grey"]))
        Ms.append(z["meta"].astype(np.float32))
    t = pd.concat(ts, ignore_index=True)
    t["match"] = t.window.str.split("_").str[1]
    M = np.concatenate(Ms)
    for s in drop:
        M[:, SOURCES.index(s)] = 0.0
    return np.concatenate(Xs), M, t


def _uid(d: pd.DataFrame) -> pd.Series:
    return (d.window.astype(str) + "/" + d.frame.astype(int).astype(str) + "/"
            + d.u.astype(float).round(1).astype(str) + "/" + d.v.astype(float).round(1).astype(str))


def cmd_label(args):
    """The crops of judged tiles (from `spot_check_tiles.py collect`) with
    their labels, out of a crop file of unlabelled candidates."""
    z = np.load(args.crops, allow_pickle=False)
    table = pd.DataFrame(json.loads(str(z["table"])))
    j = pd.concat([pd.read_parquet(p) for p in args.judged], ignore_index=True)
    j = j.assign(uid=_uid(j)).drop_duplicates("uid", keep="last").set_index("uid")
    uid = _uid(table)
    keep = uid.isin(j.index).to_numpy()
    if keep.sum() < len(j):
        raise SystemExit(f"{len(j) - keep.sum()} judged tiles not among the crops")
    t = table[keep].reset_index(drop=True)
    jj = j.loc[uid[keep].to_numpy()]
    t["label"], t["ball"] = jj.label.to_numpy(), jj.ball.to_numpy(bool)
    t["stratum"] = jj.stratum.fillna("spot").to_numpy() if "stratum" in jj else "spot"
    np.savez_compressed(args.out, rgb=z["rgb"][keep], grey=z["grey"][keep], meta=z["meta"][keep],
                        table=t.to_json(orient="records"))
    print(f"  {len(t)} judged tiles ({int(t.ball.sum())} ball) -> {args.out}")


def cmd_cv(args):
    """Leave one group (window or match) out; each fold's predictions go to
    FOLDS/<group>.parquet and are reused when present (a restart loses at
    most the fold in progress); `--groups` runs only those folds."""
    X, M, t = _load(args.crops, args.drop_meta)
    y = t.ball.to_numpy(bool)
    folds = Path(args.folds or Path(args.out).with_suffix(".folds"))
    folds.mkdir(parents=True, exist_ok=True)
    groups = sorted(t[args.by].unique())
    for g in args.groups or groups:
        f = folds / f"{g}.parquet"
        if f.exists():
            continue
        te = (t[args.by] == g).to_numpy()
        net = train(X[~te], M[~te], y[~te], epochs=args.epochs, bright=args.bright,
                    threads=args.threads)
        t[te].assign(p=predict(net, X[te], M[te])).to_parquet(f.with_suffix(".part"))
        f.with_suffix(".part").rename(f)
        print(f"  held out {g}: {scores(y[te], pd.read_parquet(f).p.to_numpy())}", flush=True)
    if args.groups or not all((folds / f"{g}.parquet").exists() for g in groups):
        return
    p = pd.concat([pd.read_parquet(folds / f"{g}.parquet") for g in groups])
    per = {g: scores(d.ball.to_numpy(bool), d.p.to_numpy()) for g, d in p.groupby(args.by)}
    out = {"held out by": args.by, "drop meta": args.drop_meta, "bright": args.bright,
           f"per {args.by}": per,
           "pooled": scores(p.ball.to_numpy(bool), p.p.to_numpy())}
    print(json.dumps(out["pooled"], indent=1))
    Path(args.out).write_text(json.dumps(out, indent=1))
    p.to_parquet(Path(args.out).with_suffix(".parquet"))


def cmd_fit(args):
    import torch

    X, M, t = _load(args.crops, args.drop_meta)
    net = train(X, M, t.ball.to_numpy(bool), epochs=args.epochs, bright=args.bright,
                threads=args.threads)
    torch.save({"state": net.state_dict(), "n_meta": M.shape[1], "hi": HI, "lo": LO,
                "drop meta": list(args.drop_meta), "bright": list(args.bright),
                "trained on": sorted(t.window.unique())}, args.out)
    print(f"  model -> {args.out}")


def cmd_apply(args):
    import torch

    ck = torch.load(args.model, map_location="cpu")
    net = build(ck["n_meta"])
    net.load_state_dict(ck["state"])
    X, M, t = _load(args.crops, ck.get("drop meta", ()))
    p = predict(net, X, M)
    t = t.assign(p=p)
    t["auto"] = auto_labels(t, ck["hi"], ck["lo"])
    t.to_parquet(args.out)
    print(f"  {len(t)} tiles: {t.auto.value_counts().to_dict()} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crops")
    c.add_argument("--tiles", required=True)
    c.add_argument("--out", required=True)
    c = sub.add_parser("label")
    c.add_argument("--crops", required=True)
    c.add_argument("--judged", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c = sub.add_parser("cv")
    c.add_argument("--crops", nargs="+", required=True)
    c.add_argument("--epochs", type=int, default=EPOCHS)
    c.add_argument("--drop-meta", nargs="*", choices=SOURCES, default=[])
    c.add_argument("--bright", nargs=2, type=float, default=list(BRIGHT))
    c.add_argument("--threads", type=int, default=2)
    c.add_argument("--by", choices=("window", "match"), default="window")
    c.add_argument("--groups", nargs="*")
    c.add_argument("--folds")
    c.add_argument("--out", required=True)
    c = sub.add_parser("fit")
    c.add_argument("--crops", nargs="+", required=True)
    c.add_argument("--epochs", type=int, default=EPOCHS)
    c.add_argument("--drop-meta", nargs="*", choices=SOURCES, default=[])
    c.add_argument("--bright", nargs=2, type=float, default=list(BRIGHT))
    c.add_argument("--threads", type=int, default=2)
    c.add_argument("--out", required=True)
    c = sub.add_parser("apply")
    c.add_argument("--model", required=True)
    c.add_argument("--crops", required=True)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    {"crops": cmd_crops, "label": cmd_label, "cv": cmd_cv, "fit": cmd_fit,
     "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
