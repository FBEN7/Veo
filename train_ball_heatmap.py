"""Train the panorama ball detector (`src/ball_heatmap.py`) on weak labels
(`build_ball_patches.py`, then `build_ball_labels.py`).

Each patch has a code per 4 x 4 cell (1 the ball is somewhere here, 0
ignore, -1 not the ball, -3 not the ball and easily taken for it).
Multiple-instance loss: per positive bag, a smooth maximum of the logits
over its region (log-mean-exp, r = 5) should be high; negative cells low,
by focal loss (gamma 2), hard ones weighted 3, summed over the batch and
divided by its number of bags. Between-action bags carry no negatives and
half the weight. From step 1000 the 15% action bags and 25% between bags
with the highest loss in a batch are dropped -- a hidden ball makes a bag
wrong, and those are the hardest.

Batches of 16: 6 action bags, 2 between, 8 ball-free patches. Flips (the
codes too), brightness, contrast, saturation, blur, JPEG re-encoding. The
stem and layer1 are frozen for the first 800 steps. Checkpoints at 1500,
2250 and 3000 steps (`<out>.<step>.pt`), chosen afterwards on tuning
windows by `eval_ball_detector.py`.

    python train_ball_heatmap.py --patches .cache/soccertrack_v2/ball_patches \\
        --out .cache/ball_heatmap

Version 2 (`--strips`; `build_ball_strips.py`, then `label_ball_strips.py`)
trains the motion model (`build(motion=True)`; `--no-motion` is the
RGB-only fallback, the same net without the motion stem) on 320 px strips
[t-2 | t | t+2 | background B_t], codes on the t crop. Per strip, summed
over the batch and divided by its 16 strips:

- a flight bag (the stretch of a pass line where the ball should be):
  BCE to 1 of z = (1/8) log((1/4) sum exp(8 z)), a smooth maximum
  normalised by the ball's footprint (4 cells), not by the bag's size: a
  ball of 4 cells at z scores z, one cell z - 0.17, a glow over a whole
  1,000-cell bag only 0.7 more than its level -- a peak pays, a corridor
  does not;
- an action or dribble bag: v1's log-mean-exp (r = 5); a dribble bag is
  positive only, at weight 0.5 (`labels.parquet`'s weight);
- negatives: v1's focal term, x 3 on -3 cells;
- peak suppression: BCE to 0 of the 16 highest 5 x 5 local maxima among
  the -1 / -3 cells, 1/16 each.

From step 1200 the 25% flight, 15% action and 25% dribble bags of a batch
with the highest loss are dropped (rounded at random, so the shares hold
on average with 1-5 bags of a kind in a batch).

Batches: 5 flight, 3 action, 1 dribble, 1 lofted, 6 free (2 landmark, 2
cluster, 1 boundary, 1 random; a category without strips is drawn from
all free strips). Augmentation is the same on the four crops: flip (the
codes too), v1's brightness and contrast, saturation x 0.8-1.2 as a blend
with the grey image (luminance, all the motion channels see, is kept:
the background is stored grey), scale x 0.8-1.25 (a whole number of
cells, the codes nearest; cropped back to 320 px keeping the bag where it
fits, or padded by reflection with codes 0), blur (p 0.2). Then each crop
is re-encoded as JPEG at one quality (p 0.3, q 80-95), and each of t-2, t,
t+2 gets its own luminance noise (sigma 0-2), so the differences see
codec-like noise. Time is reversed with p 0.5 (t-2 and t+2 swap) and the
motion dropped with p 0.1 (zero motion, as at a window's edge). The motion
channels are `motion_bg.motion_channels`: zero where the strip has no
t-2 / t+2 (`has_prev`, `has_next`) or its background has fewer than 8
samples (`n_bg`). Each strip's draws come from its own generator, and the
t crop's do not depend on the other crops, so `--motion` and `--no-motion`
see the same batches and the same t crops. The next batch is prepared in
a thread while a step runs.

AdamW (weight decay 1e-4): head and lateral 3e-4, `conv1_motion` 3e-4 from
step 1, layer2 1e-4, the RGB conv1, bn1 and layer1 frozen to step 800 then
3e-5; warm-up 100 steps, cosine to 0. Every 250 steps the held-out strips
(`labels.parquet`'s holdout) are scored: bag hit (the map's top cell is in
the bag), peakiness (bag cells with p > 0.5; above 30 from step 1500 the
run aborts, exit status 3), negative peaks with p > 0.15 per strip, and
the motion weights' norm (it must grow). Checkpoints at 2500, 3250 and
4000 steps, each with `<ckpt>.provenance.json` (`ball_heatmap.
write_provenance`): the matches and halves trained on, the ranges left
out, the label constants (with the labels' COCO flag and seed), the
motion sigmas (inference scales by these: `ball_heatmap.load_v2`), the
recipe hash (this file and `label_ball_strips.py`) and the seed. Only strips of
`ball_heatmap.V2_TRAIN` less `V2_EXCLUDE` are accepted, and only those cut
from the source stream are kept (window strips dropped and counted;
`--allow-window-strips` for smoke runs). A run refuses to start on the
placeholder motion scales: `ball_heatmap.SIGMA_D` / `SIGMA_F` must be the
measured ones, with `SIGMA_MEASURED = True` (`--placeholder-sigma` for
smoke runs; `--bench` and `--check` do not ask).

    python train_ball_heatmap.py --strips .cache/soccertrack_v2/ball_strips \\
        --motion --steps 4000 --out .cache/ball_v2
    python train_ball_heatmap.py --strips DIR --motion --bench 20 [--bench-at 801]
    python train_ball_heatmap.py --check

`--bench N` times N steps from step `--bench-at` of the schedule (1: the
stem frozen; 801: all trainable), saving nothing; with `--motion` it
alternates them with the RGB-only step on the same batches and prints the
ratio (the go/no-go: at most 1.25).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.ball_heatmap import build, to_tensor

MIX = {"action": 6, "between": 2, "negative": 8}
CHECKPOINTS = (1500, 2250, 3000)


def augment(image, codes, rng):
    if rng.random() < 0.5:
        image, codes = image[:, ::-1], codes[:, ::-1]
    im = image.astype(np.float32)
    im = (im - im.mean()) * rng.uniform(0.75, 1.25) + im.mean() * rng.uniform(0.75, 1.25)
    hsv = cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32)
                           * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if rng.random() < 0.2:
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.3:
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY,
                                            int(rng.integers(60, 96))])
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(im), np.ascontiguousarray(codes)


def bag_losses(logits, codes, positive_only):
    """Per patch: the bag loss (nan without a bag) and the negative loss."""
    import torch

    F = torch.nn.functional
    pos, neg = [], []
    for lg, c, po in zip(logits, codes, positive_only):
        inside = c == 1
        if inside.any():
            z = torch.logsumexp(5 * lg[inside], 0) / 5 - np.log(
                float(inside.sum())) / 5
            pos.append(F.binary_cross_entropy_with_logits(
                z, torch.ones_like(z)))
        else:
            pos.append(torch.tensor(float("nan")))
        if po:
            neg.append(torch.zeros(()))
            continue
        w = torch.where(c == -3, 3.0, torch.where(c == -1, 1.0, 0.0))
        p = torch.sigmoid(lg)
        bce = F.binary_cross_entropy_with_logits(lg, torch.zeros_like(lg),
                                                 reduction="none")
        neg.append((bce * p ** 2 * w).sum())
    return torch.stack(pos), torch.stack(neg)


def train_v1(args):
    """v1: weakly labelled patches (`--patches`), as before."""
    import torch

    torch.manual_seed(0)
    random.seed(0)
    rng = np.random.default_rng(0)
    root = Path(args.patches)
    table = pd.read_parquet(root / "labels.parquet")
    codes_all = np.load(root / "labels_codes.npy")
    table["match"] = table.match.astype(str)
    keep = ~table.match.isin(args.exclude_matches)
    table, codes_all = table[keep].reset_index(drop=True), codes_all[keep.to_numpy()]
    pools = {k: np.flatnonzero(table.kind.to_numpy() == k) for k in MIX}
    print(f"  {sorted(table.match.unique())}: " + ", ".join(
        f"{k} {len(v)}" for k, v in pools.items()), flush=True)
    torch.set_num_threads(4)
    model = build()
    early = list(model.stem.parameters()) + list(model.layer1.parameters())
    for p in early:
        p.requires_grad_(False)
    opt = torch.optim.AdamW([
        {"params": list(model.lat.parameters()) + list(model.head.parameters()),
         "lr": 3e-4},
        {"params": model.layer2.parameters(), "lr": 1e-4},
        {"params": early, "lr": 3e-5}], weight_decay=1e-4)
    base = [g["lr"] for g in opt.param_groups]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if step == 801:
            for p in early:
                p.requires_grad_(True)
        scale = min(1.0, step / 100) * 0.5 * (1 + np.cos(np.pi * step / args.steps))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * scale
        model.train()
        idx = np.concatenate([rng.choice(pools[k], n) for k, n in MIX.items()])
        ims, cs = [], []
        for i in idx:
            im, c = augment(cv2.imread(str(root / table.file[i])),
                            codes_all[i], rng)
            ims.append(im)
            cs.append(torch.from_numpy(c.astype(np.int64)))
        logits = model(to_tensor(ims))
        pos, neg = bag_losses(logits, cs, table.positive_only.to_numpy()[idx])
        kinds = table.kind.to_numpy()[idx]
        w = torch.ones(len(idx))
        w[torch.from_numpy(kinds == "between")] = 0.5
        has = ~torch.isnan(pos)
        if step > 1000:
            for kind, share in (("action", 0.15), ("between", 0.25)):
                m = torch.from_numpy(kinds == kind) & has
                n = int(round(share * int(m.sum())))
                if n:
                    worst = torch.topk(torch.where(m, pos.detach(),
                                                   torch.tensor(-1.0)), n).indices
                    has[worst] = False
        n_bags = max(int(has.sum()), 1)
        loss = ((pos[has] * w[has]).sum() + neg.sum()) / n_bags
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  step {step}: loss {loss.item():.3f}  "
                  f"[{time.time() - t0:.0f} s]", flush=True)
        if step in CHECKPOINTS or step == args.steps:
            torch.save(model.state_dict(), f"{args.out}.{step}.pt")
            print(f"  saved {args.out}.{step}.pt", flush=True)


# ---- v2: strips, with motion ------------------------------------------------------

SIZE, STRIDE = 320, 4
CELLS = SIZE // STRIDE
MIX_V2 = (("flight", None, 5), ("action", None, 3), ("dribble", None, 1),
          ("lofted", None, 1), ("free", "landmark", 2), ("free", "cluster", 2),
          ("free", "boundary", 1), ("free", "random", 1))
STEPS_V2 = 4000
CHECKPOINTS_V2 = (2500, 3250, 4000)
LR_V2 = {"head": 3e-4, "motion": 3e-4, "layer2": 1e-4, "early": 3e-5}
WEIGHT_DECAY = 1e-4
FREEZE_TO, WARMUP = 800, 100          # RGB conv1, bn1, layer1 frozen to this step
DROP_FROM = 1200
DROP_V2 = {"flight": 0.25, "action": 0.15, "dribble": 0.25}
FOOTPRINT, POOL_R = 4, 8.0            # flight: (1/8) log((1/4) sum exp(8 z))
LME_R = 5.0                           # action, dribble: v1's log-mean-exp
GAMMA, HARD = 2, 3.0                  # focal negatives, x 3 on -3 cells
N_PEAKS, PEAK_WIN = 16, 5             # peak suppression: 16 maxima of 5 x 5, 1/16 each
AUG = {"flip": 0.5, "contrast": (0.75, 1.25), "brightness": (0.75, 1.25),
       "saturation": (0.8, 1.2), "scale": (0.8, 1.25), "blur_p": 0.2,
       "blur_sigma": (0.3, 1.0), "jpeg_p": 0.3, "jpeg_q": (80, 95),
       "noise": (0.0, 2.0), "reverse_p": 0.5, "dropout_p": 0.1}
MONITOR_EVERY, PEAKY_MAX, PEAKY_FROM, NEG_P = 250, 30.0, 1500, 0.15
GREY_W = np.array([0.114, 0.587, 0.299])         # B, G, R: cv2's BGR2GRAY


class _RowsOf:
    """Rows `index` of an array (the memory-mapped codes), read on demand:
    the codes of the strips kept, without copying them all."""

    def __init__(self, a, index):
        self.a, self.index = a, np.asarray(index)
        self.shape = (len(self.index),) + tuple(a.shape[1:])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return self.a[self.index[i]]


def load_strips(root: Path, allow_window_strips: bool = False, log=print):
    """`labels.parquet`, `labels_codes.npy` (memory-mapped) and the label
    constants of a strips dir (`labels_meta.json`, else the code's).
    Refuses a strip of a half not in `ball_heatmap.V2_TRAIN`, of a `NEVER`
    match, or whose t-2 .. t+2 touch a `V2_EXCLUDE` range: the provenance
    says what was trained on, so the data must agree with it. Then keeps
    only strips cut from the source stream (`source` "stream"): one cut
    from a window video (x264 re-encoded, background from the window only)
    is dropped, and the number logged, unless `allow_window_strips` (smoke
    tests)."""
    from build_ball_strips import excluded
    from src.ball_heatmap import NEVER, V2_TRAIN

    path = root / "labels.parquet"
    if not path.exists():
        raise SystemExit(f"{path} missing: label_ball_strips.py --strips {root} first")
    table = pd.read_parquet(path)
    codes = np.load(root / "labels_codes.npy", mmap_mode="r")
    if len(codes) != len(table) or codes.shape[1:] != (CELLS, CELLS):
        raise SystemExit(f"labels_codes.npy {codes.shape} does not match "
                         f"labels.parquet ({len(table)} rows)")
    table["match"] = table.match.astype(str)
    table["category"] = table.category.astype(str)
    bad = [(m, int(h), int(f)) for m, h, f in zip(table.match, table.half, table.frame)
           if m in NEVER or int(h) not in V2_TRAIN.get(m, ())
           or excluded(int(f), m, int(h))]
    if bad:
        raise SystemExit(f"{len(bad)} strips outside the v2 training halves "
                         f"(V2_TRAIN less V2_EXCLUDE), e.g. {bad[:3]}")
    if "source" not in table:
        if not allow_window_strips:
            raise SystemExit(f"{path} has no source column: cannot tell stream strips "
                             f"from window strips (--allow-window-strips to take all)")
    elif not allow_window_strips:
        keep = (table.source.astype(str) == "stream").to_numpy()
        if not keep.all():
            log(f"  {int((~keep).sum())} window strips dropped (sources "
                f"{sorted(set(table.source[~keep].astype(str)))}); training on "
                f"{int(keep.sum())} stream strips")
            table = table[keep].reset_index(drop=True)
            codes = _RowsOf(codes, np.flatnonzero(keep))
        if not len(table):
            raise SystemExit(f"{path}: no stream strips (--allow-window-strips for "
                             f"window strips)")
    meta_path = root / "labels_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return table, codes, meta


class Sampler:
    """Batches of `MIX_V2` from the training strips: per kind (and free
    category) uniform with replacement, as v1; a free category with no
    strips is drawn from all free strips."""

    def __init__(self, table: pd.DataFrame, index: np.ndarray, mix=MIX_V2):
        kind = table.kind.to_numpy()[index]
        cat = table.category.to_numpy()[index]
        self.pools, self.fallback = [], []
        for k, c, n in mix:
            pool = index[(kind == k) & ((cat == c) if c else True)]
            if c and not len(pool):
                pool = index[kind == k]
                self.fallback.append(c)
            if not len(pool):
                raise SystemExit(f"no {k} strips to train on")
            self.pools.append((k, c, n, pool))

    def draw(self, rng) -> np.ndarray:
        return np.concatenate([rng.choice(pool, n) for _, _, n, pool in self.pools])


def read_strip(root: Path, r, motion: bool = True):
    """A strip's crops: t-2, t, t+2 (BGR) and the background (grey), None
    where the strip has none (an edge: stored as t; fewer than `BG_MIN`
    background samples: stored as grey t) -- zero motion, as at inference.
    Without `motion`, the t crop alone."""
    from label_ball_strips import split
    from src.motion_bg import BG_MIN

    strip = cv2.imread(str(root / r.file))
    if strip is None or strip.shape[:2] != (SIZE, 4 * SIZE):
        raise SystemExit(f"{root / r.file}: not a {4 * SIZE} x {SIZE} strip")
    if not motion:
        return None, strip[:, SIZE:2 * SIZE], None, None
    prev, cur, nxt, bg = split(strip)
    return (prev if r.has_prev else None, cur, nxt if r.has_next else None,
            bg if r.n_bg >= BG_MIN else None)


def draw_aug(g) -> dict:
    """One strip's augmentation draws, from its own generator `g`."""
    a = AUG
    return {"flip": g.random() < a["flip"],
            "contrast": g.uniform(*a["contrast"]),
            "brightness": g.uniform(*a["brightness"]),
            "saturation": g.uniform(*a["saturation"]),
            "scale": float(np.exp(g.uniform(*np.log(a["scale"])))),
            "offset": g.random(2),
            "blur": g.uniform(*a["blur_sigma"]) if g.random() < a["blur_p"] else 0.0,
            "jpeg": (int(g.integers(a["jpeg_q"][0], a["jpeg_q"][1] + 1))
                     if g.random() < a["jpeg_p"] else 0),
            "noise": g.uniform(*a["noise"], 3),               # t-2, t, t+2
            "reverse": g.random() < a["reverse_p"],
            "dropout": g.random() < a["dropout_p"],
            "seed": int(g.integers(2 ** 62))}


def _span(lo: int, hi: int, n: int, frac: float) -> int:
    """Offset (cells) of a CELLS window in n cells keeping cells lo..hi
    (the bag) inside where they fit, else inside them; `frac` in [0, 1)."""
    if lo < 0:                                    # no bag
        a, b = 0, n - CELLS
    else:
        a, b = sorted((hi - CELLS + 1, lo))
        a, b = max(a, 0), min(b, n - CELLS)
    return a + int(frac * (b - a + 1))


def augment_strip(prev, cur, nxt, bg, codes, p: dict, motion: bool = True):
    """Draws `p` (`draw_aug`) applied to a strip's crops (t-2, t, t+2 BGR;
    background grey; None where missing) and its codes: (prev, cur, nxt,
    bg, codes), prev / nxt / bg None without motion or at motion dropout."""
    if not motion or p["dropout"]:
        prev = nxt = bg = None
    elif p["reverse"]:
        prev, nxt = nxt, prev
    ims = [None if im is None else np.ascontiguousarray(im)
           for im in (prev, cur, nxt, bg)]
    if p["flip"]:
        ims = [None if im is None else cv2.flip(im, 1) for im in ims]
        codes = codes[:, ::-1]
    n = int(round(CELLS * p["scale"]))
    if n != CELLS:
        S = n * STRIDE
        interp = cv2.INTER_AREA if n < CELLS else cv2.INTER_LINEAR
        ims = [None if im is None else cv2.resize(im, (S, S), interpolation=interp)
               for im in ims]
        k = ((np.arange(n) + 0.5) * CELLS / n).astype(int)
        codes = codes[np.ix_(k, k)]
        if n > CELLS:
            rows, cols = np.nonzero(codes == 1)
            oy = _span(rows.min() if len(rows) else -1, rows.max() if len(rows) else -1,
                       n, p["offset"][0])
            ox = _span(cols.min() if len(cols) else -1, cols.max() if len(cols) else -1,
                       n, p["offset"][1])
            y, x = oy * STRIDE, ox * STRIDE
            ims = [None if im is None else im[y:y + SIZE, x:x + SIZE] for im in ims]
            codes = codes[oy:oy + CELLS, ox:ox + CELLS]
        else:
            oy, ox = (int(f * (CELLS - n + 1)) for f in p["offset"])
            y, x = oy * STRIDE, ox * STRIDE
            ims = [None if im is None else
                   cv2.copyMakeBorder(im, y, SIZE - S - y, x, SIZE - S - x,
                                      cv2.BORDER_REFLECT_101) for im in ims]
            c = np.zeros((CELLS, CELLS), codes.dtype)
            c[oy:oy + n, ox:ox + n] = codes
            codes = c
    # Brightness and contrast about the t crop's mean, and the saturation as
    # a blend with the grey image: one affine colour map, the same on every
    # crop, which keeps luminance -- so the grey background takes the
    # brightness and contrast alone. One rounding (cv2; numpy floats cost
    # 20 x as much).
    m = float(ims[1].mean())
    c, s = p["contrast"], p["saturation"]
    beta = m * (p["brightness"] - c)
    colour = np.hstack([c * (s * np.eye(3) + (1 - s) * GREY_W[None, :]),
                        np.full((3, 1), beta)]).astype(np.float32)
    ims = [None if im is None else
           cv2.transform(im, colour) if im.ndim == 3 else
           cv2.addWeighted(im, c, im, 0.0, beta) for im in ims]
    if p["blur"]:
        ims = [None if im is None else cv2.GaussianBlur(im, (0, 0), p["blur"])
               for im in ims]
    if p["jpeg"]:
        flag = [cv2.IMREAD_COLOR] * 3 + [cv2.IMREAD_GRAYSCALE]
        ims = [None if im is None else cv2.imdecode(
            cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])[1], f)
            for im, f in zip(ims, flag)]
    for k in range(3):                        # each frame its own noise
        if ims[k] is None or p["noise"][k] <= 0:
            continue
        g = np.random.default_rng([p["seed"], k])
        e = g.standard_normal(ims[k].shape[:2], dtype=np.float32)
        e *= np.float32(p["noise"][k])
        ims[k] = cv2.add(ims[k], cv2.merge([e, e, e]), dtype=cv2.CV_8U)
    return (*ims, np.ascontiguousarray(codes))


def _rgb_lut() -> np.ndarray:
    """`to_tensor`'s normalisation of each uint8 value, per RGB channel:
    the same float32 operations, so a lookup gives its values bit for bit."""
    from src.ball_heatmap import MEAN, STD

    v = np.arange(256, dtype=np.float32)
    return np.stack([(v / 255.0 - MEAN[c]) / STD[c] for c in range(3)])


RGB_LUT = _rgb_lut()


def make_batch(root: Path, rows, codes_all, idx, seeds=None, motion: bool = True):
    """Network input and codes of strips `idx`: augmented with each strip's
    generator (`seeds`), or as they are (`seeds` None: the monitor). The
    input is `to_tensor_v2` (with motion) or `to_tensor` of the t crops,
    built in place."""
    import torch

    from src.motion_bg import motion_planes

    x = np.empty((len(idx), 6 if motion else 3, SIZE, SIZE), np.float32)
    cs = np.empty((len(idx), CELLS, CELLS), np.int64)
    for j, i in enumerate(idx):
        prev, cur, nxt, bg = read_strip(root, rows[i], motion)
        c = np.asarray(codes_all[i])
        if seeds is not None:
            p = draw_aug(np.random.default_rng(int(seeds[j])))
            prev, cur, nxt, bg, c = augment_strip(prev, cur, nxt, bg, c, p, motion)
        for k in range(3):                              # BGR -> RGB, normalised
            np.take(RGB_LUT[k], cur[:, :, 2 - k], out=x[j, k])
        if motion:
            motion_planes(prev, cur, nxt, bg, out=x[j, 3:])
        cs[j] = c
    return torch.from_numpy(x), torch.from_numpy(cs)


def strip_losses(logits, codes, flight, weight, positive_only):
    """Per strip: the bag term (weighted; 0 where there is no bag, `has`
    False), the focal negative term and the peak suppression term."""
    import torch

    F = torch.nn.functional
    bag = codes == 1
    n = bag.flatten(1).sum(1)
    has = n > 0
    r = torch.where(flight, POOL_R, LME_R)
    off = torch.where(flight, float(np.log(FOOTPRINT)),
                      torch.log(n.clamp(min=1).to(logits.dtype)))
    # Cells outside the bag at a large finite value, not -inf: a strip
    # without a bag must not put NaN into the gradient.
    lse = torch.logsumexp(torch.where(bag, r[:, None, None] * logits,
                                      torch.tensor(-1e4)).flatten(1), 1)
    z = (lse - off) / r
    pos = torch.where(has, F.binary_cross_entropy_with_logits(
        z, torch.ones_like(z), reduction="none") * weight, torch.zeros_like(z))
    negative = ((codes == -1) | (codes == -3)) & ~positive_only[:, None, None]
    w = torch.where(codes == -3, HARD, 1.0) * negative
    bce0 = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits),
                                              reduction="none")
    neg = (bce0 * torch.sigmoid(logits) ** GAMMA * w).sum((1, 2))
    lg = logits.detach()
    peak_at = negative & (lg >= F.max_pool2d(lg[:, None], PEAK_WIN, 1,
                                             PEAK_WIN // 2)[:, 0])
    top = torch.where(peak_at, lg, torch.tensor(-float("inf"))).flatten(1).topk(N_PEAKS, 1)
    sel = torch.isfinite(top.values)
    peak = (bce0.flatten(1).gather(1, top.indices) * sel).sum(1) / N_PEAKS
    return pos, has, neg, peak


def drop_worst(pos, has, kinds, rng) -> np.ndarray:
    """Bags left out of a batch: per kind, its `DROP_V2` share of the
    batch's bags with the highest loss, the count rounded at random (1.25
    flight bags: 1, or 2 a quarter of the time)."""
    loss = pos.detach().numpy()
    drop = np.zeros(len(loss), bool)
    has = has.numpy()
    for kind, share in DROP_V2.items():
        cand = np.flatnonzero((kinds == kind) & has)
        x = share * len(cand)
        n = int(x) + int(rng.random() < x - int(x))
        if n:
            drop[cand[np.argsort(-loss[cand], kind="stable")[:n]]] = True
    return drop


def optimiser(model):
    """AdamW in the groups of the recipe; returns (optimiser, early): the RGB
    conv1, bn1 and layer1, frozen to step `FREEZE_TO`."""
    import torch

    early = (list(model.stem[0].parameters()) + list(model.stem[1].parameters())
             + list(model.layer1.parameters()))
    groups = [{"params": list(model.lat.parameters()) + list(model.head.parameters()),
               "lr": LR_V2["head"], "name": "head"},
              {"params": list(model.layer2.parameters()), "lr": LR_V2["layer2"],
               "name": "layer2"},
              {"params": early, "lr": LR_V2["early"], "name": "early"}]
    if getattr(model, "motion", False):
        groups.append({"params": [model.conv1_motion.weight], "lr": LR_V2["motion"],
                       "name": "motion"})
    for p in early:
        p.requires_grad_(False)
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY), early


def lr_scale(step: int, steps: int) -> float:
    return min(1.0, step / WARMUP) * 0.5 * (1 + np.cos(np.pi * step / steps))


def monitor(model, root: Path, rows, codes_all, idx, motion: bool) -> dict:
    """The held-out strips, as they are: bag hit (the map's top cell is in
    the bag), peakiness (bag cells with p > 0.5), negative peaks (5 x 5
    maxima on -1 / -3 cells with p > `NEG_P`) per strip, by kind, the loss
    without drops, and the motion weights' norm."""
    import torch

    F = torch.nn.functional
    per = {"hit": {}, "peaky": {}, "negpk": {}}
    losses = []
    model.eval()
    with torch.no_grad():
        for b in range(0, len(idx), 16):
            ids = idx[b:b + 16]
            x, c = make_batch(root, rows, codes_all, ids, None, motion)
            z = model(x)
            kinds = np.array([rows[i].kind for i in ids])
            po = torch.tensor([bool(rows[i].positive_only) for i in ids])
            wt = torch.tensor([float(rows[i].weight) for i in ids])
            pos, has, neg, peak = strip_losses(z, c, torch.from_numpy(kinds == "flight"),
                                               wt, po)
            losses.append((pos.sum() + neg.sum() + peak.sum()).item())
            p = torch.sigmoid(z)
            negative = ((c == -1) | (c == -3)) & ~po[:, None, None]
            at = negative & (z >= F.max_pool2d(z[:, None], PEAK_WIN, 1,
                                               PEAK_WIN // 2)[:, 0]) & (p > NEG_P)
            for j, kind in enumerate(kinds):
                bag = c[j] == 1
                if bag.any():
                    per["hit"].setdefault(kind, []).append(
                        bool(bag.flatten()[z[j].flatten().argmax()]))
                    per["peaky"].setdefault(kind, []).append(int((p[j][bag] > 0.5).sum()))
                if not po[j]:
                    per["negpk"].setdefault(kind, []).append(int(at[j].sum()))
    model.train()

    def pooled(d):
        v = [x for xs in d.values() for x in xs]
        return float(np.mean(v)) if v else float("nan")

    out = {"n": len(idx), "loss": float(np.sum(losses) / max(len(idx), 1))}
    for key in per:
        out[key] = pooled(per[key])
        out[key + "_by_kind"] = {k: float(np.mean(v)) for k, v in sorted(per[key].items())}
    out["motion_norm"] = (float(model.conv1_motion.weight.detach().norm())
                          if getattr(model, "motion", False) else None)
    return out


def _fmt_monitor(step: int, m: dict) -> str:
    kinds = lambda d: ", ".join(f"{k} {v:.2f}" for k, v in d.items())
    norm = "" if m["motion_norm"] is None else f", conv1_motion |w| {m['motion_norm']:.4f}"
    return (f"  monitor {step} ({m['n']} held out): bag hit {m['hit']:.2f} "
            f"({kinds(m['hit_by_kind'])}), peakiness {m['peaky']:.1f} "
            f"({kinds(m['peaky_by_kind'])}), negative peaks > {NEG_P} "
            f"{m['negpk']:.2f} a strip ({kinds(m['negpk_by_kind'])}){norm}, "
            f"loss {m['loss']:.3f}")


def _sha(*paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def recipe() -> dict:
    """The recipe hash (this file, then `label_ball_strips.py`) and the
    hashes of the code the strips and the net come from."""
    here = Path(__file__).resolve()
    files = [here, here.with_name("label_ball_strips.py"),
             here.with_name("build_ball_strips.py"),
             here.parent / "src" / "ball_heatmap.py", here.parent / "src" / "motion_bg.py"]
    return {"recipe_sha256": _sha(*files[:2]),
            "files_sha256": {f.name: _sha(f) for f in files if f.exists()}}


def recipe_constants() -> dict:
    """`label_ball_strips.LABEL_CONSTANTS` as JSON gives it back."""
    import label_ball_strips as ls
    from src.ball_heatmap import _jsonable

    return json.loads(json.dumps(_jsonable(ls.LABEL_CONSTANTS)))


def provenance_info(root: Path, table: pd.DataFrame, meta, motion: bool, seed: int,
                    step: int, steps: int, history: list, mix=MIX_V2,
                    allow_window_strips: bool = False) -> dict:
    """What `<ckpt>.provenance.json` records (`ball_heatmap.write_provenance`).
    "sigma" is what inference scales the motion channels by
    (`ball_heatmap.load_v2`)."""
    from src import ball_heatmap as bh
    from src.ball_heatmap import V2_EXCLUDE

    train = {}
    for m, h in sorted(set(zip(table.match, table.half.astype(int)))):
        train.setdefault(m, []).append(h)
    code = recipe_constants()
    held = table.holdout.to_numpy().astype(bool)
    return {
        "train": train, "exclude": V2_EXCLUDE,
        "label_constants": meta["constants"] if meta else code,
        "label_constants_from": ("labels_meta.json" if meta
                                 else "label_ball_strips.py (no labels_meta.json)"),
        "label_constants_as_code": (meta["constants"] == code) if meta else True,
        # COCO ignore disks on free strips (resting balls -1 or not), labels' seed.
        "labels_coco": meta.get("coco") if meta else None,
        "labels_seed": meta.get("seed") if meta else None,
        "sigma": {"SIGMA_D": bh.SIGMA_D, "SIGMA_F": bh.SIGMA_F},
        "sigma_measured": bool(bh.SIGMA_MEASURED),
        **recipe(), "seed": seed, "motion": motion, "step": step, "steps": steps,
        "strips": str(root), "window_strips_allowed": allow_window_strips,
        "labels_sha256": _sha(root / "labels.parquet", root / "labels_codes.npy"),
        "n_train": table[~held].kind.value_counts().sort_index().to_dict(),
        "n_holdout": table[held].kind.value_counts().sort_index().to_dict(),
        "sources": table.source.value_counts().to_dict() if "source" in table else None,
        "mix": [list(m) for m in mix], "lr": LR_V2, "weight_decay": WEIGHT_DECAY,
        "freeze_to": FREEZE_TO, "warmup": WARMUP, "drop_from": DROP_FROM,
        "drop": DROP_V2, "loss": {"footprint": FOOTPRINT, "pool_r": POOL_R,
                                  "lme_r": LME_R, "gamma": GAMMA, "hard": HARD,
                                  "peaks": N_PEAKS, "peak_window": PEAK_WIN},
        "augment": AUG, "monitor": history}


class Batches:
    """The training batches of a seed: strip indices, per-strip generator
    seeds, augmented inputs and codes. The next batch is prepared in a
    thread while a step runs; the draws stay in order, so the batches do
    not depend on it."""

    def __init__(self, root: Path, table, codes, motion: bool, seed: int,
                 mix=MIX_V2, prefetch: bool = True):
        from concurrent.futures import ThreadPoolExecutor

        self.root, self.codes, self.motion = root, codes, motion
        self.rows = list(table.itertuples(index=False))
        held = table.holdout.to_numpy().astype(bool)
        self.sampler = Sampler(table, np.flatnonzero(~held), mix)
        self.rng = np.random.default_rng(seed)
        self.pool = ThreadPoolExecutor(1) if prefetch else None
        self.pending = None

    def _make(self):
        t0 = time.perf_counter()
        idx = self.sampler.draw(self.rng)
        seeds = self.rng.integers(2 ** 62, size=len(idx))
        x, c = make_batch(self.root, self.rows, self.codes, idx, seeds, self.motion)
        return idx, x, c, time.perf_counter() - t0

    def next(self):
        """(strip indices, input, codes, seconds it took to prepare)."""
        if self.pool is None:
            return self._make()
        if self.pending is None:
            self.pending = self.pool.submit(self._make)
        out = self.pending.result()
        self.pending = self.pool.submit(self._make)
        return out

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


class Run:
    """One model's training state: its batches, optimiser, drop generator
    and step (`train_v2`; `bench_v2` runs two)."""

    def __init__(self, root: Path, table, codes, motion: bool, seed: int, steps: int,
                 mix=MIX_V2):
        import torch

        self.motion, self.steps = motion, steps
        self.kinds = table.kind.to_numpy()
        self.flight = self.kinds == "flight"
        self.po = table.positive_only.to_numpy().astype(bool)
        self.weight = table.weight.to_numpy().astype(np.float32)
        self.holdout = np.flatnonzero(table.holdout.to_numpy().astype(bool))
        self.batches = Batches(root, table, codes, motion, seed, mix)
        self.rows, self.sampler = self.batches.rows, self.batches.sampler
        self.drop_rng = np.random.default_rng([seed, 1])
        torch.manual_seed(seed)
        self.model = build(motion=motion)
        self.opt, self.early = optimiser(self.model)
        self.base = [g["lr"] for g in self.opt.param_groups]

    def batch(self):
        return self.batches.next()

    def close(self):
        self.batches.close()

    def step(self, step: int, idx, x, c) -> dict:
        import torch

        if step > FREEZE_TO and not self.early[0].requires_grad:
            for p in self.early:
                p.requires_grad_(True)
        s = lr_scale(step, self.steps)
        for g, b in zip(self.opt.param_groups, self.base):
            g["lr"] = b * s
        self.model.train()
        logits = self.model(x)
        pos, has, neg, peak = strip_losses(
            logits, c, torch.from_numpy(self.flight[idx]),
            torch.from_numpy(self.weight[idx]), torch.from_numpy(self.po[idx]))
        keep = has.clone()
        if step > DROP_FROM:
            keep &= ~torch.from_numpy(drop_worst(pos, has, self.kinds[idx], self.drop_rng))
        bags = torch.where(keep, pos, torch.zeros_like(pos)).sum()
        loss = (bags + neg.sum() + peak.sum()) / len(idx)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        n = len(idx)
        return {"loss": loss.item(), "bags": bags.item() / n, "neg": neg.sum().item() / n,
                "peak": peak.sum().item() / n}


def check_sigma(placeholder_ok: bool = False, log=print) -> bool:
    """Refuse to train v2 on the placeholder motion scales: SIGMA_D and
    SIGMA_F must have been measured (`label_ball_strips.py --sigma`) and
    written into src/ball_heatmap.py together with SIGMA_MEASURED = True.
    `placeholder_ok` (smoke runs) passes, noted. True when measured."""
    from src import ball_heatmap as bh

    values = (bh.SIGMA_D, bh.SIGMA_F)
    if bh.SIGMA_MEASURED and values != tuple(bh.SIGMA_PLACEHOLDERS):
        return True
    why = ("SIGMA_MEASURED is True but SIGMA_D / SIGMA_F are still the placeholders "
           f"{values}" if bh.SIGMA_MEASURED else
           f"SIGMA_MEASURED is False: SIGMA_D / SIGMA_F = {values} are placeholders")
    if not placeholder_ok:
        raise SystemExit(f"refused: {why}. Measure them (label_ball_strips.py --strips "
                         f"DIR --sigma), write both into src/ball_heatmap.py and set "
                         f"SIGMA_MEASURED = True there; --placeholder-sigma for a smoke run")
    log(f"  note: {why} (smoke run; the provenance records it)")
    return False


def train_v2(root, out, steps: int = STEPS_V2, motion: bool = True, seed: int = 0,
             checkpoints=CHECKPOINTS_V2, monitor_every: int = MONITOR_EVERY,
             mix=MIX_V2, log=print, placeholder_sigma: bool = False,
             allow_window_strips: bool = False) -> list:
    """Train on the strips in `root`; checkpoints `<out>.<step>.pt` with
    their provenance. Returns the monitor history. Refuses the placeholder
    motion scales unless `placeholder_sigma` (`check_sigma`), and window
    strips unless `allow_window_strips` (`load_strips`)."""
    import torch

    from src.ball_heatmap import write_provenance

    check_sigma(placeholder_sigma, log)
    root = Path(root)
    table, codes, meta = load_strips(root, allow_window_strips, log)
    torch.set_num_threads(4)
    run = Run(root, table, codes, motion, seed, steps, mix)
    tr = table[~table.holdout.astype(bool)]
    log(f"  {'motion' if motion else 'RGB-only'} model, {len(tr)} strips "
        f"({tr.kind.value_counts().sort_index().to_dict()}) of "
        f"{sorted(set(tr.match))}, {len(run.holdout)} held out"
        + (f"; free {run.sampler.fallback} drawn from all free strips"
           if run.sampler.fallback else ""))
    if meta:
        code, made = recipe_constants(), meta["constants"]
        differ = sorted(k for k in set(code) | set(made) if code.get(k) != made.get(k))
        if differ:
            log(f"  note: label constants {differ} of labels_meta.json differ from "
                f"label_ball_strips.py's now; the provenance records the former")
    history, acc = [], []
    t0 = time.time()
    try:
        for step in range(1, steps + 1):
            idx, x, c, _ = run.batch()
            acc.append(run.step(step, idx, x, c))
            if step % 100 == 0 or step == steps:
                mean = {k: np.mean([a[k] for a in acc]) for k in acc[0]}
                log(f"  step {step}: loss {mean['loss']:.4f} (bags {mean['bags']:.4f}, "
                    f"negatives {mean['neg']:.4f}, peaks {mean['peak']:.4f})  "
                    f"[{time.time() - t0:.0f} s]")
                acc = []
            if (step % monitor_every == 0 or step == steps) and len(run.holdout):
                m = monitor(run.model, root, run.rows, codes, run.holdout, motion)
                history.append(dict(m, step=step))
                log(_fmt_monitor(step, m))
                if step >= PEAKY_FROM and m["peaky"] > PEAKY_MAX:
                    log(f"  abort: peakiness {m['peaky']:.1f} > {PEAKY_MAX:.0f} "
                        f"at step {step} (the bags glow instead of peaking)")
                    raise SystemExit(3)
            if step in checkpoints or step == steps:
                ckpt = f"{out}.{step}.pt"
                torch.save(run.model.state_dict(), ckpt)
                prov = write_provenance(ckpt, provenance_info(
                    root, table, meta, motion, seed, step, steps, history, mix,
                    allow_window_strips))
                log(f"  saved {ckpt}, {prov.name}")
    finally:
        run.close()
    return history


def bench_v2(root, n: int, motion: bool = True, at: int = 1, seed: int = 0,
             rounds: int = 5, mix=MIX_V2, log=print,
             allow_window_strips: bool = False) -> dict:
    """Seconds per training step as training runs them (loading,
    augmentation and motion channels in the prefetch thread; forward,
    backward, update) over `n` steps from step `at` of the schedule
    (freezing, drops), the first one left out; nothing saved. With
    `motion`, the RGB-only model runs the same batches in alternating
    rounds: the ratio of the medians is the go/no-go's (at most 1.25)."""
    import torch

    root = Path(root)
    table, codes, _ = load_strips(root, allow_window_strips, log)
    torch.set_num_threads(4)
    runs = {m: Run(root, table, codes, m, seed, STEPS_V2, mix)
            for m in ((True, False) if motion else (False,))}
    times = {m: {"step": [], "wait": [], "prep": []} for m in runs}
    done = {m: 0 for m in runs}
    try:
        while min(done.values()) < n:
            for m, run in runs.items():
                for _ in range(min(rounds, n - done[m])):
                    t0 = time.perf_counter()
                    idx, x, c, prep = run.batch()
                    t1 = time.perf_counter()
                    run.step(at + done[m], idx, x, c)
                    if done[m]:                      # the first step warms up
                        times[m]["step"].append(time.perf_counter() - t0)
                        times[m]["wait"].append(t1 - t0)
                        times[m]["prep"].append(prep)
                    done[m] += 1
    finally:
        for run in runs.values():
            run.close()
    out = {}
    for m, t in times.items():
        name = "motion" if m else "RGB-only"
        out[name] = {k: float(np.median(v)) for k, v in t.items()}
        out[name]["mean"] = float(np.mean(t["step"]))
        log(f"  {name}: {out[name]['step']:.3f} s/step median ({out[name]['mean']:.3f} "
            f"mean) over {len(t['step'])} steps from step {at}; batch preparation "
            f"{out[name]['prep']:.3f} s in the thread, waited for {out[name]['wait']:.3f}")
    if motion:
        out["ratio"] = out["motion"]["step"] / out["RGB-only"]["step"]
        log(f"  motion / RGB-only: {out['ratio']:.2f} (go/no-go: at most 1.25)")
    return out


def main():
    ap = argparse.ArgumentParser()
    data = ap.add_mutually_exclusive_group()
    data.add_argument("--patches", help="v1: build_ball_patches.py + build_ball_labels.py")
    data.add_argument("--strips", help="v2: build_ball_strips.py + label_ball_strips.py")
    ap.add_argument("--steps", type=int, help="default 3000 (v1), 4000 (v2)")
    ap.add_argument("--out", help="checkpoint prefix")
    ap.add_argument("--exclude-matches", nargs="*", default=["117092"],
                    help="v1: never trained on (M1: another camera)")
    ap.add_argument("--motion", action=argparse.BooleanOptionalAction, default=None,
                    help="v2: with the motion stem (default), or RGB only")
    ap.add_argument("--seed", type=int, default=0, help="v2")
    ap.add_argument("--bench", type=int, default=0, metavar="N",
                    help="v2: time N steps, save nothing")
    ap.add_argument("--bench-at", type=int, default=1, metavar="STEP",
                    help="v2: the schedule's step the benchmark starts at")
    ap.add_argument("--placeholder-sigma", action="store_true",
                    help="v2 smoke runs: train on the placeholder SIGMA_D / SIGMA_F")
    ap.add_argument("--allow-window-strips", action="store_true",
                    help="v2 smoke runs: also strips cut from window videos")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        _check()
        return
    if args.patches:
        if (args.motion is not None or args.bench or args.seed or args.placeholder_sigma
                or args.allow_window_strips):
            ap.error("--motion, --seed, --bench, --placeholder-sigma and "
                     "--allow-window-strips are for --strips")
        if not args.out:
            ap.error("--out is needed")
        args.steps = args.steps or 3000
        train_v1(args)
        return
    if not args.strips:
        ap.error("--patches (v1) or --strips (v2) is needed")
    motion = True if args.motion is None else args.motion
    if args.bench:
        bench_v2(args.strips, args.bench, motion, args.bench_at, args.seed,
                 log=lambda s: print(s, flush=True),
                 allow_window_strips=args.allow_window_strips)
        return
    if not args.out:
        ap.error("--out is needed")
    train_v2(args.strips, args.out, args.steps or STEPS_V2, motion, args.seed,
             log=lambda s: print(s, flush=True), placeholder_sigma=args.placeholder_sigma,
             allow_window_strips=args.allow_window_strips)


# ---- Self-check -----------------------------------------------------------------

def _synthetic_strips(root: Path, seed: int = 0) -> pd.DataFrame:
    """A strips dir as `label_ball_strips.py` writes it: grass, a still
    dark player, a white ball moving 6 px per 2 frames (absent from free
    strips), the background without the ball; codes per kind. The free
    category "random" is left empty."""
    from label_ball_strips import LABEL_CONSTANTS
    from src.ball_heatmap import _jsonable

    rng = np.random.default_rng(seed)
    plan = ([("flight", "completed")] * 5 + [("action", "PASS")] * 4
            + [("dribble", "DRIVE")] * 2 + [("lofted", "HIGH PASS")] * 2
            + [("free", "landmark")] * 2 + [("free", "cluster")] * 3
            + [("free", "boundary")] * 2)
    holdout = {0, 5, 13}                       # a flight, an action, a free strip
    rows, codes = [], []
    for j, (kind, cat) in enumerate(plan):
        grass = np.clip(rng.normal((60, 125, 70), 8, (SIZE, SIZE, 3)), 0, 255).astype(np.uint8)
        cv2.rectangle(grass, (40, 200), (52, 240), (40, 30, 30), -1)          # player
        bx, by = int(rng.integers(100, 220)), int(rng.integers(80, 240))
        frames = []
        for dx in (-6, 0, 6):
            im = grass.copy()
            if kind != "free":
                cv2.circle(im, (bx + dx, by), 3, (250, 250, 250), -1)
            frames.append(im)
        bg = cv2.cvtColor(cv2.cvtColor(grass, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        name = f"{kind}_118575_1_{2000 + 10 * j:06d}_0_0.jpg"
        cv2.imwrite(str(root / name), np.hstack(frames + [bg]),
                    [cv2.IMWRITE_JPEG_QUALITY, 98])
        c = np.full((CELLS, CELLS), -1, np.int8)
        c[50:60, 10:13] = -3
        gy, gx = by // STRIDE, bx // STRIDE
        if kind == "flight":
            c[gy - 6:gy + 7, gx - 12:gx + 13] = 0
            c[gy - 2:gy + 3, gx - 8:gx + 9] = 1
        elif kind in ("action", "dribble"):
            c[gy - 3:gy + 4, gx - 3:gx + 4] = 0
            c[gy - 2:gy + 3, gx - 2:gx + 3] = 1
        elif kind == "lofted":
            c[gy - 10:gy + 3, gx - 10:gx + 11] = 0
        if kind == "dribble":
            c[c != 1] = 0
        codes.append(c)
        rows.append({"file": name, "match": "118575", "half": 1, "frame": 2000 + 10 * j,
                     "kind": kind, "category": cat, "label": cat, "x0": 0, "y0": 0,
                     "positive_only": kind == "dribble",
                     "weight": 0.5 if kind == "dribble" else 1.0, "n_bg": 15,
                     "has_prev": j != 1, "has_next": True, "narrow": False,
                     "coco_balls": 0, "source": "stream", "holdout": j in holdout})
    table = pd.DataFrame(rows)
    table.to_parquet(root / "labels.parquet")
    np.save(root / "labels_codes.npy", np.stack(codes))
    (root / "labels_meta.json").write_text(json.dumps(_jsonable(
        {"constants": LABEL_CONSTANTS, "coco": False, "seed": seed}), indent=1))
    return table


def _check():
    """On a synthetic strips dir: the data guard, window strips dropped
    (codes kept in register), the placeholder motion scales refused, the
    batch mix, the augmentation (crops kept in register, codes following
    the image, time reversal, dropout, missing frames), the same t crops
    with and without motion, the input equal to `to_tensor_v2`,
    prefetching, the losses (footprint pool, v1's log-mean-exp and focal
    terms, peak suppression), the random drops, the optimiser groups,
    start and schedule; then two
    steps of the motion model and one of the RGB-only one (with the
    monitor, checkpoints and provenance; both read back by
    `ball_heatmap.load_v2`), and a two-step benchmark."""
    import tempfile

    import torch

    from src.ball_heatmap import build as build_net
    from src.ball_heatmap import read_provenance, to_tensor_v2
    from src.motion_bg import motion_channels

    t_start = time.time()
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "strips"
        root.mkdir()
        table = _synthetic_strips(root)
        tab, codes, meta = load_strips(root)
        assert len(tab) == len(table) and meta["seed"] == 0
        # The guard: a never-train match, a half not trained on, a frame in
        # a left-out range.
        for col, val, row in (("match", "118578", 0), ("half", 3, 0)):
            bad = table.copy()
            bad.loc[row, col] = val
            _refused(root, bad, codes)
        bad = table.copy()
        bad.loc[0, ["match", "half", "frame"]] = ["118576", 2, 13499]
        _refused(root, bad, codes)
        # Window strips: dropped and counted, the codes still in register;
        # kept when allowed; one of a never-train match still refused.
        win = table.copy()
        win.loc[[3, 7], "source"] = "st2_118575_1st_f000000"
        win.to_parquet(root / "labels.parquet")
        said = []
        wt, wc, _ = load_strips(root, log=said.append)
        kept = [i for i in range(len(table)) if i not in (3, 7)]
        assert len(wt) == len(wc) == len(kept) and set(wt.source) == {"stream"}
        assert wt.file.tolist() == table.file[kept].tolist()
        assert all(np.array_equal(wc[j], codes[i]) for j, i in enumerate(kept))
        assert any(s.startswith("  2 window strips dropped") for s in said), said
        assert len(load_strips(root, allow_window_strips=True)[0]) == len(table)
        win.loc[3, "match"] = "118578"
        _refused(root, win, codes)
        table.to_parquet(root / "labels.parquet")
        rows = list(tab.itertuples(index=False))
        # The motion scales: placeholders refused, unless a smoke run says
        # so; measured ones (SIGMA_MEASURED, and other values) pass.
        from src import ball_heatmap as bh

        saved = bh.SIGMA_D, bh.SIGMA_F, bh.SIGMA_MEASURED
        quiet = lambda *a: None  # noqa: E731
        try:
            for vals, flag, ok in (((0.02, 0.05), False, False), ((0.02, 0.05), True, False),
                                   ((0.07, 0.03), False, False), ((0.07, 0.03), True, True)):
                (bh.SIGMA_D, bh.SIGMA_F), bh.SIGMA_MEASURED = vals, flag
                try:
                    assert check_sigma(log=quiet) and ok
                except SystemExit:
                    assert not ok
                assert check_sigma(True, log=quiet) == ok
        finally:
            bh.SIGMA_D, bh.SIGMA_F, bh.SIGMA_MEASURED = saved

        # The mix: 16 a batch; the empty "random" category from all free strips.
        train = np.flatnonzero(~tab.holdout.to_numpy())
        sampler = Sampler(tab, train)
        assert sampler.fallback == ["random"]
        idx = sampler.draw(np.random.default_rng(0))
        kinds = tab.kind.to_numpy()[idx]
        assert len(idx) == 16 and not tab.holdout.to_numpy()[idx].any()
        assert [int((kinds == k).sum()) for k in ("flight", "action", "dribble",
                                                  "lofted", "free")] == [5, 3, 1, 1, 6]
        cats = tab.category.to_numpy()[idx[10:15]]
        assert list(cats) == ["landmark"] * 2 + ["cluster"] * 2 + ["boundary"]
        try:
            Sampler(tab, np.flatnonzero(tab.kind.to_numpy() != "lofted"))
            raise AssertionError("a batch without lofted strips was accepted")
        except SystemExit:
            pass

        # Augmentation. Four identical crops stay identical (no JPEG, no
        # noise): D exactly 0, F within 2 grey levels (where no channel is
        # clipped: there luminance is not kept); the bag follows the
        # ball through flips and scales.
        g = np.random.default_rng(1)
        a = rows[2]                                # flight, has_prev
        prev, cur, nxt, bg = read_strip(root, a)
        assert prev is not None and bg is not None and bg.ndim == 2
        assert read_strip(root, rows[1])[0] is None              # has_prev False
        c0 = np.asarray(codes[2])
        for trial in range(12):
            p = draw_aug(g)
            p.update(jpeg=0, noise=np.zeros(3), dropout=False,
                     scale=(0.8, 1.25, 1.0)[trial % 3])
            same = augment_strip(cur, cur, cur, cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY),
                                 c0, p)
            m = motion_channels(*same[:4], sigma_d=1.0, sigma_f=1.0)
            assert np.abs(m[:, :, :2]).max() == 0
            clipped = ((same[1] == 0) | (same[1] == 255)).any(2)       # a channel saturated
            assert np.abs(m[:, :, 2][~clipped]).max() <= 2 / 255
            aug = augment_strip(prev, cur, nxt, bg, c0, p)
            assert aug[1].shape == (SIZE, SIZE, 3) and aug[4].shape == (CELLS, CELLS)
            gr = cv2.cvtColor(aug[1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            light = gr > (gr.max() + np.median(gr)) / 2          # the ball
            ball = np.argwhere(light).mean(0) / STRIDE
            assert aug[4][int(ball[0]), int(ball[1])] == 1, (trial, p["flip"], p["scale"])
            if p["scale"] == 0.8:                 # padded: codes 0 there
                assert (aug[4] == 0).sum() > (c0 == 0).sum()
            # The ball moves 6 px per 2 frames: D- and D+ light up at it.
            mo = motion_channels(*aug[:4], sigma_d=1.0, sigma_f=1.0)
            yb, xb = np.argwhere(light).mean(0).astype(int)
            assert mo[yb, xb, 0] > 0.15 and mo[yb, xb, 1] > 0.15 and mo[yb, xb, 2] > 0.15
        # Time reversal swaps D- and D+; dropout and missing frames are zero.
        p = draw_aug(g)
        p.update(jpeg=0, noise=np.zeros(3), dropout=False, reverse=False, flip=False,
                 scale=1.0, blur=0.0)
        fw = motion_channels(*augment_strip(prev, cur, nxt, bg, c0, p)[:4])
        bw = motion_channels(*augment_strip(prev, cur, nxt, bg, c0, dict(p, reverse=True))[:4])
        assert np.array_equal(fw[:, :, ::-1][:, :, 1:], bw[:, :, :2])
        assert np.array_equal(fw[:, :, 2], bw[:, :, 2])
        dr = augment_strip(prev, cur, nxt, bg, c0, dict(p, dropout=True))
        assert dr[0] is None and dr[2] is None and dr[3] is None
        assert not motion_channels(*dr[:4]).any()
        miss = augment_strip(None, cur, nxt, bg, c0, p)
        assert not motion_channels(*miss[:4])[:, :, 0].any()
        # JPEG and noise reach each crop separately: identical crops differ.
        q = dict(p, jpeg=85, noise=np.array([1.5, 1.5, 1.5]))
        nz = augment_strip(cur, cur, cur, None, c0, q)
        assert not np.array_equal(nz[0], nz[1]) and not np.array_equal(nz[1], nz[2])
        # With and without motion: the same batch, the same t crops.
        seeds = np.random.default_rng(2).integers(2 ** 62, size=len(idx))
        x6, c6 = make_batch(root, rows, codes, idx, seeds, True)
        x3, c3 = make_batch(root, rows, codes, idx, seeds, False)
        assert x6.shape == (16, 6, SIZE, SIZE) and x3.shape == (16, 3, SIZE, SIZE)
        assert torch.equal(x6[:, :3], x3) and torch.equal(c6, c3)
        assert x6[:, 3:].abs().sum() > 0
        # The input built in place is to_tensor_v2 / to_tensor, bit for bit.
        ids = np.arange(4)
        crops = [read_strip(root, rows[i]) for i in ids]
        ref = to_tensor_v2([cr[1] for cr in crops], [motion_channels(*cr) for cr in crops])
        assert torch.equal(make_batch(root, rows, codes, ids, None, True)[0], ref)
        assert torch.equal(make_batch(root, rows, codes, ids, None, False)[0],
                           to_tensor([cr[1] for cr in crops]))
        # Prefetching does not change the batches.
        bs = [Batches(root, tab, codes, True, 0, prefetch=pf) for pf in (True, False)]
        for _ in range(3):
            a_, b_ = (b.next() for b in bs)
            assert np.array_equal(a_[0], b_[0]) and torch.equal(a_[1], b_[1])
        for b in bs:
            b.close()

        # Losses. The flight pool: 4 cells at z score z, one z - log(4)/8,
        # N cells z + log(N/4)/8; action / dribble: v1's log-mean-exp; the
        # focal term is v1's.
        z0 = 1.5
        lg = torch.full((4, CELLS, CELLS), -8.0)
        cd = torch.full((4, CELLS, CELLS), -1, dtype=torch.int64)
        cd[:, 10:30, 10:30] = 1                               # 400-cell bags
        lg[0, 12:14, 12:14] = z0                              # a ball of 4 cells
        lg[1, 12, 12] = z0                                    # one cell
        lg[2, 10:30, 10:30] = z0                              # a glow over the bag
        lg[3, 12:14, 12:14] = z0
        fl = torch.tensor([True, True, True, False])
        one = torch.ones(4)
        pos, has, neg, peak = strip_losses(lg, cd, fl, one, torch.zeros(4, dtype=torch.bool))
        zb = -torch.log(torch.expm1(pos))                     # back from BCE to 1
        assert has.all()
        assert np.allclose(zb[:3], [z0, z0 - np.log(4) / 8, z0 + np.log(100) / 8], atol=1e-4)
        v1_pos, v1_neg = bag_losses(lg, cd, [False] * 4)
        assert torch.allclose(pos[3], v1_pos[3], atol=1e-6)
        assert torch.allclose(neg, v1_neg, rtol=1e-5)
        # Peak suppression: the 16 highest 5 x 5 maxima on -1 / -3 cells,
        # 1/16 each; none on a positive-only strip, nor from bag cells.
        lg = torch.full((2, CELLS, CELLS), -6.0)
        heights = torch.linspace(-3.0, 2.0, 20)
        for k, h in enumerate(heights):
            lg[:, 40 + (k // 5) * 8, 5 + (k % 5) * 8] = h
        lg[:, 20, 20] = 5.0                                   # in the bag
        cd = torch.full((2, CELLS, CELLS), -1, dtype=torch.int64)
        cd[:, 18:23, 18:23] = 1
        pos, has, neg, peak = strip_losses(lg, cd, torch.zeros(2, dtype=torch.bool),
                                           torch.ones(2), torch.tensor([False, True]))
        want = torch.nn.functional.softplus(heights[-16:]).sum() / 16
        assert torch.allclose(peak[0], want, atol=1e-5) and peak[1] == 0 and neg[1] == 0
        # No bag: no bag term and a finite gradient.
        lg = torch.zeros((1, CELLS, CELLS), requires_grad=True)
        pos, has, neg, peak = strip_losses(lg, torch.full((1, CELLS, CELLS), -1),
                                           torch.tensor([True]), torch.ones(1),
                                           torch.tensor([False]))
        (pos.sum() + neg.sum() + peak.sum()).backward()
        assert not has[0] and pos[0] == 0 and torch.isfinite(lg.grad).all()
        # Drops: per kind, the worst, about the share on average.
        kinds = np.array(["flight"] * 5 + ["action"] * 3 + ["dribble"] + ["free"] * 7)
        loss = torch.arange(16, dtype=torch.float32)
        hasb = torch.tensor([True] * 9 + [False] * 7)
        drng = np.random.default_rng(0)
        counts = np.array([[drop_worst(loss, hasb, kinds, drng)[kinds == k].sum()
                            for k in ("flight", "action", "dribble")] for _ in range(4000)])
        assert np.allclose(counts.mean(0), [1.25, 0.45, 0.25], atol=0.04), counts.mean(0)
        for s in range(20):                       # the worst of each kind first
            d = drop_worst(loss, hasb, kinds, np.random.default_rng(s))
            for lo, hi in ((0, 5), (5, 8), (8, 9)):
                k = np.flatnonzero(d[lo:hi])
                assert list(k) == list(range(hi - lo - len(k), hi - lo)), d
            assert not d[9:].any()

        # Optimiser: the groups cover the net once; the RGB stem and layer1
        # frozen, conv1_motion trainable from step 1; warm-up and cosine.
        torch.manual_seed(0)
        net = build_net(motion=True)
        torch.manual_seed(0)
        rgb = build_net().state_dict()
        assert all(torch.equal(v, rgb[k]) for k, v in net.state_dict().items()
                   if k != "conv1_motion.weight")        # the same start
        opt, early = optimiser(net)
        ids = [id(p) for g_ in opt.param_groups for p in g_["params"]]
        assert sorted(ids) == sorted(id(p) for p in net.parameters())
        lrs = {g_["name"]: g_["lr"] for g_ in opt.param_groups}
        assert lrs == {"head": 3e-4, "layer2": 1e-4, "early": 3e-5, "motion": 3e-4}
        assert not any(p.requires_grad for p in early) and net.conv1_motion.weight.requires_grad
        assert all(p.requires_grad for p in net.layer2.parameters())
        assert np.isclose(lr_scale(50, 4000), 0.5 * 0.5 * (1 + np.cos(np.pi * 50 / 4000)))
        assert lr_scale(4000, 4000) < 1e-12
        assert np.isclose(lr_scale(100, 4000), 0.5 * (1 + np.cos(np.pi / 40)))

        # Two steps of the motion model, one of the RGB-only one: monitor,
        # checkpoints, provenance; the motion weights move at once. Batches
        # of 8 (one of each), to keep the check short.
        small = tuple((k, c, 1) for k, c, _ in MIX_V2)
        logs = []
        hist = {}
        bh.SIGMA_MEASURED = False
        try:
            train_v2(root, Path(tmp) / "no", steps=1, mix=small, log=logs.append)
            raise AssertionError("a v2 run started on the placeholder motion scales")
        except SystemExit:
            pass
        finally:
            bh.SIGMA_MEASURED = saved[2]
        assert not list(Path(tmp).glob("no.*"))
        for motion, steps in ((True, 2), (False, 1)):
            out = Path(tmp) / ("m" if motion else "rgb")
            hist[motion] = train_v2(root, out, steps=steps, motion=motion, checkpoints=(1,),
                                    monitor_every=1, mix=small, log=logs.append,
                                    placeholder_sigma=True)
            for step in range(1, steps + 1):
                ck = f"{out}.{step}.pt"
                prov = read_provenance(ck)
                assert prov["train"] == {"118575": [1]} and prov["seed"] == 0
                assert prov["motion"] == motion and prov["step"] == step
                assert ["118576", 2, 13500, 20999] in prov["exclude"]
                assert prov["label_constants_as_code"] and len(prov["recipe_sha256"]) == 64
                assert set(prov["sigma"]) == {"SIGMA_D", "SIGMA_F"}
                assert prov["sigma_measured"] == bh.SIGMA_MEASURED
                assert prov["window_strips_allowed"] is False
                assert prov["labels_coco"] is False and prov["labels_seed"] == 0
                assert len(prov["monitor"]) == step and len(prov["mix"]) == 8
            build_net(motion=motion).load_state_dict(torch.load(f"{out}.{steps}.pt"))
        m = hist[True][-1]
        assert m["n"] == 3 and m["motion_norm"] > 0 and hist[False][-1]["motion_norm"] is None
        assert set(m["hit_by_kind"]) == {"flight", "action"} and np.isfinite(m["loss"])
        assert hist[True][0]["motion_norm"] > 0
        assert any(s.startswith("  monitor 2") for s in logs)
        assert any("from all free strips" in s for s in logs)
        a = torch.load(f"{Path(tmp) / 'm'}.1.pt")
        b = torch.load(f"{Path(tmp) / 'rgb'}.1.pt")
        assert set(a) - set(b) == {"conv1_motion.weight"}
        # Both load for inference (`ball_heatmap.load_v2`, as eval and detect
        # do): the motion one with the scales it was trained with.
        _, im = bh.load_v2(f"{Path(tmp) / 'm'}.2.pt")
        _, ir = bh.load_v2(f"{Path(tmp) / 'rgb'}.1.pt")
        assert not im["rgb_only"] and (im["sigma_d"], im["sigma_f"]) == (bh.SIGMA_D, bh.SIGMA_F)
        assert ir["rgb_only"]
        bench = bench_v2(root, 2, True, at=1201, rounds=1, mix=small, log=logs.append)
        assert bench["ratio"] > 0
    print(f"  train_ball_heatmap: guard, window strips dropped, placeholder sigmas "
          f"refused, mix, augmentation in register, input, prefetch, losses, drops, "
          f"optimiser, steps of both models with monitor and provenance (both "
          f"through load_v2), bench; {time.time() - t_start:.0f} s")


def _refused(root: Path, table: pd.DataFrame, codes):
    table.to_parquet(root / "labels.parquet")
    try:
        load_strips(root)
        raise AssertionError("strips outside the training halves were accepted")
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
