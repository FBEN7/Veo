"""Ball detector v3 (protocol v5, part 1): v2's architecture fine-tuned on
the verified labels of the seen windows (`build_verified_strips.py`) and
v2's own stream strips, from v2.4000 (run "a") or from v1 (run "b": v1's
weights, the motion stem at zero). The choice between the checkpoints is
registered in `.cache/soccertrack_v2/ball_verified/v3_selection.json`.

    python train_ball_v3.py --init v2 --out .cache/ball_v3a
    python train_ball_v3.py --init v1 --out .cache/ball_v3b
    python train_ball_v3.py --smoke     # 20 steps into the scratchpad

Recipe (v2's `train_ball_heatmap.py` where not said):
- batches of 16: 8 verified (a ball), 3 verified_neg, 2 flight, 1 action,
  1 free cluster, 1 free landmark; the verified strips drawn with weights
  1 / sqrt(strips of their match) (`draw_w`), the others uniformly;
- loss `strip_losses`, verified bags pooled as flight bags (footprint 4,
  r 8: a peak, not a glow); v2's worst-bag drops for its own kinds;
- 2,000 steps; AdamW (weight decay 1e-4), learning rates head / lateral
  1e-4, motion stem 1e-4, layer2 3e-5, conv1 / bn1 / layer1 1e-5 (no
  freeze); warm-up 100 steps, cosine to 0; the batch norms of the stem and
  layer1 kept in eval mode (16 ball-centred crops must not move the
  statistics used on the full panorama);
- an exponential moving average of the weights (decay 0.999, warmed up as
  (1 + step) / (10 + step)): the checkpoints at 1,000, 1,500 and 2,000
  steps are the averaged weights (`<out>.<step>.pt`, provenance with
  `NEVER_V3` / `NEVER_V3_FROM_V1`);
- every 250 steps the state is saved (`<out>.resume.pt`: weights, average,
  optimiser, step); a restarted run continues from it with the batch
  generator reseeded by (seed, step), recorded in the provenance;
- monitor: the held-out strips (v2's 300 and the window
  st2_128058_2nd_f015000) every 250 steps, on the averaged weights.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import train_ball_heatmap as tb
from src import ball_heatmap as bh

V3_STRIPS = Path(".cache/soccertrack_v2/ball_strips_v3")
INITS = {"v2": Path(".cache/ball_v2.4000.pt"), "v1": Path(".cache/ball_heatmap.3000.pt")}
INIT_SHA = {"v2": "31f9a5eb"}
# v1 (protocol v1): first halves of M3, M4, M5, M6, M8 (SOCCERTRACK_V2.md, "A ball
# detector for the panorama"); it has no provenance file.
V1_TRAIN = {"118575": [1], "118576": [1], "118577": [1], "118578": [1], "128058": [1]}
SEALED = {"128057", "132831"}
MIX_V3 = (("verified", None, 8), ("verified_neg", None, 3), ("flight", None, 2),
          ("action", None, 1), ("free", "cluster", 1), ("free", "landmark", 1))
STEPS = 2000
CHECKPOINTS = (1000, 1500, 2000)
LR_V3 = {"head": 1e-4, "motion": 1e-4, "layer2": 3e-5, "early": 1e-5}
EMA_DECAY = 0.999
RESUME_EVERY = 250
BAG_KINDS = ("flight", "verified")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_strips_v3(root: Path, log=print):
    """The combined table and codes, refusing what v3 must not train on:
    any strip of a sealed match, M1, or M6 (the selection match); stream
    strips outside v2's training halves; window strips of a window not in
    the verified set."""
    from build_ball_strips import excluded

    table = pd.read_parquet(root / "labels.parquet")
    codes = np.load(root / "labels_codes.npy", mmap_mode="r")
    if len(codes) != len(table) or codes.shape[1:] != (tb.CELLS, tb.CELLS):
        raise SystemExit("labels_codes.npy does not match labels.parquet")
    table["match"] = table.match.astype(str)
    table["category"] = table.category.astype(str)
    bad = table.match.isin(bh.NEVER_V3)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} strips of never-train matches "
                         f"{sorted(set(table.match[bad]))}")
    stream = (table.source.astype(str) == "stream").to_numpy()
    s = table[stream]
    wrong = [(m, h, f) for m, h, f in zip(s.match, s.half.astype(int), s.frame.astype(int))
             if h not in bh.V2_TRAIN.get(m, ()) or excluded(f, m, h)]
    if wrong:
        raise SystemExit(f"{len(wrong)} stream strips outside v2's training halves: {wrong[:3]}")
    meta = json.loads((root / "labels_meta.json").read_text())
    w = table[~stream]
    if not w.kind.isin(["verified", "verified_neg"]).all() or (w.source != w.window).any():
        raise SystemExit("window strips other than the verified ones")
    log(f"  {len(table)} strips: {table.kind.value_counts().sort_index().to_dict()}; "
        f"{int(table.holdout.sum())} held out")
    return table, codes, meta


class WeightedSampler:
    """`train_ball_heatmap.Sampler` with the verified strips drawn by `draw_w`."""

    def __init__(self, table: pd.DataFrame, index: np.ndarray, mix=MIX_V3):
        kind = table.kind.to_numpy()[index]
        cat = table.category.to_numpy()[index]
        w = table.draw_w.to_numpy(float)[index]
        self.pools, self.fallback = [], []
        for k, c, n in mix:
            sel = (kind == k) & ((cat == c) if c else True)
            if not sel.any():
                raise SystemExit(f"no {k} {c or ''} strips to train on")
            p = w[sel] / w[sel].sum()
            self.pools.append((k, c, n, index[sel], p))

    def draw(self, rng) -> np.ndarray:
        return np.concatenate([rng.choice(pool, n, p=p) for _, _, n, pool, p in self.pools])


def optimiser_v3(model):
    import torch

    early = (list(model.stem[0].parameters()) + list(model.stem[1].parameters())
             + list(model.layer1.parameters()))
    groups = [{"params": list(model.lat.parameters()) + list(model.head.parameters()),
               "lr": LR_V3["head"], "name": "head"},
              {"params": list(model.layer2.parameters()), "lr": LR_V3["layer2"], "name": "layer2"},
              {"params": early, "lr": LR_V3["early"], "name": "early"},
              {"params": [model.conv1_motion.weight], "lr": LR_V3["motion"], "name": "motion"}]
    return torch.optim.AdamW(groups, weight_decay=tb.WEIGHT_DECAY)


def frozen_bn(model):
    """The batch norms whose statistics stay as initialised."""
    import torch

    mods = [model.stem[1]] + [m for m in model.layer1.modules()
                              if isinstance(m, torch.nn.BatchNorm2d)]
    return mods


def ema_update(ema, model, step: int):
    import torch

    d = min(EMA_DECAY, (1 + step) / (10 + step))
    with torch.no_grad():
        for (k, e), v in zip(ema.state_dict().items(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(v, alpha=1 - d)
            else:
                e.copy_(v)


def init_model(init: str):
    import torch

    model = bh.build(motion=True)
    sd = torch.load(INITS[init], map_location="cpu")
    r = model.load_state_dict(sd, strict=False)
    allowed = [] if init == "v2" else ["conv1_motion.weight"]
    if r.unexpected_keys or sorted(r.missing_keys) != allowed:
        raise SystemExit(f"init {init}: missing {r.missing_keys}, unexpected {r.unexpected_keys}")
    if init == "v2" and not sha256(INITS["v2"]).startswith(INIT_SHA["v2"]):
        raise SystemExit("ball_v2.4000.pt is not the registered checkpoint")
    return model


def train_matches(table: pd.DataFrame, init: str) -> dict:
    """{match: halves}: the strips trained on and the init's own training."""
    tr = table[~table.holdout.astype(bool)]
    out = {}
    for m, h in set(zip(tr.match, tr.half.astype(int))):
        out.setdefault(m, set()).add(h)
    parent = (bh.read_provenance(INITS["v2"])["train"] if init == "v2" else V1_TRAIN)
    for m, hs in parent.items():
        out.setdefault(str(m), set()).update(int(h) for h in hs)
    return {m: sorted(hs) for m, hs in sorted(out.items())}


def _save_atomic(obj, path: Path):
    import torch

    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(init: str, out: str, root: Path = V3_STRIPS, steps: int = STEPS,
          checkpoints=CHECKPOINTS, seed: int = 0, monitor_every: int = 250, log=print):
    import torch

    if bh.SIGMA_D != 0.0535 or bh.SIGMA_F != 0.0174:
        raise SystemExit("SIGMA_D / SIGMA_F differ from v2's")
    table, codes, meta = load_strips_v3(root, log)
    torch.set_num_threads(4)
    never = bh.NEVER_V3 if init == "v2" else bh.NEVER_V3_FROM_V1
    train_set = train_matches(table, init)
    if set(train_set) & (set(never) | SEALED):
        raise SystemExit(f"training set {sorted(train_set)} meets never-train {sorted(never)}")
    model = init_model(init)
    ema = copy.deepcopy(model)
    opt = optimiser_v3(model)
    base = [g["lr"] for g in opt.param_groups]
    kinds = table.kind.to_numpy()
    bag = np.isin(kinds, BAG_KINDS)
    po = table.positive_only.to_numpy().astype(bool)
    weight = table.weight.to_numpy().astype(np.float32)
    held = np.flatnonzero(table.holdout.to_numpy().astype(bool))
    rows = list(table.itertuples(index=False))
    resume = Path(f"{out}.resume.pt")
    start, resumed, history = 1, [], []
    if resume.exists():
        st = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        start, resumed, history = st["step"] + 1, st["resumed"] + [st["step"]], st["history"]
        log(f"  resuming after step {st['step']}")

    def batches():
        b = tb.Batches(root, table, codes, True, seed)
        b.sampler = WeightedSampler(table, np.flatnonzero(~table.holdout.to_numpy().astype(bool)))
        b.rng = np.random.default_rng([seed, start])
        return b

    bt = batches()
    drop_rng = np.random.default_rng([seed, 1, start])
    bns = frozen_bn(model)
    acc, t0 = [], time.time()
    try:
        for step in range(start, steps + 1):
            idx, x, c, _ = bt.next()
            s = tb.lr_scale(step, steps)
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * s
            model.train()
            for m in bns:
                m.eval()
            logits = model(x)
            pos, has, neg, peak = tb.strip_losses(
                logits, c, torch.from_numpy(bag[idx]), torch.from_numpy(weight[idx]),
                torch.from_numpy(po[idx]))
            keep = has.clone()
            if step > tb.DROP_FROM:
                keep &= ~torch.from_numpy(tb.drop_worst(pos, has, kinds[idx], drop_rng))
            loss = (torch.where(keep, pos, torch.zeros_like(pos)).sum() + neg.sum()
                    + peak.sum()) / len(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(ema, model, step)
            acc.append([loss.item(), pos[keep].sum().item() / len(idx), neg.sum().item() / len(idx),
                        peak.sum().item() / len(idx)])
            if step % 100 == 0 or step == steps:
                a = np.mean(acc, 0)
                log(f"  step {step}: loss {a[0]:.4f} (bags {a[1]:.4f}, negatives {a[2]:.4f}, "
                    f"peaks {a[3]:.4f})  [{time.time() - t0:.0f} s]")
                acc = []
            if (step % monitor_every == 0 or step == steps) and len(held):
                m = tb.monitor(ema, root, rows, codes, held, True)
                history.append(dict(m, step=step))
                log(tb._fmt_monitor(step, m))
            if step in checkpoints or step == steps:
                ckpt = Path(f"{out}.{step}.pt")
                _save_atomic(ema.state_dict(), ckpt)
                bh.write_provenance(ckpt, provenance(
                    init, out, root, table, meta, train_set, step, steps, seed, history,
                    resumed), never=never)
                log(f"  saved {ckpt}")
            if step % RESUME_EVERY == 0 and step < steps:
                _save_atomic({"model": model.state_dict(), "ema": ema.state_dict(),
                              "opt": opt.state_dict(), "step": step, "resumed": resumed,
                              "history": history}, resume)
    finally:
        bt.close()
    resume.unlink(missing_ok=True)
    return history


def provenance(init, out, root, table, meta, train_set, step, steps, seed, history,
               resumed) -> dict:
    held = table.holdout.to_numpy().astype(bool)
    ver = table[table.kind.isin(["verified", "verified_neg"])]
    return {
        "protocol": "v5 part 1 (v3)", "run": Path(out).name, "train": train_set,
        "exclude": {}, "never": sorted(bh.NEVER_V3 if init == "v2" else bh.NEVER_V3_FROM_V1),
        "init": {"from": init, "path": str(INITS[init]), "sha256": sha256(INITS[init]),
                 "train": bh.read_provenance(INITS["v2"])["train"] if init == "v2" else V1_TRAIN,
                 "chosen on": "118578 (M6): v2's checkpoint by FF-auto, v1's on T windows"},
        "selection": ".cache/soccertrack_v2/ball_verified/v3_selection.json",
        "train windows": sorted(set(ver[~held[ver.index]].window)),
        "monitor window": sorted(set(ver[held[ver.index]].window)),
        "label sources": meta.get("v3"), "label_constants": meta["constants"],
        "sigma": {"SIGMA_D": bh.SIGMA_D, "SIGMA_F": bh.SIGMA_F}, "sigma_measured": True,
        "files_sha256": {f: sha256(f) for f in ("train_ball_v3.py", "train_ball_heatmap.py",
                                                "build_verified_strips.py", "src/ball_heatmap.py",
                                                "src/motion_bg.py")},
        "seed": seed, "motion": True, "step": step, "steps": steps, "strips": str(root),
        "window_strips_allowed": True,
        "labels_sha256": hashlib.sha256((root / "labels.parquet").read_bytes()).hexdigest(),
        "n_train": table[~held].kind.value_counts().sort_index().to_dict(),
        "n_holdout": table[held].kind.value_counts().sort_index().to_dict(),
        "sources": {"stream": int((table.source == "stream").sum()),
                    "windows": int((table.source != "stream").sum())},
        "mix": [list(m) for m in MIX_V3], "lr": LR_V3, "weight_decay": tb.WEIGHT_DECAY,
        "freeze_to": 0, "frozen batch norms": "stem bn1, layer1", "warmup": tb.WARMUP,
        "ema": EMA_DECAY, "weights saved": "EMA", "drop_from": tb.DROP_FROM, "drop": tb.DROP_V2,
        "bag kinds (footprint pooling)": list(BAG_KINDS),
        "loss": {"footprint": tb.FOOTPRINT, "pool_r": tb.POOL_R, "lme_r": tb.LME_R,
                 "gamma": tb.GAMMA, "hard": tb.HARD, "peaks": tb.N_PEAKS},
        "augment": tb.AUG, "resumed after steps": resumed, "monitor": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=sorted(INITS))
    ap.add_argument("--out")
    ap.add_argument("--strips", default=str(V3_STRIPS))
    ap.add_argument("--smoke", metavar="OUT_PREFIX",
                    help="20 steps, checkpoints at 10 and 20, monitor every 10")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)
    if args.smoke:
        train(args.init or "v2", args.smoke, Path(args.strips), steps=20, checkpoints=(10, 20),
              monitor_every=10, log=log)
        return
    if not (args.init and args.out):
        ap.error("--init and --out are needed")
    train(args.init, args.out, Path(args.strips), log=log)


if __name__ == "__main__":
    main()
