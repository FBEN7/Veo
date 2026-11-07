"""Ball detector v4: ball_v3b.1500 fine-tuned on one fold's clicked strips
(prepare.py, cut on Kaggle) and a subset of v2's stream strips (so the
detector keeps what it knew about passes and empty pitch).

Recipe: train_ball_v3.py's (optimiser, learning rates, frozen early batch
norms, EMA 0.999, strip losses, worst-bag drops off) with:
- init ball_v3b.1500.pt (all keys, strict);
- batches of 16: 8 clicked balls ("verified": footprint bags), 2 clicked
  negatives ("verified_neg"), 2 flight, 1 action, 3 free (stream strips);
- 600 steps (warm-up 100, cosine), EMA checkpoints at 300 and 600.
No strip of M6 (the guard) or of the sealed M7 / M9 is trained on (M6's
first half is in v3b's provenance already, through its v1 init); M1 may be
trained on now (its tests are spent).

    python train_v4.py --fold A --strips STRIPS_DIR --out .cache/ball_v4A [--recipe v4r]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import train_ball_heatmap as tb  # noqa: E402
import train_ball_v3 as t3  # noqa: E402
from src import ball_heatmap as bh  # noqa: E402

INIT = ROOT / ".cache/ball_v3b.1500.pt"
NEVER_V4 = {"118578", "128057", "132831"}     # no strip of these in v4's batches
SEALED = {"128057", "132831"}                # and never in a provenance (v3b's has M6 half 1, from v1)
MIX_V4 = (("verified", None, 8), ("verified_neg", None, 2), ("flight", None, 2),
          ("action", None, 1), ("free", None, 3))
MIX_V4R = (("verified", None, 6), ("verified_neg", None, 2), ("flight", None, 2),
           ("action", None, 2), ("free", None, 4))
# v4r (protocol_v4r.json): v4 forgot ordinary play (M6 guard), so more stream
# strips per batch, half the learning rates, and the stream strips' logits
# kept near v3b's (distillation: mean squared difference per strip, logits
# floored at DISTILL_FLOOR so that the many sure-negative cells do not count).
# Weight 5: v4's checkpoints sit at 0.10 per stream strip from v3b (180 stream
# strips, no augmentation), so with 8 stream strips in 16 that drift costs
# 0.25 a batch, against the ~0.6 the clicked strips' loss fell in v4.
RECIPES = {"v4": {"mix": MIX_V4, "lr": 1.0, "distill": 0.0},
           "v4r": {"mix": MIX_V4R, "lr": 0.5, "distill": 5.0}}
DISTILL_FLOOR = -6.0
STEPS = 600                                  # EMA checkpoints at half and at the end


def load(strips: Path, fold: str):
    t = pd.read_parquet(strips / "labels.parquet")
    c = np.load(strips / "labels_codes.npy", mmap_mode="r")
    t["match"], t["category"] = t.match.astype(str), t.category.astype(str)
    keep = (t.source == "stream") | (t.fold.astype(str) == fold)
    idx = np.flatnonzero(keep.to_numpy())
    t = t.iloc[idx].reset_index(drop=True)
    c = np.asarray(c[idx])
    bad = sorted(set(t.match) & NEVER_V4)
    if bad:
        raise SystemExit(f"never-train matches in the strips: {bad}")
    return t, c


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--strips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="v4")
    args = ap.parse_args()
    rec = RECIPES[args.recipe]
    log = lambda s: print(s, flush=True)
    strips = Path(args.strips)
    table, codes = load(strips, args.fold)
    log(f"fold {args.fold}: {len(table)} strips {table.kind.value_counts().to_dict()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = bh.build(motion=True)
    model.load_state_dict(torch.load(INIT, map_location="cpu"), strict=True)
    model.to(device)
    ema = copy.deepcopy(model)
    teacher = copy.deepcopy(model).eval() if rec["distill"] else None
    opt = t3.optimiser_v3(model)
    base = [g["lr"] * rec["lr"] for g in opt.param_groups]
    stream = (table.source.astype(str) == "stream").to_numpy()
    kinds = table.kind.to_numpy()
    bag = np.isin(kinds, t3.BAG_KINDS)
    po = table.positive_only.to_numpy().astype(bool)
    weight = table.weight.to_numpy().astype(np.float32)
    bt = tb.Batches(strips, table, codes, True, args.seed, mix=rec["mix"])
    bns = t3.frozen_bn(model)
    acc, t0 = [], time.time()
    checkpoints = (args.steps // 2, args.steps)
    try:
        for step in range(1, args.steps + 1):
            idx, x, c, _ = bt.next()
            s = tb.lr_scale(step, args.steps)
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * s
            model.train()
            for m in bns:
                m.eval()
            logits = model(x.to(device)).float().cpu()      # the losses as v3's, on the CPU
            pos, has, neg, peak = tb.strip_losses(
                logits, c, torch.from_numpy(bag[idx]), torch.from_numpy(weight[idx]),
                torch.from_numpy(po[idx]))
            loss = (torch.where(has, pos, torch.zeros_like(pos)).sum() + neg.sum() + peak.sum()) / len(idx)
            dist = torch.zeros(())
            if teacher is not None and stream[idx].any():
                with torch.no_grad():
                    t_logits = teacher(x.to(device)).float().cpu()
                sel = torch.from_numpy(stream[idx])
                d = logits[sel].clamp(min=DISTILL_FLOOR) - t_logits[sel].clamp(min=DISTILL_FLOOR)
                dist = rec["distill"] * (d ** 2).flatten(1).mean(1).sum() / len(idx)
                loss = loss + dist
            opt.zero_grad()
            loss.backward()
            opt.step()
            t3.ema_update(ema, model, step)
            acc.append([loss.item(), neg.sum().item() / len(idx), dist.item()])
            if step % 50 == 0 or step == args.steps:
                a = np.mean(acc, 0)
                log(f"  step {step}: loss {a[0]:.4f} (negatives {a[1]:.4f}, distillation {a[2]:.4f}) "
                    f"[{time.time() - t0:.0f} s]")
                acc = []
            if step in checkpoints:
                ckpt = Path(f"{args.out}.{step}.pt")
                torch.save({k: v.cpu() for k, v in ema.state_dict().items()}, ckpt)
                prov = json.loads(INIT.with_name(INIT.name + ".provenance.json").read_text())
                train = {str(m): sorted(set(int(h) for h in hs)) for m, hs in prov["train"].items()}
                for m, h in set(zip(table.match, table.half.astype(int))):
                    train.setdefault(m, [])
                    if h not in train[m]:
                        train[m] = sorted(train[m] + [h])
                bh.write_provenance(ckpt, {
                    "protocol": f"ball {args.recipe} (clicked balls)", "fold": args.fold, "train": train,
                    "init": str(INIT.name), "steps": args.steps, "step": step, "seed": args.seed,
                    "motion": True, "sigma": {"SIGMA_D": bh.SIGMA_D, "SIGMA_F": bh.SIGMA_F},
                    "mix": [list(m) for m in rec["mix"]], "lr scale": rec["lr"],
                    "distill": rec["distill"], "distill floor": DISTILL_FLOOR,
                    "n_train": table.kind.value_counts().sort_index().to_dict()}, never=SEALED)
                cal = INIT.with_name(INIT.name + ".calib.json")
                if cal.exists():             # v3's Platt calibration, an approximation for v4
                    Path(str(ckpt) + ".calib.json").write_text(cal.read_text())
                log(f"  saved {ckpt}")
    finally:
        bt.close()


if __name__ == "__main__":
    main()
