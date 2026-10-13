import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/user/Veo")
import train_ball_heatmap as tr
from src.ball_heatmap import build

root = Path(sys.argv[1])
table = pd.read_parquet(root / "strips.parquet")
rows = list(table.itertuples(index=False))
torch.manual_seed(0)
torch.set_num_threads(4)
m = build(motion=True)
opt, early = tr.optimiser(m)
idx = np.arange(min(8, len(rows)))
codes = np.full((len(rows), 80, 80), -1, np.int8)
codes[:, 38:42, 38:42] = 1
flight = torch.tensor([r.kind == "flight" for r in rows])[idx]
seeds = np.random.default_rng(0).integers(2 ** 62, size=len(idx))
for step in (1, 2, 3):
    x, c = tr.make_batch(root, rows, codes, idx, seeds + step, True)
    print("input motion abs max", float(x[:, 3:].abs().max()), "finite", bool(torch.isfinite(x).all()))
    s = tr.lr_scale(step, 4000)
    base = [3e-4, 1e-4, 3e-5, 3e-4]
    for g, b in zip(opt.param_groups, [gg["lr"] for gg in opt.param_groups]):
        pass
    for g in opt.param_groups:
        g["lr"] = tr.LR_V2[g["name"]] * s
    m.train()
    z = m(x)
    pos, has, neg, peak = tr.strip_losses(z, c, flight, torch.ones(len(idx)),
                                          torch.zeros(len(idx), dtype=torch.bool))
    loss = (pos.sum() + neg.sum() + peak.sum()) / len(idx)
    opt.zero_grad()
    loss.backward()
    g = m.conv1_motion.weight.grad
    print(f"step {step}: loss {loss.item():.4f} grad norm {g.norm().item():.3e} "
          f"conv1 grad {m.stem[0].weight.grad}")
    opt.step()
    print("   conv1_motion |w|", m.conv1_motion.weight.norm().item())
