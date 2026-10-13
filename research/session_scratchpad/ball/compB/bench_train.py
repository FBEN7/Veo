"""One training step (forward + backward, no optimiser) of batch 16 at
320 px, RGB-only vs motion, early layers frozen (steps 1-800) or all
trainable (801-4000). 4 torch threads, interleaved; min and median."""
import sys, time
import numpy as np, torch
sys.path.insert(0, "/home/user/Veo")
from src import ball_heatmap as bh
torch.set_num_threads(4)
F = torch.nn.functional
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 6
torch.manual_seed(0)
v1, m = bh.build(), bh.build(motion=True)
with torch.no_grad():
    m.conv1_motion.weight.normal_(0, 0.05)
xb6 = torch.randn(16, 6, 320, 320); xb3 = xb6[:, :3].contiguous()

def freeze(model, frozen):
    for p in model.parameters():
        p.requires_grad_(True)
    if frozen:
        for p in list(model.stem.parameters()) + list(model.layer1.parameters()):
            p.requires_grad_(False)

def step(model, x):
    model.zero_grad(set_to_none=True)
    z = model(x)
    F.binary_cross_entropy_with_logits(z, torch.zeros_like(z), reduction="sum").backward()

cfg = {"rgb frozen": (v1, xb3, True), "motion frozen": (m, xb6, True),
       "rgb full": (v1, xb3, False), "motion full": (m, xb6, False)}
v1.train(); m.train()
T = {k: [] for k in cfg}
for k, (model, x, fr) in cfg.items():
    freeze(model, fr); step(model, x)
for _ in range(reps):
    for k, (model, x, fr) in cfg.items():
        freeze(model, fr)
        t0 = time.perf_counter(); step(model, x); T[k].append(time.perf_counter() - t0)
for k, v in T.items():
    print(f"  step {k:14s} median {np.median(v):.3f} s, min {min(v):.3f} s  all {[round(t, 2) for t in v]}")
for tag in ("frozen", "full"):
    rr = np.array(T[f"motion {tag}"]) / np.array(T[f"rgb {tag}"])
    print(f"  per-round ratio motion/rgb {tag}: median {np.median(rr):.3f} {[round(x, 2) for x in rr]}")
rf = np.median(np.array(T["motion frozen"]) / np.array(T["rgb frozen"]))
ru = np.median(np.array(T["motion full"]) / np.array(T["rgb full"]))
print(f"  per-round 4000-step mix (time-weighted by rgb mins): "
      f"{(0.2 * rf * min(T['rgb frozen']) + 0.8 * ru * min(T['rgb full'])) / (0.2 * min(T['rgb frozen']) + 0.8 * min(T['rgb full'])):.3f}")
for stat, f in (("median", np.median), ("min", np.min)):
    r = {k: f(v) for k, v in T.items()}
    mix = lambda p: 0.2 * r[f"{p} frozen"] + 0.8 * r[f"{p} full"]
    print(f"  ({stat}) motion/rgb: frozen {r['motion frozen'] / r['rgb frozen']:.3f}, "
          f"full {r['motion full'] / r['rgb full']:.3f}, 4000-step mix {mix('motion') / mix('rgb'):.3f}")
