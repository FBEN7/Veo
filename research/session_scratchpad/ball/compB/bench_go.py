"""Go/no-go timing on synthetic input, 4 torch threads: full-frame forward
(1000 x 4096) v1 vs motion; detect vs detect_v2 on a 1080 x 4096 frame
whose pitch box is ~1000 x 4096; one training step (forward + backward) of
batch 16 at 320 px, RGB-only vs motion, with the early layers frozen (steps
1-800) and all trainable (801-4000). Interleaved repeats, medians."""
import sys, time
import numpy as np, torch
sys.path.insert(0, "/home/user/Veo")
from src import ball_heatmap as bh
from src import motion_bg as mb
torch.set_num_threads(4)
F = torch.nn.functional
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 4
torch.manual_seed(0)
v1 = bh.build()
m = bh.build(motion=True)
with torch.no_grad():
    m.conv1_motion.weight.normal_(0, 0.05)

def two_conv_forward(model, x):        # the plan's literal form, for reference
    s = model.stem
    y = s[3](s[2](s[1](s[0](x[:, :3]) + model.conv1_motion(x[:, 3:]))))
    c1 = model.layer1(y)
    up = F.interpolate(model.lat(model.layer2(c1)), size=c1.shape[2:], mode="bilinear", align_corners=False)
    return model.head(c1 + up)[:, 0]

def med(ts):
    return float(np.median(ts))

# ---- inference ----
x6 = torch.randn(1, 6, 1000, 4096); x3 = x6[:, :3].contiguous()
rng = np.random.default_rng(0)
frame = rng.integers(0, 256, (1080, 4096, 3), dtype=np.uint8)
prev = rng.integers(0, 256, (1080, 4096, 3), dtype=np.uint8)
nxt = rng.integers(0, 256, (1080, 4096, 3), dtype=np.uint8)
outline = [[40, 64], [4056, 64], [4056, 1032], [40, 1032]]     # box 0..4096 x 0..1064
box = bh.pitch_box(frame.shape, outline)
bg = mb.grey(frame)[box[1]:box[3], box[0]:box[2]].copy()
inf = {"forward v1": lambda: v1(x3), "forward motion": lambda: m(x6),
       "forward motion (two convs)": lambda: two_conv_forward(m, x6),
       "detect v1": lambda: bh.detect(v1, frame, outline, tau=0.05, top_k=8),
       "detect_v2": lambda: bh.detect_v2(m, prev, frame, nxt, bg, outline, tau=0.05, top_k=8),
       "motion_channels+to_tensor_v2 only": lambda: bh.to_tensor_v2(
           [frame[box[1]:box[3], box[0]:box[2]]],
           [mb.motion_channels(prev[box[1]:box[3], box[0]:box[2]], frame[box[1]:box[3], box[0]:box[2]],
                               nxt[box[1]:box[3], box[0]:box[2]], bg)]),
       "to_tensor only": lambda: bh.to_tensor([frame[box[1]:box[3], box[0]:box[2]]])}
v1.eval(); m.eval()
T = {k: [] for k in inf}
with torch.inference_mode():
    for f in inf.values():
        f()
    for _ in range(reps):
        for k, f in inf.items():
            t0 = time.perf_counter(); f(); T[k].append(time.perf_counter() - t0)
print(f"pitch box {box} -> crop {(box[3]-box[1], box[2]-box[0])}")
for k, v in T.items():
    print(f"  {k:36s} {med(v):.3f} s (min {min(v):.3f})")
print(f"  RATIO forward motion / v1          {med(T['forward motion']) / med(T['forward v1']):.3f}  (two convs: "
      f"{med(T['forward motion (two convs)']) / med(T['forward v1']):.3f})")
print(f"  RATIO detect_v2 / detect           {med(T['detect_v2']) / med(T['detect v1']):.3f}")

# ---- training step ----
def freeze(model, motion, frozen):
    for p in model.parameters():
        p.requires_grad_(True)
    if frozen:
        early = list(model.stem.parameters()) + list(model.layer1.parameters())
        for p in early:
            p.requires_grad_(False)

xb6 = torch.randn(16, 6, 320, 320); xb3 = xb6[:, :3].contiguous()
def step(model, x, fwd=None):
    model.zero_grad(set_to_none=True)
    z = model(x) if fwd is None else fwd(model, x)
    loss = F.binary_cross_entropy_with_logits(z, torch.zeros_like(z), reduction="sum")
    loss.backward()

cfg = {}
for frozen in (True, False):
    tag = "frozen" if frozen else "full"
    cfg[f"rgb {tag}"] = (v1, xb3, None, frozen)
    cfg[f"motion {tag}"] = (m, xb6, None, frozen)
    cfg[f"motion two-conv {tag}"] = (m, xb6, two_conv_forward, frozen)
v1.train(); m.train()
TT = {k: [] for k in cfg}
n_steps = 0
for k, (model, x, fwd, frozen) in cfg.items():
    freeze(model, model is m, frozen); step(model, x, fwd); n_steps += 1
for _ in range(reps):
    for k, (model, x, fwd, frozen) in cfg.items():
        freeze(model, model is m, frozen)
        t0 = time.perf_counter(); step(model, x, fwd); TT[k].append(time.perf_counter() - t0); n_steps += 1
for k, v in TT.items():
    print(f"  step {k:28s} {med(v):.3f} s (min {min(v):.3f})")
for tag in ("frozen", "full"):
    print(f"  RATIO step motion / rgb, {tag:6s}    {med(TT[f'motion {tag}']) / med(TT[f'rgb {tag}']):.3f}  (two convs: "
          f"{med(TT[f'motion two-conv {tag}']) / med(TT[f'rgb {tag}']):.3f})")
w = lambda pre: 0.2 * med(TT[f"{pre} frozen"]) + 0.8 * med(TT[f"{pre} full"])
print(f"  RATIO step motion / rgb, 4000-step mix (800 frozen + 3200 full): {w('motion') / w('rgb'):.3f}")
print(f"  ({n_steps} forward+backward passes, no optimiser steps)")
