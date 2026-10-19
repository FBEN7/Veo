"""Full-frame inference, 4 threads, synthetic: forward on a 1000 x 4096
crop (v1 vs motion; the two-conv form for reference), and detect vs
detect_v2 on a 1080 x 4096 frame whose pitch box is 1064 x 4080 (with the
motion channels and padding). Per-round ratios, median."""
import sys, time
import numpy as np, torch
sys.path.insert(0, "/home/user/Veo")
from src import ball_heatmap as bh
from src import motion_bg as mb
torch.set_num_threads(4)
F = torch.nn.functional
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
torch.manual_seed(0)
v1, m = bh.build().eval(), bh.build(motion=True).eval()
with torch.no_grad():
    m.conv1_motion.weight.normal_(0, 0.05)

def two_conv(model, x):
    s = model.stem
    y = s[3](s[2](s[1](s[0](x[:, :3]) + model.conv1_motion(x[:, 3:]))))
    c1 = model.layer1(y)
    up = F.interpolate(model.lat(model.layer2(c1)), size=c1.shape[2:], mode="bilinear", align_corners=False)
    return model.head(c1 + up)[:, 0]

# detect/detect_v2 compare with conv1_motion at zero (the motion path is
# computed in full; the weights only change which peaks pass tau, and a
# random motion stem on random frames gives thousands of peaks to outline-test).
m0 = bh.build(motion=True).eval()
m0.load_state_dict({**v1.state_dict(), "conv1_motion.weight": m0.conv1_motion.weight})
x6 = torch.randn(1, 6, 1000, 4096); x3 = x6[:, :3].contiguous()
rng = np.random.default_rng(0)
frame, prev, nxt = (rng.integers(0, 256, (1080, 4096, 3), dtype=np.uint8) for _ in range(3))
outline = [[40, 64], [4056, 64], [4056, 1032], [40, 1032]]
box = bh.pitch_box(frame.shape, outline)
bg = mb.grey(frame)[box[1]:box[3], box[0]:box[2]].copy()
fs = {"fwd v1": lambda: v1(x3), "fwd motion": lambda: m(x6), "fwd two-conv": lambda: two_conv(m, x6),
      "detect": lambda: bh.detect(v1, frame, outline, tau=0.05, top_k=8),
      "detect_v2": lambda: bh.detect_v2(m0, prev, frame, nxt, bg, outline, tau=0.05, top_k=8)}
T = {k: [] for k in fs}
with torch.inference_mode():
    for f in fs.values():
        f()
    for _ in range(reps):
        for k, f in fs.items():
            t0 = time.perf_counter(); f(); T[k].append(time.perf_counter() - t0)
for k, v in T.items():
    print(f"  {k:12s} median {np.median(v):.3f} s, min {min(v):.3f} s  all {[round(t, 2) for t in v]}")
for a, b in (("fwd motion", "fwd v1"), ("fwd two-conv", "fwd v1"), ("detect_v2", "detect")):
    rr = np.array(T[a]) / np.array(T[b])
    print(f"  {a} / {b}: per-round median {np.median(rr):.3f}, ratio of mins {min(T[a]) / min(T[b]):.3f}")
