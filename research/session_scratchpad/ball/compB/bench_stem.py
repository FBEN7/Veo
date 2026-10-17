import sys, time
import numpy as np, torch
sys.path.insert(0, "/home/user/Veo")
from src import ball_heatmap as bh
torch.set_num_threads(4)
F = torch.nn.functional
torch.manual_seed(0)
v1 = bh.build().eval()
m = bh.build(motion=True).eval()
with torch.no_grad():
    m.conv1_motion.weight.normal_(0, 0.05)
H, W = 1000, 4096
x6 = torch.randn(1, 6, H, W)
x3 = x6[:, :3].contiguous()

def rest(model, y):
    s = model.stem
    y = s[3](s[2](s[1](y)))
    c1 = model.layer1(y)
    up = F.interpolate(model.lat(model.layer2(c1)), size=c1.shape[2:], mode="bilinear", align_corners=False)
    return model.head(c1 + up)[:, 0]

variants = {
    "v1": lambda: v1(x3),
    "A two convs (current)": lambda: m(x6),
    "B contiguous slices": lambda: rest(m, m.stem[0](x6[:, :3].contiguous()) + m.conv1_motion(x6[:, 3:].contiguous())),
    "C one conv, cat weights": lambda: rest(m, F.conv2d(x6, torch.cat([m.stem[0].weight, m.conv1_motion.weight], 1), stride=2, padding=3)),
    "stem only v1": lambda: v1.stem[0](x3),
    "stem only C": lambda: F.conv2d(x6, torch.cat([m.stem[0].weight, m.conv1_motion.weight], 1), stride=2, padding=3),
    "stem only A": lambda: m.stem[0](x6[:, :3]) + m.conv1_motion(x6[:, 3:]),
}
times = {k: [] for k in variants}
with torch.inference_mode():
    for f in variants.values():
        f()
    for rep in range(int(sys.argv[1]) if len(sys.argv) > 1 else 3):
        for k, f in variants.items():
            t0 = time.perf_counter(); f(); times[k].append(time.perf_counter() - t0)
    a = variants["A two convs (current)"](); c = variants["C one conv, cat weights"]()
    print("max |A - C|", float((a - c).abs().max()))
base = np.median(times["v1"])
for k, v in times.items():
    print(f"{k:28s} median {np.median(v):.3f} s  min {np.min(v):.3f}  ratio(median) {np.median(v) / base:.3f}")
