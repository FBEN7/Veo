import sys, time
import numpy as np, torch, cv2
sys.path.insert(0, "/home/user/Veo")
from src import ball_heatmap as bh
from src import motion_bg as mb
torch.set_num_threads(4)
torch.manual_seed(0)
v1 = bh.build().eval()
m0 = bh.build(motion=True).eval()
m0.load_state_dict({**v1.state_dict(), "conv1_motion.weight": m0.conv1_motion.weight})
rng = np.random.default_rng(0)
frame, prev, nxt = (rng.integers(0, 256, (1080, 4096, 3), dtype=np.uint8) for _ in range(3))
outline = [[40, 64], [4056, 64], [4056, 1032], [40, 1032]]
x0, y0, x1, y1 = bh.pitch_box(frame.shape, outline)
poly = np.asarray(outline, np.float32)
bg = mb.grey(frame)[y0:y1, x0:x1].copy()
cut = lambda im: im[y0:y1, x0:x1]
crop = cut(frame); ph, pw = (-crop.shape[0]) % 8, (-crop.shape[1]) % 8
for rnd in range(3):
    t = [time.perf_counter()]
    xa = bh.to_tensor([cv2.copyMakeBorder(crop, 0, ph, 0, pw, cv2.BORDER_CONSTANT)]); t.append(time.perf_counter())
    with torch.inference_mode():
        za = v1(xa)[0]; t.append(time.perf_counter())
        pa = bh._peaks(za, x0, y0, poly, 0.05, 8, 10.0); t.append(time.perf_counter())
    mo = mb.motion_channels(cut(prev), crop, cut(nxt), bg); t.append(time.perf_counter())
    xb = bh.to_tensor_v2([cv2.copyMakeBorder(crop, 0, ph, 0, pw, cv2.BORDER_CONSTANT)], [np.pad(mo, ((0, ph), (0, pw), (0, 0)))]); t.append(time.perf_counter())
    with torch.inference_mode():
        zb = m0(xb)[0]; t.append(time.perf_counter())
        pb = bh._peaks(zb, x0, y0, poly, 0.05, 8, 10.0); t.append(time.perf_counter())
    d = np.diff(t)
    print(f"v1: tensor {d[0]:.3f} fwd {d[1]:.3f} peaks {d[2]:.3f} | v2: motion {d[3]:.3f} tensor {d[4]:.3f} fwd {d[5]:.3f} peaks {d[6]:.3f}"
          f" | n peaks {0 if pa is None else len(pa)} {0 if pb is None else len(pb)} | v1 {sum(d[:3]):.2f} v2 {sum(d[3:]):.2f}")
