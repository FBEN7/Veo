import sys, time
import numpy as np, torch, cv2
sys.path.insert(0, "/home/user/Veo")
from src import ball_heatmap as bh
from src import motion_bg as mb
torch.set_num_threads(4)
rng = np.random.default_rng(0)
frame, prev, nxt = (rng.integers(0, 256, (1080, 4096, 3), dtype=np.uint8) for _ in range(3))
outline = [[40, 64], [4056, 64], [4056, 1032], [40, 1032]]
x0, y0, x1, y1 = bh.pitch_box(frame.shape, outline)
bg = mb.grey(frame)[y0:y1, x0:x1].copy()
cut = lambda im: im[y0:y1, x0:x1]
def tm(name, f, n=3):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t0)
    print(f"  {name:40s} {min(ts) * 1000:.0f} ms"); return r
crop = cut(frame)
ph, pw = (-crop.shape[0]) % 8, (-crop.shape[1]) % 8
mo = tm("motion_channels", lambda: mb.motion_channels(cut(prev), crop, cut(nxt), bg))
tm("grey of a crop view", lambda: mb.grey(cut(prev)))
cp = tm("copyMakeBorder", lambda: cv2.copyMakeBorder(crop, 0, ph, 0, pw, cv2.BORDER_CONSTANT))
mp = tm("np.pad motion", lambda: np.pad(mo, ((0, ph), (0, pw), (0, 0))))
tm("to_tensor (v1)", lambda: bh.to_tensor([cp]))
tm("to_tensor_v2", lambda: bh.to_tensor_v2([cp], [mp]))
tm("detect v1 total", lambda: bh.detect(bh_v1, frame, outline, tau=0.05), 1) if False else None
