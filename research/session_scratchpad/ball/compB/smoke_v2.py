"""detect_v2 on a training window: zero-initialised motion stem loaded with
v1 weights must give v1's candidates; background by sequential decode vs by
seeking; raw D/F scales on pitch pixels (informational)."""
import json, sys, time
import cv2, numpy as np, torch
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
from src import ball_heatmap as bh
from src import motion_bg as mb
from src.paths import DATA_DIR

w = sys.argv[1] if len(sys.argv) > 1 else "st2_118577_1st_f030000"
torch.set_num_threads(4)
info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
cal = st.calibration(info["match"])
first = info["start_frame"]; last = first + info["frames"] - 1
sd = torch.load("/home/user/Veo/.cache/ball_heatmap.3000.pt")
v1 = bh.build(); v1.load_state_dict(sd); v1.eval()
m = bh.build(motion=True); m.load_state_dict(sd, strict=False); m.eval()
H, W = info["height"], info["width"]
box = bh.pitch_box((H, W), info["pitch"])
print("pitch box", box, "crop", (box[3] - box[1], box[2] - box[0]))
cache = mb.BackgroundCache(first, last, crop=box)
ring = mb.RingBuffer()
targets = [first + 400, first + 402, first + 404]
cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{w}.mp4"))
t0 = time.time(); kept = {}
local = 0
while True:
    ok, im = cap.read()
    if not ok:
        break
    a = first + local
    cache.add(a, im); ring.push(a, im)
    for t in targets:
        if a == t + mb.OFFSET:
            kept[t] = ring.triple(t)
    if all(cache.ready(t) for t in targets) and len(kept) == len(targets):
        break
    local += 1
print(f"decoded {local + 1} frames in {time.time() - t0:.1f} s; samples {len(cache.samples)}, "
      f"{cache.nbytes / 1e6:.0f} MB")
# Pass-1 style: seek to every 50th frame.
seek = mb.BackgroundCache(first, last, crop=box)
cap2 = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{w}.mp4"))
for f in mb.bg_sample_frames(targets[0], first, last):
    cap2.set(cv2.CAP_PROP_POS_FRAMES, f - first); ok, im = cap2.read(); assert ok
    seek.add(f, im)
seek.finish()
for t in targets:
    b1, n1 = cache.get(t); b2, n2 = seek.get(t)
    print(f"  frame {t}: centre {mb.bg_centre(t)}, n {n1}/{n2}, "
          f"seek == sequential: {np.array_equal(b1, b2)}, max |diff| "
          f"{np.abs(b1.astype(int) - b2.astype(int)).max()}")
tv1, tv2 = [], []
for t in targets:
    p, c, nx = kept[t]
    bg, n = cache.get(t)
    s = time.time(); a = bh.detect(v1, c, info["pitch"], tau=0.05, top_k=8, cal=cal); tv1.append(time.time() - s)
    s = time.time(); b = bh.detect_v2(m, p, c, nx, bg, info["pitch"], tau=0.05, top_k=8, cal=cal); tv2.append(time.time() - s)
    bz = bh.detect_v2(m, p, c, nx, bg, info["pitch"], tau=0.05, top_k=8, cal=cal, motion_zero=True)
    same = np.array_equal(a[:, :3], b[:, :3]) and np.array_equal(b, bz, equal_nan=True)
    print(f"  frame {t}: {len(b)} candidates; zero-init v2 == v1 (u, v, p): {same}; "
          f"size v2/v1 median {np.nanmedian(b[:, 3] / a[:, 3]):.2f}")
    # Raw scales of D and F on pitch pixels (sigma 1 = units of 255).
    x0, y0, x1, y1 = box
    raw = mb.motion_channels(p[y0:y1, x0:x1], c[y0:y1, x0:x1], nx[y0:y1, x0:x1], bg, 1.0, 1.0)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(info["pitch"]) - (x0, y0)).astype(np.int32)], 1)
    vals = raw[mask.astype(bool)]
    mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
    print(f"    raw D- MAD-std {mad(vals[:, 0]):.4f} (frac exactly 0: {np.mean(vals[:, 0] == 0):.2f}), "
          f"std {vals[:, 0].std():.4f}; F MAD-std {mad(vals[:, 2]):.4f}, std {vals[:, 2].std():.4f}, "
          f"p99.9 |D| {np.percentile(np.abs(vals[:, 0]), 99.9):.3f}")
print(f"time per frame: v1 detect {np.median(tv1):.2f} s, detect_v2 {np.median(tv2):.2f} s "
      f"(ratio {np.median(tv2) / np.median(tv1):.2f}; shared CPU)")
