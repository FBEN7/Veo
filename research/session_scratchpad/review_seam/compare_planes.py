"""Trainer planes (strip JPEG -> read_strip -> make_batch) vs detect_v2 planes
(window video, OpenCV, BackgroundCache over the window) at the same frame/crop."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/Veo")
import train_ball_heatmap as tr
from src import ball_heatmap as bh
from src.motion_bg import BackgroundCache, bg_sample_frames, motion_planes

root = Path(sys.argv[1])
W = "st2_118577_1st_f030000"
info = json.loads(Path(f"/home/user/Veo/data/soccertrack_v2/{W}.json").read_text())
first = info["start_frame"]
last = first + info["frames"] - 1
table = pd.read_parquet(root / "strips.parquet")
pick = [int(i) for i in sys.argv[2:]] or [11, 12]
rows = list(table.itertuples(index=False))

box = bh.pitch_box((info["height"], info["width"]), info["pitch"])
bx0, by0, bx1, by1 = box
print("pitch box", box)
frames_needed = set()
need = set()
for i in pick:
    t = int(rows[i].frame)
    frames_needed |= {t - 2, t, t + 2}
    need |= set(bg_sample_frames(t, first, last))
cache = BackgroundCache(first, last, crop=box)
got = {}
cap = cv2.VideoCapture(info["path"])
stop = max(max(need), max(frames_needed)) - first
idx = 0
while idx <= stop:
    if not cap.grab():
        break
    f = first + idx
    if f in need or f in frames_needed:
        ok, im = cap.retrieve()
        if f in need:
            cache.add(f, im)
        if f in frames_needed:
            got[f] = im
    idx += 1
cap.release()
cache.finish()

for i in pick:
    r = rows[i]
    t, x0, y0 = int(r.frame), int(r.x0), int(r.y0)
    # Trainer: make_batch without augmentation (seeds None), with codes dummy.
    codes = np.zeros((1, 80, 80), np.int8)
    xt, _ = tr.make_batch(root, [r], codes, [0], None, True)
    xt = xt.numpy()[0]
    # detect_v2: its in-place input on the pitch box, then the strip's region.
    bg, n = cache.get(t)
    cur = got[t]
    crop = cur[by0:by1, bx0:bx1]
    ph, pw = (-crop.shape[0]) % 8, (-crop.shape[1]) % 8
    x = bh._input_v2(crop, got[t - 2][by0:by1, bx0:bx1], got[t + 2][by0:by1, bx0:bx1],
                     bg, ph, pw).numpy()[0]
    yy, xx = y0 - by0, x0 - bx0
    xi = x[:, yy:yy + 320, xx:xx + 320].copy()
    print(f"\nstrip {r.file}: n_bg strip {r.n_bg}, window {n}")
    for c, name in enumerate(["R", "G", "B", "D-", "D+", "F"]):
        a, b = xt[c], xi[c]
        if c >= 3:
            s = bh.SIGMA_D if c < 5 else bh.SIGMA_F
            a, b = a * s * 255, b * s * 255     # grey levels
        d = a - b
        print(f"  {name:3s} train mean {a.mean():+.3f} std {a.std():.3f} zero {np.mean(a == 0):.2f}"
              f" | infer mean {b.mean():+.3f} std {b.std():.3f} zero {np.mean(b == 0):.2f}"
              f" | diff mean {d.mean():+.3f} absmean {np.abs(d).mean():.3f} max {np.abs(d).max():.2f}"
              f" corr {np.corrcoef(a.ravel(), b.ravel())[0, 1]:.3f}")
    # Sign check: D- train vs infer sign agreement at large |D|.
    for c in (3, 4, 5):
        a, b = xt[c], xi[c]
        big = np.abs(b) > np.quantile(np.abs(b), 0.99)
        print(f"  ch{c} sign agreement at top 1% |infer|: "
              f"{np.mean(np.sign(a[big]) == np.sign(b[big])):.3f}")
