"""eval_ball_detector.decode (background pass by grab + ring buffer) against
an independent computation (every frame into a BackgroundCache, as the
dense path does), at a frame with a full background and at the last frame
of the window (no t+2). Training window only."""
import json
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/user/Veo")
import eval_ball_detector as ev  # noqa: E402
from src import ball_heatmap as bh  # noqa: E402
from src import motion_bg as mb  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402

torch.set_num_threads(4)
w = "st2_118577_1st_f030000"
ck = sys.argv[1]
m = bh.build(motion=True)
m.load_state_dict(torch.load(ck))
m.eval()
wd = ev.window_data(w)
info = wd["info"]
sets = wd["sets"]
p = next(p for p in sets["passes"] if p["frames"][0] >= 400)
wanted = [p["frames"][1], info["frames"] - 1]
print("frames", wanted, "pass", p["i"])
fn = lambda prev, cur, nxt, bg, pitch, cal: bh.detect_v2(m, prev, cur, nxt, bg, pitch, tau=0.05,
                                                         top_k=8, cal=cal)
t0 = time.time()
cands, n_bg = ev.decode(wd, {}, {"v2": fn}, wanted)
print(f"decode: {time.time() - t0:.0f} s, n_bg {n_bg}")
# Independent: every frame offered to the cache, triples kept.
first, last = info["start_frame"], info["start_frame"] + info["frames"] - 1
box = bh.pitch_box((info["height"], info["width"]), info["pitch"])
cache = mb.BackgroundCache(first, last, crop=box)
ring = mb.RingBuffer()
kept = {}
cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{w}.mp4"))
idx = 0
while True:
    ok, im = cap.read()
    if not ok:
        break
    cache.add(first + idx, im)
    ring.push(idx, im)
    for t in wanted:
        if idx == t + 2 or (idx == t and t == info["frames"] - 1):
            kept[t] = ring.triple(t)
    idx += 1
cache.finish()
for t in wanted:
    prev, cur, nxt = kept[t]
    bg, n = cache.get(first + t)
    ref = fn(prev, cur, nxt, bg, info["pitch"], wd["cal"])
    got = cands["v2"][t]
    print(f"  frame {t}: n_bg {n} (decode {n_bg[t]}), nxt None: {nxt is None}, "
          f"rows {len(ref)}, identical: {np.array_equal(ref, got, equal_nan=True)}")
