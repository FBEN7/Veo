"""Read-only probe on a training window (M3): frame duplication, background
median, foreground fraction inside the pitch, tile occupancy for gating."""
import json, time, sys
import cv2, numpy as np
sys.path.insert(0, "/home/user/Veo")
from src.paths import DATA_DIR
w = "st2_118575_1st_f015000"
info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{w}.mp4"))
t0 = time.time()
samples, near = [], {}
for i in range(751):
    ok = cap.grab()
    if not ok: break
    if i % 50 == 0 or 296 <= i <= 304:
        ok, im = cap.retrieve()
        if i % 50 == 0: samples.append(im)
        if 296 <= i <= 304: near[i] = im
print("decode", round(time.time() - t0, 1), "s", len(samples), "bg samples", samples[0].shape)
t1 = time.time()
bg = np.median(np.stack(samples), axis=0).astype(np.uint8)
print("median of 15 full frames", round(time.time() - t1, 1), "s")
poly = np.asarray(info["pitch"], np.int32)
mask = np.zeros(bg.shape[:2], np.uint8); cv2.fillPoly(mask, [poly], 1)
inside = mask.astype(bool)
for a in range(296, 304):
    d = np.abs(near[a + 1].astype(np.int16) - near[a].astype(np.int16)).max(axis=2)
    print("consecutive", a, a + 1, "mean|diff| %.2f  >20: %.4f" % (d[inside].mean(), (d[inside] > 20).mean()))
t = 300
F = np.abs(near[t].astype(np.int16) - bg.astype(np.int16)).max(axis=2)
Dm = np.abs(near[t].astype(np.int16) - near[t - 2].astype(np.int16)).max(axis=2)
Dp = np.abs(near[t].astype(np.int16) - near[t + 2].astype(np.int16)).max(axis=2)
for th in (15, 25, 40):
    fg = (F > th) & inside
    mv = ((Dm > th) | (Dp > th)) & inside
    print("th", th, "fg frac %.4f" % fg[inside].mean(), "moving frac %.4f" % mv[inside].mean())
fg = ((F > 25) & inside).astype(np.uint8)
fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
x0, x1 = poly[:, 0].min(), poly[:, 0].max(); y0, y1 = poly[:, 1].min(), poly[:, 1].max()
print("pitch bbox", x0, x1, y0, y1, "area Mpx %.2f" % ((x1 - x0) * (y1 - y0) / 1e6), "inside Mpx %.2f" % (inside.sum() / 1e6))
for T in (128, 256):
    occ = tot = 0
    for yy in range(y0, y1, T):
        for xx in range(x0, x1, T):
            m = inside[yy:yy + T, xx:xx + T]
            if not m.any(): continue
            tot += 1
            occ += fg[yy:yy + T, xx:xx + T].any()
    print("tile", T, "tiles in pitch", tot, "with foreground", occ, "%.2f" % (occ / tot))
n, lab, st_, cen = cv2.connectedComponentsWithStats(fg)
areas = st_[1:, cv2.CC_STAT_AREA]
print("components", n - 1, "area pct 10/50/90", np.percentile(areas, [10, 50, 90]) if len(areas) else None)
print("total", round(time.time() - t0, 1), "s")
