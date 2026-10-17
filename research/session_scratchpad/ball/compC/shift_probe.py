"""Best image shift (dx, dy) of the release's GSR boxes onto non-grass
pixels, per region of the panorama (left / centre / right, far / near)."""
import sys, json
sys.path.insert(0, "/home/user/Veo")
import cv2, numpy as np
from src import soccertrack_v2 as st
w = sys.argv[1]
info = json.loads(open(f"/home/user/Veo/data/soccertrack_v2/{w}.json").read())
g = st.gsr_table(info["match"], info["half"])
locs = [200, 900, 1600, 2300, 3000, 3700]
cap = cv2.VideoCapture(f"/home/user/Veo/data/soccertrack_v2/{w}.mp4")
k, acc = 0, {}
shifts = [(dx, dy) for dx in range(-60, 61, 6) for dy in range(-60, 61, 6)]
while k <= max(locs):
    ok, im = cap.read()
    if k in locs:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        nong = (~((hsv[..., 0] > 35) & (hsv[..., 0] < 85) & (hsv[..., 1] > 60))).astype(np.float32)
        ii = cv2.integral(nong)
        p = g[g.frame == info["start_frame"] + k]
        for r in p.itertuples():
            u = r.bx + r.bw / 2
            reg = ("L" if u < 1400 else "C" if u < 2700 else "R") + ("far" if r.by + r.bh < 300 else "near")
            # a fixed 14 x 40 px body window above the box's bottom-middle
            x0, y1 = int(u - 7), int(r.by + r.bh)
            row = []
            for dx, dy in shifts:
                a, b, c, d = y1 - 40 + dy, y1 + dy, x0 + dx, x0 + 14 + dx
                if a < 0 or c < 0 or b >= im.shape[0] or d >= im.shape[1]:
                    row.append(np.nan); continue
                row.append((ii[b, d] - ii[a, d] - ii[b, c] + ii[a, c]) / (40 * 14))
            acc.setdefault(reg, []).append(row)
    k += 1
for reg, rows in sorted(acc.items()):
    m = np.nanmean(np.array(rows), axis=0)
    i = int(np.nanargmax(m))
    print(w, reg, len(rows), "best shift", shifts[i], "score %.2f" % m[i], "at (0,0) %.2f" % m[shifts.index((0, 0))])
