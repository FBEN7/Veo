"""Which GSR frame offset best matches the window video: score = mean
non-grass share inside the release's image boxes (players are not green)."""
import sys
sys.path.insert(0, "/home/user/Veo")
import cv2, numpy as np, pandas as pd
from src import soccertrack_v2 as st
w = sys.argv[1]
import json
info = json.loads(open(f"/home/user/Veo/data/soccertrack_v2/{w}.json").read())
g = st.gsr_table(info["match"], info["half"]) if (st.cache_dir() / "gsr" / f"{info['match']}_{st._half(info['half'])}.parquet").exists() else None
g = g[g.frame.between(info["start_frame"] - 200, info["start_frame"] + 4700)]
byf = {f: d for f, d in g.groupby("frame")}
cap = cv2.VideoCapture(f"/home/user/Veo/data/soccertrack_v2/{w}.mp4")
locs = [int(x) for x in sys.argv[2:]] or [600, 1500, 2500, 3500]
imgs = {}
k = 0
while k <= max(locs):
    ok, im = cap.read()
    if k in locs:
        imgs[k] = im
    k += 1
for k, im in imgs.items():
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] > 35) & (hsv[..., 0] < 85) & (hsv[..., 1] > 60)
    res = []
    for d in range(-75, 76, 5):
        p = byf.get(info["start_frame"] + k + d)
        if p is None:
            continue
        sc = []
        for r in p.itertuples():
            x0, y0, x1, y1 = int(r.bx), int(r.by), int(r.bx + r.bw), int(r.by + r.bh)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            sc.append(1 - grass[max(y0, 0):y1, max(x0, 0):x1].mean())
        res.append((d, float(np.mean(sc))))
    best = max(res, key=lambda t: t[1])
    print(w, "local", k, "best offset", best[0], "score %.3f" % best[1], "at 0: %.3f" % dict(res)[0],
          " ".join(f"{d}:{s:.2f}" for d, s in res if d % 25 == 0))
