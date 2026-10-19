import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/Veo")
from label_ball_strips import split
from src.motion_bg import grey, motion_channels

base = Path(sys.argv[1])
A = base / "stream"
B = base / "strips"
ta = pd.read_parquet(A / "strips.parquet")
tb = pd.read_parquet(B / "strips.parquet")
info = json.loads(Path("/home/user/Veo/data/soccertrack_v2/st2_118577_1st_f030000.json").read_text())
first = info["start_frame"]
# Raw window frames, for the inference-side D.
want = set()
for r in ta.itertuples():
    want |= {r.frame - 2, r.frame, r.frame + 2}
frames = {}
cap = cv2.VideoCapture(info["path"])
idx = 0
while idx <= max(want) - first:
    ok = cap.grab()
    if first + idx in want:
        frames[first + idx] = cap.retrieve()[1]
    idx += 1
cap.release()
for r in ta.itertuples():
    rb = tb[(tb.frame == r.frame) & (tb.x0 == r.x0) & (tb.y0 == r.y0) & (tb.kind == r.kind)].iloc[0]
    sa = cv2.imread(str(A / r.file))
    sb = cv2.imread(str(B / rb.file))
    pa, ca, na, _ = split(sa)
    pb, cb, nb, _ = split(sb)
    raw = [frames[f][r.y0:r.y0 + 320, r.x0:r.x0 + 320] for f in (r.frame - 2, r.frame, r.frame + 2)]
    print(f"\n{r.file}")
    for name, a, b in (("t-2", pa, pb), ("t", ca, cb), ("t+2", na, nb)):
        d = a.astype(int) - b.astype(int)
        # Channel-swap test: stream crop with B and R swapped vs window crop.
        sw = a[:, :, ::-1].astype(int) - b.astype(int)
        print(f"  {name}: stream-window per channel mean {d.reshape(-1, 3).mean(0).round(2)}, "
              f"absmean {np.abs(d).mean():.2f}; with R/B swapped absmean {np.abs(sw).mean():.2f}; "
              f"grey absmean {np.abs(grey(a).astype(int) - grey(b).astype(int)).mean():.2f}")
    # Frame alignment: stream t crop vs window crops at t-1, t, t+1 grey.
    for off in (-2, 0, 2):
        f = r.frame + off
        g = grey(frames[f][r.y0:r.y0 + 320, r.x0:r.x0 + 320]).astype(int)
        print(f"  stream t vs raw window frame t{off:+d}: grey absmean "
              f"{np.abs(grey(ca).astype(int) - g).mean():.2f}")
    ma = motion_channels(pa, ca, na, None, sigma_d=1.0, sigma_f=1.0) * 255
    mb = motion_channels(pb, cb, nb, None, sigma_d=1.0, sigma_f=1.0) * 255
    mr = motion_channels(*raw, None, sigma_d=1.0, sigma_f=1.0) * 255
    for k, nm in ((0, "D-"), (1, "D+")):
        a, b, c = ma[:, :, k], mb[:, :, k], mr[:, :, k]
        mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
        print(f"  {nm} grey levels: stream strip std {a.std():.3f} mad {mad(a):.3f} zero {np.mean(a == 0):.2f}"
              f" | window strip std {b.std():.3f} mad {mad(b):.3f} zero {np.mean(b == 0):.2f}"
              f" | raw window (inference) std {c.std():.3f} mad {mad(c):.3f} zero {np.mean(c == 0):.2f}"
              f" | corr stream/raw {np.corrcoef(a.ravel(), c.ravel())[0, 1]:.3f}")
