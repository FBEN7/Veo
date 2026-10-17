"""Synthetic: a grid made denser by odd extra frames lets a restart (no
motion check) fall within gap 12, and fill_gaps interpolates across it."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import ball_track as bt
import eval_ball_detector as ev
K = np.array([[1e6, 0, 2048], [0, 1e6, 540], [0, 0, 1]]); s = 20.0
Hp = np.array([[s, 0, 2048 - 52.5 * s], [0, s, 540 - 34 * s], [0, 0, 1]])
cal = {"K": K, "D": np.zeros(4), "Knew": K, "Hinv": np.linalg.inv(Hp)}
px = lambda x, y: (2048 + x * s, 540 + y * s)
rows = []
for f in range(0, 42, 2):                       # object A at x=-30
    u, v = px(-30, 0); rows.append(dict(frame=f, cls="ball", px=u, py=v, crop_h=4.0, confidence=0.6))
for f in range(50, 90, 2):                      # object B at x=+30 (60 m away)
    u, v = px(30, 0); rows.append(dict(frame=f, cls="ball", px=u, py=v, crop_h=4.0, confidence=0.6))
rows = pd.DataFrame(rows)
stride_only = np.arange(0, 90, 2)
extras = np.array([41, 43, 45, 47, 49])         # odd frames "processed" (--extra-frames)
for name, fr in (("stride only", stride_only), ("with odd extras", np.union1d(stride_only, extras))):
    tr = bt.track(rows, cal, tau=0.5, gap=12, frames=fr)
    span = tr[(tr.frame > 40) & (tr.frame < 50)]
    print(f"{name}: frames 41-49 filled {len(span)}; u range {span.u.min() if len(span) else None}..{span.u.max() if len(span) else None}")
# covered(): extras make odd frames 'processed' -> fallback eligible
print("covered with extras, 45:", 45 in ev.covered([45], np.union1d(stride_only, extras), 2))
