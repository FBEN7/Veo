"""Regression (2): score_events_paired reproduces e5_score.py's TP / pred /
labels on events_coco and events_heatmap of the two V-old windows (already
reported); the bootstrap with e5's 2000 draws is compared for equality only.
Prints counts and booleans, no F1."""
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/home/user/Veo")
import score_events_paired as sp  # noqa: E402

W = ["st2_117093_1st_f015000", "st2_132877_1st_f015000"]
E5 = "/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/e5_score.py"
out = subprocess.run([sys.executable, E5, *W], capture_output=True, text=True,
                     cwd="/home/user/Veo", check=True).stdout
e5 = {m[0]: tuple(map(int, m[1:])) for m in re.findall(
    r"(events_\w+) TP (\d+) pred (\d+) labels (\d+)", out)}
e5_ci = re.search(r"95% \[\s*(\S+)\s+(\S+)\s*\]", out)
truths = {w: sp.labels_of(w) for w in W}
ev = {c: {w: sp.load_events(c, w) for w in W}
      for c in ("events_coco", "events_heatmap")}
for c in ev:
    mine = sp.counts({}, {}, 0) if False else None
    tp = n_pred = n_true = 0
    for w in W:
        a, b, d = sp.counts(sp.preds_of(ev[c][w]), truths[w], 1.0)
        tp, n_pred, n_true = tp + a, n_pred + b, n_true + d
    print(c, "mine TP/pred/labels", (tp, n_pred, n_true), "e5", e5[c],
          "equal:", (tp, n_pred, n_true) == e5[c])
d = sp.paired(ev["events_coco"], ev["events_heatmap"], truths, n=2000)
print("bootstrap (2000 draws) percentiles equal to e5's:",
      np.allclose([d["lower"], d["upper"]],
                  [float(e5_ci[1]), float(e5_ci[2])], atol=6e-4))
