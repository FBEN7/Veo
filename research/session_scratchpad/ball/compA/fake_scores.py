"""A scores file for a training window with two fake detectors (no
detector is run): detA on the pass line at s, detB (with logits) 3 px from
it at even passes, 100 px off at odd ones, none at pass 1; detB confident
on random frames. For exercising ball_tiles' new options and
validate_ff_auto --sealed-check."""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/user/Veo")
import eval_ball_detector as ev  # noqa: E402
from src.ball_heatmap import pitch_to_image  # noqa: E402

w, out = sys.argv[1], sys.argv[2]
wd = ev.window_data(w)
sets, cal = wd["sets"], wd["cal"]
A, B = {}, {}
for f in ev.measured_frames(sets):
    A[f], B[f] = np.zeros((0, 4)), np.zeros((0, 5))
for p in sets["passes"]:
    a, b = np.array(p["from"]), np.array(p["to"])
    for q, f in enumerate(p["frames"]):
        u, v = pitch_to_image([a + p["s"][q] * (b - a)], cal)[0]
        A[f] = np.array([[u, v - 2, 0.9, 4.0]])
        if p["i"] == 1:
            continue
        du = 3.0 if p["i"] % 2 == 0 else 100.0
        B[f] = np.array([[u + du, v - 2, 0.7, 4.0, 0.85]])
for f in sets["random"]:
    B[f] = np.array([[2000.0 + f % 100, 400.0, 0.88, 4.0, 2.0]])
r = ev.measure(wd, {"detA": A, "detB": B})
blob = {"summary": {}, "detectors": {"detA": {"kind": "v1"}, "detB": {"kind": "v2"}},
        "results": [r]}
open(out, "w").write(json.dumps(blob))
print("passes", len(sets["passes"]), "frames", len(A))
