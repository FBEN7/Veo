"""A scores file for a training window with one fake v2 detector (no
detector is run): at each measured frame 6 candidates with logits ~ N(-1, 2),
each placed on the pass line at s (pass frames) / at the actor (PASS, DRIVE)
with probability sigmoid(0.7 z - 0.8), else 15 m off; random frames get
8 rows of low logits. Real frames, people and calibration."""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/user/Veo")
import eval_ball_detector as ev  # noqa: E402
from src.ball_heatmap import pitch_to_image  # noqa: E402

w, out, name = sys.argv[1], sys.argv[2], sys.argv[3]
wd = ev.window_data(w)
sets, cal = wd["sets"], wd["cal"]
rng = np.random.default_rng(0)
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def row(g, z, size=5.0):
    u, v = pitch_to_image([g], cal)[0]
    return [u, v - size / 2, sig(z), size, z]


C = {f: np.zeros((0, 5)) for f in ev.measured_frames(sets)}
for p in sets["passes"]:
    a, b = np.array(p["from"]), np.array(p["to"])
    n = (b - a)[::-1] * (1, -1) / np.hypot(*(b - a))
    for q, f in enumerate(p["frames"]):
        on = a + p["s"][q] * (b - a)
        z = np.sort(rng.normal(-1, 2, 6))[::-1]
        C[f] = np.array([row(on + (rng.uniform(-1, 1, 2) if rng.random() < sig(0.7 * zk - 0.8)
                                   else 15 * n), zk) for zk in z])
for e in sets["actions"].itertuples():
    if np.isnan(e.x):
        continue
    z = np.sort(rng.normal(-1, 2, 6))[::-1]
    g = np.array([e.x, e.y])
    C[int(e.frame)] = np.array([row(g + (0.5 if rng.random() < sig(0.7 * zk - 0.8) else 6.0, 0), zk)
                                for zk in z])
for f in sets["random"]:
    z = np.sort(rng.normal(-3, 1.2, 8))[::-1]
    C[f] = np.array([row((rng.uniform(-40, 40), rng.uniform(-25, 25)), zk) for zk in z if sig(zk) >= 0.05])
r = ev.measure(wd, {name: C})
blob = {"summary": {}, "detectors": {name: {"kind": "v2"}}, "results": [r]}
open(out, "w").write(json.dumps(blob))
print("passes", len(sets["passes"]), "actions", len(sets["actions"]), "random", len(sets["random"]))
