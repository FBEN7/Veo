"""The original eval_ball_detector.run_window (git HEAD) against the new one
on the same few frames of a training window, COCO and v1: legacy measures
and saved candidates must be the same."""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/user/Veo")
sys.path.insert(0, "/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/compA")
import eval_ball_detector as new  # noqa: E402
import eval_ball_detector_orig as old  # noqa: E402
from src.ball_heatmap import build, detect  # noqa: E402

torch.set_num_threads(4)
w = sys.argv[1] if len(sys.argv) > 1 else "st2_118576_1st_f030000"
full = old.frames_of(w)


def small(sets):
    p = [q for q in sets["passes"] if q["frames"][0] > 30][:1]
    return dict(sets, passes=p, actions=sets["actions"].iloc[:2], random=sets["random"][:2])


old.frames_of = lambda window, seed=0: small(old.__dict__["_full"](window, seed))
old._full = lambda window, seed=0: full
new_full = new.frames_of(w)
new.frames_of = lambda window, seed=0: dict(small(new_full), placebo_pre=[], placebo_post=[])
from ultralytics import YOLO  # noqa: E402

yolo = YOLO("yolov8s.pt")
m = build()
m.load_state_dict(torch.load("/home/user/Veo/.cache/ball_heatmap.3000.pt"))
m.eval()
dets = {"coco": lambda im, pitch, cal: old.coco_candidates(yolo, im, pitch),
        "ball_heatmap.3000.pt": lambda im, pitch, cal: detect(m, im, pitch, tau=0.01, top_k=8,
                                                              cal=cal)}
a = old.run_window(w, dets)
b = new.run_window(w, dets, log=lambda *x: None)
for n in dets:
    same = {k: a[n][k] == b[n][k] for k in ("free-flight hit", "per pass", "feet share",
                                             "action hit", "candidates per frame")}
    ca, cb = a[n]["candidates"], b[n]["candidates"]
    cand = set(ca) == set(cb) and all(np.allclose(np.array(ca[f]), np.array(cb[f]), atol=6e-4)
                                      if len(ca[f]) else not len(cb[f]) for f in ca)
    print(n, "frames", sorted(map(int, ca)), "legacy fields equal:", all(same.values()),
          "| candidates equal (3 decimals):", cand)
print("window-level legacy equal:", all(a[k] == b[k] for k in ("window", "passes", "actions",
                                                              "random point action hit")))
