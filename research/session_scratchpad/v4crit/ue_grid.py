"""U-E only, counts only (no ground truth, no measure): across the draft's
36-config grid, at how many measured frames does the tracked pick differ
from the default's; with F fixed, how many passes have a frame where the
tracked pick differs from raw v2 top-1 (the McNemar-informative passes)."""
import itertools
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/Veo")
from eval_ball_detector import frames_of  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.ball_link import link  # noqa: E402

UE = ["st2_118578_1st_f030000", "st2_118578_2nd_f015000"]
A, B, TAU = 0.895072, -0.927959, 0.25
SCORES = "/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/U_a.json"
NAME = "ball_v2.4000.pt"
EMPTY = pd.DataFrame({"frame": pd.Series([], dtype=int), "cls": pd.Series([], dtype=str),
                      "px": pd.Series([], dtype=float), "py": pd.Series([], dtype=float),
                      "crop_h": pd.Series([], dtype=float)})
saved = {r["window"]: r[NAME]["candidates"] for r in json.load(open(SCORES))["results"]}


def tracked(kept, G, F, dense_top1):
    fr, u, v = kept.frame.to_numpy(int), kept.px.to_numpy(float), kept.py.to_numpy(float)
    out = {}
    for i in range(len(fr)):
        out[fr[i]] = (u[i], v[i], "det")
        if i + 1 < len(fr) and fr[i + 1] - fr[i] <= G:
            for f in range(fr[i] + 1, fr[i + 1]):
                w = (f - fr[i]) / (fr[i + 1] - fr[i])
                out[f] = (u[i] + w * (u[i + 1] - u[i]), v[i] + w * (v[i + 1] - v[i]), "interp")
    if F:
        for f, r in dense_top1.items():
            out.setdefault(f, (r[0], r[1], "F"))
    return out


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return np.hypot(a[0] - b[0], a[1] - b[1]) <= 6


grid = list(itertools.product((0.5, 1.0, 2.0), (0.0, 2.0, 6.0), (6, 12), (False, True)))
default = (1.0, 2.0, 6, False)
diff_from_default = {g: 0 for g in grid}
n_frames = 0
inform = {}
src_at_q1 = {}
for w in UE:
    pq = f"/home/user/Veo/.cache/v2_balls_{w}.parquet"
    meta = json.load(open(pq + ".meta.json"))
    done = sorted(int(f) for f in meta["frames"])
    df = pd.read_parquet(pq)
    top1 = {int(f): (g.px.iloc[0], g.py.iloc[0]) for f, g in
            df.sort_values("p_raw", ascending=False, kind="stable").groupby("frame")}
    rows = df[df.confidence >= TAU].sort_values("confidence", ascending=False,
                                                 kind="stable").groupby("frame").head(4)
    cal = st.calibration(meta["match"])
    s = frames_of(w)
    pass_frames = [(p["i"], q, f) for p in s["passes"] for q, f in enumerate(p["frames"])]
    plac = [e["frame"] for e in s["placebo_pre"] + s["placebo_post"]]
    kept = {}
    for cn, cs in itertools.product((0.5, 1.0, 2.0), (0.0, 2.0, 6.0)):
        kept[(cn, cs)] = link(rows, cal, EMPTY, c_null=cn, c_static=cs, tau=TAU,
                              frames=done).sort_values("frame")
    tr = {g: tracked(kept[g[:2]], g[2], g[3], top1) for g in grid}
    frames_all = [f for _, _, f in pass_frames] + plac
    n_frames += len(frames_all)
    for g in grid:
        diff_from_default[g] += sum(not same(tr[g].get(f), tr[default].get(f)) for f in frames_all)
    for g in ((1.0, 2.0, 6, True), (1.0, 2.0, 6, False), (1.0, 2.0, 12, True)):
        for p in s["passes"]:
            ks = []
            for q, f in enumerate(p["frames"]):
                raw = saved[w][str(f)]
                rawp = (raw[0][0], raw[0][1]) if raw else None
                ks.append(not same(tr[g].get(f), rawp))
            d = inform.setdefault(g, {"passes": 0, "q1 differs": 0, "any of 4 differs": 0,
                                      "frames differing": 0})
            d["passes"] += 1
            d["q1 differs"] += ks[1]
            d["any of 4 differs"] += any(ks)
            d["frames differing"] += sum(ks)
print("measured frames (pass + placebo) on U-E:", n_frames)
vals = sorted(diff_from_default.values())
print("frames where a config's tracked pick differs from the default's: "
      f"min {vals[0]} median {np.median(vals)} max {vals[-1]}")
for g in grid:
    print("  ", g, diff_from_default[g])
for g, d in inform.items():
    print("c_null, c_static, G, F =", g, d)
