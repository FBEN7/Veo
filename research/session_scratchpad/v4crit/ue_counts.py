"""U-E only, counts only (no ground truth, no measure): how often a minimal
v4 tracker (link + linear gap fill) differs from raw v2 top-1 at pass
frames -- an upper bound on McNemar's discordant pairs -- and how the
draft's literal call (frames=None, players empty) changes the link grid."""
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


def pcal(z):
    return 1 / (1 + np.exp(-(A * np.asarray(z, float) + B)))


def track(rows, cal, frames, c_null, c_static, G):
    """{frame: (u, v, source, fa, fb)} over every frame."""
    kept = link(rows, cal, EMPTY, c_null=c_null, c_static=c_static, tau=TAU,
                frames=frames).sort_values("frame")
    out = {}
    fr, u, v = kept.frame.to_numpy(int), kept.px.to_numpy(float), kept.py.to_numpy(float)
    for i in range(len(fr)):
        out[fr[i]] = (u[i], v[i], "det", fr[i], fr[i])
        if i + 1 < len(fr) and fr[i + 1] - fr[i] <= G:
            for f in range(fr[i] + 1, fr[i + 1]):
                w = (f - fr[i]) / (fr[i + 1] - fr[i])
                out[f] = (u[i] + w * (u[i + 1] - u[i]), v[i] + w * (v[i + 1] - v[i]),
                          "interp", fr[i], fr[i + 1])
    return out, len(kept)


saved = {r["window"]: r[NAME]["candidates"] for r in json.load(open(SCORES))["results"]}
tot = {}
for w in UE:
    pq = f"/home/user/Veo/.cache/v2_balls_{w}.parquet"
    meta = json.load(open(pq + ".meta.json"))
    done = sorted(int(f) for f in meta["frames"])
    df = pd.read_parquet(pq)
    rows = df[df.confidence >= TAU].sort_values("confidence", ascending=False,
                                                 kind="stable").groupby("frame").head(4)
    cal = st.calibration(meta["match"])
    s = frames_of(w)
    print(w, "processed frames", len(done), "with a candidate >= tau",
          rows.frame.nunique(), "rows >= tau", len(rows))
    # Draft-literal grid vs the processed-frame grid.
    lit = link(rows, cal, EMPTY, c_null=1.0, c_static=2.0, tau=TAU)
    grd = link(rows, cal, EMPTY, c_null=1.0, c_static=2.0, tau=TAU, frames=done)
    lit_tau05 = link(rows, cal, EMPTY, c_null=1.0, c_static=2.0)       # link's default tau 0.5
    same = len(set(lit.index) & set(grd.index))
    print(f"  picks: frames=None {len(lit)}, frames=processed {len(grd)}, shared rows {same}; "
          f"link default tau=0.5: {len(lit_tau05)}")
    # Longest time between consecutive picks of one chain under frames=None:
    # gaps the grid does not count.
    f_lit = np.sort(lit.frame.to_numpy(int))
    gaps = np.diff(f_lit)
    print(f"  frames=None: consecutive picks > 12 frames apart: {(gaps > 12).sum()}, "
          f"max gap {gaps.max() if len(gaps) else 0}")
    for cn, cs, G in ((1.0, 2.0, 6), (1.0, 2.0, 12), (0.5, 0.0, 12), (2.0, 6.0, 6)):
        tr, n = track(rows, cal, done, cn, cs, G)
        key = (cn, cs, G)
        t = tot.setdefault(key, {})
        for p in s["passes"]:
            for q, f in enumerate(p["frames"]):
                raw = saved[w].get(str(f))
                if raw is None:
                    t["raw unmeasured"] = t.get("raw unmeasured", 0) + 1
                    continue
                k = []
                k.append("q1" if q == 1 else "q*")
                k.append("odd" if f % 2 else "even")
                pick = tr.get(f)
                raw_ok = len(raw) > 0
                raw_tau = raw_ok and pcal(raw[0][4]) >= TAU
                if pick is None:
                    st_ = "trk none, raw " + ("none" if not raw_ok else ">=tau" if raw_tau else "<tau")
                else:
                    if not raw_ok:
                        st_ = f"trk {pick[2]}, raw none"
                    else:
                        d = np.hypot(pick[0] - raw[0][0], pick[1] - raw[0][1])
                        st_ = f"trk {pick[2]}, " + ("same<=6px" if d <= 6 else "differs")
                for kk in (tuple(k), (k[0],)):
                    t.setdefault(kk, {}).setdefault(st_, 0)
                    t[kk][st_] += 1
                if pick is not None and pick[2] == "interp":
                    span = pick[4] - pick[3]
                    D = p["reception"] - p["kick"]
                    whole = pick[3] <= p["kick"] + 0.15 * D and pick[4] >= p["reception"] - 0.15 * D
                    t.setdefault(("span", k[0]), []).append((span, whole))
for key, t in tot.items():
    print("config c_null, c_static, G =", key)
    for kk in (("q1",), ("q1", "even"), ("q1", "odd"), ("q*",)):
        if kk in t:
            print("   ", kk, dict(sorted(t[kk].items())))
    for kk in (("span", "q1"), ("span", "q*")):
        if kk in t:
            sp = np.array([a for a, _ in t[kk]])
            print("   ", kk, "interp spans: n", len(sp), "median", np.median(sp), "max", sp.max(),
                  "bridging ~kick->~reception:", sum(b for _, b in t[kk]))
    if "raw unmeasured" in t:
        print("    raw unmeasured", t["raw unmeasured"])
