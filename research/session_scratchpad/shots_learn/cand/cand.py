"""Shots spotter with ball-candidate cues (protocol.json): the v4 spotter's
features on the v3b tracks plus cues read from raw ball candidates near
the goals, arm B from v3b's kept rows, arm C from v4r's rows, both only at
the hybrid's window frames; nested leave-one-match-out each.

    VEO_UNSEAL=... python cand.py B
    VEO_UNSEAL=... python cand.py C
    python cand.py --compare
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball/events_tune"))
for p in ("", "v2", "v3", "v4"):
    sys.path.insert(0, str(HERE.parent / p))
import learn_shots as LS  # noqa: E402
import learn_v2 as V  # noqa: E402
import learn_v3 as V3  # noqa: E402
import learn_v4 as V4  # noqa: E402
import tune_rules as tr  # noqa: E402
from src import ball_link as bl  # noqa: E402
from src import ball_spots as bs  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402

V4R_ROWS = ROOT / ".cache/hybrid_v4r_rows"
CACHE = ROOT / ".cache/shots_cand"
NEXT, PREV = 38, 12                     # frames: 1.5 s ahead, 0.5 s back
BOX_X, BOX_Y = 36.0, 20.16
MOUTH_X, MOUTH_Y = 47.0, 6.0
C_MIN = 0.10


def rows_of(arm: str, w: str, frames: np.ndarray) -> pd.DataFrame:
    """The arm's candidates at the unit's window frames, with ground points."""
    m = w.split("_")[1]
    cal = st.calibration(m)
    if arm == "B":
        base = ROOT / (".cache/gpu_nograss" if m in V3.NEW_PLAYERS else ".cache/gpu") / w / f"output_{w}"
        r = pd.read_parquet(base / "v3s_rows.parquet")
        r = r[r.frame.isin(frames)]
    else:
        r = pd.read_parquet(V4R_ROWS / f"{w}.parquet")
        n = len(frames)
        r = bs.drop_marks(r, bs.spot_marks(r, cal, n))
        keep = np.ones(len(r), bool)
        for x, y in tr.static_points(r, n):
            keep &= np.hypot(r.px.to_numpy() - x, r.py.to_numpy() - y) > 1.5 * tr.STATIC_PX
        r = r[keep]
    r = r[r.cls == "ball"].reset_index(drop=True)
    g = bl.ground_points(r, cal)
    return r.assign(gx=g[:, 0], gy=g[:, 1]).dropna(subset=["gx", "gy"])


def cues(r: pd.DataFrame, frames_with_rows: np.ndarray, t_frames: np.ndarray) -> pd.DataFrame:
    ax, ay, c, f = (np.abs(r.gx.to_numpy()), np.abs(r.gy.to_numpy()), r.confidence.to_numpy(),
                    r.frame.to_numpy())
    box = (ax >= BOX_X) & (ay <= BOX_Y)
    mouth = (ax >= MOUTH_X) & (ay <= MOUTH_Y)
    dg = np.hypot(52.5 - ax, ay)
    sure = c >= C_MIN
    have = np.asarray(sorted(set(frames_with_rows)))
    out = []
    for t in t_frames:
        nx = (f > t) & (f <= t + NEXT)
        pv = (f >= t - PREV) & (f <= t)
        n_have = int(((have > t) & (have <= t + NEXT)).sum())
        if n_have == 0:
            out.append([np.nan] * 6 + [0.0])
            continue
        box_frames = np.unique(f[nx & box & sure])
        dn = dg[nx & sure]
        dp = dg[pv & sure]
        d_next = float(dn.min()) if len(dn) else np.nan
        d_prev = float(dp.min()) if len(dp) else np.nan
        out.append([float(c[nx & box].max()) if (nx & box).any() else 0.0,
                    float(c[nx & mouth].max()) if (nx & mouth).any() else 0.0,
                    len(box_frames) / n_have, d_next, d_prev, d_prev - d_next, n_have / (NEXT / 2)])
    return pd.DataFrame(out, index=t_frames, columns=["k_box_c", "k_mouth_c", "k_box_share", "k_dgoal_next",
                                                      "k_dgoal_prev", "k_approach", "k_cover"])


def features(arm: str, u: dict, fr: pd.DataFrame) -> pd.DataFrame:
    cache = CACHE / arm / f"{u['name']}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    base = V4.features_v4(u)
    win = fr[fr.unit == u["name"]].frame.to_numpy()
    if len(win):
        k = cues(rows_of(arm, u["name"], win), win, base.index.to_numpy())
    else:
        k = pd.DataFrame(index=base.index, columns=["k_box_c", "k_mouth_c", "k_box_share", "k_dgoal_next",
                                                    "k_dgoal_prev", "k_approach"], dtype=float).assign(k_cover=0.0)
    out = pd.concat([base, k], axis=1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


def run(arm: str):
    from src.unseal import unsealed
    for m in V3.NEW_PLAYERS:
        if not unsealed(m, f"shots cand {arm}: M7 / M9 clips as spotter training data"):
            raise SystemExit("VEO_UNSEAL needed")
    us = V3.new_tree([u for u in V.units() if (LS.GPU / u["name"]).exists()])
    LS.GPU = V.GPU = V3.NEW
    fr = pd.read_parquet(HERE.parent / "hybrid/frames.parquet")
    d = {u["name"]: (features(arm, u, fr), LS.labels(u), u["match"]) for u in us}
    print("features done", len(d), flush=True)
    names = list(d)
    probs, thr = LS.nested(d, names, LS.cols_of(d))
    res = LS.score(d, names, probs, thr)
    (HERE / f"cv_{arm}.json").write_text(json.dumps({"protocol sha256": (HERE / "protocol.json.sha256").read_text().split()[0],
                                                     "arm": arm, "units": len(us), "result": res}, indent=1))
    print(arm, {k: res[k] for k in ("tp", "fp", "fn", "recall", "precision")})


def diff(a: dict, b: dict, k: int, idx) -> dict:
    ms = sorted(a)
    d = np.array([a[m][k] - b[m][k] for m in ms], float)
    s = [d[i].sum() for i in idx]
    return {"total": float(d.sum()), "bootstrap 95% over matches": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]}


def compare():
    b = json.loads((HERE / "cv_B.json").read_text())["result"]
    c = json.loads((HERE / "cv_C.json").read_text())["result"]
    ref = json.loads((HERE.parent / "v4/cv_v4.json").read_text())["v4"]
    pm = {k: v["per match [tp, fp, fn]"] for k, v in (("B", b), ("C", c), ("v4", ref))}
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(pm["v4"]), len(pm["v4"])) for _ in range(5000)]
    out = {"protocol sha256": (HERE / "protocol.json.sha256").read_text().split()[0],
           "counts [tp, fp, fn]": {k: [v["tp"], v["fp"], v["fn"]] for k, v in (("v4 (cv_v4.json)", ref), ("B: v3b cues", b), ("C: v4r cues", c))},
           "primary: C - B hits": diff(pm["C"], pm["B"], 0, idx), "C - B false alarms": diff(pm["C"], pm["B"], 1, idx),
           "B - v4 hits": diff(pm["B"], pm["v4"], 0, idx), "C - v4 hits": diff(pm["C"], pm["v4"], 0, idx),
           "C - v4 false alarms": diff(pm["C"], pm["v4"], 1, idx),
           "per match [tp, fp, fn]": {m: {"v4": pm["v4"][m], "B": pm["B"][m], "C": pm["C"][m]} for m in sorted(pm["v4"])}}
    (HERE / "compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per match [tp, fp, fn]"}, indent=1))


if __name__ == "__main__":
    compare() if sys.argv[1] == "--compare" else run(sys.argv[1])
