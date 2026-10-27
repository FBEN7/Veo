"""Learned shots v4 (plan.json): v2 features plus the players' reactions,
on the no-grass tracks of shots_learn/v3, nested leave-one-match-out.

    VEO_UNSEAL=... python learn_v4.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "v2"))
sys.path.insert(0, str(HERE.parent / "v3"))
import learn_shots as LS  # noqa: E402
import learn_v2 as V  # noqa: E402
import learn_v3 as V3  # noqa: E402

GOALS = (0.0, 105.0)
GY = 34.0
NEAR_M, BOX_M = 25.0, 20.0


def reaction(u: dict, frames: np.ndarray) -> pd.DataFrame:
    m = V._metric(u)
    p = m[(m.cls == "player") & m.team.isin(["team_A", "team_B"])].sort_values(["track_id", "frame"])
    dt = p.groupby("track_id").frame.diff() / LS.FPS
    p = p.assign(vx=p.groupby("track_id").x.diff() / dt, vy=p.groupby("track_id").y.diff() / dt)
    # track switches give impossible jumps: player speeds are clipped at 12 m/s
    p = p.assign(vx=p.vx.clip(-12, 12), vy=p.vy.clip(-12, 12))
    p = p.assign(v=np.hypot(p.vx, p.vy).clip(upper=12))
    by = {f: g for f, g in p.groupby("frame")}
    rec = []
    for f in frames:
        g = by.get(f)
        if g is None or g.empty:
            rec.append([np.nan] * 7)
            continue
        x, y, vx, vy, v = (g[c].to_numpy() for c in ("x", "y", "vx", "vy", "v"))
        n_near = [int((np.hypot(x - gx, y - GY) <= NEAR_M).sum()) for gx in GOALS]
        gx = GOALS[int(np.argmax(n_near))]
        d = np.hypot(x - gx, y - GY)
        k = int(np.argmin(d))
        box = d <= BOX_M
        toward = np.sign(gx - 52.5) * vx              # velocity component toward the chosen goal
        rec.append([v[k], abs(vy[k]), float(box.sum()),
                    float(np.nanmean(v[box])) if box.any() else np.nan,
                    float(np.nanmean(toward[box])) if box.any() else np.nan,
                    float(np.nanmean(v)), gx])
    r = pd.DataFrame(rec, index=frames, columns=["gk_s", "gk_lat", "box_n", "box_s", "box_to", "all_s", "goal"])
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 2
    nb, na = max(1, int(0.5 * LS.FPS / step)), max(1, int(1.0 * LS.FPS / step))

    def before(s):
        return s.shift(1).rolling(nb, min_periods=1).mean()

    def after_max(s):
        return s[::-1].shift(1).rolling(na, min_periods=1).max()[::-1]

    def after_mean(s):
        return s[::-1].shift(1).rolling(na, min_periods=1).mean()[::-1]

    out = pd.DataFrame(index=frames)
    out["r_gk_before"], out["r_gk_after"] = before(r.gk_s), after_max(r.gk_s)
    out["r_gk_delta"] = out.r_gk_after - out.r_gk_before
    out["r_gk_lat_after"] = after_max(r.gk_lat)
    out["r_box_n"] = r.box_n
    out["r_box_s_before"], out["r_box_s_after"] = before(r.box_s), after_mean(r.box_s)
    out["r_box_s_delta"] = out.r_box_s_after - out.r_box_s_before
    out["r_box_to_before"], out["r_box_to_after"] = before(r.box_to), after_mean(r.box_to)
    out["r_box_to_delta"] = out.r_box_to_after - out.r_box_to_before
    out["r_all_s_before"], out["r_all_s_after"] = before(r.all_s), after_mean(r.all_s)
    out["r_all_s_delta"] = out.r_all_s_after - out.r_all_s_before
    return out


def features_v4(u: dict) -> pd.DataFrame:
    cache = V.GPU / u["name"] / "shot_features_v4.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    f = V.features_v2(u)
    out = pd.concat([f, reaction(u, f.index.to_numpy())], axis=1)
    out.to_parquet(cache)
    return out


def main():
    from src.unseal import unsealed
    for m in V3.NEW_PLAYERS:
        if not unsealed(m, "shots v4: M7 / M9 clips as training data"):
            raise SystemExit("VEO_UNSEAL needed")
    us = V3.new_tree([u for u in V.units() if (LS.GPU / u["name"]).exists()])
    LS.GPU = V.GPU = V3.NEW
    d = {u["name"]: (features_v4(u), LS.labels(u), u["match"]) for u in us}
    print("features done", len(d), flush=True)
    names = list(d)
    probs, thr = LS.nested(d, names, LS.cols_of(d))
    res = LS.score(d, names, probs, thr)
    ref = json.loads((HERE.parent / "v3/cv_v3.json").read_text())["no grass tests"]
    ms = sorted(ref["per match [tp, fp, fn]"])
    diff = np.array([res["per match [tp, fp, fn]"][m][0] - ref["per match [tp, fp, fn]"][m][0] for m in ms], float)
    rng = np.random.default_rng(0)
    boots = [diff[rng.integers(0, len(ms), len(ms))].sum() for _ in range(5000)]
    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0], "units": len(us),
           "v4": res, "v2 (reference, same units and tracks)": ref,
           "hits v4 - v2": {"total": float(diff.sum()),
                            "bootstrap 95% over matches": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}}
    (HERE / "cv_v4.json").write_text(json.dumps(out, indent=1))
    print({k: res[k] for k in ("tp", "fp", "fn", "recall", "precision")}, json.dumps(out["hits v4 - v2"]))


if __name__ == "__main__":
    main()
