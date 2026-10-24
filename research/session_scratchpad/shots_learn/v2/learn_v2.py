"""Learned shots v2 (plan.json, exploratory): v1 features + player context,
more units (M5 and the M7 / M9 gate clips), v1 against v2 by nested
leave-one-match-out on the same units.

    python learn_v2.py cv        # -> cv_v2.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import learn_shots as LS  # noqa: E402

ROOT = LS.ROOT
GPU = LS.GPU
STEP_S = 2 / LS.FPS
BOX_X, BOX_Y = 16.5, 20.16          # penalty box depth and half width (m)
GOAL_Y = 34.0


def units() -> list[dict]:
    us = LS.units(LS.HERE / "clips_train.json") + LS.units(LS.HERE / "clips_test.json")
    dev = json.loads((ROOT / "research/session_scratchpad/kaggle/shot_clips_dev.json").read_text())
    us += [{"name": c["name"], "match": c["match"], "span": c["span_local"]} for c in dev["clips"]]
    us += [{"name": w, "match": w.split("_")[1], "span": None} for w in LS.TRAIN_WINDOWS]
    return us


def _metric(u: dict) -> pd.DataFrame:
    made = GPU / u["name"] / f"output_{u['name']}" / "metric_v3s_link.parquet"
    return pd.read_parquet(made if made.exists() else GPU / u["name"] / "metric_v3s_link.parquet")


def context(u: dict, frames: np.ndarray, px: np.ndarray, py: np.ndarray) -> pd.DataFrame:
    """Per detection frame (a clip's span: the ball runs only there, so the
    next-10-s look stops at the span's end, +10 s after the last shot):
    goalkeeper proxy, box occupancy, box speed, the ball's next-10-s approach
    to a goal line / the centre spot, and longer ball windows."""
    m = _metric(u)
    p = m[(m.cls == "player") & m.team.isin(["team_A", "team_B"])].sort_values(["track_id", "frame"])
    # player speed per row from the same track one step apart
    p = p.assign(v=np.hypot(p.groupby("track_id").x.diff(), p.groupby("track_id").y.diff())
                 / (p.groupby("track_id").frame.diff() / LS.FPS))
    by = {f: g for f, g in p.groupby("frame")}
    rec = []
    for f, bx, byy in zip(frames, px, py):
        g = by.get(f)
        if g is None:
            rec.append([np.nan] * 10)
            continue
        x, y, v, t = g.x.to_numpy(), g.y.to_numpy(), g.v.to_numpy(), g.team.to_numpy()
        row = []
        for gx in (0.0, 105.0):                         # nearest player to each goal: keeper proxy
            d = np.hypot(x - gx, y - GOAL_Y)
            i = int(np.argmin(d))
            row += [d[i], v[i] if np.isfinite(v[i]) else np.nan]
        near = 0.0 if (np.isfinite(bx) and bx < 52.5) else 105.0
        inbox = (np.abs(x - near) <= BOX_X) & (np.abs(y - GOAL_Y) <= BOX_Y)
        row += [float((inbox & (t == "team_A")).sum()), float((inbox & (t == "team_B")).sum()),
                float(np.nanmean(v[inbox])) if inbox.any() else np.nan]
        if np.isfinite(bx):
            dd = np.hypot(x - bx, y - byy)
            close = dd <= 5
            row += [float(np.nanmean(v[close])) if close.any() else np.nan, float(close.sum())]
        else:
            row += [np.nan, np.nan]
        row += [min(abs(near - x).min(), 99.0)]          # deepest player towards that goal line
        rec.append(row)
    c = pd.DataFrame(rec, index=frames, columns=[
        "gk0_d", "gk0_v", "gk105_d", "gk105_v", "box_A", "box_B", "box_v",
        "near5_v", "near5_n", "deep_x"])
    c["gk_d"] = np.where(px < 52.5, c.gk0_d, c.gk105_d)
    c["gk_v"] = np.where(px < 52.5, c.gk0_v, c.gk105_v)
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 2
    n10 = int(10 / (step / LS.FPS))
    goal_line = pd.Series(np.minimum(np.abs(px - 0.0), np.abs(px - 105.0)), index=frames)
    centre = pd.Series(np.hypot(px - 52.5, py - GOAL_Y), index=frames)
    c["line_min_next10"] = goal_line[::-1].rolling(n10, min_periods=1).min()[::-1]
    c["centre_min_next10"] = centre.shift(-int(3 / (step / LS.FPS)))[::-1].rolling(n10, min_periods=1).min()[::-1]
    c["gk_v_max_next2"] = c.gk_v[::-1].rolling(int(2 / (step / LS.FPS)), min_periods=1).max()[::-1]
    return c


def features_v2(u: dict) -> pd.DataFrame:
    cache = GPU / u["name"] / "shot_features_v2.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    f = LS.features(u)               # v1, already cropped to a clip's span
    frames = f.index.to_numpy()
    px, py = f.x_0.to_numpy(), f.y_0.to_numpy()
    c = context(u, frames, px, py)
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 2
    n3 = int(3 / (step / LS.FPS))
    long = pd.DataFrame({
        "speed_max_pm3": f.speed_0.rolling(2 * n3 + 1, center=True, min_periods=1).max(),
        "gvto_max_pm3": f.g_vto.rolling(2 * n3 + 1, center=True, min_periods=1).max(),
        "has_frac_pm3": f.has_0.rolling(2 * n3 + 1, center=True, min_periods=1).mean()}, index=frames)
    out = pd.concat([f, c, long], axis=1)
    out.to_parquet(cache)
    return out


def main():
    from src.unseal import unsealed
    for m in ("128057", "132831"):       # the gate clips join the training data (owner, 2026-10-22)
        if not unsealed(m, "shots v2: gate clips as training data"):
            raise SystemExit("VEO_UNSEAL needed for the M7 / M9 clips")
    us = [u for u in units() if (GPU / u["name"]).exists()]
    # v1 on the same units: LS.features caches the span-cropped v1 per unit
    d1 = {u["name"]: (LS.features(u), LS.labels(u), u["match"]) for u in us}
    d2 = {u["name"]: (features_v2(u), LS.labels(u), u["match"]) for u in us}
    out = {"units": len(us), "matches": sorted({u["match"] for u in us}),
           "labels": sum(len(v[1]) for v in d1.values())}
    for name, d in (("v1", d1), ("v2", d2)):
        names = list(d)
        cols = LS.cols_of(d)
        probs, thr = LS.nested(d, names, cols)
        out[name] = LS.score(d, names, probs, thr)
        print(name, {k: out[name][k] for k in ("tp", "fp", "fn", "recall", "precision")}, flush=True)
    a = out["v1"]["per match [tp, fp, fn]"]
    b = out["v2"]["per match [tp, fp, fn]"]
    ms = sorted(a)
    diff = np.array([b[m][0] - a[m][0] for m in ms], float)
    rng = np.random.default_rng(0)
    boots = [diff[rng.integers(0, len(ms), len(ms))].sum() for _ in range(5000)]
    out["hits v2 - v1"] = {"total": float(diff.sum()), "bootstrap 95% over matches":
                           [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    (HERE / "cv_v2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["hits v2 - v1"]))


if __name__ == "__main__":
    main()
