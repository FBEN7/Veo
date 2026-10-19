"""Refit a match's camera model from the players (M1's release calibration puts
the near touchline in mid-pitch, and its painted lines are too faint at night
for a line fit).

Same model as the release (fisheye K, D, Knew, then the homography from
undistorted pixels to pitch metres). Free: the inverse homography (8 dof) and
K's vertical focal length and principal point. Data: our player detections'
feet (output_<w>/tracks.parquet) and the ground truth's pitch positions, every
10th frame of the fit windows; correspondences by nearest neighbour in pitch
metres under the current model, the radius shrinking 8 -> 1.5 m, robust least
squares. Checked on other windows, and by the pitch lines drawn on a frame.

This repairs the measuring instrument for a test match; the product still
needs a calibration from the lines (or a calibration step) on its own cameras.

    python fit_players.py MATCH --fit W1 [W2 ...] --check W3 [W4 ...] [--out-dir DIR]

(a diagnostic fit of another match goes to another --out-dir, or
calib_override.py would use it)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ROOT = Path("/home/user/Veo")
sys.path.insert(0, str(ROOT))
from src import soccertrack_v2 as st  # noqa: E402


def to_pitch(uv, K, D, Knew, Hinv):
    uv = np.asarray(uv, float).reshape(-1, 2)
    if not len(uv):
        return np.zeros((0, 2))
    und = cv2.fisheye.undistortPoints(np.ascontiguousarray(np.asarray(uv, float).reshape(-1, 1, 2)),
                                      K, D, P=Knew).reshape(-1, 2)
    q = np.c_[und, np.ones(len(und))] @ Hinv.T
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = q[:, :2] / q[:, 2:] - (52.5, 34.0)
    xy[q[:, 2] <= 0] = np.nan
    return xy


def pairs_data(w: str, step: int = 10):
    raw = pd.read_parquet(ROOT / f"output_{w}/tracks.parquet")
    det = raw[(raw.cls == "player")]
    gt = pd.read_parquet(ROOT / f"data/soccertrack_v2/{w}_gt.parquet")
    frames = sorted(set(det.frame) & set(gt.frame))[::step]
    D = {f: g[["px", "py"]].to_numpy(float) for f, g in det[det.frame.isin(frames)].groupby("frame")}
    G = {f: g[["x", "y"]].to_numpy(float) for f, g in gt[gt.frame.isin(frames)].groupby("frame")}
    return [(D[f], G[f]) for f in frames if f in D and f in G]


class Model:
    def __init__(self, cal):
        self.K0, self.D, self.Knew = cal["K"].copy(), cal["D"].copy(), cal["Knew"].copy()
        H = cal["Hinv"] / abs(cal["Hinv"][2, 2])
        self.H0 = H

    def unpack(self, v):
        Hinv = self.H0.copy().ravel()
        Hinv[:8] = self.H0.ravel()[:8] * (1 + v[:8])
        K = self.K0.copy()
        K[1, 1] = self.K0[1, 1] * (1 + v[8])
        K[1, 2] = self.K0[1, 2] + 100 * v[9]
        return K, Hinv.reshape(3, 3)

    def pitch(self, v, uv):
        K, Hinv = self.unpack(v)
        return to_pitch(uv, K, self.D, self.Knew, Hinv)


def match(model, v, data, radius):
    src, dst = [], []
    for uv, g in data:
        xy = model.pitch(v, uv)
        ok = np.isfinite(xy).all(1)
        if not ok.any():
            continue
        d, i = cKDTree(g).query(xy[ok])
        keep = d <= radius
        src.append(uv[ok][keep])
        dst.append(g[i[keep]])
    return (np.vstack(src), np.vstack(dst)) if src else (np.zeros((0, 2)), np.zeros((0, 2)))


def error(model, v, data, radius=3.0):
    uv, g = match(model, v, data, radius)
    e = np.hypot(*(model.pitch(v, uv) - g).T)
    return {"pairs": int(len(e)), "median m": round(float(np.median(e)), 3) if len(e) else None,
            "share of detections matched within 3 m": round(len(e) / max(sum(len(u) for u, _ in data), 1), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match")
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--check", nargs="*", default=[])
    ap.add_argument("--out-dir", default=str(ROOT / ".cache/calib_fix"))
    args = ap.parse_args()
    cal = st.calibration(args.match)
    model = Model(cal)
    fit_data = sum((pairs_data(w) for w in args.fit), [])
    check = {w: pairs_data(w) for w in args.check}
    v = np.zeros(10)
    before = {"fit": error(model, v, fit_data), **{w: error(model, v, d) for w, d in check.items()}}
    print("before", before, flush=True)
    for radius in (8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.5):
        uv, g = match(model, v, fit_data, radius)
        res = least_squares(lambda p: (model.pitch(p, uv) - g).ravel(), v, loss="soft_l1", f_scale=0.5,
                            max_nfev=200)
        v = res.x
        print(radius, len(uv), error(model, v, fit_data), flush=True)
    after = {"fit": error(model, v, fit_data), **{w: error(model, v, d) for w, d in check.items()}}
    print("after", after, flush=True)
    K, Hinv = model.unpack(v)
    out = Path(args.out_dir) / f"{args.match}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, K=K, D=model.D, Knew=model.Knew, Hinv=Hinv, v=v)
    out.with_suffix(".json").write_text(json.dumps(
        {"match": args.match, "fit windows": args.fit, "check windows": args.check,
         "before": before, "after": after, "v": v.tolist()}, indent=1))


if __name__ == "__main__":
    main()
