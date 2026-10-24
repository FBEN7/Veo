"""Refit a match's camera model from its painted lines (M1's release calibration
puts the near touchline in mid-pitch).

The model is the release's: fisheye K, D, Knew, then a pitch-to-image
homography. Free here: the homography (8 dof) and K's vertical focal length and
principal point (the release's M1 values look made for another frame height);
D and the horizontal intrinsics stay. The data: white line pixels of a median
frame (players averaged out) of each window; the cost: the mean distance, in
pixels, from points sampled along the pitch's lines (IFAB layout, 105 x 68 m) to
the nearest white pixel, truncated, from coarse to fine truncation.

    python fit_lines.py MATCH WINDOW [WINDOW ...]   # writes .cache/calib_fix/<match>.npz

Checked independently afterwards by players' ground-truth pitch positions
against our detections' projected feet (`check_fit.py`): nothing here reads them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

ROOT = Path("/home/user/Veo")
sys.path.insert(0, str(ROOT))
from src import soccertrack_v2 as st  # noqa: E402

L, W = 52.5, 34.0


def model_lines():
    """Points along the pitch markings, centre-origin metres."""
    segs = [((-L, -W), (L, -W)), ((-L, W), (L, W)), ((-L, -W), (-L, W)), ((L, -W), (L, W)),
            ((0, -W), (0, W))]
    for s in (-1, 1):
        x0, x1, x2 = s * L, s * (L - 16.5), s * (L - 5.5)
        segs += [((x0, -20.16), (x1, -20.16)), ((x1, -20.16), (x1, 20.16)), ((x0, 20.16), (x1, 20.16)),
                 ((x0, -9.16), (x2, -9.16)), ((x2, -9.16), (x2, 9.16)), ((x0, 9.16), (x2, 9.16))]
    pts = [np.c_[np.linspace(a[0], b[0], int(np.hypot(b[0] - a[0], b[1] - a[1]) * 2) + 2),
                 np.linspace(a[1], b[1], int(np.hypot(b[0] - a[0], b[1] - a[1]) * 2) + 2)] for a, b in segs]
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    pts.append(np.c_[9.15 * np.cos(th), 9.15 * np.sin(th)])
    return np.vstack(pts)


def project(xy, K, D, Knew, H):
    p = np.c_[xy + (52.5, 34.0), np.ones(len(xy))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        und = p[:, :2] / p[:, 2:]
    ok = (p[:, 2] > 0) & np.isfinite(und).all(1)
    norm = (und - Knew[:2, 2]) / np.diag(Knew)[:2]
    out = np.full((len(xy), 2), np.nan)
    if ok.any():
        out[ok] = cv2.fisheye.distortPoints(np.ascontiguousarray(norm[ok].reshape(-1, 1, 2)),
                                            K, D).reshape(-1, 2)
    return out


def line_mask(w: str, n: int = 31) -> np.ndarray:
    cap = cv2.VideoCapture(str(ROOT / f"data/soccertrack_v2/{w}.mp4"))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    med = np.median(np.stack(frames), axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (25, 30, 25), (95, 255, 255))
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tophat = gray - cv2.GaussianBlur(gray, (0, 0), 6)
    white = ((tophat > 12) & (hsv[..., 1] < 90) & (grass > 0)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return white, med


def fit(match: str, windows: list[str]):
    cal = st.calibration(match)
    K0, D, Knew = cal["K"].copy(), cal["D"].copy(), cal["Knew"].copy()
    H0 = np.linalg.inv(cal["Hinv"])
    H0 = H0 / H0[2, 2]
    dts, shape = [], None
    for w in windows:
        white, med = line_mask(w)
        shape = white.shape
        dts.append(cv2.distanceTransform((1 - white).astype(np.uint8), cv2.DIST_L2, 5))
        cv2.imwrite(str(ROOT / f".cache/calib_fix/{w}_lines.png"), white * 255)
    xy = model_lines()
    h, wd = shape

    def unpack(v):
        H = H0.copy().ravel()
        H[:8] = H0.ravel()[:8] * (1 + v[:8])
        K = K0.copy()
        K[1, 1] = K0[1, 1] * (1 + v[8])
        K[1, 2] = K0[1, 2] + 100 * v[9]
        return K, H.reshape(3, 3)

    def cost(v, cap):
        K, H = unpack(v)
        uv = project(xy, K, D, Knew, H)
        ok = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < wd - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        if ok.mean() < 0.5:
            return cap * 2
        u, vv = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
        d = np.mean([np.minimum(dt[vv, u], cap).mean() for dt in dts])
        return d + cap * (1 - ok.mean())

    v = np.zeros(10)
    log = []
    for cap in (200, 80, 30, 12):
        r = minimize(cost, v, args=(cap,), method="Powell", options={"maxiter": 6000, "xtol": 1e-5, "ftol": 1e-4})
        v = r.x
        log.append({"cap": cap, "cost": float(r.fun), "start cost": float(cost(np.zeros(10), cap))})
        print(log[-1], flush=True)
    K, H = unpack(v)
    out = ROOT / f".cache/calib_fix/{match}.npz"
    np.savez(out, K=K, D=D, Knew=Knew, H=H, v=v)
    (ROOT / f".cache/calib_fix/{match}.json").write_text(json.dumps({"match": match, "windows": windows, "log": log}, indent=1))
    return K, D, Knew, H


if __name__ == "__main__":
    (ROOT / ".cache/calib_fix").mkdir(parents=True, exist_ok=True)
    fit(sys.argv[1], sys.argv[2:])
