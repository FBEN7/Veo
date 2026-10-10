"""Weakly labelled ball patches from SoccerTrack v2, to train a ball
detector for fixed panoramas (`train_ball_heatmap.py`).

The release has no ball boxes, but every ball action says where it
happened, and its ball track (`ball/`) is anchored there and drawn in
straight lines between actions. Projected into the picture with the
match's calibration, that point is within ~60 px of the ball at an action
(it is the actor's position on a 1.05 x 0.68 m grid) and further off
between actions. So each patch carries a point and a radius: the ball is
somewhere within it, or hidden.

For each half, streamed once from the dataset, at every action frame and
at the midpoint of every gap of more than `GAP_S` between actions: a
`SIZE` x `SIZE` patch around the projected point, shifted by up to
`JITTER` so the ball is not always central, stored as JPEG with the point
and radius (`patches.parquet`).

    python build_ball_patches.py --match 118575 --half 1 --out DIR
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import soccertrack_v2 as st

SIZE = 256
JITTER = 64
GAP_S = 1.5
R_ACTION = 60.0
R_BETWEEN = 100.0


def pitch_to_image(xy, cal: dict) -> np.ndarray:
    """(N, 2) pitch metres (centre origin) to panorama pixels: the inverse
    of `soccertrack_v2.image_to_pitch`."""
    H = np.linalg.inv(cal["Hinv"])
    p = np.c_[np.asarray(xy, float) + (52.5, 34.0), np.ones(len(xy))] @ H.T
    und = p[:, :2] / p[:, 2:]
    norm = (und - cal["Knew"][:2, 2]) / np.diag(cal["Knew"])[:2]
    return cv2.fisheye.distortPoints(
        np.ascontiguousarray(norm.reshape(-1, 1, 2)), cal["K"],
        cal["D"]).reshape(-1, 2)


def targets(match: str, half: int) -> pd.DataFrame:
    """Frames to cut and the ball's projected point there: frame, u, v,
    radius, kind ('action' / 'between')."""
    ev = st.events(match)
    ev = ev[ev.half == half].sort_values("frame")
    z = np.load(io.BytesIO(st.open_stream(
        f"ball/{match}_{st._half(half)}_ball.npz").read()), allow_pickle=False)
    track = pd.DataFrame({"x": z["x"], "y": z["y"]},
                         index=z["frame"].astype(int) - 1)
    f = ev.frame.to_numpy()
    rows = [(int(a), "action") for a in f]
    rows += [(int((a + b) // 2), "between") for a, b in zip(f[:-1], f[1:])
             if b - a > GAP_S * st.FPS]
    out = pd.DataFrame(rows, columns=["frame", "kind"]).drop_duplicates("frame")
    out = out[out.frame.isin(track.index)]
    xy = track.loc[out.frame, ["x", "y"]].to_numpy()
    ok = np.isfinite(xy).all(axis=1)
    out, xy = out[ok], xy[ok]
    uv = pitch_to_image(xy, st.calibration(match))
    out["u"], out["v"] = uv[:, 0], uv[:, 1]
    out["radius"] = np.where(out.kind == "action", R_ACTION, R_BETWEEN)
    return out.sort_values("frame").reset_index(drop=True)


def build(match: str, half: int, out: Path, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    want = targets(match, half)
    by_frame = {int(r.frame): r for r in want.itertuples()}
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    container = st.open_video(match, half)
    try:
        stream = container.streams.video[0]
        tb = float(stream.time_base)
        t0 = float(stream.start_time or 0) * tb
        last = max(by_frame)
        for frame in container.decode(stream):
            k = int(round((float(frame.pts) * tb - t0) * st.FPS))
            if k > last:
                break
            r = by_frame.get(k)
            if r is None:
                continue
            image = frame.to_ndarray(format="bgr24")
            H, W = image.shape[:2]
            cx = int(np.clip(r.u + rng.integers(-JITTER, JITTER + 1),
                             SIZE // 2, W - SIZE // 2))
            cy = int(np.clip(r.v + rng.integers(-JITTER, JITTER + 1),
                             SIZE // 2, H - SIZE // 2))
            x0, y0 = cx - SIZE // 2, cy - SIZE // 2
            name = f"{match}_{half}_{k:06d}.jpg"
            cv2.imwrite(str(out / name), image[y0:y0 + SIZE, x0:x0 + SIZE],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"file": name, "match": match, "half": half,
                         "frame": k, "kind": r.kind, "u": r.u - x0,
                         "v": r.v - y0, "radius": r.radius,
                         "x0": x0, "y0": y0})
    finally:
        container.close()
    table = pd.DataFrame(rows)
    path = out / "patches.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        table = pd.concat([old[~((old.match == match) & (old.half == half))],
                           table])
    table.to_parquet(path)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t = build(args.match, args.half, Path(args.out))
    mine = t[(t.match == args.match) & (t.half == args.half)]
    print(f"  {args.match} half {args.half}: {len(mine)} patches "
          f"({mine.kind.value_counts().to_dict()})", flush=True)


if __name__ == "__main__":
    main()
