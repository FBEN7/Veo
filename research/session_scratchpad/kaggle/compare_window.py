"""GPU (Kaggle, run_windows.py) against CPU (this repo) on one window, from
the returned tarball: player detections, tracks with teams, events_v1, and
the dense v3 ball rows. Only agreement is measured; no label is read.

    python compare_window.py W GPU_DIR [--out compare_W.json]

GPU_DIR holds the extracted tarball (output_<w>/, .cache/v3_balls_<w>.*).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def feet_match(a: pd.DataFrame, b: pd.DataFrame, tol: float = 3.0) -> dict:
    """Player rows matched frame by frame on the foot point (px, py) within
    `tol` pixels, greedily nearest first."""
    n_a, n_b, hit, d_all = len(a), len(b), 0, []
    gb = dict(tuple(b.groupby("frame")))
    for f, fa in a.groupby("frame"):
        fb = gb.get(f)
        if fb is None:
            continue
        pa, pb = fa[["px", "py"]].to_numpy(), fb[["px", "py"]].to_numpy()
        d = np.hypot(pa[:, None, 0] - pb[None, :, 0], pa[:, None, 1] - pb[None, :, 1])
        used_a, used_b = set(), set()
        for k in np.argsort(d, axis=None):
            i, j = divmod(int(k), d.shape[1])
            if d[i, j] > tol:
                break
            if i in used_a or j in used_b:
                continue
            used_a.add(i), used_b.add(j)
            d_all.append(d[i, j])
        hit += len(used_a)
    return {"rows_gpu": n_a, "rows_cpu": n_b, "matched": hit,
            "share_of_gpu": hit / max(n_a, 1), "share_of_cpu": hit / max(n_b, 1),
            "median_px": float(np.median(d_all)) if d_all else None,
            "p99_px": float(np.percentile(d_all, 99)) if d_all else None}


def events_of(path: Path) -> list:
    return json.loads(path.read_text()) if path.exists() else []


def event_match(a: list, b: list, tol: float = 0.5) -> dict:
    out = {}
    for kind in sorted({e["event_type"] for e in a + b}):
        ta = sorted(e["timestamp_s"] for e in a if e["event_type"] == kind)
        tb = sorted(e["timestamp_s"] for e in b if e["event_type"] == kind)
        used, hit = set(), 0
        for t in ta:
            j = min((j for j in range(len(tb)) if j not in used), key=lambda j: abs(tb[j] - t),
                    default=None)
            if j is not None and abs(tb[j] - t) <= tol:
                used.add(j)
                hit += 1
        out[kind] = {"gpu": len(ta), "cpu": len(tb), "matched_0.5s": hit}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("window")
    ap.add_argument("gpu_dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    w, g, c = args.window, Path(args.gpu_dir), ROOT
    res = {"window": w}
    for name in ("tracks.parquet", "events_v1/tracks_teams.parquet"):
        tg = pd.read_parquet(g / f"output_{w}" / name)
        tc = pd.read_parquet(c / f"output_{w}" / name)
        pg, pc = tg[tg.cls == "player"], tc[tc.cls == "player"]
        r = feet_match(pg, pc)
        r["tracks_gpu"], r["tracks_cpu"] = int(pg.track_id.nunique()), int(pc.track_id.nunique())
        if "team" in tg:
            r["team_counts_gpu"] = pg.team.value_counts().to_dict()
            r["team_counts_cpu"] = pc.team.value_counts().to_dict()
        res[name] = r
    res["events_v1"] = event_match(events_of(g / f"output_{w}/events_v1/events.json"),
                                   events_of(c / f"output_{w}/events_v1/events.json"))
    bg = pd.read_parquet(g / f".cache/v3_balls_{w}.parquet")
    bc = pd.read_parquet(c / f".cache/v3_balls_{w}.parquet")
    r = {"rows_gpu": len(bg), "rows_cpu": len(bc)}
    if len(bg) == len(bc):
        a = bg.sort_values(["frame", "logit"]).reset_index(drop=True)
        b = bc.sort_values(["frame", "logit"]).reset_index(drop=True)
        r["same_frames"] = bool((a.frame == b.frame).all())
        r["max_abs_logit_diff"] = float(np.abs(a.logit - b.logit).max())
        r["max_px_diff"] = float(np.hypot(a.px - b.px, a.py - b.py).max())
    res["v3_balls"] = r
    text = json.dumps(res, indent=1, default=str)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
