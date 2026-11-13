"""Quick feasibility of inferring the ball where it is not detected
(plan.json): the existing link (step 1), then gap filling in the picture
(step 3) and a ball-at-feet state (step 2), scored on hand-verified balls
and on simulated occlusions.

    python feasibility.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball/events_tune"))
import tune_rules as tr  # noqa: E402
from detect_ball_heatmap import read_sidecar  # noqa: E402
from src import ball_link as bl  # noqa: E402
from src import ball_spots as bs  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402

WINDOWS = ["st2_118578_2nd_f015000", "st2_118578_2nd_f037500"]
TRUTH = ROOT / ".cache/soccertrack_v2/ball_verified/A/frames.parquet"
GAPS = [12, 25, 50]
HOLES = [6, 12, 25]
HIT, NEAR = 12.0, 30.0
SPAN = 200


def candidates(w, cal):
    path = ROOT / f".cache/v3_balls_{w}.parquet"
    n = len(read_sidecar(path)["frames"])
    rows = pd.read_parquet(path)
    rows = bs.drop_marks(rows, bs.spot_marks(rows, cal, n))
    keep = np.ones(len(rows), bool)
    for x, y in tr.static_points(rows, n):
        keep &= np.hypot(rows.px.to_numpy() - x, rows.py.to_numpy() - y) > 1.5 * tr.STATIC_PX
    return rows[keep].reset_index(drop=True)


def players_of(w):
    p = ROOT / f"output_{w}/tracks.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["frame", "track_id", "cls", "px", "py", "crop_h"])
    t = pd.read_parquet(p)
    return t[t.cls == "player"].reset_index(drop=True)


def do_link(rows, cal, players, frames):
    return bl.link(rows, cal, players, c_null=1.0, c_static=2.0, tau=0.10, frames=frames)


def owner(players, f, u, v):
    """The player track whose feet are within one box height of (u, v) at
    the nearest player frame (players every 2nd frame)."""
    if players.empty:
        return None
    for g in (f, f - 1, f + 1):
        q = players[players.frame == g]
        if len(q):
            d = np.hypot(q.px - u, q.py - v)
            i = d.idxmin()
            return int(q.track_id[i]) if d[i] <= q.crop_h[i] else None
    return None


def feet_path(players, tid, f0, f1):
    q = players[(players.track_id == tid) & (players.frame >= f0) & (players.frame <= f1)]
    if len(q) < 2:
        return None
    fr = np.arange(f0, f1 + 1)
    return np.interp(fr, q.frame, q.px), np.interp(fr, q.frame, q.py - 0.05 * q.crop_h)


def fill(track, gap, players=None):
    """Positions per frame: linked picks, then gaps up to `gap` frames filled
    (linear, or along the same player's feet when both ends are at them)."""
    t = track.sort_values("frame")
    pos = {int(r.frame): (r.px, r.py, "det") for r in t.itertuples()}
    fr = t.frame.to_numpy(int)
    u, v = t.px.to_numpy(), t.py.to_numpy()
    for i in range(len(fr) - 1):
        a, b = fr[i], fr[i + 1]
        if b - a <= 1 or b - a > gap:
            continue
        path = None
        if players is not None:
            oa, ob = owner(players, a, u[i], v[i]), owner(players, b, u[i + 1], v[i + 1])
            if oa is not None and oa == ob:
                path = feet_path(players, oa, a, b)
        for k, f in enumerate(range(a + 1, b)):
            if path is not None:
                pos[f] = (path[0][k + 1], path[1][k + 1], "feet")
            else:
                s = (f - a) / (b - a)
                pos[f] = (u[i] + s * (u[i + 1] - u[i]), v[i] + s * (v[i + 1] - v[i]), "line")
    return pos


def score(pos, truth):
    ball = truth[truth.status == "ball"]
    out = {"n ball frames": len(ball)}
    e = []
    for r in ball.itertuples():
        p = pos.get(int(r.frame))
        e.append(np.hypot(p[0] - r.u, p[1] - r.v) if p else np.nan)
    e = np.array(e)
    given = np.isfinite(e)
    out.update({"coverage": round(given.mean(), 3), "hit 12px": round((e <= HIT).mean(), 3),
                "precision 12px": round((e[given] <= HIT).mean(), 3) if given.any() else None,
                "precision 30px": round((e[given] <= NEAR).mean(), 3) if given.any() else None})
    unk = truth[truth.status == "unknown"]
    out["unknown frames given"] = round(np.mean([int(f) in pos for f in unk.frame]), 3) if len(unk) else None
    return out


def occlusion(rows, cal, players, truth, base_pos, use_feet):
    res = {}
    ball = truth[truth.status == "ball"]
    hits = [r for r in ball.itertuples() if (p := base_pos.get(int(r.frame)))
            and np.hypot(p[0] - r.u, p[1] - r.v) <= HIT]
    for h in HOLES:
        errs, kinds = [], []
        for r in hits:
            f = int(r.frame)
            lo, hi = f - SPAN, f + SPAN
            sub = rows[(rows.frame >= lo) & (rows.frame <= hi) & ((rows.frame < f - h) | (rows.frame > f + h))]
            pl = players[(players.frame >= lo) & (players.frame <= hi)]
            tr_ = do_link(sub, cal, pl, np.arange(max(lo, 0), hi + 1))
            pos = fill(tr_, 2 * h + 2, pl if use_feet else None)
            p = pos.get(f)
            errs.append(np.hypot(p[0] - r.u, p[1] - r.v) if p else np.nan)
            kinds.append(p[2] if p else "none")
        e = np.array(errs)
        res[f"hole {2 * h / 25:.1f} s"] = {
            "n": len(e), "recovered (any)": round(np.isfinite(e).mean(), 3),
            "within 12px": round((e <= HIT).mean(), 3), "within 30px": round((e <= NEAR).mean(), 3),
            "median error px": round(float(np.nanmedian(e)), 1) if np.isfinite(e).any() else None,
            "by kind": pd.Series(kinds).value_counts().to_dict()}
    return res


def main():
    truth_all = pd.read_parquet(TRUTH)
    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0]}
    for w in WINDOWS:
        info = json.loads((ROOT / "data/soccertrack_v2" / f"{w}.json").read_text())
        cal = st.calibration(str(info["match"]))
        rows = candidates(w, cal)
        players = players_of(w)
        truth = truth_all[truth_all.window == w]
        frames = np.arange(int(info["frames"]))
        base = do_link(rows, cal, players, frames)
        base_pos = fill(base, 1)
        r = {"base (link)": score(base_pos, truth)}
        for g in GAPS:
            r[f"fill line <= {g / 25:.1f} s"] = score(fill(base, g), truth)
            if len(players):
                r[f"fill feet+line <= {g / 25:.1f} s"] = score(fill(base, g, players), truth)
        r["simulated occlusion, line"] = occlusion(rows, cal, players, truth, base_pos, False)
        if len(players):
            r["simulated occlusion, feet+line"] = occlusion(rows, cal, players, truth, base_pos, True)
        out[w] = r
        print(w, json.dumps(r, indent=1, default=str), flush=True)
    (HERE / "feasibility.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
