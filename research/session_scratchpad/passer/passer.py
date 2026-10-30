"""Who made the pass (plan.json): the actor of each labelled BAS action
chosen from detections only -- the v3s_link ball and player tracks in pitch
metres -- and scored against the label's player_id through feet matching.

    VEO_UNSEAL=... python passer.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/shots_goals"))
import eval_soccertrack_v2 as ev  # noqa: E402
from score_goals_test import wilson  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.unseal import unsealed  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

GPU = ROOT / ".cache/gpu"
CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
UNSEEN = {"117092", "128057", "132831"}
SEALED = {"128057", "132831"}
PASSES = {"PASS", "HIGH PASS", "CROSS"}
NOT_ACTOR = {"OUT", "GOAL"}
RADIUS = 3.0
METRIC = "metric_v3s_link.parquet"   # the pitch tracks read (ball_grass/: the relinked ones)
FILL = 0            # ball gaps up to FILL frames filled linearly (ball_interp/; 0: off)
FIX = ROOT / ".cache/calib_fix"         # M1's refitted camera (calib_fix/fit_players.py)


def calibration(match: str) -> dict:
    f = FIX / f"{match}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in ("K", "D", "Knew", "Hinv")}
    return st.calibration(match)


def match_pitch(det: pd.DataFrame, gt: pd.DataFrame, limit_m: float = 3.0) -> pd.Series:
    """Ground-truth track per detection by pitch distance (as stats_fix.py)."""
    out = pd.Series(np.nan, index=det.index)
    gt_by = {f: g for f, g in gt.groupby("frame")}
    for f, d in det.groupby("frame"):
        g = gt_by.get(f)
        if g is None:
            continue
        dist = np.linalg.norm(d[["x", "y"]].to_numpy()[:, None] - g[["x", "y"]].to_numpy()[None], axis=2)
        dist = np.nan_to_num(dist, nan=1e6)
        r, c = linear_sum_assignment(np.where(dist <= limit_m, dist, 1e6))
        for i, j in zip(r, c):
            if dist[i, j] <= limit_m:
                out.loc[d.index[i]] = g.track_id.iloc[j]
    return out


def nearest(pl: pd.DataFrame, bx: float, by: float):
    if pl.empty:
        return None, np.inf
    d = np.hypot(pl.x - bx, pl.y - by)
    i = d.idxmin()
    return i, float(d[i])


def unit(w: str) -> list[dict]:
    info = json.loads((CLIPS / f"{w}.json").read_text())
    m = pd.read_parquet(GPU / w / f"output_{w}" / "metric_v3s_link.parquet")
    # pitch metres, centre origin, from the image through the match's camera
    # (the Kaggle metric step used the release's camera, broken on M1)
    m[["x", "y"]] = st.image_to_pitch(m[["px_raw", "py_raw"]].to_numpy(), calibration(str(info["match"])))
    ball = m[m.cls == "ball"].set_index("frame")
    if FILL:
        ball = ball[~ball.index.duplicated()].sort_index()
        fr = ball.index.to_numpy()
        new = [f for a, b in zip(fr[:-1], fr[1:]) if 1 < b - a <= FILL for f in range(a + 1, b)]
        full = np.sort(np.r_[fr, new]).astype(int)
        ball = pd.DataFrame({"x": np.interp(full, fr, ball.x), "y": np.interp(full, fr, ball.y)}, index=full)
    pl = m[m.cls == "player"]
    by_f = {f: g for f, g in pl.groupby("frame")}
    g = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
    pid = g.drop_duplicates("track_id").set_index("track_id").player_id.astype(str)
    if (FIX / f"{info['match']}.npz").exists():  # M1: truth boxes off, pitch metres under the refit
        person = match_pitch(pl, g[g.x.notna() & g.role.isin(["player", "goalkeeper"])])
    else:                                        # elsewhere: feet, as eval_soccertrack_v2
        person = ev.match_feet(ev.boxes(pl), ev.truth(CLIPS / f"{w}_gt.parquet"))
    who = person.map(pid)                       # detection index -> player_id
    labels = pd.read_csv(CLIPS / f"{w}_events.csv")
    rows = []
    for e in labels.itertuples():
        if e.label in NOT_ACTOR or pd.isna(e.player_id):
            continue
        truth = str(int(e.player_id))
        f = int(e.frame)
        if f not in by_f:               # players every 2nd frame (stride 2): the nearest tracked frame
            ks = np.array(sorted(by_f)) if by_f else np.array([f])
            f = int(ks[np.argmin(abs(ks - f))])
        r = {"match": info["match"], "half": info["half"], "abs": int(info["start_frame"]) + f,
             "label": e.label, "unit": w, "truth": truth,
             "detected": bool((who[pl.frame == f] == truth).any()) if f in by_f else False}
        # A: nearest ball row within +-5 frames, nearest player at f
        a = None
        near = ball.index[(ball.index >= f - 5) & (ball.index <= f + 5)]
        if len(near) and f in by_f:
            bf = near[np.argmin(abs(near - f))]
            b = ball.loc[[bf]].iloc[0]
            i, d = nearest(by_f[f], b.x, b.y)
            a = i if d <= RADIUS else None
        # B: best (player, frame) over f-10 .. f+5
        best, bd = None, np.inf
        for bf in ball.index[(ball.index >= f - 10) & (ball.index <= f + 5)]:
            if bf not in by_f:
                continue
            b = ball.loc[[bf]].iloc[0]
            i, d = nearest(by_f[bf], b.x, b.y)
            if d < bd:
                best, bd = i, d
        bsel = best if bd <= RADIUS else None
        for k, i in (("A", a), ("B", bsel)):
            r[f"given_{k}"] = i is not None
            r[f"right_{k}"] = i is not None and who.get(i) == truth
        rows.append(r)
    return rows


def summary(df: pd.DataFrame) -> dict:
    n = len(df)
    out = {"n": n, "detected (ceiling)": round(df.detected.mean(), 3) if n else None}
    for k in "AB":
        right, given = int(df[f"right_{k}"].sum()), int(df[f"given_{k}"].sum())
        out[k] = {"right": right, "share right": round(right / max(n, 1), 3), "wilson": wilson(right, n),
                  "coverage": round(given / max(n, 1), 3), "precision": round(right / max(given, 1), 3)}
    return out


def main():
    for s in SEALED:
        if not unsealed(s, "passer: actor of each labelled action"):
            raise SystemExit("VEO_UNSEAL needed for M7 / M9")
    units = sorted(p.name for p in GPU.iterdir() if (CLIPS / f"{p.name}_events.csv").exists()
                   and (GPU / p.name / f"output_{p.name}" / "metric_v3s_link.parquet").exists())
    df = pd.DataFrame([r for w in units for r in unit(w)])
    df = df.drop_duplicates(["match", "half", "abs", "label"])
    df["unseen"] = df.match.astype(str).isin(UNSEEN)
    out = {"units": len(units), "plan sha256": (HERE / "plan.json.sha256").read_text().split()[0]}
    for name, part in (("unseen", df[df.unseen]), ("seen by v3", df[~df.unseen])):
        out[name] = {"passes (primary)" if name == "unseen" else "passes": summary(part[part.label.isin(PASSES)]),
                     "all actions": summary(part),
                     "per label": {l: summary(g) for l, g in part.groupby("label")}}
    out["unseen passes per match"] = {str(m): summary(g) for m, g in
                                      df[df.unseen & df.label.isin(PASSES)].groupby("match")}
    print(json.dumps(out, indent=1, default=str))
    (HERE / "passer.json").write_text(json.dumps(out, indent=1, default=str))
    df.to_parquet(HERE / "passer_rows.parquet")


if __name__ == "__main__":
    main()
