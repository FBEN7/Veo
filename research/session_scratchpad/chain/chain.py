"""chain plan v1 (plan.json, hashed first): where the chain ball -> player on
the ball -> team breaks, on the labelled actions of the 223 shot clips.

    VEO_UNSEAL=... python chain.py
"""
import glob
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
TREE = ROOT / ".cache/gpu_shots3"
SEALED = {"128057", "132831"}
LO, HI, BALL_DF, BALL_M, MATCH_M = 350, 1075, 4, 3.0, 2.0
TEAMS = ("team_A", "team_B")


def match(a, b, lim):
    """Hungarian on pitch distance; dict index in a -> index in b."""
    if not len(a) or not len(b):
        return {}
    d = np.linalg.norm(a[:, None] - b[None], axis=2)
    r, c = linear_sum_assignment(np.where(d <= lim, d, 1e6))
    return {int(i): int(j) for i, j in zip(r, c) if d[i, j] <= lim}


def unit(w):
    m = pd.read_parquet(TREE / w / f"output_{w}" / "metric_v3s_link.parquet")
    pl = m[(m.cls == "player") & m.x.notna()]
    ball = m[(m.cls == "ball") & m.x.notna()]
    gt = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
    gt = gt[gt.x.notna()]
    ev = pd.read_csv(CLIPS / f"{w}_events.csv")
    ev = ev[(ev.frame >= LO) & (ev.frame <= HI) & ev.player_id.notna() & ev.x.notna()]
    pl_f = {f: g for f, g in pl.groupby("frame")}
    gt_f = {f: g for f, g in gt.groupby("frame")}
    # team map over the clip: our label -> majority truth side among matched detections
    votes = {}
    for f, g in pl_f.items():
        t = gt_f.get(f)
        if t is None:
            continue
        mm = match(g[["x", "y"]].to_numpy(float), t[["x", "y"]].to_numpy(float), MATCH_M)
        for i, j in mm.items():
            lab = g.team.iloc[i]
            if lab in TEAMS:
                votes.setdefault(lab, []).append(t.side.iloc[j])
    side_of = {lab: pd.Series(v).mode().iloc[0] for lab, v in votes.items()}
    bf = ball.frame.to_numpy()
    out = []
    for r in ev.itertuples():
        f = int(r.frame) - int(r.frame) % 2
        g, t = pl_f.get(f), gt_f.get(f)
        if g is None or t is None:
            continue
        a = t[t.player_id.astype(str) == str(int(r.player_id))]
        if a.empty:
            continue
        P = g[["x", "y"]].to_numpy(float)
        T = t[["x", "y"]].to_numpy(float)
        mm = match(P, T, MATCH_M)
        j_actor = int(np.flatnonzero(t.player_id.astype(str).to_numpy() == str(int(r.player_id)))[0])
        i_actor = next((i for i, j in mm.items() if j == j_actor), None)
        pos = np.array([r.x, r.y], float)
        # S1: our ball near the action
        k = np.flatnonzero(np.abs(bf - f) <= BALL_DF)
        bxy = None
        if len(k):
            kk = k[np.argmin(np.abs(bf[k] - f))]
            cand = ball[ball.frame == bf[kk]][["x", "y"]].to_numpy(float)
            cand = cand[np.argmin(np.linalg.norm(cand - pos, axis=1))]
            if np.linalg.norm(cand - pos) <= BALL_M:
                bxy = cand
        side = lambda i: side_of.get(g.team.iloc[i]) if i is not None and g.team.iloc[i] in TEAMS else None
        truth_side = lambda i: t.side.iloc[mm[i]] if i is not None and i in mm else None
        nearest = lambda xy: int(np.argmin(np.linalg.norm(P - xy, axis=1))) if xy is not None and len(P) else None
        i_ball = nearest(bxy)
        i_pos = nearest(pos)
        row = {"unit": w, "match": w.split("_")[1], "frame": int(r.frame), "kind": r.kind, "side": r.side,
               "S1 ball found": bxy is not None,
               "actor detected": i_actor is not None,
               "S2 nearest is actor": bxy is not None and i_ball is not None and i_ball == i_actor,
               "S3 team given": bxy is not None and i_ball == i_actor and side(i_ball) is not None,
               "S4 team right": bxy is not None and i_ball == i_actor and side(i_ball) == r.side,
               "end to end": side(i_ball) == r.side,
               "swap ball": side(i_pos) == r.side,
               "swap player": bxy is not None and side(i_actor) == r.side,
               "swap team": bxy is not None and truth_side(i_ball) == r.side}
        out.append(row)
    return out


def boot(df, col, base=None, n=5000):
    ms = sorted(df.match.unique())
    by = {m: g for m, g in df.groupby("match")}
    rng = np.random.default_rng(0)
    val = lambda s: (sum(by[m][col].sum() for m in s) - (sum(by[m][base].sum() for m in s) if base else 0)) / sum(len(by[m]) for m in s)
    draws = [val(rng.choice(ms, len(ms))) for _ in range(n)]
    return {"value": round(float(val(ms)), 4), "95% over matches": [round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)]}


def main():
    from src.unseal import unsealed
    for m in sorted(SEALED):
        if not unsealed(m, "chain plan v1: development use (seal lifted by the owner on 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    units = sorted(os.path.basename(p) for p in glob.glob(str(TREE / "st2_*")) if "_117092_" not in p)
    with Pool(os.cpu_count()) as pool:
        rows = [r for rs in pool.imap_unordered(unit, units) for r in rs]
    df = pd.DataFrame(rows)
    df.to_parquet(HERE / "chain.parquet")
    cols = ["S1 ball found", "actor detected", "S2 nearest is actor", "S3 team given", "S4 team right", "end to end",
            "swap ball", "swap player", "swap team"]
    share = lambda g: {c: round(float(g[c].mean()), 4) for c in cols}
    kinds = df.kind.where(df.kind.isin(["pass", "carry"]), "other")
    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0], "units": len(units), "actions": len(df),
           "shares (of all actions)": share(df),
           "conditional": {"S2 | S1": round(float(df["S2 nearest is actor"].sum() / df["S1 ball found"].sum()), 4),
                           "S3 | S2": round(float(df["S3 team given"].sum() / df["S2 nearest is actor"].sum()), 4),
                           "S4 | S3": round(float(df["S4 team right"].sum() / df["S3 team given"].sum()), 4)},
           "end to end": boot(df, "end to end"),
           "swap gains over end to end": {c: boot(df, c, "end to end") for c in ("swap ball", "swap player", "swap team")},
           "per kind": {k: dict(share(g), actions=len(g)) for k, g in df.groupby(kinds)},
           "per match": {m: dict(share(g), actions=len(g)) for m, g in df.groupby("match")}}
    (HERE / "chain.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per match",)}, indent=1))


if __name__ == "__main__":
    main()
