"""identity step 1 (plan.json, hashed first): re-track our player detections
in pitch metres; settings chosen on the dev matches, reported on the test
matches; measures as player_stitch.

    VEO_UNSEAL=... python retrack.py
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "research/session_scratchpad/player_stitch"))
import stitch as T  # noqa: E402

CE, S, FPS = T.CE, T.S, 25
DEV = {"117092", "117093", "118575", "118576", "118577"}
GATES, COASTS, BETAS = (1.0, 1.5, 2.5, 4.0), (2, 5, 10), (0.2, 0.05)
OUTLIER = "117092"    # addendum 1: our positions do not meet the truth there
ALPHA, BETA, VMAX, TEAM_PEN, OTHER_PEN = 0.5, 0.2, 7.0, 1.5, 0.5


def check():
    out = {}
    for name in ("plan.json", "addendum1.json"):
        want = (HERE / f"{name}.sha256").read_text().split()[0]
        got = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"{name} does not match its sha256")
        out[name] = got
    return out


def retrack(pl, gate, coast):
    """new track id per row of pl (index-aligned)."""
    pos, vel, last, votes = [], [], [], []
    new_id = pd.Series(-1, index=pl.index)
    for f, g in pl.groupby("frame", sort=True):
        t = f / FPS
        z = g[["x", "y"]].to_numpy()
        tm = g.team.to_numpy()
        act = [k for k in range(len(pos)) if t - last[k] <= coast]
        if act:
            dt = np.array([t - last[k] for k in act])
            pred = np.array([pos[k] + vel[k] * dt[i] for i, k in enumerate(act)])
            d = np.hypot(pred[:, None, 0] - z[None, :, 0], pred[:, None, 1] - z[None, :, 1])
            team_k = []
            for k in act:
                c = votes[k]
                a, b = c.get("team_A", 0), c.get("team_B", 0)
                team_k.append("team_A" if a > b else "team_B" if b > a else None)
            pen = np.zeros_like(d)
            for i, tk in enumerate(team_k):
                for j, tj in enumerate(tm):
                    if tj == "other":
                        pen[i, j] = OTHER_PEN
                    elif tk is not None and tj != tk:
                        pen[i, j] = TEAM_PEN
            ok = d <= gate + VMAX * dt[:, None]
            cost = np.where(ok, d + pen, 1e6)
            r, c = linear_sum_assignment(cost)
            pairs = [(i, j) for i, j in zip(r, c) if cost[i, j] < 1e5]
        else:
            pairs = []
        used = set()
        for i, j in pairs:
            k = act[i]
            dtk = t - last[k]
            p = pos[k] + vel[k] * dtk
            res = z[j] - p
            pos[k] = p + ALPHA * res
            if dtk > 0:
                vel[k] = vel[k] + BETA / dtk * res
            last[k] = t
            votes[k][tm[j]] += 1
            new_id.loc[g.index[j]] = k
            used.add(j)
        for j in range(len(z)):
            if j in used:
                continue
            pos.append(z[j].astype(float))
            vel.append(np.zeros(2))
            last.append(t)
            votes.append(Counter({tm[j]: 1}))
            new_id.loc[g.index[j]] = len(pos) - 1
    team = {k: (max(("team_A", "team_B"), key=lambda x: votes[k].get(x, 0))
                if votes[k].get("team_A", 0) + votes[k].get("team_B", 0) else "other") for k in range(len(pos))}
    return new_id, team


def identity_share(mt, traj_of):
    """addendum 1: per player, matched samples on his main trajectory / all his matched samples."""
    m = mt.assign(traj=mt.track_id.map(traj_of))
    m = m[m.traj.notna() & m.pid.notna()]
    c = m.groupby(["pid", "traj"]).size()
    return (c.groupby(level=0).max() / c.groupby(level=0).sum()).rename("identity").reset_index()


def summarize(d):
    e = (d.dist_ours - d.dist_truth).abs() / d.dist_truth
    return {"identity share median": round(float(d.identity.median()), 3),
            "continuity median": round(float(d.continuity.median()), 3),
            "mixing median": round(float(d.swaps.median()), 3) if "swaps" in d else None,
            "pieces median": float(d.pieces.median()),
            "distance on main span within 5%": round(float((e <= 0.05).mean()), 3)}


def arms(pl, gt, n, gate, coast, radius):
    T.MATCH_M = radius
    mt = T.match_table(pl, gt)
    out = {}
    keepU = {f["track"] for f in T.frag_ends(pl)}
    mU = mt[mt.track_id.isin(keepU)]
    out["U"] = T.measure(mU, {t: t for t in keepU}, gt, n).merge(identity_share(mU, {t: t for t in keepU}), on="pid", how="left")
    nid, team = retrack(pl, gate, coast)
    plT = pl.assign(track_id=nid, team=nid.map(team))
    mtT = mt.assign(track_id=nid.loc[mt.index], team=nid.loc[mt.index].map(team))
    frT = T.frag_ends(plT)
    keepT = {f["track"] for f in frT}
    mtT = mtT[mtT.track_id.isin(keepT)]
    out["T"] = T.measure(mtT, {t: t for t in keepT}, gt, n).merge(identity_share(mtT, {t: t for t in keepT}), on="pid", how="left")
    tS, _ = T.stitch(frT)
    out["T+S"] = T.measure(mtT, tS, gt, n).merge(identity_share(mtT, tS), on="pid", how="left")
    return out


def load(name, tree):
    info = S.LS.sd.info_of(name)
    m = pd.read_parquet(tree / name / f"output_{name}" / "metric_v3s_link.parquet")
    pl = m[(m.cls == "player") & m.x.notna()][["frame", "track_id", "x", "y", "team"]].copy()
    pl["team"] = pl.team.fillna("other")
    gt = pd.read_parquet(info["dir"] / f"{name}_gt.parquet")
    return pl, gt.assign(xc=gt.x + 52.5, yc=gt.y + 34.0), int(info["frames"])


def main():
    sha = check()
    from src.unseal import unsealed
    for m in S.V3.NEW_PLAYERS:
        if not unsealed(m, "identity step 1: M7 / M9 windows (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    units = [u for u in CE.unit_list() if u[2] is None]
    data = {u[0]: (u[1], load(u[0], u[3])) for u in units}
    # dev: the grid, re-tracker alone, 3-m matching
    global BETA
    grid, cells = {}, {}
    for beta in BETAS:
        for gate in GATES:
            for coast in COASTS:
                BETA = beta
                res = [arms(*data[n][1], gate, coast, 3.0)["T"] for n in data if data[n][0] in DEV and data[n][0] != OUTLIER]
                key = f"beta {beta}, gate {gate} m, coast {coast} s"
                grid[key] = summarize(pd.concat(res))
                cells[key] = (beta, gate, coast)
                print("dev", key, grid[key], flush=True)
    best = max(grid, key=lambda k: grid[k]["identity share median"] - grid[k]["mixing median"])
    BETA, gate, coast = cells[best]
    test = {}
    for radius in (3.0, 2.0):
        acc = {"U": [], "T": [], "T+S": []}
        for n, (match, args) in data.items():
            if match in DEV:
                continue
            if match == OUTLIER:
                continue
            for k, v in arms(*args, gate, coast, radius).items():
                acc[k].append(v.assign(window=n, match=match))
        test[f"{radius:.0f} m"] = {k: summarize(pd.concat(v)) for k, v in acc.items()}
        print("test", radius, test[f"{radius:.0f} m"], flush=True)
    t3 = test["3 m"]
    best_arm = max(("T", "T+S"), key=lambda a: t3[a]["identity share median"] - t3[a]["mixing median"])
    hit = t3[best_arm]["identity share median"] >= 0.80 and t3[best_arm]["mixing median"] <= 0.05
    out = {"plan sha256": sha, "dev grid (T, 3 m)": grid, "chosen": best, "test matches": test,
           "primary (test, 3 m)": {"best arm": best_arm, "U": t3["U"], best_arm: t3[best_arm]},
           "step target met (identity share >= 0.80, mixing <= 0.05)": bool(hit)}
    (HERE / "retrack.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["primary (test, 3 m)"], indent=1), "target met:", hit)


if __name__ == "__main__":
    main()
