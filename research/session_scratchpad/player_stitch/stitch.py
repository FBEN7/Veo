"""player stitch (plan.json, hashed first): our track fragments linked into
players on the 47 ordinary-play windows; continuity, swaps and distance per
player against unstitched tracks and an oracle identity.

    VEO_UNSEAL=... python stitch.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "research/session_scratchpad/shots_learn/tdeed_probe"))
import centres as CE  # noqa: E402

S = CE.S
FPS, MIN_DET, MATCH_M = 25, 5, 2.0
G_MIN, G_MAX, R0, VMAX, NOLINK = -0.4, 10.0, 2.0, 7.0, 1.0


def check():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit("plan.json does not match its sha256")
    return got


def match_table(pl, gt):
    """each of our detections -> the truth player within 2 m (Hungarian per frame), or None."""
    g = dict(tuple(gt.groupby("frame")))
    out = []
    for f, a in pl.groupby("frame"):
        b = g.get(f)
        pid = np.full(len(a), None, object)
        if b is not None and len(b):
            d = np.hypot(a.x.to_numpy()[:, None] - b.xc.to_numpy()[None], a.y.to_numpy()[:, None] - b.yc.to_numpy()[None])
            r, c = linear_sum_assignment(np.where(d <= MATCH_M, d, 1e6))
            for i, j in zip(r, c):
                if d[i, j] <= MATCH_M:
                    pid[i] = b.player_id.iloc[j]
        out.append(a.assign(pid=pid))
    return pd.concat(out)


def frag_ends(pl):
    rows = []
    for t, g in pl.groupby("track_id"):
        if len(g) < MIN_DET:
            continue
        g = g.sort_values("frame")
        team = g.team.mode().iloc[0]
        fr = g.frame.to_numpy()
        sm = g[["x", "y"]].rolling(13, center=True, min_periods=1).mean().to_numpy()   # ~1 s at stride 2
        t0, t1 = fr[0] / FPS, fr[-1] / FPS
        k = fr >= fr[-1] - FPS
        dt = (fr[-1] - fr[k][0]) / FPS
        v = (sm[-1] - sm[k][0]) / dt if dt > 0 else np.zeros(2)
        rows.append({"track": t, "team": team, "t0": t0, "t1": t1, "p0": sm[0], "p1": sm[-1], "v1": v})
    return rows


def stitch(fr_):
    n = len(fr_)
    C = np.full((n, n), 1e6)
    for i, a in enumerate(fr_):
        for j, b in enumerate(fr_):
            if i == j or a["team"] != b["team"]:
                continue
            g = b["t0"] - a["t1"]
            if not (G_MIN <= g <= G_MAX):
                continue
            pred = a["p1"] + a["v1"] * min(max(g, 0.0), 1.0)
            d = float(np.hypot(*(b["p0"] - pred)))
            if d <= R0 + VMAX * max(g, 0.0):
                C[i, j] = d / (R0 + VMAX * max(g, 0.0)) + 0.05 * g
    M = np.full((2 * n, 2 * n), 1e6)
    M[:n, :n] = C
    M[:n, n:] = np.where(np.eye(n, dtype=bool), NOLINK, 1e6)
    M[n:, :n] = np.where(np.eye(n, dtype=bool), NOLINK, 1e6)
    M[n:, n:] = 0.0
    r, c = linear_sum_assignment(M)
    nxt = {int(i): int(j) for i, j in zip(r, c) if i < n and j < n and M[i, j] < 1e5}
    has_prev = set(nxt.values())
    traj = {}
    for i in range(n):
        if i in has_prev:
            continue
        seen, k, chain = set(), i, []
        while k is not None and k not in seen:
            seen.add(k)
            chain.append(fr_[k]["track"])
            k = nxt.get(k)
        for t in chain:
            traj[t] = fr_[i]["track"]
    for f in fr_:                       # anything left (a cycle guard) keeps its own id
        traj.setdefault(f["track"], f["track"])
    return traj, len(nxt)


def path(fr, x, y, span=None):
    """gaps interpolated, centred 1-s mean, sampled every 1 s; span = (a, b) frames."""
    o = np.argsort(fr)
    fr, x, y = fr[o], x[o], y[o]
    a, b = (fr[0], fr[-1]) if span is None else span
    grid = np.arange(a, b + 1)
    xi, yi = np.interp(grid, fr, x), np.interp(grid, fr, y)
    s = pd.DataFrame({"x": xi, "y": yi}).rolling(FPS, center=True, min_periods=1).mean().iloc[::FPS]
    return float(np.hypot(np.diff(s.x), np.diff(s.y)).sum())


def measure(mt, traj_of, gt, n_frames):
    mt = mt.assign(traj=mt.track_id.map(traj_of))
    mt = mt[mt.traj.notna()]
    frames = np.sort(mt.frame.unique())
    out = []
    for pid, gp in gt.groupby("player_id"):
        cnt = mt[mt.pid == pid].groupby("traj").size()
        if not len(cnt):
            out.append({"pid": pid, "continuity": 0.0, "pieces": 0})
            continue
        dom = cnt.idxmax()
        rows = mt[mt.traj == dom].drop_duplicates("frame")
        matched = rows.pid.notna().sum()
        swaps = float(((rows.pid != pid) & rows.pid.notna()).sum() / max(matched, 1))
        a, b = int(rows.frame.min()), int(rows.frame.max())
        gp = gp.sort_values("frame")
        gpk = gp[(gp.frame >= a) & (gp.frame <= b)]
        ours = path(rows.frame.to_numpy(), rows.x.to_numpy(), rows.y.to_numpy())
        tru = path(gpk.frame.to_numpy(), gpk.xc.to_numpy(), gpk.yc.to_numpy()) if len(gpk) > 1 else np.nan
        out.append({"pid": pid, "continuity": float(cnt.max() / len(frames)), "pieces": int((cnt >= 2).sum()),
                    "swaps": swaps, "span_s": (b - a) / FPS, "dist_ours": ours, "dist_truth": tru})
    return pd.DataFrame(out)


def oracle(mt, gt, n_frames):
    out = []
    for pid, gp in gt.groupby("player_id"):
        r = mt[mt.pid == pid].drop_duplicates("frame")
        gp = gp.sort_values("frame")
        tru = path(gp.frame.to_numpy(), gp.xc.to_numpy(), gp.yc.to_numpy(), (0, n_frames - 1))
        if len(r) < 2:
            out.append({"pid": pid, "dist_oracle": np.nan, "dist_truth_full": tru})
            continue
        ours = path(r.frame.to_numpy(), r.x.to_numpy(), r.y.to_numpy(), (0, n_frames - 1))
        out.append({"pid": pid, "dist_oracle": ours, "dist_truth_full": tru})
    return pd.DataFrame(out)


def main():
    sha = check()
    from src.unseal import unsealed
    for m in S.V3.NEW_PLAYERS:
        if not unsealed(m, "player stitch: M7 / M9 windows (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    units = [u for u in CE.unit_list() if u[2] is None]
    allU, allS, allO, links = [], [], [], {}
    for name, match, _, tree in units:
        info = S.LS.sd.info_of(name)
        n = int(info["frames"])
        m = pd.read_parquet(tree / name / f"output_{name}" / "metric_v3s_link.parquet")
        pl = m[(m.cls == "player") & m.x.notna()][["frame", "track_id", "x", "y", "team"]].copy()
        pl["team"] = pl.team.fillna("other")
        gt = pd.read_parquet(info["dir"] / f"{name}_gt.parquet")
        gt = gt.assign(xc=gt.x + 52.5, yc=gt.y + 34.0)
        mt = match_table(pl, gt)
        fr_ = frag_ends(pl)
        keep = {f["track"] for f in fr_}
        mt = mt[mt.track_id.isin(keep)]
        tU = {t: t for t in keep}
        tS, nl = stitch(fr_)
        links[name] = {"fragments": len(fr_), "links": nl, "trajectories": len(set(tS.values()))}
        for res, tm in ((allU, tU), (allS, tS)):
            res.append(measure(mt, tm, gt, n).assign(window=name, match=match))
        allO.append(oracle(mt, gt, n).assign(window=name, match=match))
        print(name, links[name], flush=True)
    U, Sx, O = (pd.concat(x, ignore_index=True) for x in (allU, allS, allO))

    def summary(d):
        e = (d.dist_ours - d.dist_truth).abs() / d.dist_truth
        return {"continuity median": round(float(d.continuity.median()), 3),
                "continuity mean": round(float(d.continuity.mean()), 3),
                "pieces per player median": float(d.pieces.median()),
                "swaps (mixing) median / mean": [round(float(d.swaps.median()), 3), round(float(d.swaps.mean()), 3)],
                "dominant span s median": round(float(d.span_s.median()), 1),
                "distance on dominant span: median abs rel error": round(float(e.median()), 3),
                "within 5%": round(float((e <= 0.05).mean()), 3)}
    eo = (O.dist_oracle - O.dist_truth_full).abs() / O.dist_truth_full
    s_sum = summary(Sx)
    reading = ("stitching works on 3 min" if s_sum["continuity median"] >= 0.80 and s_sum["within 5%"] >= 0.70
               else "detections limit" if float((eo <= 0.05).mean()) < 0.5 else "otherwise")
    out = {"plan sha256": sha, "player-windows": len(U), "U (unstitched)": summary(U), "S (stitched)": s_sum,
           "O (oracle identity, whole window)": {"median abs rel error": round(float(eo.median()), 3),
                                                  "within 5%": round(float((eo <= 0.05).mean()), 3),
                                                  "ours / truth median": round(float((O.dist_oracle / O.dist_truth_full).median()), 3)},
           "reading": reading, "links per window": links}
    (HERE / "stitch.json").write_text(json.dumps(out, indent=1, default=str))
    pd.concat([U.assign(arm="U"), Sx.assign(arm="S")]).round(4).to_csv(HERE / "per_player.csv", index=False)
    print(json.dumps({k: v for k, v in out.items() if k != "links per window"}, indent=1, default=str))


if __name__ == "__main__":
    main()
