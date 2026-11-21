"""identity step 2b (plan.json, hashed first): partial numbers + the lineup +
look cues (relative height, colours) on 20-s segments of truth-matched players.

    VEO_UNSEAL=... python fusion.py
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/jersey_read"))
import sheets as J  # noqa: E402  (SOURCES, tree_of, MATCH_M, ALONE_M, T)

STEP, SLOT, MIN_N, DRAWS, P_ON = 50, 500, 4, 200, 0.5
SLICES = {"hair": (0.0, 0.12, 0.25, 0.75), "shorts": (0.45, 0.60, 0.0, 1.0),
          "socks": (0.72, 0.88, 0.0, 1.0), "shoes": (0.88, 1.0, 0.0, 1.0)}
COLS = [f"{s}_{c}" for s in SLICES for c in "Lab"]
BANDS = [(0, 40, "<40"), (40, 60, "40-60"), (60, 80, "60-80"), (80, 1e9, ">=80")]


def check():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit("plan.json does not match its sha256")
    return got


def matched(gt, ours, frames):
    """jersey_read addendum 1, keeping the side."""
    out = []
    gt = gt.assign(xc=gt.x + 52.5, yc=gt.y + 34.0)
    G, O = dict(tuple(gt.groupby("frame"))), dict(tuple(ours.groupby("frame")))
    for f in frames:
        if f not in G or f not in O:
            continue
        g, o = G[f], O[f]
        d = np.hypot(g.xc.to_numpy()[:, None] - o.x.to_numpy()[None], g.yc.to_numpy()[:, None] - o.y.to_numpy()[None])
        r, c = linear_sum_assignment(d)
        for i, j in zip(r, c):
            if d[i, j] > J.MATCH_M or g.role.iloc[i] != "player":
                continue
            if np.sort(d[:, j])[1:2].min(initial=1e9) <= J.ALONE_M:
                continue
            q = o.iloc[j]
            out.append({"frame": f, "player_id": g.player_id.iloc[i], "jersey": int(g.jersey.iloc[i]), "side": g.side.iloc[i],
                        "px": q.px_raw, "py": q.py_raw, "bw": q.crop_w, "bh": q.crop_h})
    return pd.DataFrame(out)


def colours(img, r):
    x0, x1 = int(max(r.px - r.bw / 2, 0)), int(min(r.px + r.bw / 2, img.shape[1]))
    y0, y1 = int(max(r.py - r.bh, 0)), int(min(r.py, img.shape[0]))
    box = img[y0:y1, x0:x1]
    out = {}
    if box.size == 0:
        return {c: np.nan for c in COLS}
    hsv, lab = cv2.cvtColor(box, cv2.COLOR_BGR2HSV), cv2.cvtColor(box, cv2.COLOR_BGR2LAB).astype(float)
    grass = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 90) & (hsv[..., 1] > 50) & (hsv[..., 2] > 40)
    h, w = box.shape[:2]
    for s, (a, b, l, rr) in SLICES.items():
        ys, xs = slice(int(a * h), max(int(b * h), int(a * h) + 1)), slice(int(l * w), max(int(rr * w), int(l * w) + 1))
        m = ~grass[ys, xs]
        px = lab[ys, xs][m]
        for k, c in enumerate("Lab"):
            out[f"{s}_{c}"] = float(px[:, k].mean()) if len(px) >= 3 else np.nan
    return out


def per_video(k):
    match, vid, gtp = J.SOURCES[k]
    name = vid.stem
    m = pd.read_parquet(J.tree_of(name) / name / f"output_{name}" / "metric_v3s_link.parquet")
    ours = m[(m.cls == "player") & m.team.isin(["team_A", "team_B"]) & m.x.notna()]
    r0 = int(ours.frame.min()) % 2
    gt = pd.read_parquet(gtp)
    n = int(gt.frame.max()) + 1
    frames = [f for f in sorted(ours.frame.unique()) if f % STEP == r0]
    mt = matched(gt, ours, frames)
    # relative height: log box height against a quadratic in the foot position
    x, y = mt.px / 1000, mt.py / 1000
    A = np.c_[np.ones(len(mt)), x, y, x * x, y * y, x * y]
    coef = np.linalg.lstsq(A, np.log(mt.bh), rcond=None)[0]
    mt["rel_h"] = np.exp(np.log(mt.bh) - A @ coef)
    # colours, frame by frame
    want = set(mt.frame)
    cap, f, rows = cv2.VideoCapture(str(vid)), 0, {}
    by = dict(tuple(mt.groupby("frame")))
    while f <= max(want):
        if not cap.grab():
            break
        if f in want:
            img = cap.retrieve()[1]
            for i, r in by[f].iterrows():
                rows[i] = colours(img, r)
        f += 1
    cap.release()
    mt = mt.join(pd.DataFrame.from_dict(rows, orient="index"))
    mt["slot"] = (mt.frame - r0) // SLOT
    full = (mt.slot + 1) * SLOT + r0 <= n
    mt = mt[full]
    seg = mt.groupby(["player_id", "slot"]).agg(jersey=("jersey", "first"), side=("side", "first"), n=("frame", "size"),
                                                rel_h=("rel_h", "median"), bhs=("bh", list), **{c: (c, "median") for c in COLS})
    seg = seg[seg.n >= MIN_N].reset_index()
    # truth distance over each segment's slot (1-s smoothing, 1-s sampling)
    G = dict(tuple(gt[gt.role == "player"].groupby("player_id")))
    dist = []
    for r in seg.itertuples():
        g = G[r.player_id]
        g = g[(g.frame >= r.slot * SLOT + r0) & (g.frame < (r.slot + 1) * SLOT + r0)].sort_values("frame")
        xy = g[["x", "y"]].rolling(25, center=True, min_periods=1).mean().to_numpy()[::25]
        dist.append(float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0)
    seg["dist"] = dist
    seg["match"], seg["time"] = match, [f"{k}:{s}" for s in seg.slot]
    return seg


def zscore(seg):
    cues = ["rel_h"] + COLS
    z = seg.groupby(["match", "side"])[cues].transform(lambda c: (c - c.mean()) / (c.std() if c.std() > 0 else 1))
    return z.fillna(0.0)


def rank1(seg, Z, cols):
    hit, n = 0, 0
    for (_m, _s), g in seg.groupby(["match", "side"]):
        for i in g.index:
            others = g.drop(i)
            protos = Z.loc[others.index, cols].groupby(others.player_id).mean()
            if seg.at[i, "player_id"] not in protos.index or len(protos) < 2:
                continue
            d = np.linalg.norm(protos.to_numpy() - Z.loc[i, cols].to_numpy(), axis=1)
            hit += int(protos.index[int(np.argmin(d))] == seg.at[i, "player_id"])
            n += 1
    return round(hit / n, 3), n


def band(h):
    return next(b[2] for b in BANDS if b[0] <= h < b[1])


def draw_numbers(seg, rates, lineup, rng):
    """the simulated number of each segment (a lineup player id or None)."""
    out = []
    for r in seg.itertuples():
        team = lineup[(r.match, r.side)]
        reads = []
        for h in r.bhs:
            c, w = rates[band(h)]
            u = rng.random()
            if u < c:
                reads.append(r.player_id)
            elif u < c + w and rng.random() < P_ON:
                others = [p for p in team if p != r.player_id]
                reads.append(others[int(rng.integers(len(others)))])
        cnt = Counter(reads).most_common(2)
        out.append(cnt[0][0] if cnt and (len(cnt) == 1 or cnt[0][1] > cnt[1][1]) else None)
    return out


def assign(seg, Z, num, lineup, use_cues):
    cues = ["rel_h"] + COLS
    got = pd.Series(index=seg.index, dtype=object)
    seg = seg.assign(num=num)
    for (m, s), g in seg.groupby(["match", "side"]):
        team = lineup[(m, s)]
        numbered = g[g.num.notna()]
        protos = Z.loc[numbered.index, cues].groupby(numbered.num).mean() if len(numbered) else pd.DataFrame(columns=cues)
        for _t, q in g.groupby("time"):
            cost = np.zeros((len(q), len(team)))
            dmat = np.full((len(q), len(team)), np.nan)
            for a, i in enumerate(q.index):
                for b, p in enumerate(team):
                    if use_cues and p in protos.index:
                        dmat[a, b] = np.linalg.norm(protos.loc[p].to_numpy() - Z.loc[i, cues].to_numpy())
            med = np.nanmedian(dmat) if np.isfinite(dmat).any() else 1.0
            for a, i in enumerate(q.index):
                for b, p in enumerate(team):
                    if q.at[i, "num"] is not None and pd.notna(q.at[i, "num"]):
                        cost[a, b] = 0.0 if q.at[i, "num"] == p else 1e6
                    else:
                        cost[a, b] = dmat[a, b] if np.isfinite(dmat[a, b]) else med
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c):
                got[q.index[a]] = team[b]
    return got


def distance_ok(seg, got):
    ok = []
    for (_m, _s), g in seg.groupby(["match", "side"]):
        for p in g.player_id.unique():
            t = g[g.player_id == p].dist.sum()
            o = g[got.loc[g.index] == p].dist.sum()
            if t > 0:
                ok.append(abs(o - t) / t <= 0.05)
    return float(np.mean(ok))


def main():
    sha = check()
    from src.unseal import unsealed
    for m in J.T.S.V3.NEW_PLAYERS:
        if not unsealed(m, "identity step 2b: 132831 cues (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    seg = pd.concat([per_video(k) for k in range(len(J.SOURCES))], ignore_index=True)
    print("segments", seg.groupby("match").size().to_dict(), flush=True)
    Z = zscore(seg)
    lineup = {k: sorted(g.player_id.unique().tolist()) for k, g in seg.groupby(["match", "side"])}
    a = {name: {m: rank1(seg[seg.match == m], Z, cols) for m in sorted(seg.match.unique())}
         for name, cols in (("all cues", ["rel_h"] + COLS), ("height only", ["rel_h"]), ("colours only", COLS))}
    a["team-mates per team (median)"] = float(np.median([len(v) for v in lineup.values()]))
    sc = json.loads((ROOT / "research/session_scratchpad/jersey_read/score.json").read_text())["part 2 per band (our box height)"]
    rates = {b: (v["correct"], v["read"] - v["correct"]) for b, v in sc.items()}
    rng = np.random.default_rng(0)
    res = {m: {"fusion": [], "no cues": [], "numbered": [], "dist fusion": [], "dist no cues": []} for m in sorted(seg.match.unique())}
    for _ in range(DRAWS):
        num = draw_numbers(seg, rates, lineup, rng)
        gf, g0 = assign(seg, Z, num, lineup, True), assign(seg, Z, num, lineup, False)
        for m in res:
            idx = seg.index[seg.match == m]
            res[m]["fusion"].append(float((gf.loc[idx] == seg.player_id.loc[idx]).mean()))
            res[m]["no cues"].append(float((g0.loc[idx] == seg.player_id.loc[idx]).mean()))
            res[m]["numbered"].append(float(pd.Series(num, index=seg.index).loc[idx].notna().mean()))
            res[m]["dist fusion"].append(distance_ok(seg.loc[idx], gf))
            res[m]["dist no cues"].append(distance_ok(seg.loc[idx], g0))
    b = {m: {k: round(float(np.mean(v)), 3) for k, v in d.items()} for m, d in res.items()}
    hit = all(b[m]["fusion"] >= 0.80 for m in ("117093", "132831"))
    out = {"plan sha256": sha, "segments per match": seg.groupby("match").size().to_dict(),
           "box height median per match (px)": seg.assign(h=seg.bhs.map(np.median)).groupby("match").h.median().round(1).to_dict(),
           "(a) cues alone, rank-1 among team-mates [rank-1, segments]": a,
           "(b) fusion, mean of 200 draws: segment identity, numbered share, players' distance within 5%": b,
           "worth building (fusion >= 0.80 on 117093 and 132831)": bool(hit)}
    (HERE / "fusion.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
