"""possession v2 (plan.json, hashed first): share of time on the ball on 47
ordinary-play windows; the ball rule R, a learned estimator L, a constant C.

    VEO_UNSEAL=... python possession.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "research/session_scratchpad/shots_learn/tdeed_probe"))
import centres as CE  # noqa: E402  (its unit_list: the 47 windows and their trees)

S = CE.S
FPS, DEAD_S, HOLD_F, NEAR_M, MATCH_M = 25, 6.0, 50, 4.0, 2.0
TEAMS = ("team_A", "team_B")


def check():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit("plan.json does not match its sha256")
    return got


def truth(ev, n):
    """per frame: 0 dead / not covered, 1 left, 2 right."""
    ev = ev[ev.side.isin(["left", "right"])].sort_values("frame", kind="stable")
    out = np.zeros(n, np.int8)
    f, sd, lab = ev.frame.to_numpy(), ev.side.map({"left": 1, "right": 2}).to_numpy(), ev.label.to_numpy()
    for i in range(len(f) - 1):
        a, b = int(f[i]), int(min(f[i + 1], n))
        if lab[i] == "OUT" or (b - a) / FPS > DEAD_S or b <= a:
            continue
        if sd[i] == sd[i + 1]:
            out[a:b] = sd[i]
        else:
            mid = (a + b) // 2
            out[a:mid], out[mid:b] = sd[i], sd[i + 1]
    return out


def side_map(pl, gt):
    """our team -> 'left' / 'right' by matched detections (evaluation only)."""
    tally = {(t, s): 0 for t in TEAMS for s in ("left", "right")}
    gq = gt.assign(xc=gt.x + 52.5, yc=gt.y + 34.0)
    for f in range(0, int(pl.frame.max()) + 1, 25):
        a = pl[pl.frame == f]
        b = gq[gq.frame == f]
        if not len(a) or not len(b):
            continue
        d = np.hypot(a.x.to_numpy()[:, None] - b.xc.to_numpy()[None], a.y.to_numpy()[:, None] - b.yc.to_numpy()[None])
        r, c = linear_sum_assignment(np.where(d <= MATCH_M, d, 1e6))
        for i, j in zip(r, c):
            if d[i, j] <= MATCH_M:
                tally[(a.team.iloc[i], b.side.iloc[j])] += 1
    straight = tally[("team_A", "left")] + tally[("team_B", "right")]
    crossed = tally[("team_A", "right")] + tally[("team_B", "left")]
    if straight + crossed == 0:
        return None
    return {"team_A": "left", "team_B": "right"} if straight >= crossed else {"team_A": "right", "team_B": "left"}


def rule(pl, ball, frames):
    """R per player frame: 1 team_A, 2 team_B, 0 none (held up to 2 s)."""
    bxy = ball.drop_duplicates("frame").set_index("frame")[["x", "y"]]
    g = dict(tuple(pl.groupby("frame")))
    out, dA, dB = np.zeros(len(frames), np.int8), np.full(len(frames), np.nan), np.full(len(frames), np.nan)
    last, last_f = 0, -10 ** 9
    for k, f in enumerate(frames):
        t = 0
        if f in bxy.index and f in g:
            p = g[f]
            d = np.hypot(p.x.to_numpy() - bxy.at[f, "x"], p.y.to_numpy() - bxy.at[f, "y"])
            for tm, col in (("team_A", dA), ("team_B", dB)):
                k2 = (p.team == tm).to_numpy()
                if k2.any():
                    col[k] = d[k2].min()
            if len(d) and d.min() <= NEAR_M:
                t = 1 if p.team.iloc[int(np.argmin(d))] == "team_A" else 2
        if t:
            last, last_f = t, f
        elif f - last_f <= HOLD_F:
            t = last
        out[k] = t
    return out, dA, dB


def features(pl, frames, r, dA, dB, n):
    secs = np.arange(n // FPS)
    rows = []
    g = dict(tuple(pl.groupby("frame")))
    for s in secs:
        k = (frames >= s * FPS) & (frames < (s + 1) * FPS)
        fs = frames[k]
        row = {"sec": s}
        for tm in TEAMS:
            st = []
            for f in fs:
                p = g.get(f)
                if p is None:
                    continue
                q = p[p.team == tm]
                if len(q):
                    st.append((len(q), q.x.mean(), q.y.mean(), q.x.std(ddof=0), q.y.std(ddof=0)))
            st = np.array(st) if st else np.full((1, 5), np.nan)
            for j, nm in enumerate(("n", "cx", "cy", "sx", "sy")):
                row[f"{nm}_{tm[-1]}"] = np.nanmean(st[:, j])
            area = np.nan
            if len(fs) and fs[0] in g:
                q = g[fs[0]]
                q = q[q.team == tm]
                if len(q) >= 3:
                    try:
                        area = ConvexHull(q[["x", "y"]].to_numpy()).volume
                    except Exception:
                        pass
            row[f"area_{tm[-1]}"] = area
        rk = r[k]
        row["rA"], row["rB"], row["r0"] = (rk == 1).mean() if len(rk) else np.nan, (rk == 2).mean() if len(rk) else np.nan, (rk == 0).mean() if len(rk) else np.nan
        row["dA"], row["dB"] = np.nanmean(dA[k]) if np.isfinite(dA[k]).any() else np.nan, np.nanmean(dB[k]) if np.isfinite(dB[k]).any() else np.nan
        rows.append(row)
    f = pd.DataFrame(rows).set_index("sec")
    for nm in ("n", "cx", "cy", "sx", "sy", "area"):
        f[f"{nm}_diff"] = f[f"{nm}_A"] - f[f"{nm}_B"]
    f["cdist"] = np.hypot(f.cx_A - f.cx_B, f.cy_A - f.cy_B)
    return f


def boot(per_w, col_a, col_b, matches, n=5000):
    rng = np.random.default_rng(0)
    d = per_w.assign(diff=per_w[col_a] - per_w[col_b])
    by = {m: d[d.match == m]["diff"].to_numpy() for m in matches}
    out = []
    for _ in range(n):
        ms = rng.choice(matches, len(matches))
        out.append(np.concatenate([by[m] for m in ms]).mean())
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def main():
    sha = check()
    from src.unseal import unsealed
    for m in S.V3.NEW_PLAYERS:
        if not unsealed(m, "possession v2: M7 / M9 windows (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    units = [u for u in CE.unit_list() if u[2] is None]          # windows only
    W, skipped = {}, []
    for name, match, _, tree in units:
        info = S.LS.sd.info_of(name)
        n = int(info["frames"])
        m = pd.read_parquet(tree / name / f"output_{name}" / "metric_v3s_link.parquet")
        pl = m[(m.cls == "player") & m.team.isin(TEAMS) & m.x.notna()]
        ball = m[(m.cls == "ball") & m.x.notna()]
        ev = pd.read_csv(info["dir"] / f"{name}_events.csv")
        gt = pd.read_parquet(info["dir"] / f"{name}_gt.parquet")
        mp = side_map(pl, gt)
        if mp is None:
            skipped.append(name)
            continue
        tr = truth(ev, n)
        frames = np.sort(pl.frame.unique())
        r, dA, dB = rule(pl, ball, frames)
        f = features(pl, frames, r, dA, dB, n)
        # per-second truth side (majority of live frames), as team_A = 1
        sec_side = [np.bincount(tr[s * FPS:(s + 1) * FPS], minlength=3) for s in f.index]
        y = []
        for c in sec_side:
            live = c[1] + c[2]
            if live < FPS // 2 + 1:
                y.append(np.nan)
            else:
                left = c[1] >= c[2]
                y.append(float((mp["team_A"] == "left") == left))
        f["y"] = y
        W[name] = {"match": match, "map": mp, "truth": tr, "r": r, "feat": f}
        print(name, "ok", flush=True)
    cols = [c for c in next(iter(W.values()))["feat"].columns if c != "y"]
    matches = sorted({w["match"] for w in W.values()})
    from sklearn.ensemble import HistGradientBoostingClassifier
    PA = {}
    for m in matches:
        tr_ = pd.concat([w["feat"] for w in W.values() if w["match"] != m])
        tr_ = tr_[tr_.y.notna()]
        clf = HistGradientBoostingClassifier(random_state=0).fit(tr_[cols], tr_.y.astype(int))
        for nm, w in W.items():
            if w["match"] == m:
                PA[nm] = clf.predict_proba(w["feat"][cols])[:, 1]
    rows = []
    for nm, w in W.items():
        tr = w["truth"]
        live = (tr > 0).sum()
        t_left = (tr == 1).sum() / live if live else np.nan
        a_left = w["map"]["team_A"] == "left"
        rA, rB = (w["r"] == 1).sum(), (w["r"] == 2).sum()
        r_left = (rA if a_left else rB) / (rA + rB) if rA + rB else 0.5
        l_left = PA[nm].mean() if a_left else 1 - PA[nm].mean()
        yk = w["feat"].y.notna().to_numpy()
        acc = float(((PA[nm][yk] > 0.5) == w["feat"].y.to_numpy()[yk].astype(bool)).mean()) if yk.any() else np.nan
        rows.append({"window": nm, "match": w["match"], "live_frames": int(live), "truth_left": t_left,
                     "C": 0.5, "R": r_left, "L": l_left, "L_sec_acc": acc,
                     "t_left_frames": int((tr == 1).sum()), "r_left_frames": int(rA if a_left else rB),
                     "r_frames": int(rA + rB), "l_left_sum": float(PA[nm].sum() if a_left else (1 - PA[nm]).sum()),
                     "secs": len(PA[nm])})
    pw = pd.DataFrame(rows)
    for k in ("C", "R", "L"):
        pw[f"err_{k}"] = (pw[k] - pw.truth_left).abs()
    per_m = []
    for m, g in pw.groupby("match"):
        t = g.t_left_frames.sum() / g.live_frames.sum()
        r_ = g.r_left_frames.sum() / g.r_frames.sum() if g.r_frames.sum() else 0.5
        l_ = g.l_left_sum.sum() / g.secs.sum()
        per_m.append({"match": m, "truth_left": round(t, 3), "minutes": round(g.live_frames.sum() / FPS / 60, 1),
                      "err C (points)": round(100 * abs(0.5 - t), 1), "err R (points)": round(100 * abs(r_ - t), 1),
                      "err L (points)": round(100 * abs(l_ - t), 1)})
    pm = pd.DataFrame(per_m)
    gain = float(pw.err_L.mean() - pw.err_C.mean())
    ci = boot(pw, "err_L", "err_C", matches)
    out = {"plan sha256": sha, "windows": len(pw), "skipped (no team mapping)": skipped,
           "primary: per-window abs error of the left share, L - C": round(gain, 4), "95% bootstrap over matches": ci,
           "adopt L (<= -0.03, interval below 0)": bool(gain <= -0.03 and ci[1] < 0),
           "R - C": round(float(pw.err_R.mean() - pw.err_C.mean()), 4), "R - C, 95%": boot(pw, "err_R", "err_C", matches),
           "mean per-window abs error": {k: round(float(pw[f"err_{k}"].mean()), 4) for k in ("C", "R", "L")},
           "L per-second accuracy on live seconds (mean over windows)": round(float(pw.L_sec_acc.mean()), 3),
           "truth's left share: per window min / median / max": [round(float(x), 3) for x in pw.truth_left.quantile([0, 0.5, 1])],
           "per match (windows pooled)": pm.to_dict("records"),
           "matches within 5 points": {k: int((pm[f"err {k} (points)"] <= 5).sum()) for k in ("C", "R", "L")}}
    (HERE / "possession.json").write_text(json.dumps(out, indent=1, default=str))
    pw.round(4).to_csv(HERE / "per_window.csv", index=False)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
