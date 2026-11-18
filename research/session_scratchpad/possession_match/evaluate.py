"""possession on a whole match (plan.json, hashed first): score the frozen
ball rule R on 128057's 31 windows, pooled over the match, per half and per
15-min period; team consistency across windows and team distance
(descriptive).

    VEO_UNSEAL=... python evaluate.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/possession_v2"))
import possession as P  # noqa: E402  (truth, side_map, rule: frozen)
import drive as D  # noqa: E402

FPS, PERIOD = 25, 22500
CODE = {"left": 1, "right": 2}


def check():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit("plan.json does not match its sha256")
    return got


def speeds(p):
    """team_distance.speeds: per row, speed from the track smoothed over 1 s (13 rows at even frames)."""
    out = pd.Series(np.nan, index=p.index)
    for _, g in p.groupby("track_id"):
        g = g.drop_duplicates("frame").sort_values("frame")
        fr = g.frame.to_numpy()
        full = g[["x", "y"]].set_axis(fr).reindex(np.arange(fr.min(), fr.max() + 1, 2))
        sm = full.rolling(13, center=True, min_periods=13).mean()
        a, b = sm.shift(6), sm.shift(-6)
        v = np.hypot(b.x - a.x, b.y - a.y) / (24 / 25)
        out.loc[g.index] = v.reindex(fr).to_numpy()
    return out


def truth_path(gt, n):
    """per side: outfield players' path lengths, smoothed over 1 s, sampled each second."""
    g = gt[(gt.role == "player") & gt.x.notna()]
    out = {"left": 0.0, "right": 0.0}
    samples = np.arange(12, n, 25)
    for (_pid, side), d in g.groupby(["player_id", "side"]):
        d = d.drop_duplicates("frame").set_index("frame").sort_index()[["x", "y"]]
        sm = d.reindex(np.arange(d.index.min(), d.index.max() + 1)).rolling(25, center=True, min_periods=25).mean()
        s = sm.reindex(samples)
        out[side] += float(np.nansum(np.hypot(np.diff(s.x), np.diff(s.y))))
    return out


def our_distance(pl, n):
    """per our team: 10 x the mean 1-s speed of its detections, each second (team_distance's estimator)."""
    p = pl.reset_index(drop=True)
    p["speed"] = speeds(p)
    centres = np.arange(12, n - 12, 25)
    centres = centres - centres % 2 + (int(p.frame.min()) % 2)
    c = p[p.frame.isin(centres) & p.speed.notna()]
    out = {}
    for tm in P.TEAMS:
        per = c[c.team == tm].groupby("frame").speed.mean().reindex(centres)
        per = per.fillna(per.mean()) if per.notna().any() else per.fillna(0.0)
        out[tm] = float(10 * per.sum())
    return out, len(centres)


def kit_means(w):
    """per our team, the mean kit feature of its tracks (team_features.json of the v3s run)."""
    base = D.OUT / w / f"output_{w}"
    m = pd.read_parquet(base / "metric_v3s_link.parquet")
    for sub in ("shots_v3s_link", "events_v1", "."):
        f = base / sub / "team_features.json"
        if f.exists():
            feats = json.loads(f.read_text())
            break
    else:
        return None
    out = {}
    for tm in P.TEAMS:
        ids = m[(m.cls == "player") & (m.team == tm)].track_id.dropna().astype(int).astype(str).unique()
        k = [feats[i]["kit"] for i in ids if i in feats]
        out[tm] = np.mean(k, axis=0) if k else None
    return out if all(v is not None for v in out.values()) else None


def main():
    sha = check()
    from src.unseal import unsealed
    if not unsealed(D.MATCH, "possession on a whole match: score 128057 (development use, seal lifted 2026-11-05)"):
        raise SystemExit("VEO_UNSEAL needed")
    wins = D.windows()
    # the team on the left in the 1st half, by the truth's player ids
    t1 = set()
    for w in wins:
        if w["half"] == 1:
            gt = pd.read_parquet(D.DATA / f"{w['name']}_gt.parquet")
            t1 |= set(gt[(gt.side == "left") & (gt.role == "player")].player_id)
    rows, failed, skipped, kits = [], [], [], {}
    for w in wins:
        name, n = w["name"], w["frames"]
        gt = pd.read_parquet(D.DATA / f"{name}_gt.parquet")
        ev = pd.read_csv(D.DATA / f"{name}_events.csv")
        tr = P.truth(ev, n)
        g1 = gt[gt.player_id.isin(t1) & (gt.role == "player")]
        t1_side = g1.side.mode().iloc[0]
        row = {"window": name, "half": w["half"], "period": f"{w['half']}-{w['start'] // PERIOD}", "live": int((tr > 0).sum()),
               "t1_truth": int((tr == CODE[t1_side]).sum()), "t1_side": t1_side}
        tp = truth_path(gt, n)
        row["dist_truth_t1"], row["dist_truth_t2"] = tp[t1_side], tp["right" if t1_side == "left" else "left"]
        f = D.OUT / name / f"output_{name}" / "metric_v3s_link.parquet"
        if not f.exists():
            failed.append(name)
            rows.append(row)
            continue
        m = pd.read_parquet(f)
        pl = m[(m.cls == "player") & m.team.isin(P.TEAMS) & m.x.notna()]
        ball = m[(m.cls == "ball") & m.x.notna()]
        mp = P.side_map(pl, gt)
        if mp is None:
            skipped.append(name)
            rows.append(row)
            continue
        frames = np.sort(pl.frame.unique())
        r, _, _ = P.rule(pl, ball, frames)
        ours_t1 = next(t for t, s in mp.items() if s == t1_side)
        code_t1 = 1 if ours_t1 == "team_A" else 2
        row.update(ok=True, r_t1=int((r == code_t1).sum()), r_all=int((r > 0).sum()), ours_t1=ours_t1)
        od, secs = our_distance(pl, n)
        row.update(dist_ours_t1=od[ours_t1], dist_ours_t2=od["team_B" if ours_t1 == "team_A" else "team_A"], seconds=secs)
        kits[name] = (kit_means(name), ours_t1)
        rows.append(row)
        print(name, "ok", flush=True)
    d = pd.DataFrame(rows)
    ok = d[d.get("ok", pd.Series(False, index=d.index)).fillna(False).astype(bool)]

    def score(g):
        t = g.t1_truth.sum() / g.live.sum()
        r = g.r_t1.sum() / g.r_all.sum()
        return {"truth share (team on the left in the 1st half)": round(float(t), 3), "R": round(float(r), 3),
                "error R (points)": round(100 * abs(r - t), 1), "error constant 0.5 (points)": round(100 * abs(0.5 - t), 1),
                "live minutes": round(g.live.sum() / FPS / 60, 1)}
    match = score(ok)
    halves = {f"half {h}": score(g) for h, g in ok.groupby("half")}
    periods = {p: score(g) for p, g in ok.groupby("period")}
    # team consistency across windows: link each window's teams to the first ok window's by kit
    first = next((k for k in ok.window if kits.get(k, (None,))[0] is not None), None)
    agree, linked = 0, 0
    if first:
        k1, t1_first = kits[first]
        for name, (km, ot1) in kits.items():
            if km is None:
                continue
            same = (np.linalg.norm(km["team_A"] - k1["team_A"]) + np.linalg.norm(km["team_B"] - k1["team_B"])
                    <= np.linalg.norm(km["team_A"] - k1["team_B"]) + np.linalg.norm(km["team_B"] - k1["team_A"]))
            truth_same = ot1 == t1_first
            agree += int(same == truth_same)
            linked += 1
    # team distance (descriptive)
    dd = json.loads((ROOT / "research/session_scratchpad/team_distance/team_distance.json").read_text())
    dist = pd.read_parquet(ROOT / "research/session_scratchpad/team_distance/distance.parquet")
    o = dist[dist.match != D.MATCH]
    rate = float(o.truth.sum() / o.seconds.sum())
    td = {}
    for t, (oc, tc) in {"t1": ("dist_ours_t1", "dist_truth_t1"), "t2": ("dist_ours_t2", "dist_truth_t2")}.items():
        tru, our, secs = ok[tc].sum(), ok[oc].sum(), ok.seconds.sum()
        td[t] = {"truth (km)": round(tru / 1000, 2), "ours (km)": round(our / 1000, 2), "ours rel. error": round(abs(our - tru) / tru, 3),
                 "constant (km)": round(rate * secs / 1000, 2), "constant rel. error": round(abs(rate * secs - tru) / tru, 3)}
    out = {"plan sha256": sha, "windows": len(d), "ran": len(ok), "failed": failed, "skipped (no side mapping)": skipped,
           "live time covered": round(float(ok.live.sum() / d.live.sum()), 3),
           "primary: the match": match, "confirmed (error <= 5 points)": bool(match["error R (points)"] <= 5.0),
           "per half": halves, "per 15-min period": periods,
           "team consistency across windows by kit (descriptive)": {"windows linked": linked, "agree with the truth": agree},
           "team distance (descriptive)": td, "team distance constant rate (m per team-second, other matches' clips)": round(rate, 2),
           "per window": d.round(3).to_dict("records")}
    (HERE / "possession_match.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "per window"}, indent=1, default=str))


if __name__ == "__main__":
    main()
