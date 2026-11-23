"""pass accuracy (plan.json, hashed first): team pass accuracy on 47
ordinary-play windows, a possession-chain estimator against a constant.

    VEO_UNSEAL=... python pass_accuracy.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "research/session_scratchpad/possession_v2"))
import possession as P  # noqa: E402  (side_map, the windows' units)

S, CE, FPS = P.S, P.CE, 25
NEAR_M, GAP_F, MIN_F, PASS_S, PASS_M = 4.0, 10, 10, 4.0, 5.0
PASSES = ("PASS", "HIGH PASS", "CROSS")


def check():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit("plan.json does not match its sha256")
    return got


def truth(ev):
    ev = ev[ev.side.isin(["left", "right"])].sort_values("frame", kind="stable").reset_index(drop=True)
    out = {"left": [0, 0], "right": [0, 0]}
    for i in range(len(ev) - 1):
        if ev.label[i] in PASSES:
            ok = ev.label[i + 1] != "OUT" and ev.side[i + 1] == ev.side[i]
            out[ev.side[i]][0] += int(ok)
            out[ev.side[i]][1] += 1
    return out


def holders(pl, ball):
    """per player frame with our ball: (frame, track, team, x, y) of the nearest team-labelled player within 4 m."""
    bxy = ball.drop_duplicates("frame").set_index("frame")[["x", "y"]]
    rows = []
    for f, p in pl.groupby("frame"):
        if f not in bxy.index:
            continue
        d = np.hypot(p.x.to_numpy() - bxy.at[f, "x"], p.y.to_numpy() - bxy.at[f, "y"])
        k = int(np.argmin(d))
        if d[k] <= NEAR_M:
            r = p.iloc[k]
            rows.append((f, r.track_id, r.team, r.x, r.y))
    return pd.DataFrame(rows, columns=["frame", "track", "team", "x", "y"])


def spells(h):
    out, cur = [], None
    for r in h.itertuples():
        if cur and r.track == cur["track"] and r.frame - cur["f1"] <= GAP_F:
            cur.update(f1=r.frame, x1=r.x, y1=r.y)
            continue
        if cur:
            out.append(cur)
        cur = {"track": r.track, "team": r.team, "f0": r.frame, "f1": r.frame, "x0": r.x, "y0": r.y, "x1": r.x, "y1": r.y}
    if cur:
        out.append(cur)
    return [s for s in out if s["f1"] - s["f0"] >= MIN_F]


def ours(sp):
    out = {"team_A": [0, 0], "team_B": [0, 0]}
    for a, b in zip(sp[:-1], sp[1:]):
        if b["track"] == a["track"] or (b["f0"] - a["f1"]) / FPS > PASS_S:
            continue
        if np.hypot(b["x0"] - a["x1"], b["y0"] - a["y1"]) < PASS_M:
            continue
        out[a["team"]][0] += int(b["team"] == a["team"])
        out[a["team"]][1] += 1
    return out


def boot(d, a, b, matches, n=5000):
    rng = np.random.default_rng(0)
    by = {m: (d[d.match == m][a] - d[d.match == m][b]).to_numpy() for m in matches}
    v = [np.concatenate([by[m] for m in rng.choice(matches, len(matches))]).mean() for _ in range(n)]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def main():
    sha = check()
    from src.unseal import unsealed
    for m in S.V3.NEW_PLAYERS:
        if not unsealed(m, "pass accuracy: M7 / M9 windows (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    units = [u for u in CE.unit_list() if u[2] is None]
    acc = {}
    for name, match, _, tree in units:
        info = S.LS.sd.info_of(name)
        m = pd.read_parquet(tree / name / f"output_{name}" / "metric_v3s_link.parquet")
        pl = m[(m.cls == "player") & m.team.isin(P.TEAMS) & m.x.notna()]
        ball = m[(m.cls == "ball") & m.x.notna()]
        ev = pd.read_csv(info["dir"] / f"{name}_events.csv")
        gt = pd.read_parquet(info["dir"] / f"{name}_gt.parquet")
        mp = P.side_map(pl, gt)
        if mp is None:
            continue
        t = truth(ev)
        o = ours(spells(holders(pl, ball)))
        a = acc.setdefault(match, {s: {"t": [0, 0], "o": [0, 0]} for s in ("left", "right")})
        for side in ("left", "right"):
            a[side]["t"] = [a[side]["t"][i] + t[side][i] for i in (0, 1)]
        for tm, side in mp.items():
            a[side]["o"] = [a[side]["o"][i] + o[tm][i] for i in (0, 1)]
        print(name, "ok", flush=True)
    rows = [{"match": m, "side": s, "truth": v["t"][0] / v["t"][1], "truth_passes": v["t"][1],
             "P": v["o"][0] / v["o"][1] if v["o"][1] else np.nan, "our_passes": v["o"][1],
             "t_ok": v["t"][0], "t_n": v["t"][1]}
            for m, sides in acc.items() for s, v in sides.items()]
    d = pd.DataFrame(rows)
    matches = sorted(d.match.unique())
    for m in matches:
        o = d[d.match != m]
        d.loc[d.match == m, "C"] = o.t_ok.sum() / o.t_n.sum()
        d.loc[d.match == m, "Pc"] = d.loc[d.match == m, "P"] - float((o.P - o.truth).median())
    for k in ("C", "P", "Pc"):
        d[f"err_{k}"] = (d[k] - d.truth).abs()
    gain = float(d.err_Pc.mean() - d.err_C.mean())
    ci = boot(d, "err_Pc", "err_C", matches)
    out = {"plan sha256": sha, "team-matches": len(d),
           "primary: mean abs error, Pc - C": round(gain, 4), "95% bootstrap over matches": ci,
           "beats the constant": bool(gain < 0 and ci[1] < 0),
           "P - C": round(float(d.err_P.mean() - d.err_C.mean()), 4), "P - C, 95%": boot(d, "err_P", "err_C", matches),
           "mean abs error (points)": {k: round(100 * float(d[f"err_{k}"].mean()), 1) for k in ("C", "P", "Pc")},
           "within 10 points": {k: int((d[f"err_{k}"] <= 0.10).sum()) for k in ("C", "P", "Pc")},
           "within 5 points": {k: int((d[f"err_{k}"] <= 0.05).sum()) for k in ("C", "P", "Pc")},
           "pass counts, ours vs truth: median relative error": round(float(((d.our_passes - d.truth_passes).abs() / d.truth_passes).median()), 3),
           "P - truth: median, min, max": [round(float(x), 3) for x in ((d.P - d.truth).median(), (d.P - d.truth).min(), (d.P - d.truth).max())],
           "per team-match": d.round(3).to_dict("records")}
    (HERE / "pass_accuracy.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "per team-match"}, indent=1, default=str))


if __name__ == "__main__":
    main()
