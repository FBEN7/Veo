"""goals v7 (plan.json + addendum1.json, hashed first): src/match_periods.py
on whole recordings simulated from the SoccerTrack 1-fps scans.

    VEO_UNSEAL=... python simulate.py
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
_spec = importlib.util.spec_from_file_location("goals_v5_score", ROOT / "research/session_scratchpad/goals_v5/score.py")
V5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V5)
F, E = V5.F, V5.E
from src import goal_candidates as GC  # noqa: E402
from src import match_periods as MP  # noqa: E402

FPS = 25
HL, HW = 52.5, 34.0
VARIANTS = ["V0", "V1", "V2", "V3"]


def check(name):
    want = (HERE / f"{name}.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"{name} does not match its sha256")
    return got


def rows_uniform(rng, t0, secs, teams_x):
    """teams_x: [(team, n, xlo, xhi)] -> rows redrawn each second."""
    out = []
    for t in range(t0, t0 + secs):
        for team, n, lo, hi in teams_x:
            if n:
                out.append(pd.DataFrame({"frame": t * FPS, "x": rng.uniform(lo, hi, n),
                                         "y": rng.uniform(-HW, HW, n), "team": team}))
    return out


def held(rng, first, t0, secs):
    out = []
    for t in range(t0, t0 + secs):
        out.append(pd.DataFrame({"frame": t * FPS, "x": first.x.to_numpy() + rng.normal(0, 0.5, len(first)),
                                 "y": first.y.to_numpy() + rng.normal(0, 0.5, len(first)),
                                 "team": first.team.to_numpy()}))
    return out


def build(h1, h2, variant, seed):
    rng = np.random.default_rng(seed)
    L1 = next(t for t, e in h1["ends"].items() if e < 0)
    R1 = next(t for t, e in h1["ends"].items() if e > 0)
    L2 = next(t for t, e in h2["ends"].items() if e < 0)
    R2 = next(t for t, e in h2["ends"].items() if e > 0)
    ren2 = {R2: L1, L2: R1}                       # the teams change ends
    p1 = h1["people"][h1["people"].team.isin([L1, R1])][["frame", "x", "y", "team"]]
    p2 = h2["people"][h2["people"].team.isin([L2, R2])][["frame", "x", "y", "team"]].copy()
    p2["team"] = p2.team.map(ren2)
    parts, t = [], 0
    if variant != "V0":
        warm = [(L1, 11, -HL, 0), (R1, 11, 0, HL)] if variant != "V2" else [(L1, 11, 0, HL), (R1, 11, -HL, 0)]
        parts += rows_uniform(rng, t, 600, warm)
        t += 600 + 360
        parts += held(rng, p1[p1.frame == p1.frame.min()], t, 45)
        t += 45
    k1 = t
    q = p1.copy(); q["frame"] = q.frame + k1 * FPS
    parts.append(q)
    t = k1 + int(np.ceil((p1.frame.max() + 1) / FPS))
    if variant == "V3":
        parts += rows_uniform(rng, t, 720, [(L1, 7, -HL, HL), (R1, 7, -HL, HL)])
    else:
        for s in range(t, t + 720):
            k = int(rng.integers(0, 4))
            if k:
                parts.append(pd.DataFrame({"frame": s * FPS, "x": rng.uniform(-HL, HL, k),
                                           "y": rng.uniform(-HW, HW, k), "team": rng.choice([L1, R1], k)}))
    t += 720
    if variant != "V0":
        first2 = p2[p2.frame == p2.frame.min()]
        parts += held(rng, first2, t, 45)
        t += 45
    k2 = t
    q = p2.copy(); q["frame"] = q.frame + k2 * FPS
    parts.append(q)
    tracks = pd.concat(parts, ignore_index=True)
    # goals, restartable flags (per half), actions, the scorer in the timeline's names
    g1, g2 = h1["goals"].frame.to_numpy() + k1 * FPS, h2["goals"].frame.to_numpy() + k2 * FPS
    rest = list(F.restartable(h1["goals"].frame.to_numpy(), h1["actions"])) + \
        list(F.restartable(h2["goals"].frame.to_numpy(), h2["actions"]))
    want = [h1["side_team"].get(sd) for sd in h1["goals"].side] + \
        [ren2.get(h2["side_team"].get(sd)) for sd in h2["goals"].side]
    return tracks, [k1, k2], (L1, R1), np.r_[g1, g2], rest, want


def score_list(cands, gf, rest, want):
    starts = [int(round(c["formation_from_s"] * FPS)) for c in cands]
    m = F.match_runs(starts, gf)
    hits = [gi for gi in m if rest[gi]]
    right = sum(cands[m[gi]]["proposed_scorer"] == want[gi] for gi in hits)
    return {"hits": len(hits), "restartable": int(sum(rest)), "false runs": len(cands) - len(m),
            "scorer right": int(right), "candidates": len(cands)}


def main():
    sha = {n: check(n) for n in ("plan.json", "addendum1.json")}
    from src.unseal import unsealed
    for m in sorted(E.SEALED):
        if not unsealed(m, "goals v7: simulated recordings (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    res, errs, equal = [], [], True
    for mi, m in enumerate(E.MATCHES):
        h1, h2 = V5.load(m, 1), V5.load(m, 2)
        for vi, v in enumerate(VARIANTS):
            tracks, truth, teams, gf, rest, want = build(h1, h2, v, seed=100 * mi + vi)
            r = MP.find(tracks, fps=FPS, origin="centre", teams=teams)
            k = r["kickoffs_s"]
            ok_n = len(k) == 2
            err = [round(a - b, 1) for a, b in zip(k, truth)] if ok_n else None
            c_true = GC.find(tracks, fps=FPS, half_starts_s=truth, origin="centre", teams=teams)
            c_det = GC.find(tracks, fps=FPS, half_starts_s=k, origin="centre", teams=teams) if k else []
            st, sd = score_list(c_true, gf, rest, want), score_list(c_det, gf, rest, want)
            same = all(st[x] == sd[x] for x in ("hits", "false runs", "scorer right"))
            equal &= same
            ht = sum(truth[0] + (h1["people"].frame.max() / FPS) < c["formation_from_s"] < truth[1] for c in c_true)
            row = {"match": m, "variant": v, "kick-offs found": len(k), "error s": err, "truth s": truth,
                   "how": [p["how"] for p in r["periods"]], "dropped": [d["why"] for d in r["dropped"]],
                   "list, true starts": st, "list, detected starts": sd, "lists equal": same,
                   "true-start candidates inside the half-time": int(ht)}
            res.append(row)
            if err is not None:
                errs += err
            print(m, v, "found", len(k), "err", err, "equal", same, flush=True)
    ok_all = all(r["kick-offs found"] == 2 and all(abs(e) <= 10 for e in r["error s"]) for r in res)
    by_v = {v: {"recordings with both kick-offs within 10 s": sum(r["kick-offs found"] == 2 and all(abs(e) <= 10 for e in r["error s"]) for r in res if r["variant"] == v),
                "lists equal": sum(r["lists equal"] for r in res if r["variant"] == v),
                "half-time candidates (true starts), per match": round(np.mean([r["true-start candidates inside the half-time"] for r in res if r["variant"] == v]), 2)}
            for v in VARIANTS}
    out = {"sha256": sha, "recordings": len(res), "by variant": by_v,
           "kick-off error s, all found": {"median |e|": float(np.median(np.abs(errs))) if errs else None,
                                           "max |e|": float(np.max(np.abs(errs))) if errs else None},
           "bar met (every kick-off within 10 s, lists equal)": bool(ok_all and equal), "per recording": res}
    (HERE / "goals_v7.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "per recording"}, indent=1, default=str))


if __name__ == "__main__":
    main()
