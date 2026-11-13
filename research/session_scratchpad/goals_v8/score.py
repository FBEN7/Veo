"""goals v8 (plan.json, hashed first): the goal rule with >= 6 team-labelled
players present per sample, on the 20 halves and the simulated recordings.

    VEO_UNSEAL=... python score.py
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V5 = _load("goals_v5_score", "research/session_scratchpad/goals_v5/score.py")
SIM = _load("goals_v7_sim", "research/session_scratchpad/goals_v7/simulate.py")
F, E = V5.F, V5.E
from src import goal_candidates as GC  # noqa: E402

NMIN = 6
CELL = (0.75, None, 3)


def check_plan():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit("plan.json does not match its sha256")
    return got


def frozen(halves, mp):
    hits = rest_n = cand = false = right = 0
    for h in halves:
        teams = tuple(h["ends"])
        c = GC.find(h["people"], fps=25, origin="centre", teams=teams, min_present=mp)
        r = SIM.score_list(c, h["goals"].frame.to_numpy(), F.restartable(h["goals"].frame.to_numpy(), h["actions"]),
                           [h["side_team"].get(sd) for sd in h["goals"].side])
        hits += r["hits"]; rest_n += r["restartable"]; cand += r["candidates"]; false += r["false runs"]; right += r["scorer right"]
    return {"hits": f"{hits} / {rest_n}", "candidates per match": cand / 10, "false runs per match": false / 10,
            "scorer right": f"{right} / {hits}", "_hits": hits, "_right": right}


def held_out(halves, mp):
    table = {}
    for h in halves:
        s = h["s"].copy()
        if mp:
            q = h["people"][h["people"].team.isin(list(h["ends"])) & h["people"].x.notna() & h["people"].y.notna()]
            n = q.groupby("frame").size().reindex(s.index).fillna(0)
            s.loc[n < mp, "share"] = np.nan
        for c in V5.FORM:
            rs = E.runs(s, *c)
            table[(h["name"], c)] = V5.score_runs(h, rs, [None] * len(rs))
    matches = sorted({h["match"] for h in halves})
    tot = {"hits": 0, "restartable": 0, "false runs": 0}
    for mo in matches:
        tr = [h for h in halves if h["match"] != mo]
        best = None
        for c in V5.FORM:
            hits = sum(table[(h["name"], c)]["hits"] for h in tr)
            fr = sum(table[(h["name"], c)]["false runs"] for h in tr)
            key = (hits, -fr, c[0], -(c[1] if c[1] is not None else 99), -c[2])
            if best is None or key > best[0]:
                best = (key, c)
        for h in halves:
            if h["match"] == mo:
                for k in tot:
                    tot[k] += table[(h["name"], best[1])][k]
    return dict(tot, **{"false runs per match": round(tot["false runs"] / len(matches), 2)})


def simulated(loaded, mp):
    per_v = {v: {"half-time candidates": 0, "hits": 0, "false runs": 0, "scorer right": 0, "restartable": 0}
             for v in SIM.VARIANTS}
    for mi, m in enumerate(E.MATCHES):
        h1, h2 = loaded[(m, 1)], loaded[(m, 2)]
        for vi, v in enumerate(SIM.VARIANTS):
            tracks, truth, teams, gf, rest, want = SIM.build(h1, h2, v, seed=100 * mi + vi)
            c = GC.find(tracks, fps=25, half_starts_s=truth, origin="centre", teams=teams, min_present=mp)
            end1 = truth[0] + h1["people"].frame.max() / 25
            per_v[v]["half-time candidates"] += sum(end1 < x["formation_from_s"] < truth[1] for x in c)
            r = SIM.score_list(c, gf, rest, want)
            for k in ("hits", "false runs", "scorer right", "restartable"):
                per_v[v][k] += r[k]
    for v in per_v:
        per_v[v]["half-time candidates per match"] = per_v[v]["half-time candidates"] / 10
    return per_v


def main():
    sha = check_plan()
    from src.unseal import unsealed
    for m in sorted(E.SEALED):
        if not unsealed(m, "goals v8: minimum players present (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    halves, loaded = [], {}
    for m in E.MATCHES:
        for half in (1, 2):
            h = V5.load(m, half)
            h["name"] = f"{m}_{half}"
            halves.append(h)
            loaded[(m, half)] = h
    out = {"plan sha256": sha}
    for mp in (0, NMIN):
        out[f"min_present {mp}"] = {"frozen cell (in-sample)": frozen(halves, mp),
                                    "held out (formation only, LOMO)": held_out(halves, mp),
                                    "simulated recordings, true starts": simulated(loaded, mp)}
    new = out[f"min_present {NMIN}"]
    fz, ho, sim = new["frozen cell (in-sample)"], new["held out (formation only, LOMO)"], new["simulated recordings, true starts"]
    bar = {"frozen: hits 41/41 and scorer >= 39": fz["_hits"] == 41 and fz["_right"] >= 39,
           "held out: hits >= 40 and false per match <= 7.1": ho["hits"] >= 40 and ho["false runs per match"] <= 7.1,
           "simulated: half-time candidates <= 0.5 per match, every variant":
               all(sim[v]["half-time candidates per match"] <= 0.5 for v in sim)}
    out["bar"] = bar
    out["adopt"] = all(bar.values())
    (HERE / "goals_v8.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
