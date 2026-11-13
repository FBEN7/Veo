"""goals v6 (plan.json, hashed first): the scoring team proposed for each
possible goal of the formation-only candidate list, chosen leaving one match out.

    VEO_UNSEAL=... python score.py
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
_spec = importlib.util.spec_from_file_location("goals_v5_score", ROOT / "research/session_scratchpad/goals_v5/score.py")
V5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V5)          # sets F.MATCH_S = (10, 60), kick-off ends in load()
F, E, FPS = V5.F, V5.E, V5.FPS
SPOT_R, CIRCLE_R = 3.0, 9.15
RULES = ["kick_spot", "attack_end", "spot_vote", "circle_count", "circle_count_last10"]


def check_plan():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit("plan.json does not match its sha256")
    return got


def samples(h, run):
    fr = h["pf_frames"]
    return [int(f) for f in fr[(fr >= run[0]) & (fr <= run[1])]]


def kick_spot(h, run):
    return V5.kicker_other(h, run[1]) if run[1] in h["pf"] else F.scorer_kick_spot(h["people"], h["ends"], run)


def spot_vote(h, run):
    votes, near = [], []
    for f in samples(h, run):
        g = h["pf"][f]
        d = np.hypot(g.x.to_numpy(), g.y.to_numpy())
        k = int(d.argmin())
        votes.append(g.team.iloc[k])
        if d[k] <= SPOT_R:
            near.append(g.team.iloc[k])
    v = near or votes
    if not v:
        return kick_spot(h, run)
    teams, n = np.unique(v, return_counts=True)
    if (n == n.max()).sum() > 1:
        return kick_spot(h, run)
    return F._other(teams[n.argmax()], h["ends"])


def circle(h, run, last=None):
    fs = samples(h, run)
    if last:
        fs = fs[-last:]
    c = {t: 0 for t in h["ends"]}
    for f in fs:
        g = h["pf"][f]
        inside = g[np.hypot(g.x.to_numpy(), g.y.to_numpy()) <= CIRCLE_R]
        for t, n in inside.team.value_counts().items():
            c[t] += int(n)
    a, b = sorted(c.items(), key=lambda kv: -kv[1])
    if a[1] == b[1]:
        return kick_spot(h, run)
    return F._other(a[0], h["ends"])


def propose(rule, h, run):
    if rule == "kick_spot":
        return kick_spot(h, run)
    if rule == "attack_end":
        return F.scorer_attack_end(h["people"], h["ends"], run)
    if rule == "spot_vote":
        return spot_vote(h, run)
    if rule == "circle_count":
        return circle(h, run)
    return circle(h, run, last=10)


def candidate_cell(halves):
    best = None
    for c in V5.FORM:
        r = [V5.score_runs(h, h["runs"][c], [None] * len(h["runs"][c])) for h in halves]
        hits, fr = sum(x["hits"] for x in r), sum(x["false runs"] for x in r)
        key = (hits, -fr, c[0], -(c[1] if c[1] is not None else 99), -c[2])
        if best is None or key > best[0]:
            best = (key, c)
    return best[1]


def main():
    sha = check_plan()
    from src.unseal import unsealed
    for m in sorted(E.SEALED):
        if not unsealed(m, "goals v6: team proposal scoring (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    halves = []
    for m in E.MATCHES:
        for half in (1, 2):
            h = V5.load(m, half)
            h["name"] = f"{m}_{half}"
            halves.append(h)
    cell = candidate_cell(halves)
    # hits of the candidate cell: (match, rule -> right?) per hit
    rows, n_cand, n_false = [], 0, 0
    for h in halves:
        runs = h["runs"][cell]
        gf = h["goals"].frame.to_numpy()
        rest = F.restartable(gf, h["actions"])
        mt = F.match_runs([r[0] for r in runs], gf)
        n_cand += len(runs)
        n_false += len(runs) - len(mt)
        for gi, ri in mt.items():
            if not rest[gi]:
                continue
            want = h["side_team"].get(h["goals"].side.iloc[gi])
            got = {r: propose(r, h, runs[ri]) for r in RULES}
            rows.append({"match": h["match"], "half": h["name"], "goal frame": int(gf[gi]),
                         "want": want, **{r: (got[r] == want) if got[r] is not None else None for r in RULES}})
    matches = sorted({h["match"] for h in halves})
    held, per = 0, {}
    for mo in matches:
        tr = [r for r in rows if r["match"] != mo]
        score = {ru: sum(bool(r[ru]) for r in tr) for ru in RULES}
        pick = max(RULES, key=lambda ru: (score[ru], -RULES.index(ru)))
        te = [r for r in rows if r["match"] == mo]
        right = sum(bool(r[pick]) for r in te)
        held += right
        per[mo] = {"hits": len(te), "right": right, "rule": pick}
    n = len(rows)
    out = {"plan sha256": sha, "candidate cell": dict(zip(["T", "R", "N"], cell)),
           "candidates per match": n_cand / len(matches), "false runs per match": n_false / len(matches),
           "restartable goals hit": n,
           "held out: proposed team right": f"{held} / {n}", "held out share": round(held / max(n, 1), 4),
           "bar met (>= 0.9)": bool(held / max(n, 1) >= 0.9),
           "in-sample right share per rule": {ru: round(sum(bool(r[ru]) for r in rows) / max(n, 1), 4) for ru in RULES},
           "no team per rule": {ru: sum(r[ru] is None for r in rows) for ru in RULES},
           "per match": per, "per hit": rows}
    (HERE / "goals_v6.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "per hit"}, indent=1, default=str))


if __name__ == "__main__":
    main()
