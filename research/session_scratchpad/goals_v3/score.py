"""goals v3 (plan.json, hashed first): the kick-off formation rule on the
1-fps scans of all 20 halves, its cell chosen leaving one match out.

    VEO_UNSEAL=... python score.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/goals_v2"))
import explore as E  # noqa: E402
import on_detections as D  # noqa: E402

F, st = E.F, E.st
SCANS = ROOT / ".cache/formation_scan"
CELLS = list(itertools.product(E.GRID["T"], E.GRID["R"], E.GRID["N"]))


def load(match, half):
    name = f"{match}_{'1st' if half == 1 else '2nd'}"
    meta = json.loads((SCANS / f"{name}.json").read_text())
    if not meta.get("complete"):
        raise SystemExit(f"{name}: scan not complete")
    h = D.half_det(SCANS / f"{name}.parquet")
    s = pd.read_parquet(SCANS / f"{name}.parquet")
    p = s[(s.x.abs() <= 54.5) & (s.y.abs() <= 36) & s.team.isin(["team_A", "team_B"])]
    ends = F.defended_ends(p)
    h.update(people=p, ends=ends, half=half,
             side_team={"left": next(t for t, e in ends.items() if e < 0),
                        "right": next(t for t, e in ends.items() if e > 0)})
    return h


def held(h, cell):
    T, R, N = cell
    rs = E.runs(h["s"], T, R, N)
    gf = h["goals"].frame.to_numpy()
    rest = F.restartable(gf, h["actions"])
    m = F.match_runs([r[0] for r in rs], gf)
    hits = [gi for gi in m if rest[gi]]
    team = sum(F.scorer_kick_spot(h["people"], h["ends"], rs[m[gi]]) == h["side_team"].get(h["goals"].side.iloc[gi])
               for gi in hits)
    return {"hits": len(hits), "restartable": int(sum(rest)), "goals": len(gf), "false runs": len(rs) - len(m),
            "team right": int(team), "runs": len(rs), "last": int(h["s"].index.max()), "rest flags": list(map(bool, rest)),
            "goal frames": gf.tolist()}


def main():
    from src.unseal import unsealed
    for m in sorted(E.SEALED):
        if not unsealed(m, "goals v3: scans of all 20 halves (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    halves = [load(m, h) for m in E.MATCHES for h in (1, 2)]
    matches = sorted({h["match"] for h in halves})
    per = {}
    tot = {"hits": 0, "restartable": 0, "false runs": 0, "team right": 0, "goals": 0}
    base_halves = []
    for mo in matches:
        cell = E.choose([h for h in halves if h["match"] != mo], CELLS)
        rows = [held(h, cell) for h in halves if h["match"] == mo]
        r = {k: sum(x[k] for x in rows) for k in tot}
        per[mo] = dict(r, cell={"T": cell[0], "R": cell[1], "N": cell[2]})
        for k in tot:
            tot[k] += r[k]
        base_halves += [{"n_runs": x["runs"], "goals": x["goal frames"], "restartable": x["rest flags"], "last": x["last"]} for x in rows]
    reg = [held(h, (0.85, None, 2)) for h in halves]
    rb = F.random_baseline(base_halves, 1000, 0)
    chosen_all = E.choose(halves, CELLS)
    found = tot["hits"] / max(tot["restartable"], 1)
    fr_per_match = tot["false runs"] / len(matches)
    team = tot["team right"] / max(tot["hits"], 1)
    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0], "halves": len(halves),
           "held out": {"restartable goals found": f"{tot['hits']} / {tot['restartable']}", "share": round(found, 4),
                        "false runs": tot["false runs"], "false runs per match": round(fr_per_match, 2),
                        "scoring team right on hits (kick_spot)": round(team, 4),
                        "goals in all (incl. not restartable)": tot["goals"], "per match": per},
           "random-time baseline (same run counts), hits": {"mean": round(float(rb.mean()), 2), "95th percentile": float(np.percentile(rb, 95))},
           "registered cell (T 0.85, N 2) on all 20 halves": {k: int(sum(x[k] for x in reg)) for k in ("hits", "restartable", "false runs", "team right")},
           "cell chosen on all 10 matches": {"T": chosen_all[0], "R": chosen_all[1], "N": chosen_all[2]},
           "bar met (found >= 0.95, false runs <= 1 per match)": bool(found >= 0.95 and fr_per_match <= 1),
           "team bar met (right on >= 0.9 of hits)": bool(team >= 0.9)}
    (HERE / "goals_v3.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
