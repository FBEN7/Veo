"""Goals v2 on detections (exploratory): the 1-fps scans of the halves that
exist here (.cache/formation_scan/<match>_<half>.parquet), the formation
share with the scan's teams (as score_goals_test.py: team-labelled people
within |x| <= 54.5 m and |y| <= 36 m), the centre-spot distance d0 of the
nearest of them, and the rule grid of explore.py (T x R x N). Reports, per
cell, hits on restartable goals and false runs pooled over the halves, and
the registered cell (T 0.85, N 2, no R).

    python on_detections.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import explore as E  # noqa: E402

F, st = E.F, E.st
SCANS = E.ROOT / ".cache/formation_scan"


def half_det(path: Path) -> dict:
    match, h = path.stem.split("_")
    half = 1 if h == "1st" else 2
    s = pd.read_parquet(path)
    p = s[(s.x.abs() <= 54.5) & (s.y.abs() <= 36) & s.team.isin(["team_A", "team_B"])]
    ends = F.defended_ends(p)
    share = F.own_half_share(p, ends, frames=sorted(s.frame.unique()))
    d0 = np.hypot(p.x, p.y).groupby(p.frame).min().reindex(share.index)
    ev = st.events(match)
    ev = ev[ev.half == half]
    goals = ev[ev.label == "GOAL"][["frame", "side"]].reset_index(drop=True)
    return {"s": pd.DataFrame({"share": share, "d0": d0}), "goals": goals,
            "actions": ev.frame.to_numpy(), "match": match, "name": path.stem}


def main():
    halves = [half_det(p) for p in sorted(SCANS.glob("*.parquet"))]
    cells = list(itertools.product(E.GRID["T"], E.GRID["R"], E.GRID["N"]))
    rows = []
    for c in cells:
        sc = [E.score(h, *c) for h in halves]
        rows.append({"T": c[0], "R": c[1], "N": c[2], "hits": sum(x[0] for x in sc),
                     "false runs": sum(x[1] for x in sc), "restartable": sum(x[2] for x in sc),
                     "per half": {h["name"]: list(x[:2]) for h, x in zip(halves, sc)}})
    df = pd.DataFrame(rows)
    reg = df[(df["T"] == 0.85) & df.R.isna() & (df.N == 2)].iloc[0]
    print("halves:", [h["name"] for h in halves], "restartable:", int(reg.restartable))
    print("registered:", int(reg.hits), "hits,", int(reg["false runs"]), "false")
    top = df.sort_values(["hits", "false runs"], ascending=[False, True]).head(12)
    print(top[["T", "R", "N", "hits", "false runs"]].to_string(index=False))
    (HERE / "on_detections.json").write_text(json.dumps(
        {"halves": [h["name"] for h in halves], "registered": reg.to_dict(),
         "grid": json.loads(df.to_json(orient="records"))}, indent=1, default=str))


if __name__ == "__main__":
    main()
