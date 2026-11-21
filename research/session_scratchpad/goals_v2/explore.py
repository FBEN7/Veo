"""Goals v2 (exploratory, owner decision 2026-10-22; no untouched half is left
for a registered test): the formation rule plus a kick-off cue -- a person
standing on the centre spot -- chosen by leave-one-match-out on the ground
truth of all 20 halves, then checked on detections where a 1-fps scan exists.

Per half and 1-s sample, from the GSR (players and goalkeepers; team = side):
  share = people in their own half (formation.own_half_share, ends by mean x)
  d0    = the nearest person's distance to the centre spot (m)
A run: >= N consecutive samples with share >= T and d0 <= R; then
formation.runs' start exclusion and merging; matched to BAS GOALs as
formation.match_runs (15-60 s). Scored as formation.score_half: hits on
restartable goals, false runs.

Grid T x R x N. Choice (as protocol.json): most hits, then fewest false
runs, then the larger T, then the smaller R. Leave-one-match-out: chosen on
nine matches, scored on the tenth; the pooled held-out counts are the result.
The registered rule (T 0.85, N 2, no R) is scored the same way.

    VEO_UNSEAL=... python explore.py [--detections]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/shots_goals"))
import formation as F  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.unseal import unsealed  # noqa: E402

MATCHES = sorted(st.MATCHES)
SEALED = {"128057", "132831"}
GRID = {"T": [0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90],
        "R": [1.0, 2.0, 3.0, 5.0, None], "N": [1, 2, 3]}


def half_gt(match: str, half: int) -> dict:
    cache = HERE / "cache" / f"{match}_{half}.parquet"
    if cache.exists():
        s = pd.read_parquet(cache)
    else:
        g = st.gsr_table(match, half)
        g = g[(g.frame % F.SAMPLE == 0) & g.role.isin(["player", "goalkeeper"]) & g.x.notna()]
        g = g.rename(columns={"side": "team"})[["frame", "x", "y", "team"]]
        ends = F.defended_ends(g)
        share = F.own_half_share(g, ends)
        d0 = np.hypot(g.x, g.y).groupby(g.frame).min()
        s = pd.DataFrame({"share": share, "d0": d0.reindex(share.index)})
        s["team_A_end"] = ends[sorted(ends)[0]]
        cache.parent.mkdir(exist_ok=True)
        s.to_parquet(cache)
    ev = st.events(match)
    ev = ev[ev.half == half]
    goals = ev[ev.label == "GOAL"][["frame", "side"]].reset_index(drop=True)
    return {"s": s, "goals": goals, "actions": ev.frame.to_numpy(), "match": match}


def runs(s: pd.DataFrame, T: float, R, N: int):
    sh = s.share.where(s.d0 <= R, -1.0) if R is not None else s.share
    return F.runs(sh, T, N)


def score(h: dict, T, R, N) -> tuple[int, int]:
    rs = runs(h["s"], T, R, N)
    gf = h["goals"].frame.to_numpy()
    rest = F.restartable(gf, h["actions"])
    m = F.match_runs([r[0] for r in rs], gf)
    hits = sum(1 for gi in m if rest[gi])
    return hits, len(rs) - len(m), sum(rest)


def choose(halves, cells):
    best = None
    for c in cells:
        hits = fr = 0
        for h in halves:
            a, b, _ = score(h, *c)
            hits, fr = hits + a, fr + b
        key = (hits, -fr, c[0], -(c[1] if c[1] is not None else 99))
        if best is None or key > best[0]:
            best = (key, c)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    for m in SEALED:
        if not unsealed(m, "goals v2: ground-truth formation features"):
            raise SystemExit("VEO_UNSEAL needed for M7 / M9")
    halves = [half_gt(m, h) for m in MATCHES for h in (1, 2)]
    cells = list(itertools.product(GRID["T"], GRID["R"], GRID["N"]))
    out = {"halves": len(halves), "restartable goals": int(sum(score(h, 0.85, None, 2)[2] for h in halves))}
    reg = [score(h, 0.85, None, 2) for h in halves]
    out["registered rule (T 0.85, N 2) on the ground truth"] = {
        "hits": sum(r[0] for r in reg), "false runs": sum(r[1] for r in reg)}
    held, chosen = [], {}
    for m in MATCHES:
        train = [h for h in halves if h["match"] != m]
        c = choose(train, cells)
        chosen[m] = {"T": c[0], "R": c[1], "N": c[2]}
        for h in halves:
            if h["match"] == m:
                a, b, n = score(h, *c)
                held.append((m, a, b, n))
    out["leave-one-match-out (ground truth)"] = {
        "hits": sum(x[1] for x in held), "false runs": sum(x[2] for x in held),
        "per match [hits, false, restartable]": {m: [sum(x[1] for x in held if x[0] == m),
                                                     sum(x[2] for x in held if x[0] == m),
                                                     sum(x[3] for x in held if x[0] == m)] for m in MATCHES},
        "chosen per fold": chosen}
    allc = choose(halves, cells)
    out["chosen on all 20 halves"] = {"T": allc[0], "R": allc[1], "N": allc[2],
                                      "hits": sum(score(h, *allc)[0] for h in halves),
                                      "false runs": sum(score(h, *allc)[1] for h in halves)}
    print(json.dumps(out, indent=1))
    (HERE / "explore_gt.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
