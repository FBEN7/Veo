"""src/goal_candidates.py against goals_v6's rule on the same 20 halves:
the same formation runs and the same proposed scorers, and hence the same
hits and team figures (41/41 in-sample, attack_end 39/41).

    VEO_UNSEAL=... python equivalence.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score as S6  # noqa: E402
from src import goal_candidates as GC  # noqa: E402
from src.unseal import unsealed  # noqa: E402

CELL = (0.75, None, 3)
for m in sorted(S6.E.SEALED):
    assert unsealed(m, "goals v6: product module equivalence (development use, seal lifted 2026-11-05)")
diff, n_runs, hits, right, rest_n = [], 0, 0, 0, 0
for m in S6.E.MATCHES:
    for half in (1, 2):
        h = S6.V5.load(m, half)
        runs = h["runs"][CELL]
        ref = [(a, S6.propose("attack_end", h, (a, b))) for a, b in runs]
        teams = tuple(h["ends"])
        c = GC.find(h["people"], fps=25, half_starts_s=(0.0,), origin="centre", teams=teams)
        got = [(int(round(x["formation_from_s"] * 25)), x["proposed_scorer"]) for x in c]
        if got != ref:
            diff.append({"half": f"{m}_{half}", "ref": ref, "module": got})
        n_runs += len(got)
        gf = h["goals"].frame.to_numpy()
        rest = S6.F.restartable(gf, h["actions"])
        rest_n += sum(rest)
        mt = S6.F.match_runs([g[0] for g in got], gf)
        for gi, ri in mt.items():
            if rest[gi]:
                hits += 1
                right += got[ri][1] == h["side_team"].get(h["goals"].side.iloc[gi])
out = {"halves differing": len(diff), "differences": diff, "candidates per match": n_runs / 10,
       "restartable goals hit": f"{hits} / {rest_n}", "proposed scorer right": f"{right} / {hits}"}
(HERE / "equivalence.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str)[:3000])
