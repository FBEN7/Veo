"""Exploratory (after goals_v5.json; owner 2026-11-12 chose a semi-automatic
'possible goals' list): formation-only rule (kick-off ends, 10-60 s), per cell
of T x R x N: candidates per match vs restartable goals found, in-sample over
the 20 halves (Pareto front), and the held-out point."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import score as S
from src.unseal import unsealed
for m in sorted(S.E.SEALED):
    assert unsealed(m, "goals v5 extra: candidates trade-off (development use, seal lifted 2026-11-05)")
halves = []
for m in S.E.MATCHES:
    for half in (1, 2):
        h = S.load(m, half); h["name"] = f"{m}_{half}"; halves.append(h)
pts = []
for c in S.FORM:
    r = [S.score_runs(h, h["runs"][c], [None] * len(h["runs"][c])) for h in halves]
    hits = sum(x["hits"] for x in r); kept = sum(x["kept runs"] for x in r)
    ts = sum(x["team right (kick_spot)"] for x in r)
    pts.append({"T": c[0], "R": c[1], "N": c[2], "hits": hits, "candidates per match": kept / 10,
                "false per match": sum(x["false runs"] for x in r) / 10, "team right": round(ts / max(hits, 1), 3)})
rest = sum(sum(S.F.restartable(h["goals"].frame.to_numpy(), h["actions"])) for h in halves)
front = []
for p in sorted(pts, key=lambda p: (p["candidates per match"], -p["hits"])):
    if not front or p["hits"] > front[-1]["hits"]:
        front.append(p)
out = {"restartable goals": int(rest), "Pareto front (in-sample)": front}
(HERE / "reading_tradeoff.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
