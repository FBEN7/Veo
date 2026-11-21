"""The goals rule's re-choice on the dev half's detections, then the freeze
(protocol.json, GOALS 'dev, detections').

On M7 2nd (128057 half 2: 5 restartable goals), with the 1-fps scan
(formation_scan.load_scan: team_A / team_B people inside the pitch box), each
(T, N) of the registered grid is scored against the half's BAS GOALs. The
choice: hits, then fewest false runs, then the larger T, then the larger N.
The team rule stays the one frozen on ground truth (formation_gt_choice.json).
A BAS side maps to the detection team defending that end (left = x < 0).

Writes frozen_goals.json and its sha256: from then on the test halves may be
scanned (formation_scan.py guard) and the gate may read M9 2nd.

    python freeze_goals.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/user/Veo")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import formation as fm  # noqa: E402
import formation_scan as fs  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402

MATCH, HALF, NAME = "128057", 2, "128057_2nd"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    proto = fs.load_protocol()
    grid = proto["GOALS"]["grid"]
    gt_choice = json.loads((HERE / "formation_gt_choice.json").read_text())["choice"]
    people, frames = fs.load_scan(NAME)
    ends = fm.defended_ends(people)
    share = fm.own_half_share(people, ends, frames)
    ev = st.events(MATCH)
    ev = ev[ev.half == HALF].sort_values("frame")
    goals = ev[ev.label == "GOAL"][["frame", "side"]].reset_index(drop=True)
    side_team = {"left": next(t for t, e in ends.items() if e < 0),
                 "right": next(t for t, e in ends.items() if e > 0)}
    table = []
    for T in grid["T"]:
        for N in grid["N"]:
            r = fm.score_half(people, ends, share, goals, ev.frame.to_numpy(), T, N,
                              team_of_side=side_team.get)
            table.append({"T": T, "N": N, "hits": r["hits"], "false_runs": r["false_runs"],
                          "team_right": r["team_right"], "runs": r["runs"],
                          "matches": r["matches"]})
    best = min(table, key=lambda c: (-c["hits"], c["false_runs"], -c["T"], -c["N"]))
    frozen = {
        "protocol sha256": sha256(HERE / "protocol.json"),
        "protocol_notes sha256": sha256(HERE / "protocol_notes.json"),
        "T": best["T"], "N": best["N"], "team rule": gt_choice["team rule"],
        "scan key": fs.SCAN_KEY,
        "dev half": NAME, "dev goals": goals.frame.tolist(),
        "dev result": {k: best[k] for k in ("hits", "false_runs", "team_right", "runs")},
        "ground-truth choice": gt_choice,
        "ends": ends, "dev grid": [{k: c[k] for k in ("T", "N", "hits", "false_runs", "team_right")}
                                   for c in table],
    }
    out = HERE / "frozen_goals.json"
    out.write_text(json.dumps(frozen, indent=1))
    out.with_name(out.name + ".sha256").write_text(sha256(out) + "\n")
    print(json.dumps({k: frozen[k] for k in ("T", "N", "team rule", "dev result")}, indent=1))


if __name__ == "__main__":
    main()
