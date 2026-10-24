"""The registered goals test (protocol.json GOALS), scored once, on the four
test halves' 1-fps scans with the frozen rule (frozen_goals.json).

Per half: the formation runs under the frozen T and N, matched one to one to
the half's BAS GOALs (15-60 s after); hits = matched restartable goals; false
runs = unmatched runs. Primary: hits over the 8 restartable goals, false runs
per 90 min over the four halves. Secondary: scoring team right per hit under
the frozen team rule (and the other rule, reported); goal-time error (run
start - 22 s - goal); a random-time baseline (the same number of runs per half
at uniform random 1-s samples after the first 60 s, 1000 draws). A BAS side
maps to the detection team defending that end (left = x < 0). Success as
registered: hits >= 7/8, false runs <= 1, team right on all hits but one.

    python score_goals_test.py      # -> goals_test.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/user/Veo")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import formation as fm  # noqa: E402
import formation_scan as fs  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402

FPS = 25


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def main():
    proto = fs.load_protocol()
    frozen_path = HERE / "frozen_goals.json"
    assert (frozen_path.with_name(frozen_path.name + ".sha256").read_text().split()[0]
            == sha256(frozen_path)), "frozen_goals.json changed"
    fr = json.loads(frozen_path.read_text())
    T, N, rule = fr["T"], fr["N"], fr["team rule"]
    rng = np.random.default_rng(0)
    halves, tot = [], {"hits": 0, "restartable": 0, "false_runs": 0, "minutes": 0.0,
                       "team_right": {k: 0 for k in fm.SCORERS}, "delays": []}
    draws = np.zeros(1000, int)
    for name in proto["GOALS"]["test halves"]:
        match, h = name.split("_")
        half = 1 if h == "1st" else 2
        people, frames = fs.load_scan(name)
        ends = fm.defended_ends(people)
        share = fm.own_half_share(people, ends, frames)
        ev = st.events(match)
        ev = ev[ev.half == half].sort_values("frame")
        goals = ev[ev.label == "GOAL"][["frame", "side"]].reset_index(drop=True)
        side_team = {"left": next(t for t, e in ends.items() if e < 0),
                     "right": next(t for t, e in ends.items() if e > 0)}
        r = fm.score_half(people, ends, share, goals, ev.frame.to_numpy(), T, N,
                          team_of_side=side_team.get)
        minutes = (frames[-1] - frames[0]) / FPS / 60
        delays = [m["delay_s"] for m in r["matches"] if m["restartable"]]
        halves.append({"half": name, "minutes": round(minutes, 1), "ends": ends, **r,
                       "goal-time error s": [round(d - 22, 1) for d in delays]})
        tot["hits"] += r["hits"]
        tot["restartable"] += len(r["restartable_goals"])
        tot["false_runs"] += r["false_runs"]
        tot["minutes"] += minutes
        for k in fm.SCORERS:
            tot["team_right"][k] += r["team_right"][k]
        tot["delays"] += delays
        # random-time baseline: as many runs as this half had, at random samples after 60 s
        cand = np.asarray([f for f in frames if f >= 60 * FPS])
        rest = [g for g, ok in zip(goals.frame, fm.restartable(goals.frame.to_numpy(), ev.frame.to_numpy())) if ok]
        n_runs = len(r["runs"])
        for d in range(len(draws)):
            starts = np.sort(rng.choice(cand, size=n_runs, replace=False)) if n_runs else []
            m = fm.match_runs(list(starts), np.asarray(rest, dtype=float))
            draws[d] += len(m)
    res = {
        "protocol sha256": sha256(HERE / "protocol.json"),
        "frozen_goals sha256": sha256(frozen_path),
        "rule": {"T": T, "N": N, "team rule": rule},
        "primary": {"hits": tot["hits"], "restartable goals": tot["restartable"],
                    "hit rate wilson95": wilson(tot["hits"], tot["restartable"]),
                    "false runs": tot["false_runs"], "minutes": round(tot["minutes"], 1),
                    "false runs per 90 min": round(tot["false_runs"] / tot["minutes"] * 90, 2)},
        "secondary": {"team right (frozen rule)": tot["team_right"][rule],
                      "team right (all rules)": tot["team_right"],
                      "goal-time error s (run start - 22 s - goal)": {
                          "median": round(float(np.median(tot["delays"])) - 22, 1) if tot["delays"] else None,
                          "mean abs": round(float(np.mean(np.abs(np.asarray(tot["delays"]) - 22))), 1) if tot["delays"] else None},
                      "random-time baseline hits": {"mean": round(float(draws.mean()), 2),
                                                     "q975": int(np.quantile(draws, 0.975)),
                                                     "p >= observed": float((draws >= tot["hits"]).mean())}},
        "success": {"hits >= 7/8": tot["hits"] >= 7, "false runs <= 1": tot["false_runs"] <= 1,
                    "team right on all hits but one": tot["team_right"][rule] >= tot["hits"] - 1},
        "halves": halves,
    }
    res["success"]["all"] = all(res["success"].values())
    (HERE / "goals_test.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("rule", "primary", "secondary", "success")}, indent=1))


if __name__ == "__main__":
    main()
