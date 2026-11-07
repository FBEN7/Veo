"""Goals protocol (protocol.json, GOALS): the dev selection on ground truth, and
the extra dev check on the processed windows.

1. Dev selection on ground truth ("dev, ground truth": 14 halves). Per half,
   the GSR at 1 fps (frames that are multiples of 25), people with role
   player or goalkeeper, team = GSR side, the end each side defends from its
   mean x over the half (formation.defended_ends). Every cell of the grid
   (T x N x team rule) is scored with formation.score_half against the BAS
   GOALs: hits (matched restartable goals), false runs (unmatched runs) and,
   per team rule, hits with the scoring team right (BAS GOAL side = the
   rule's GSR side). T, N and the team rule are chosen by the protocol's
   ordering: most hits, then fewest false runs, then the larger T, then the
   larger N. -> formation_gt_choice.json (the full grid table, the chosen
   cell's per-half runs and matches, goal-time error, random-time baseline).

2. Extra dev check ("false runs on the 9 processed non-M1 windows
   (tracks_teams, 1 fps), reported, not used to choose"): the chosen rule on
   output_<w>/tracks_teams.parquet, players labelled team_A / team_B, feet
   (px, py) to pitch metres with st.image_to_pitch and the release
   calibration (calib_override is not imported), one sample a second.
   -> formation_windows_check.json.

Readings where the protocol is silent (the rule's own are in formation.py):
- The team rule changes neither hits nor false runs, so the ordering leaves
  the two rules tied at the chosen (T, N); the tie is broken by the number
  of hits with the scoring team right at that cell (the only quantity the
  rule changes), then by grid order (kick_spot first).
- False runs per 90 min use the halves' sampled length (number of 1-s
  samples / 60).
- Goal-time error per hit: (run start - 22 s) - goal frame, in seconds.
- Random-time baseline at the chosen cell: formation.random_baseline, 1000
  draws, seed 0; reported with P(random hits >= observed).
- Windows: the window has no 60-s exclusion of its own -- run starts are
  half-video frames (window start_frame + window frame), so the half-start
  exclusion applies as on the whole half (no non-M1 window starts before
  frame 15000). The end each team defends is its mean x over the window
  (only the window is processed). The tracks are at stride 2 (even window
  frames), so the sample of second k is the latest tracked frame at or
  before window frame 25 k, labelled with half frame start_frame + 25 k.
  Runs are matched to all BAS GOALs of the half; BAS 'left' is taken as the
  team defending the x < 0 end (true of GSR 'left' on all 14 dev GT halves,
  checked here and written to the choice JSON).
- Test hygiene overrides the protocol's count of 9: of the 9 non-M1 output_
  windows one, output_st2_132831_2nd_f030000, is in the test half 132831
  half 2, whose detections may not be read, so the check runs on 8.

    PYTHONPATH=/home/user/Veo python research/session_scratchpad/shots_goals/formation_gt.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/user/Veo")
HERE = ROOT / "research/session_scratchpad/shots_goals"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from src import soccertrack_v2 as st  # noqa: E402
import formation as fm  # noqa: E402

PROTOCOL = HERE / "protocol.json"
GSR = ROOT / ".cache/soccertrack_v2/gsr"
CHOICE = HERE / "formation_gt_choice.json"
WINDOWS = HERE / "formation_windows_check.json"


def load_protocol() -> dict:
    blob = PROTOCOL.read_bytes()
    want = (HERE / "protocol.json.sha256").read_text().split()[0]
    got = hashlib.sha256(blob).hexdigest()
    assert got == want, f"protocol.json changed: {got} != {want}"
    return json.loads(blob)


def split(name: str) -> tuple[str, int]:
    match, h = name.split("_")
    return match, {"1st": 1, "2nd": 2}[h]


def guard(name: str, test: list[str]):
    """Never touch a test half (or any M1 half)."""
    if name in test or name.startswith("117092"):
        raise RuntimeError(f"{name} is a test half: not read")


def gt_people(name: str) -> pd.DataFrame:
    """1-fps GSR people of a dev half: frame, team (= side), x, y."""
    g = pd.read_parquet(GSR / f"{name}.parquet",
                        columns=["frame", "role", "side", "x", "y"])
    g = g[(g.frame % fm.SAMPLE == 0) & g.role.isin(["player", "goalkeeper"])
          & g.side.notna() & g.x.notna()]
    return g.rename(columns={"side": "team"})[["frame", "team", "x", "y"]]


def bas_half(match: str, half: int) -> tuple[pd.DataFrame, np.ndarray]:
    ev = st.events(match)
    e = ev[ev.half == half].sort_values("frame", kind="stable")
    goals = e[e.label == "GOAL"][["frame", "side"]].reset_index(drop=True)
    return goals, e.frame.to_numpy()


def prepare_gt(names: list[str], test: list[str]) -> list[dict]:
    halves = []
    for name in names:
        guard(name, test)
        match, half = split(name)
        people = gt_people(name)
        ends = fm.defended_ends(people)
        last = int(people.frame.max())
        grid = np.arange(0, last + 1, fm.SAMPLE)
        share = fm.own_half_share(people, ends, grid)
        goals, acts = bas_half(match, half)
        halves.append({"name": name, "people": people, "ends": ends,
                       "share": share, "goals": goals, "acts": acts,
                       "last": last, "minutes": len(grid) / 60,
                       "mean_x": people.groupby("team").x.mean().round(2).to_dict()})
        print(f"{name}: {len(grid)} samples, ends {ends}, "
              f"{len(goals)} goals, share median {share.median():.3f}", flush=True)
    return halves


def grid_table(halves: list[dict], grid: dict) -> tuple[list[dict], dict]:
    minutes = sum(h["minutes"] for h in halves)
    table, per = [], {}
    for T in grid["T"]:
        for N in grid["N"]:
            res = [fm.score_half(h["people"], h["ends"], h["share"], h["goals"],
                                 h["acts"], T, N) for h in halves]
            per[(T, N)] = res
            row = {"T": T, "N": N,
                   "hits": sum(r["hits"] for r in res),
                   "restartable_goals": sum(len(r["restartable_goals"]) for r in res),
                   "false_runs": sum(r["false_runs"] for r in res),
                   "runs": sum(len(r["runs"]) for r in res),
                   "matched_non_restartable": sum(r["matched_non_restartable"]
                                                  for r in res)}
            row["false_runs_per_90"] = round(row["false_runs"] * 90 / minutes, 3)
            for rule in grid["team rule"]:
                row[f"team_right_{rule}"] = sum(r["team_right"][rule] for r in res)
            table.append(row)
    return table, per


def choose(table: list[dict], rules: list[str]) -> dict:
    """Most hits, then fewest false runs, then the larger T, then the larger
    N; then the team rule with more team-right hits at that cell (grid order
    on a tie)."""
    best = min(table, key=lambda r: (-r["hits"], r["false_runs"], -r["T"], -r["N"]))
    rule = max(rules, key=lambda k: (best[f"team_right_{k}"], -rules.index(k)))
    return {"T": best["T"], "N": best["N"], "team rule": rule}


def details(halves: list[dict], res: list[dict], rule: str) -> tuple[list, dict]:
    out, errs = [], []
    for h, r in zip(halves, res):
        for m in r["matches"]:
            if m["restartable"]:
                errs.append((m["run"][0] - 22 * fm.FPS - m["goal_frame"]) / fm.FPS)
        out.append({"half": h["name"], "ends": h["ends"], "mean_x": h["mean_x"],
                    "minutes": round(h["minutes"], 2),
                    "runs_s": [[a / fm.FPS, b / fm.FPS] for a, b in r["runs"]],
                    "restartable_goals": r["restartable_goals"],
                    "missed_by_construction": r["missed_by_construction"],
                    "matches": [{k: m[k] for k in ("goal_frame", "goal_side",
                                                   "restartable", "run", "delay_s",
                                                   "kick_spot", "attack_end")}
                                | {"team_right": m[rule + "_right"]}
                                for m in r["matches"]],
                    "hits": r["hits"], "false_runs": r["false_runs"],
                    "false_runs_s": [[a / fm.FPS, b / fm.FPS]
                                     for a, b in r["false_run_list"]]})
    e = np.asarray(errs)
    err = {"n": len(e), "median_s": float(np.median(e)) if len(e) else None,
           "mean_abs_s": float(np.abs(e).mean()) if len(e) else None,
           "min_s": float(e.min()) if len(e) else None,
           "max_s": float(e.max()) if len(e) else None}
    return out, err


def baseline(halves: list[dict], res: list[dict], observed: int) -> dict:
    hs = [{"n_runs": len(r["runs"]), "goals": h["goals"].frame.to_numpy(),
           "restartable": fm.restartable(h["goals"].frame.to_numpy(), h["acts"]),
           "last": h["last"]} for h, r in zip(halves, res)]
    draws = fm.random_baseline(hs, 1000, seed=0)
    return {"draws": 1000, "seed": 0, "mean_hits": float(draws.mean()),
            "q025": float(np.quantile(draws, 0.025)),
            "q975": float(np.quantile(draws, 0.975)), "max": int(draws.max()),
            "p_ge_observed": float((draws >= observed).mean())}


def run_gt(proto: dict) -> dict:
    goals = proto["GOALS"]
    names, test, grid = goals["dev, ground truth"], goals["test halves"], goals["grid"]
    halves = prepare_gt(names, test)
    left_low = {h["name"]: h["ends"].get("left") == -1 for h in halves}
    table, per = grid_table(halves, grid)
    pick = choose(table, grid["team rule"])
    res = per[(pick["T"], pick["N"])]
    cell = next(r for r in table if r["T"] == pick["T"] and r["N"] == pick["N"])
    halves_out, err = details(halves, res, pick["team rule"])
    out = {
        "protocol sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "script": "formation_gt.py (rule: formation.py)",
        "formation.py sha256": hashlib.sha256((HERE / "formation.py").read_bytes()).hexdigest(),
        "dev halves": names,
        "minutes": round(sum(h["minutes"] for h in halves), 2),
        "ordering": "hits desc, false runs asc, T desc, N desc; team rule by "
                    "team-right hits at the chosen (T, N), grid order on a tie",
        "choice": pick,
        "chosen cell": cell,
        "GSR left defends x < 0 (per half)": left_low,
        "goal-time error (run start - 22 s - goal)": err,
        "random-time baseline (chosen cell)": baseline(halves, res, cell["hits"]),
        "grid": table,
        "per half (chosen cell)": halves_out,
    }
    CHOICE.write_text(json.dumps(out, indent=1, default=_json))
    print(f"chosen {pick}: hits {cell['hits']}/{cell['restartable_goals']}, "
          f"false runs {cell['false_runs']} ({cell['false_runs_per_90']}/90 min), "
          f"team right kick_spot {cell['team_right_kick_spot']}, "
          f"attack_end {cell['team_right_attack_end']} -> {CHOICE}", flush=True)
    return out


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


# ---- Extra dev check: processed windows ---------------------------------------

def window_people(w: Path, info: dict) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """1-fps team_A / team_B players of a processed window in pitch metres;
    frames are half-video frames of the nominal seconds."""
    tt = pd.read_parquet(w / "tracks_teams.parquet",
                         columns=["frame", "cls", "px", "py", "team"])
    tt = tt[(tt.cls == "player") & tt.team.isin(["team_A", "team_B"])]
    tracked = np.sort(tt.frame.unique())
    want = np.arange(0, int(info["frames"]), fm.SAMPLE)
    i = np.searchsorted(tracked, want, side="right") - 1
    ok = (i >= 0) & (want - tracked[np.clip(i, 0, None)] < fm.SAMPLE)
    pick = dict(zip(tracked[i[ok]], want[ok]))
    p = tt[tt.frame.isin(list(pick))].copy()
    cal = st.calibration(info["match"])
    xy = st.image_to_pitch(np.c_[p.px.to_numpy(float), p.py.to_numpy(float)], cal)
    p["x"], p["y"] = xy[:, 0], xy[:, 1]
    stats = {"tracked_frames": len(tracked), "samples": int(ok.sum()),
             "person_samples": len(p),
             "nan_metres": int(np.isnan(xy[:, 0]).sum()),
             "outside_pitch": int(((np.abs(xy[:, 0]) > 52.5 + 2)
                                   | (np.abs(xy[:, 1]) > 34 + 2)).sum())}
    p["frame"] = int(info["start_frame"]) + p.frame.map(pick)
    grid = int(info["start_frame"]) + want
    return p[p.x.notna()][["frame", "team", "x", "y"]], grid, stats


def run_windows(proto: dict, pick: dict) -> dict:
    test = proto["GOALS"]["test halves"]
    T, N, rule = pick["T"], pick["N"], pick["team rule"]
    rows, skipped = [], []
    for w in sorted(ROOT.glob("output_st2_*")):
        name = w.name.removeprefix("output_st2_")
        half_name = "_".join(name.split("_")[:2])
        if half_name.startswith("117092"):
            continue                                   # M1: not a dev window
        if half_name in test:
            skipped.append({"window": w.name, "reason": "test half "
                            f"{half_name}: its detections may not be read"})
            continue
        guard(half_name, test)
        info = json.loads((ROOT / "data/soccertrack_v2" / f"st2_{name}.json").read_text())
        people, grid, stats = window_people(w, info)
        ends = fm.defended_ends(people)
        share = fm.own_half_share(people, ends, grid)
        goals, acts = bas_half(info["match"], int(info["half"]))
        side_team = {"left": next(t for t, e in ends.items() if e == -1),
                     "right": next(t for t, e in ends.items() if e == +1)}
        r = fm.score_half(people, ends, share, goals, acts, T, N,
                          team_of_side=side_team.get)
        lo, hi = int(grid[0]), int(grid[-1])
        in_win = [int(g) for g in goals.frame if lo <= g <= hi]
        minutes = len(grid) / 60
        rows.append({
            "window": w.name, "start_frame": int(info["start_frame"]),
            "minutes": round(minutes, 2), **stats, "ends": ends,
            "mean_x": people.groupby("team").x.mean().round(2).to_dict(),
            "share_median": float(share.median()),
            "share_ge_T": float((share >= T).mean()),
            "share_max": float(share.max()),
            "runs_half_s": [[a / fm.FPS, b / fm.FPS] for a, b in r["runs"]],
            "false_runs": r["false_runs"],
            "false_runs_half_s": [[a / fm.FPS, b / fm.FPS]
                                  for a, b in r["false_run_list"]],
            "goals_in_window": in_win, "hits": r["hits"],
            "matches": [{"goal_frame": m["goal_frame"], "run": m["run"],
                         "delay_s": m["delay_s"], "goal_side": m["goal_side"],
                         "scorer": m[rule],
                         "team_right": m[rule + "_right"]} for m in r["matches"]]})
        print(f"{w.name}: {len(r['runs'])} runs, {r['false_runs']} false, "
              f"{r['hits']} hits, goals in window {in_win}, "
              f"share median {share.median():.3f} max {share.max():.3f}", flush=True)
    minutes = sum(r["minutes"] for r in rows)
    false = sum(r["false_runs"] for r in rows)
    out = {
        "protocol sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "rule": pick, "calibration": "release (st.calibration)",
        "note": "reported, not used to choose (protocol: extra dev check)",
        "windows": len(rows), "skipped": skipped,
        "minutes": round(minutes, 2), "false_runs": false,
        "false_runs_per_90": round(false * 90 / minutes, 3) if minutes else None,
        "hits": sum(r["hits"] for r in rows),
        "per window": rows,
    }
    WINDOWS.write_text(json.dumps(out, indent=1, default=_json))
    print(f"windows: {len(rows)} ({minutes:.1f} min), false runs {false} "
          f"({out['false_runs_per_90']}/90 min) -> {WINDOWS}", flush=True)
    return out


def main():
    proto = load_protocol()
    gt = run_gt(proto)
    run_windows(proto, gt["choice"])


if __name__ == "__main__":
    main()
