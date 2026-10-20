"""Shots (protocol_shots.json), dev phase: the grid over the three shot
parameters on the dev units, the choice, and frozen_shots.json.

Units (protocol "dev data"):
  - the 8 processed non-M1 windows not in a test half (DEV_WINDOWS), whole
    windows: their attempts and their minutes as negatives;
  - the 26 gate attempts (M7 1st, M9 2nd) as clips -15 s / +10 s, merged
    (research/session_scratchpad/kaggle/shot_clips_dev.json): only the
    evaluated span counts (the cut adds 400 frames each side for v3's
    background, owner decision 2026-10-20).
Detections come from the Kaggle T4 runs (backend_note.json and its
addendum): GPU/<unit>/output_<unit>/ and GPU/<unit>/.cache/v3_balls_<unit>.*.

Per unit, here on CPU (the ball input 'v3s_link' of tune_rules.py):
  1. painted-spot marks dropped (src/ball_spots, as v5p2/marks.py) -> v3m;
  2. static points dropped (tune_rules.static_points, 1.5 x 4 px) -> v3s;
  3. the rows swapped into the unit's events_v1 run (score_soccernet
     .swap_ball_rows), merged with link selection (top_k 4, tau 0.10,
     tune_rules.LINK), to pitch metres with the release calibration
     -> metric.parquet (cached per unit).
Then for every grid cell: src/events.detect_events with events_v3.json's
frozen rules and the cell's three shot parameters; shots kept only toward
the goal the shooter's team attacks (each team's mean player x over the
unit: the lower mean defends x = 0 and attacks x = 105; the shot's goal is
the one its ball moves towards, x = 105 when end x > start x); inside a
clip's span only. Matched one to one (nearest first) at 2 s against the
unit's BAS SHOT and GOAL labels (inside the span for a clip).

Choice: the cell maximising F1 pooled over all dev units; ties: fewer
events, then the higher SHOT_MIN_KMH. Written to frozen_shots.json with its
sha256, together with the per-unit table, before any M1 clip is cut.

    VEO_UNSEAL=... python shots_dev.py --gpu .cache/gpu [--metric-only]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball/events_tune"))
import score_soccernet as sc  # noqa: E402
import tune_rules as tr  # noqa: E402
from detect_ball_heatmap import read_sidecar  # noqa: E402
from src import ball_spots as bs  # noqa: E402
from src import events as ev  # noqa: E402
from src import pixel_scale  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.unseal import unsealed  # noqa: E402

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol_shots.json"
CLIPS = ROOT / "research/session_scratchpad/kaggle/shot_clips_dev.json"
DEV_WINDOWS = ["st2_117093_1st_f015000", "st2_118575_1st_f015000", "st2_118576_1st_f030000",
               "st2_118578_1st_f030000", "st2_118578_2nd_f015000", "st2_128057_1st_f030000",
               "st2_128058_1st_f030000", "st2_132877_1st_f015000"]
SEALED = {"128057", "132831"}
LABELS = ("SHOT", "GOAL")
TOL_S = 2.0
GOAL_X = (0.0, 105.0)
REC = json.loads((ROOT / "research/session_scratchpad/ball/events_v3.json").read_text())
FROZEN = {k: v for k, v in REC["frozen"].items() if k in ev.EVENT_PARAMS}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol() -> dict:
    want = (HERE / "protocol_shots.json.sha256").read_text().split()[0]
    if sha256(PROTOCOL) != want:
        raise SystemExit("protocol_shots.json does not match its sha256")
    return json.loads(PROTOCOL.read_text())


def units() -> list[dict]:
    """The dev units. A clip span that overlaps a dev window of the same half
    (M7 1st: the clips of f31370 and f33829 inside st2_128057_1st_f030000) is
    masked out of that window, so each attempt and each second counts once:
    in its clip."""
    clips = json.loads(CLIPS.read_text())["clips"]
    out = []
    for w in DEV_WINDOWS:
        info = info_of(w)
        start, n = int(info["start_frame"]), int(info["frames"])
        masks = [[c["span_half"][0] - start, c["span_half"][1] - start] for c in clips
                 if c["match"] == str(info["match"]) and c["half"] == int(info["half"])
                 and c["span_half"][1] >= start and c["span_half"][0] < start + n]
        out.append({"name": w, "kind": "window", "span": None, "mask": masks})
    for c in clips:
        out.append({"name": c["name"], "kind": "clip", "span": c["span_local"], "mask": [],
                    "attempts_half": c["attempts_half"]})
    return out


def kept(t: float, span_s, mask_s) -> bool:
    if span_s and not span_s[0] <= t <= span_s[1]:
        return False
    return not any(a <= t <= b for a, b in mask_s)


def info_of(name: str) -> dict:
    for d in (ROOT / "data/soccertrack_v2", ROOT / "data/shot_clips/soccertrack_v2"):
        if (d / f"{name}.json").exists():
            return {**json.loads((d / f"{name}.json").read_text()), "dir": d}
    raise SystemExit(f"{name}: no .json in data/")


def metric_of(u: dict, gpu: Path) -> pd.DataFrame:
    """Steps 1-3, cached in GPU/<unit>/metric_v3s_link.parquet."""
    g = gpu / u["name"]
    cache = g / "metric_v3s_link.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    info = info_of(u["name"])
    match = str(info["match"])
    if match in SEALED and not unsealed(match, f"shots dev metric {u['name']}"):
        raise SystemExit(f"{u['name']}: sealed (set VEO_UNSEAL)")
    cal = st.calibration(match)
    rows_path = g / f".cache/v3_balls_{u['name']}.parquet"
    meta = read_sidecar(rows_path)
    n = len(meta["frames"])
    rows = pd.read_parquet(rows_path)
    rows = bs.drop_marks(rows, bs.spot_marks(rows, cal, n))
    keep = np.ones(len(rows), bool)
    for x, y in tr.static_points(rows, n):
        keep &= np.hypot(rows.px.to_numpy() - x, rows.py.to_numpy() - y) > 1.5 * tr.STATIC_PX
    run = g / f"output_{u['name']}"
    dst = sc.swap_ball_rows(run / "events_v1", run / "shots_v3s_link", rows[keep])
    with contextlib.redirect_stdout(io.StringIO()):
        merged = sc.merged_from_cached(dst, cal, select="link", top_k=4, tau=0.10,
                                       link_params=tr.LINK)
    metric = pixel_scale.to_pitch_metres(merged, lambda uv: st.image_to_pitch(uv, cal))
    metric.to_parquet(cache)
    return metric


def attacked_goal(metric: pd.DataFrame) -> dict:
    """team -> the x of the goal it attacks (the lower mean x defends x = 0)."""
    p = metric[(metric.cls == "player") & metric.team.isin(["team_A", "team_B"])]
    means = p.groupby("team").x.mean()
    if len(means) < 2:
        return {}
    low = means.idxmin()
    return {t: (GOAL_X[1] if t == low else GOAL_X[0]) for t in means.index}


def shots_of(metric: pd.DataFrame, cell: dict, attack: dict, span_s, mask_s) -> list[float]:
    with contextlib.redirect_stdout(io.StringIO()):
        evs = ev.detect_events(metric, absolute_pitch=True, ball_time_aware=True,
                               params={**FROZEN, **cell})
    out = []
    for e in evs:
        if e["event_type"] != "shot" or e["team"] not in attack:
            continue
        goal = GOAL_X[1] if e["end_location_x"] > e["location_x"] else GOAL_X[0]
        if goal != attack[e["team"]]:
            continue
        if not kept(e["timestamp_s"], span_s, mask_s):
            continue
        out.append(e["timestamp_s"])
    return out


def labels_of(u: dict, span_s, mask_s) -> list[float]:
    info = info_of(u["name"])
    lab = pd.read_csv(info["dir"] / f"{u['name']}_events.csv")
    t = lab[lab.label.isin(LABELS)].frame.to_numpy() / st.FPS
    return sorted(x for x in t.tolist() if kept(x, span_s, mask_s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=str(ROOT / ".cache/gpu"))
    ap.add_argument("--metric-only", action="store_true")
    args = ap.parse_args()
    proto = protocol()
    if (HERE / "frozen_shots.json").exists():
        raise SystemExit("frozen_shots.json exists: the dev choice is made")
    gpu = Path(args.gpu)
    us = units()
    data = {}
    for u in us:
        m = metric_of(u, gpu)
        span_s = [s / st.FPS for s in u["span"]] if u["span"] else None
        mask_s = [[a / st.FPS, b / st.FPS] for a, b in u["mask"]]
        data[u["name"]] = (m, attacked_goal(m), span_s, mask_s, labels_of(u, span_s, mask_s))
        print(f"{u['name']}: metric {len(m)} rows, {len(data[u['name']][4])} labels", flush=True)
    if args.metric_only:
        return
    grid = proto["free parameters (grid)"]
    keys = list(grid)
    table = []
    for vals in itertools.product(*grid.values()):
        cell = dict(zip(keys, vals))
        tp = fp = fn = 0
        per = {}
        for name, (m, attack, span_s, mask_s, labels) in data.items():
            preds = shots_of(m, cell, attack, span_s, mask_s)
            matched, f, n = sc.match_one_to_one(preds, labels, TOL_S)
            tp, fp, fn = tp + len(matched), fp + len(f), fn + len(n)
            per[name] = {"events": len(preds), "labels": len(labels), "tp": len(matched)}
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        table.append({"cell": cell, "tp": tp, "fp": fp, "fn": fn, "events": tp + fp,
                      "F1": f1, "per unit": per})
        print(f"{cell}: F1 {f1:.3f} (tp {tp}, fp {fp}, fn {fn})", flush=True)
    best = max(table, key=lambda r: (r["F1"], -r["events"], r["cell"]["SHOT_MIN_KMH"]))
    out = {
        "protocol": "protocol_shots.json sha256 " + sha256(PROTOCOL),
        "backend": "research/session_scratchpad/kaggle/backend_note.json (+ addendum1): Kaggle T4, fp32",
        "units": [{k: v for k, v in u.items()} for u in us],
        "frozen rules": FROZEN, "ball input": "v3s_link (tune_rules.py)",
        "chosen": best["cell"], "dev F1 at 2 s": best["F1"],
        "dev counts": {k: best[k] for k in ("tp", "fp", "fn", "events")},
        "dev per unit (chosen)": best["per unit"],
        "grid": [{k: r[k] for k in ("cell", "F1", "tp", "fp", "fn", "events")} for r in table],
    }
    path = HERE / "frozen_shots.json"
    path.write_text(json.dumps(out, indent=1))
    (HERE / "frozen_shots.json.sha256").write_text(sha256(path) + "\n")
    print("chosen", best["cell"], f"F1 {best['F1']:.3f}")


if __name__ == "__main__":
    main()
