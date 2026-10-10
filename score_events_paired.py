"""Pass + carry F1 of event runs against the ball actions, pooled over
windows, with chance and a paired block bootstrap (SOCCERTRACK_V2.md,
events protocol v3: the sweep's objective E3 and the claims E6).

A configuration is one run per window, output_<window>/<NAME>/events.json
(NAME, or a path template holding {w}). For each:

- passes and carries are matched to their labels separately (PASS, HIGH
  PASS, CROSS <- pass; DRIVE <- carry), one to one, nearest first, within
  0.5, 1 or 2 s (`score_st2_actions`), and pooled over both groups and all
  windows: F1 = 2 TP / (events + labels);
- chance: as many events per window and group at uniformly random times
  over the window, 200 draws, pooled the same way; its mean F1, and p, the
  share of draws doing at least as well;
- the events sweep's objective: F1 at 1 s minus chance.

With two configurations A B, the paired difference F1(B) - F1(A) at 1 s:
each window is cut into 10 s blocks (a match counts in its label's block,
an event in its own); the blocks are resampled with replacement 4000 times
(seed 0) and both F1 recomputed on each draw. "B is better" is shown at
one-sided alpha 0.025 when the 2.5th percentile is above 0.

    python score_events_paired.py --windows W1 W2 --configs events_coco events_heatmap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from score_soccernet import match_one_to_one
from score_st2_actions import FPS, GROUPS
from src.paths import DATA_DIR

SCORED = ("pass", "carry")
TOLERANCES_S = (0.5, 1.0, 2.0)
OBJECTIVE_TOL_S = 1.0
BLOCK_S = 10.0
N_BOOT = 4000
N_CHANCE = 200
SEED = 0
ALPHA = 0.025


def labels_of(window: str) -> dict[str, list[float]]:
    """Label times (s, window-local) per scored group."""
    labels = pd.read_csv(DATA_DIR / "soccertrack_v2" / f"{window}_events.csv")
    return {g: list(labels[labels.label.isin(GROUPS[g][0])].frame / FPS)
            for g in SCORED}


def duration_of(window: str, run_dir: Path | None = None) -> float:
    """Window length (s): from a run's clip.json, else its ground truth."""
    if run_dir is not None and (Path(run_dir) / "clip.json").exists():
        clip = json.loads((Path(run_dir) / "clip.json").read_text())
        return clip["n_frames"] / clip["fps"]
    gt = pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{window}_gt.parquet",
                         columns=["frame"])
    return (int(gt.frame.max()) + 1) / FPS


def preds_of(events: list[dict]) -> dict[str, list[float]]:
    """Event times per scored group."""
    return {g: [e["timestamp_s"] for e in events
                if e["event_type"] in GROUPS[g][1]] for g in SCORED}


def run_dir_of(config: str, window: str) -> Path:
    return Path(config.format(w=window) if "{w}" in config
                else f"output_{window}/{config}")


def load_events(config: str, window: str) -> list[dict]:
    return json.loads((run_dir_of(config, window) / "events.json").read_text())


def f1_of(tp, n_pred, n_true) -> float:
    return 2 * tp / max(n_pred + n_true, 1)


def counts(preds: dict, truths: dict, tol: float) -> tuple[int, int, int]:
    """(TP, events, labels) over the scored groups."""
    tp = sum(len(match_one_to_one(preds[g], truths[g], tol)[0])
             for g in SCORED)
    return (tp, sum(len(preds[g]) for g in SCORED),
            sum(len(truths[g]) for g in SCORED))


def chance_draws(preds: dict[str, dict], truths: dict[str, dict],
                 durations: dict[str, float], tol: float,
                 n: int = N_CHANCE, seed: int = SEED) -> np.ndarray:
    """(n, len(windows) + 1) F1 of random-time events: per window, then
    pooled (last column). Draw k places every window's events at once."""
    rng = np.random.default_rng(seed)
    windows = list(preds)
    out = np.empty((n, len(windows) + 1))
    for k in range(n):
        tot = np.zeros(3)
        for i, w in enumerate(windows):
            rand = {g: sorted(rng.uniform(0, durations[w], len(preds[w][g])))
                    for g in SCORED}
            c = counts(rand, truths[w], tol)
            out[k, i] = f1_of(*c)
            tot += c
        out[k, -1] = f1_of(*tot)
    return out


def score(events: dict[str, list], truths: dict[str, dict],
          durations: dict[str, float], tols=TOLERANCES_S,
          n_chance: int = N_CHANCE, seed: int = SEED) -> dict:
    """Per tolerance, pooled and per window: TP, events, labels, F1, chance
    F1 and p; and the objective (F1 - chance at OBJECTIVE_TOL_S)."""
    windows = list(events)
    preds = {w: preds_of(events[w]) for w in windows}
    out = {}
    for tol in tols:
        per = {w: counts(preds[w], truths[w], tol) for w in windows}
        pooled = tuple(int(sum(c[i] for c in per.values())) for i in range(3))
        draws = chance_draws(preds, truths, durations, tol, n_chance, seed)
        row = {}
        for i, (name, c) in enumerate([*per.items(), ("pooled", pooled)]):
            f1 = f1_of(*c)
            row[name] = {"TP": c[0], "events": c[1], "labels": c[2],
                         "F1": round(f1, 4),
                         "chance": round(float(draws[:, i].mean()), 4),
                         "p": round(float((draws[:, i] >= f1).mean()), 4)}
        out[f"{tol:g}s"] = row
    obj = out[f"{OBJECTIVE_TOL_S:g}s"]
    out["objective"] = {k: round(v["F1"] - v["chance"], 4)
                        for k, v in obj.items()}
    return out


def block_table(events: dict[str, dict[str, list]], truths: dict[str, dict],
                tol: float = OBJECTIVE_TOL_S,
                block_s: float = BLOCK_S) -> pd.DataFrame:
    """Rows (window, block) x columns (tp|np|nt, config): matches in their
    label's block, events and labels in their own. ``events`` is
    {config: {window: events}}; blocks are those any configuration uses."""
    rows = []
    for cfg, by_w in events.items():
        for w, ev in by_w.items():
            preds = preds_of(ev)
            for g in SCORED:
                t, p = truths[w][g], preds[g]
                m, _, _ = match_one_to_one(p, t, tol)
                rows += [(w, int(t[ti] // block_s), cfg, 1, 0, 0)
                         for _, ti, _ in m]
                rows += [(w, int(x // block_s), cfg, 0, 1, 0) for x in p]
                rows += [(w, int(x // block_s), cfg, 0, 0, 1) for x in t]
    df = pd.DataFrame(rows, columns=["w", "b", "cfg", "tp", "np", "nt"])
    return (df.groupby(["w", "b", "cfg"])[["tp", "np", "nt"]].sum()
            .unstack("cfg", fill_value=0))


def paired(events_a: dict[str, list], events_b: dict[str, list],
           truths: dict[str, dict], tol: float = OBJECTIVE_TOL_S,
           block_s: float = BLOCK_S, n: int = N_BOOT,
           seed: int = SEED) -> dict:
    """F1(B) - F1(A), pooled, with its block-bootstrap percentiles."""
    tab = block_table({"a": events_a, "b": events_b}, truths, tol, block_s)
    a = tab.xs("a", axis=1, level="cfg")[["tp", "np", "nt"]].to_numpy(float)
    b = tab.xs("b", axis=1, level="cfg")[["tp", "np", "nt"]].to_numpy(float)
    nb = len(tab)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for k in range(n):
        pick = rng.integers(0, nb, nb)
        diffs[k] = f1_of(*b[pick].sum(0)) - f1_of(*a[pick].sum(0))
    lo, hi = np.percentile(diffs, [100 * ALPHA, 100 * (1 - ALPHA)])
    return {"diff": round(f1_of(*b.sum(0)) - f1_of(*a.sum(0)), 4),
            "lower": round(float(lo), 4), "upper": round(float(hi), 4),
            "blocks": nb, "draws": n, "rejected": bool(lo > 0)}


def _check():
    """Synthetic (labels 4 s apart): perfect events score F1 1 and beat
    chance; the labels shifted by 0.7 s match at 1 s but not at 0.5 s; the paired
    difference of a run with itself is exactly 0; B = perfect against A =
    nothing rejects."""
    rng = np.random.default_rng(3)
    truths = {w: {g: (np.arange(30) * 4 + rng.uniform(0, 1, 30)).tolist()
                  for g in SCORED} for w in ("w1", "w2")}
    durations = {"w1": 120.0, "w2": 120.0}

    def as_events(times, shift=0.0):
        return [{"event_type": et, "timestamp_s": t + shift}
                for g, et in (("pass", "pass"), ("carry", "carry"))
                for t in times[g]]

    perfect = {w: as_events(truths[w]) for w in truths}
    late = {w: as_events(truths[w], 0.7) for w in truths}
    none = {w: [] for w in truths}
    s = score(perfect, truths, durations, n_chance=20)
    assert s["1s"]["pooled"]["F1"] == 1.0 and s["objective"]["pooled"] > 0.3
    assert s["1s"]["pooled"]["p"] == 0.0
    s = score(late, truths, durations, tols=(0.5, 1.0), n_chance=5)
    assert s["1s"]["pooled"]["TP"] == 120 and s["0.5s"]["pooled"]["TP"] < 120
    same = paired(late, late, truths, n=200)
    assert same["diff"] == 0 and same["lower"] == 0 == same["upper"]
    win = paired(none, perfect, truths, n=200)
    assert win["diff"] == 1.0 and win["rejected"]
    print("score_events_paired check ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--windows", nargs="+")
    ap.add_argument("--configs", nargs="+",
                    help="run directory names under output_<window>/, or "
                         "templates with {w}; with exactly two, their "
                         "paired difference (second minus first)")
    ap.add_argument("--json", help="write the scores here")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        _check()
        return
    if not args.windows or not args.configs:
        ap.error("--windows and --configs are required")

    truths = {w: labels_of(w) for w in args.windows}
    events = {c: {w: load_events(c, w) for w in args.windows}
              for c in args.configs}
    durations = {w: duration_of(w, run_dir_of(args.configs[0], w))
                 for w in args.windows}
    report = {"windows": args.windows, "configs": {}}
    for c in args.configs:
        s = score(events[c], truths, durations)
        report["configs"][c] = s
        print(f"\n{c}")
        print(f"  {'tol':>5} {'window':<26} {'TP':>4} {'events':>6} "
              f"{'labels':>6} {'F1':>6} {'chance':>6} {'p':>6}")
        for tol in TOLERANCES_S:
            for name, r in s[f"{tol:g}s"].items():
                print(f"  {tol:>5g} {name:<26} {r['TP']:>4} {r['events']:>6} "
                      f"{r['labels']:>6} {r['F1']:>6.3f} {r['chance']:>6.3f} "
                      f"{r['p']:>6.3f}")
        print(f"  objective (F1 - chance at {OBJECTIVE_TOL_S:g} s): "
              f"{s['objective']}")
    if len(args.configs) == 2:
        a, b = args.configs
        d = paired(events[a], events[b], truths)
        report["paired"] = {"a": a, "b": b, **d}
        print(f"\nF1 {b} - {a} at {OBJECTIVE_TOL_S:g} s: {d['diff']:+.3f}, "
              f"{100 * ALPHA:g}-{100 * (1 - ALPHA):g}%: "
              f"[{d['lower']:+.3f}, {d['upper']:+.3f}] ({d['blocks']} blocks "
              f"of {BLOCK_S:g} s, {d['draws']} draws, seed {SEED}); "
              f"{'rejected' if d['rejected'] else 'not rejected'} at "
              f"one-sided {ALPHA}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
