"""Platt calibration of a ball heatmap v2 checkpoint, and its tau, on the
tuning windows U (protocol v3, decisions (b) and (c)).

The motion model's raw p = sigmoid(logit) ranks candidates but is the
probability of nothing in particular (it was trained with a smooth
maximum over bags, focal negatives and drop-worst). Platt scaling maps
it to one:

    p_cal = sigmoid(a * logit + b)

with (a, b) the maximum-likelihood logistic regression (Newton's method)
of a label on the logit, over every saved candidate of the checkpoint
(top 8 a frame, p_raw >= 0.05; `eval_ball_detector.py --v2 CKPT --save
U.json`) at the U frames whose answer is known:

- **pass frames** (four per completed ground pass): positive when the
  candidate's ground point (u, v + size / 2) meets FF-auto's geometry for
  that frame: within 4.0 m of the passer->receiver segment, projection
  t in [0, 1] within 0.30 of the frame's s, 1.0 m or more from every
  ground-truth person (`eval_ball_detector.ff_hit`); a frame without
  people is skipped;
- **PASS / DRIVE frames**: positive when within 1.5 m of the actor.

So p_cal reads "the chance that this candidate is where those rules put
the ball". A positive slope keeps the ranking; a non-positive one, or
labels the logit separates perfectly, is refused.

Also written: the reliability table (10 equal-width bins of p_cal) and
the ECE on the same candidates (in-sample), and tau, label-free: the
smallest p_cal on the grid 0.05, 0.10, ..., 0.95 giving at most 1.0
candidates with p_cal >= tau per frame on U's random frames. Only grid
values whose count the saved rows give exactly are eligible: rows below
p_raw 0.05 were not saved, so the threshold must lie above that logit,
and a frame whose 8 saved rows all pass might have had a 9th.

    python calibrate_ball_heatmap.py --scores U.json [U2.json ...] \\
        --ckpt .cache/ball_v2.4000.pt [--out CKPT.calib.json] [--name NAME]

writes `<ckpt>.calib.json` ({"a", "b", "tau", ...}, read by
`ball_heatmap.load_calibration`) and prints a, b, tau, the table, the
ECE and the counts on the grid. The detector is the checkpoint's file
name in the scores (`--name` otherwise). Fitting reads U windows (match
118578) only: never a report match (117093, 132877) or a test match, and
a match of the checkpoint's training provenance, or any other match,
only with `--allow-seen` (smoke tests). An existing calibration is not
overwritten without `--force`. The motion scales the scores were made
with (their metadata's "sigma", `eval_ball_detector.py --v2`, or the
parquet sidecars' under `--candidates`) must be the checkpoint's own
(its provenance), and are recorded; a checkpoint sha256 the scores
record must be this checkpoint's.

`--evaluate CALIB --out FILE` applies a frozen calibration to other
scores (V's sealed file, for the ECE of p_cal): the same table, ECE and
candidates per frame at its tau go to FILE -- never to an input file --
and only frame and window counts are printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import eval_ball_detector as ev
from src import ball_heatmap as bh
from src import soccertrack_v2 as st

U_MATCHES = {"118578"}                  # M6: tuning, the only place to fit
REPORT_MATCHES = {"117093", "132877"}   # M2, M10: V, never fitted on
ACTOR_R = 1.5                           # m from the actor at PASS / DRIVE
BINS = 10
TAU_GRID = tuple(round(0.05 * k, 2) for k in range(1, 20))
MAX_PER_FRAME = 1.0


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, float)))


def _sha256(path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- Samples ------------------------------------------------------------------------

def _rows(rows) -> np.ndarray:
    a = np.asarray(rows, float)
    if a.ndim != 2 or not len(a):
        return np.zeros((0, 5))
    if a.shape[1] < 5:
        raise ValueError("candidate rows without a logit: not a v2 detector")
    return a


def samples(results, name: str, cal_of) -> dict:
    """The labelled candidates of detector `name` in scores results
    (`eval_ball_detector.load_scores`, one dict per window): arrays
    "logit", "label" (0 / 1), "kind" ("pass" | "action"), "window", and
    the frame counts. `cal_of(match)` gives a match's calibration."""
    out = []                                # (logit, label, kind, window)
    frames = {"pass": 0, "action": 0, "pass without people": 0,
              "pass not measured": 0, "action not measured": 0}
    for r in results:
        if name not in r:
            continue
        cands = r[name]["candidates"]
        cal = cal_of(str(r["match"]))
        people = r.get("people", {})
        for p in r["frames"]["passes"]:
            for q, f in enumerate(p["frames"]):
                if str(f) not in cands:
                    frames["pass not measured"] += 1
                    continue
                if str(f) not in people:
                    frames["pass without people"] += 1
                    continue
                frames["pass"] += 1
                rows = _rows(cands[str(f)])
                ppl = np.asarray(people[str(f)], float).reshape(-1, 2)
                for g, row in zip(ev.ground(rows, cal), rows):
                    ok = np.isfinite(g).all() and ev.ff_hit(g, p, q, ppl)
                    out.append((row[4], int(ok), "pass", r["window"]))
        for a in r["frames"]["actions"]:
            f = int(a["frame"])
            if a.get("x") is None or a.get("y") is None:
                continue
            if str(f) not in cands:
                frames["action not measured"] += 1
                continue
            frames["action"] += 1
            rows = _rows(cands[str(f)])
            for g, row in zip(ev.ground(rows, cal), rows):
                ok = (np.isfinite(g).all()
                      and np.hypot(g[0] - a["x"], g[1] - a["y"]) <= ACTOR_R)
                out.append((row[4], int(ok), "action", r["window"]))
    z, y, kind, win = zip(*out) if out else ((), (), (), ())
    return {"logit": np.asarray(z, float), "label": np.asarray(y, int),
            "kind": np.asarray(kind, dtype=object), "window": np.asarray(win, dtype=object),
            "frames": frames}


def random_logits(results, name: str) -> list[np.ndarray]:
    """The logits (best first) at every measured random frame."""
    out = []
    for r in results:
        if name not in r:
            continue
        cands = r[name]["candidates"]
        for f in r["frames"]["random"]:
            if str(f) in cands:
                out.append(_rows(cands[str(f)])[:, 4])
    return out


# ---- Fit and checks --------------------------------------------------------------

def fit_platt(z, y, iters: int = 100) -> tuple[float, float, dict]:
    """(a, b) maximising the likelihood of labels `y` under p =
    sigmoid(a z + b): Newton's method with step halving, from the
    identity (1, 0). It exists (finite) unless a threshold on z
    separates the labels; raises then, when the labels are all one
    class, and when a <= 0."""
    z, y = np.asarray(z, float), np.asarray(y, float)
    if not 0 < y.sum() < len(y):
        raise ValueError(f"Platt fit needs positives and negatives "
                         f"({int(y.sum())} of {len(y)})")
    pos, neg = z[y == 1], z[y == 0]
    if pos.min() >= neg.max() or neg.min() >= pos.max():
        raise ValueError("the logit separates the labels: no finite Platt fit")
    X = np.c_[z, np.ones_like(z)]

    def nll(w):
        s = X @ w
        return float(np.sum(np.logaddexp(0.0, s) - y * s))

    w = np.array([1.0, 0.0])
    f = nll(w)
    for it in range(1, iters + 1):
        p = _sigmoid(X @ w)
        grad = X.T @ (p - y)
        hess = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(hess, grad)
        lam = 1.0
        while True:
            w_new = w - lam * step
            f_new = nll(w_new)
            if f_new <= f + 1e-12 * abs(f) or lam < 1e-6:
                break
            lam /= 2
        w, f = w_new, f_new
        if np.abs(lam * step).max() < 1e-10:
            break
    else:
        raise ValueError("Platt fit did not converge")
    a, b = float(w[0]), float(w[1])
    if a <= 0:
        raise ValueError(f"Platt slope a = {a:.4f} <= 0: the logit does not "
                         f"rank the labels")
    return a, b, {"iterations": it, "log loss": round(f / len(z), 6)}


def reliability(p, y, bins: int = BINS) -> tuple[list, float | None]:
    """Equal-width bins of p: [lo, hi), n, mean p, positive rate; and the
    ECE, sum over bins of n / N |rate - mean p|."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    k = np.minimum((p * bins).astype(int), bins - 1)
    table, ece = [], 0.0
    for i in range(bins):
        m = k == i
        n = int(m.sum())
        table.append({"bin": [round(i / bins, 3), round((i + 1) / bins, 3)], "n": n,
                      "p_cal": round(float(p[m].mean()), 4) if n else None,
                      "positive": round(float(y[m].mean()), 4) if n else None})
        if n:
            ece += n / len(p) * abs(float(y[m].mean()) - float(p[m].mean()))
    return table, (round(ece, 4) if len(p) else None)


def per_frame(logits: list[np.ndarray], calib: dict, grid=TAU_GRID,
              top_k: int = ev.TOP_K, floor: float = ev.V2_TAU) -> list[dict]:
    """Candidates per frame with p_cal >= tau on frames given by their
    saved logits, for each tau of `grid`, and whether the saved rows give
    the count exactly: tau above the p_cal of the save floor (a hair
    above logit(floor), for float32's sigmoid), and no frame with all
    `top_k` rows passing."""
    edge = float(bh.p_cal(_logit(floor) + 1e-4, calib))
    out = []
    for tau in grid:
        n = [int((bh.p_cal(z, calib) >= tau).sum()) for z in logits]
        capped = any(len(z) >= top_k and bh.p_cal(z, calib).min() >= tau
                     for z in logits)
        out.append({"tau": tau,
                    "candidates per frame": (round(float(np.mean(n)), 4)
                                             if n else None),
                    "exact": bool(edge <= tau and not capped)})
    return out


def choose_tau(grid_counts: list[dict], max_per_frame: float = MAX_PER_FRAME):
    """The smallest exact grid tau with at most `max_per_frame`, or None."""
    for row in grid_counts:
        c = row["candidates per frame"]
        if row["exact"] and c is not None and c <= max_per_frame:
            return row["tau"]
    return None


def _by_kind(p, y, kind) -> dict:
    return {k: {"candidates": int((kind == k).sum()),
                "positives": int(y[kind == k].sum()),
                "ece": reliability(p[kind == k], y[kind == k])[1]}
            for k in ("pass", "action")}


# ---- Guards and the two modes ----------------------------------------------------

def check_windows(results, ckpt, allow_seen: bool = False):
    """Fit only on U: never a report or test match; a training match of the
    checkpoint, or any non-U match, only with `allow_seen`."""
    for r in results:
        m = str(r["match"])
        if m in REPORT_MATCHES | ev.TEST_MATCHES:
            raise ValueError(f"{r['window']}: match {m} is never calibrated on")
        if m not in U_MATCHES and not allow_seen:
            raise ValueError(f"{r['window']}: match {m} is not a U match; "
                             f"only with --allow-seen")
        bh.check_unseen(ckpt, m, allow_seen)


def calibrate(results, name: str, cal_of) -> dict:
    """Platt (a, b), reliability and ECE on the labelled candidates, and
    tau from the random frames."""
    s = samples(results, name, cal_of)
    if not len(s["logit"]):
        raise ValueError(f"no candidates of {name} at pass or action frames")
    a, b, info = fit_platt(s["logit"], s["label"])
    # Rounded as written, so that the file's (a, b) give its table exactly.
    calib = {"a": round(a, 6), "b": round(b, 6)}
    p = bh.p_cal(s["logit"], calib)
    table, ece = reliability(p, s["label"])
    rnd = random_logits(results, name)
    grid = per_frame(rnd, calib)
    return {**calib, "tau": choose_tau(grid),
            "candidates": int(len(p)), "positives": int(s["label"].sum()),
            "by kind": _by_kind(p, s["label"], s["kind"]), "frames": s["frames"],
            "fit": info, "reliability": table, "ece": ece,
            "random frames": len(rnd), "tau grid": grid}


def evaluate(results, name: str, calib: dict, cal_of) -> dict:
    """A frozen calibration on other scores: reliability, ECE, and the
    candidates per frame at its tau on the random frames."""
    s = samples(results, name, cal_of)
    p = bh.p_cal(s["logit"], calib)
    table, ece = reliability(p, s["label"])
    rnd = random_logits(results, name)
    at = (per_frame(rnd, calib, grid=(calib["tau"],))[0]
          if calib.get("tau") is not None else None)
    return {"a": calib["a"], "b": calib["b"], "tau": calib.get("tau"),
            "candidates": int(len(p)), "positives": int(s["label"].sum()),
            "by kind": _by_kind(p, s["label"], s["kind"]), "frames": s["frames"],
            "reliability": table, "ece": ece, "random frames": len(rnd),
            "at tau": at}


def _rules() -> dict:
    return {"FF_W": ev.FF_W, "FF_DT": ev.FF_DT, "FF_R": ev.FF_R, "ACTOR_R": ACTOR_R,
            "TOP_K": ev.TOP_K, "V2_TAU": ev.V2_TAU, "BINS": BINS,
            "TAU_GRID": list(TAU_GRID), "MAX_PER_FRAME": MAX_PER_FRAME}


def _detector(results, meta: dict, ckpt, name: str | None) -> str:
    name = name or Path(ckpt).name
    have = sorted({k for r in results for k, v in r.items()
                   if isinstance(v, dict) and "candidates" in v})
    if name not in have:
        raise ValueError(f"detector {name!r} not in the scores (have {have}); "
                         f"use --name")
    if meta.get(name, {}).get("kind") == "v2-motion-zero":
        raise ValueError(f"{name}: the motion-zeroed diagnostic is not calibrated")
    return name


def _sigma(meta: dict, ckpt) -> dict | None:
    """The motion scales the scores' detector ran with ("sigma" in the
    scores' metadata: `eval_ball_detector.py --v2`, or each window's
    parquet sidecar under `--candidates`; else the checkpoint's
    provenance). Refuses scores whose scales are not the checkpoint's own
    (those of an RGB-only run do not matter), or which record another
    checkpoint's sha256: they are not this checkpoint's candidates."""
    runs = [meta, *(w for w in (meta.get("windows") or {}).values() if isinstance(w, dict))]
    shas = {r["ckpt sha256"] for r in runs if r.get("ckpt sha256")}
    if shas and shas != {_sha256(ckpt)}:
        raise ValueError(f"the scores were made with another checkpoint than {ckpt} "
                         f"(sha256 {sorted(shas)})")
    d, f = bh.provenance_sigma(bh.read_provenance(ckpt))
    trained = None if d is None and f is None else {"SIGMA_D": d, "SIGMA_F": f}
    used = [r["sigma"] for r in runs if r.get("sigma") is not None and not r.get("rgb_only")]
    for u in used:
        if u != (trained or used[0]):
            raise ValueError(f"the scores' {u} are not the motion scales of {ckpt}'s "
                             f"provenance {trained or used[0]}")
    return used[0] if used else trained


def run(scores, ckpt, out=None, name: str | None = None, allow_seen: bool = False,
        force: bool = False, cal_of=st.calibration, log=print) -> dict:
    """The command line's fit: read, guard, calibrate, write, print."""
    merged, meta = ev.load_scores(scores)
    results = list(merged.values())
    name = _detector(results, meta, ckpt, name)
    results = [r for r in results if name in r]
    check_windows(results, ckpt, allow_seen)
    sigma = _sigma(meta.get(name, {}), ckpt)
    out = Path(out) if out else bh.calib_path(ckpt)
    if out.exists() and not force:
        raise ValueError(f"{out} exists (a frozen calibration?); --force to replace")
    c = calibrate(results, name, cal_of)
    blob = {"a": c.pop("a"), "b": c.pop("b"), "tau": c.pop("tau"),
            "ckpt": str(ckpt), "ckpt sha256": _sha256(ckpt), "detector": name,
            "sigma": sigma, "rgb_only": meta.get(name, {}).get("rgb_only"),
            "scores": [{"path": str(p), "sha256": _sha256(p)} for p in scores],
            "windows": [r["window"] for r in results], **c, "rules": _rules()}
    out.write_text(json.dumps(blob, indent=1))
    log(f"  {name}: {blob['candidates']} candidates ({blob['positives']} positive) "
        f"on {c['frames']['pass']} pass + {c['frames']['action']} action frames of "
        f"{len(results)} windows; {c['frames']['pass without people']} pass frames "
        f"without people skipped")
    log(f"  Platt a = {blob['a']:.4f}, b = {blob['b']:.4f} "
        f"({c['fit']['iterations']} iterations, log loss {c['fit']['log loss']:.4f})")
    log("  p_cal bin      n   mean p_cal  positive")
    for row in c["reliability"]:
        lo, hi = row["bin"]
        pc = "" if row["p_cal"] is None else f"{row['p_cal']:.3f}"
        pr = "" if row["positive"] is None else f"{row['positive']:.3f}"
        log(f"  {lo:.1f}-{hi:.1f}  {row['n']:6d}  {pc:>10}  {pr:>8}")
    log(f"  ECE {c['ece']} (pass {c['by kind']['pass']['ece']}, "
        f"action {c['by kind']['action']['ece']})")
    log(f"  candidates per frame on {c['random frames']} random frames: " +
        ", ".join(f"{g['tau']:.2f}: {g['candidates per frame']}"
                  + ("" if g["exact"] else " (lower bound)") for g in c["tau grid"]))
    if blob["tau"] is None:
        log(f"  WARNING: no exact grid value gives <= {MAX_PER_FRAME} per frame; tau None")
    log(f"  tau = {blob['tau']} -> {out}")
    return blob


def run_evaluate(scores, calib_path, out, name: str | None = None,
                 cal_of=st.calibration, log=print) -> dict:
    """The command line's `--evaluate`: writes `out` only (never the
    calibration or a scores file), prints frame and window counts only --
    no candidate count, which with the frames would be a measure of the
    sealed detector."""
    calib = bh.load_calibration(calib_path)
    if calib is None:
        raise ValueError(f"no calibration at {calib_path}")
    keep = {Path(p).resolve() for p in [*scores, calib_path]}
    if Path(calib_path).suffix != ".json":
        keep.add(bh.calib_path(calib_path).resolve())
    if Path(out).resolve() in keep:
        raise ValueError(f"--out {out} is an input (the calibration or a scores file)")
    merged, meta = ev.load_scores(scores)
    results = list(merged.values())
    name = _detector(results, meta, calib.get("ckpt", "?"), name or calib.get("detector"))
    results = [r for r in results if name in r]
    for r in results:
        if str(r["match"]) in ev.TEST_MATCHES:
            raise ValueError(f"{r['window']}: test match {r['match']}")
    blob = {"calib": str(calib_path), "detector": name,
            "scores": [{"path": str(p), "sha256": _sha256(p)} for p in scores],
            "windows": [r["window"] for r in results],
            **evaluate(results, name, calib, cal_of), "rules": _rules()}
    Path(out).write_text(json.dumps(blob, indent=1))
    log(f"  {name}: {blob['frames']['pass']} pass + {blob['frames']['action']} action "
        f"frames, {blob['random frames']} random frames of {len(results)} windows -> {out}")
    return blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, metavar="SCORES.json")
    ap.add_argument("--ckpt", help="the v2 checkpoint (its name is the detector's)")
    ap.add_argument("--name", help="the detector's name in the scores")
    ap.add_argument("--out", help="default <ckpt>.calib.json")
    ap.add_argument("--allow-seen", action="store_true",
                    help="fit on non-U windows, training matches included (smoke tests)")
    ap.add_argument("--force", action="store_true", help="replace an existing calibration")
    ap.add_argument("--evaluate", metavar="CALIB.json",
                    help="apply a frozen calibration; write to --out, print counts")
    args = ap.parse_args()
    if args.evaluate and not args.out:
        ap.error("--evaluate needs --out")
    if not args.evaluate and not args.ckpt:
        ap.error("--ckpt is needed")
    try:
        if args.evaluate:
            run_evaluate(args.scores, args.evaluate, args.out, args.name)
        else:
            run(args.scores, args.ckpt, args.out, args.name, args.allow_seen, args.force)
    except ValueError as e:
        raise SystemExit(f"refused: {e}")


# ---- Self-check -------------------------------------------------------------------

def _check():
    """A synthetic U window seen by a distortion-free camera at 10 px a
    metre (pixel (u, v) is pitch (u / 10 - 52.5, v / 10 - 34)): labels
    at the edges of the pass and actor rules; a Platt fit recovering the
    (a, b) the labels were drawn with, and calibrated within its bins;
    tau against a brute count, with the save floor and the top-8 cap
    making grid values inexact; the window guards, the motion scales of
    the scores against the checkpoint's, the file round trip and
    `--evaluate` (frame counts only on the screen, inputs never written)."""
    import tempfile

    K = np.array([[1e6, 0, 525.0], [0, 1e6, 340.0], [0, 0, 1]])
    cal = {"K": K, "D": np.zeros((4, 1)), "Knew": K, "Hinv": np.diag([0.1, 0.1, 1.0])}

    def px(x, y, z, size=4.0):
        """A row whose ground point (u, v + size / 2) is (x, y), with logit z."""
        half = 0.0 if np.isnan(size) else size / 2
        return [(x + 52.5) * 10, (y + 34) * 10 - half, float(_sigmoid(z)), size, z]

    # Labels at the edges: pass 0 from (-20, 0) to (0, 0); its frame q = 0
    # is at s = 0.25 (x = -15); a person at (-17, 0.5). In order: 3.9 m
    # off, 4.1 m off, 0.5 m from the person, t - s = 0.29, 0.31; at the
    # actor (5, 0): 1.4 m, 1.6 m, 1.0 m with no size (ground at v).
    P = [{"i": 0, "frames": [10, 11, 12, 13], "s": list(ev.FF_S),
          "from": [-20.0, 0.0], "to": [0.0, 0.0], "kick": 0, "reception": 20,
          "block": "u:0"}]
    near = [[-17.0, 0.5], [30.0, 30.0]]
    rows = [px(-15, 3.9, 1.0), px(-15, 4.1, 2.0), px(-17, 0.0, 3.0),
            px(-20 + 20 * 0.54, 0.0, 0.5), px(-20 + 20 * 0.56, 0.0, 0.4)]
    act = [px(5, 1.4, 1.0), px(5, 1.6, 0.0), px(5, -1.0, -1.0, size=float("nan"))]
    r = {"window": "u", "match": "118578",
         "frames": {"passes": P, "actions": [{"frame": 40, "label": "DRIVE", "x": 5.0,
                                               "y": 0.0}],
                    "random": []},
         "people": {"10": near, "40": near},
         "det": {"candidates": {"10": rows, "40": act, "11": [[1, 2, 0.5, 4]]}}}
    s = samples([r], "det", lambda m: cal)
    assert s["label"].tolist() == [1, 0, 0, 1, 0, 1, 0, 1], s["label"]
    assert s["kind"].tolist() == ["pass"] * 5 + ["action"] * 3
    assert s["frames"]["pass without people"] == 1           # frame 11: no people
    assert s["frames"]["pass not measured"] == 2
    # A synthetic U: labels drawn from sigmoid(a0 z + b0), each placed where
    # the rules say (on the corridor at s / at the actor) or clearly not.
    a0, b0 = 0.7, -0.8
    rng = np.random.default_rng(0)
    passes, actions, cands, people, rnd = [], [], {}, {}, []
    for i in range(150):
        kick = 1000 + 100 * i
        A, B = np.array([-30.0, -20 + i % 40]), np.array([10.0, -10 + i % 30])
        fr = [kick + 10 + 5 * q for q in range(4)]
        passes.append({"i": i, "frames": fr, "s": list(ev.FF_S), "from": A.tolist(),
                       "to": B.tolist(), "kick": kick, "reception": kick + 40,
                       "block": f"u:{i // 15}"})
        for q, f in enumerate(fr):
            z = np.sort(rng.normal(-1.0, 2.0, 6))[::-1]
            on = A + ev.FF_S[q] * (B - A)
            out = []
            for zk in z:
                hit = rng.random() < _sigmoid(a0 * zk + b0)
                g = on + rng.uniform(-1, 1, 2) if hit else on + (0.0, 12.0 + 5 * rng.random())
                out.append(px(*g, zk))
            cands[str(f)] = out
            people[str(f)] = [[-50.0, 30.0]]
        x, y = rng.uniform(-40, 40), rng.uniform(-25, 25)
        actions.append({"frame": kick, "label": "PASS", "x": x, "y": y})
        z = rng.normal(-1.0, 2.0, 5)
        cands[str(kick)] = [px(x + (0.5 if rng.random() < _sigmoid(a0 * zk + b0) else 6.0),
                               y, zk) for zk in z]
        f = kick + 60
        rnd.append(f)
        cands[str(f)] = [px(0, 0, zk) for zk in np.sort(rng.normal(-2.5, 1.5, 8))[::-1]
                         if _sigmoid(zk) >= 0.05]
    U = {"window": "st2_118578_1st_f000000", "match": "118578",
         "frames": {"passes": passes, "actions": actions, "random": rnd},
         "people": people, "ball_v2.pt": {"candidates": cands}}
    c = calibrate([U], "ball_v2.pt", lambda m: cal)
    assert abs(c["a"] - a0) < 0.1 and abs(c["b"] - b0) < 0.15, (c["a"], c["b"])
    assert c["ece"] < 0.05 and c["candidates"] == 150 * 29, c["ece"]
    assert c["by kind"]["pass"]["candidates"] == 150 * 24
    # tau: the brute count at each grid value; exactness from the floor.
    calib = {"a": c["a"], "b": c["b"]}
    rz = [np.array([row[4] for row in cands[str(f)]]) for f in rnd]
    edge = float(bh.p_cal(_logit(0.05) + 1e-4, calib))
    want = None
    for row in c["tau grid"]:
        n = np.mean([(bh.p_cal(z, calib) >= row["tau"]).sum() for z in rz])
        assert abs(row["candidates per frame"] - n) < 1e-4
        assert row["exact"] == (edge <= row["tau"]), row
        if want is None and row["exact"] and n <= 1.0:
            want = row["tau"]
    assert c["tau"] == want and want is not None
    assert not c["tau grid"][0]["exact"]      # 0.05: below the save floor here
    # A frame of 8 rows all above a value makes it inexact (a 9th may pass).
    full = [np.full(8, 3.0)]
    g = per_frame(full, {"a": 1.0, "b": 0.0}, grid=(0.5, 0.96))
    assert not g[0]["exact"] and g[1]["exact"] and g[0]["candidates per frame"] == 8
    # Platt refuses a single class, a separating logit and a reversed one.
    for bad in (lambda: fit_platt([0.0, 1.0], [1, 1]),
                lambda: fit_platt([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1]),
                lambda: fit_platt(np.r_[rng.normal(1, 1, 200), rng.normal(-1, 1, 200)],
                                  np.r_[np.zeros(200), np.ones(200)])):
        try:
            bad()
            raise AssertionError("a bad Platt fit was accepted")
        except ValueError:
            pass
    # Reliability: two bins, each 0.05 off.
    tb, e = reliability([0.05, 0.05, 0.95, 0.95], [0, 0, 1, 1])
    assert np.isclose(e, 0.05) and tb[0]["n"] == 2 and tb[5]["n"] == 0
    # Files: guards, the written calibration, no silent overwrite, evaluate.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = tmp / "ball_v2.pt"
        ckpt.write_bytes(b"weights")
        bh.write_provenance(ckpt, {"train": bh.V2_TRAIN, "exclude": bh.V2_EXCLUDE})
        sc = tmp / "U.json"
        sc.write_text(json.dumps({"detectors": {"ball_v2.pt": {"kind": "v2"}},
                                  "results": [U]}))
        quiet = lambda *a: None  # noqa: E731
        blob = run([sc], ckpt, cal_of=lambda m: cal, log=quiet)
        got = bh.load_calibration(ckpt)
        assert got["a"] == blob["a"] == c["a"] and got["tau"] == c["tau"]
        assert got["ckpt sha256"] == hashlib.sha256(b"weights").hexdigest()
        assert got["windows"] == ["st2_118578_1st_f000000"] and got["sigma"] is None
        # The motion scales: the scores' must be the checkpoint's own, in the
        # eval metadata or a parquet's sidecars (an RGB-only run's do not
        # matter); a recorded checkpoint sha256 must be this one's.
        mine, other = {"SIGMA_D": 0.07, "SIGMA_F": 0.03}, {"SIGMA_D": 0.02, "SIGMA_F": 0.05}
        sha = hashlib.sha256(b"weights").hexdigest()
        bh.write_provenance(ckpt, {"train": bh.V2_TRAIN, "exclude": bh.V2_EXCLUDE,
                                   "sigma": mine})
        side = lambda s, **k: {"source": "parquet", "windows": {  # noqa: E731
            U["window"]: {"sigma": s, "ckpt sha256": sha, **k}}}
        for det, ok in (({"sigma": other}, False), ({"sigma": mine}, True),
                        ({"sigma": mine, "ckpt sha256": sha}, True),
                        ({"sigma": mine, "ckpt sha256": "0" * 64}, False),
                        ({"sigma": other, "rgb_only": True}, True),
                        (side(other), False), (side(mine), True),
                        (side(other, rgb_only=True), True),
                        (dict(side(mine), sigma=other), False)):
            (tmp / "S.json").write_text(json.dumps({
                "detectors": {"ball_v2.pt": {"kind": "v2", **det}}, "results": [U]}))
            try:
                b2 = run([tmp / "S.json"], ckpt, tmp / "s.json", force=True,
                         cal_of=lambda m: cal, log=quiet)
                assert ok and b2["sigma"] == mine, det
            except ValueError:
                assert not ok, det
        for m, allow in (("117093", True), ("128057", True), ("118577", False),
                         ("117092", False)):
            V = dict(U, window=f"st2_{m}_1st_f000000", match=m)
            bad = tmp / f"{m}.json"
            bad.write_text(json.dumps({"results": [V]}))
            try:
                run([bad], ckpt, tmp / "x.json", allow_seen=allow, cal_of=lambda m: cal,
                    log=quiet)
                raise AssertionError(f"match {m} was fitted on")
            except ValueError:
                pass
        T = dict(U, window="st2_118577_1st_f000000", match="118577")
        (tmp / "T.json").write_text(json.dumps({"results": [T]}))
        run([tmp / "T.json"], ckpt, tmp / "t.json", allow_seen=True,
            cal_of=lambda m: cal, log=quiet)
        for bad in (lambda: run([sc], ckpt, cal_of=lambda m: cal, log=quiet),
                    lambda: run([sc], ckpt, tmp / "y.json", name="nope",
                                cal_of=lambda m: cal, log=quiet)):
            try:
                bad()
                raise AssertionError("an overwrite or a wrong name passed")
            except ValueError:
                pass
        # --evaluate: the file gets everything, the screen frame counts only;
        # an input as --out is refused.
        said = []
        ev_out = run_evaluate([sc], bh.calib_path(ckpt), tmp / "e.json",
                              cal_of=lambda m: cal, log=said.append)
        assert ev_out["ece"] == c["ece"] and ev_out["at tau"]["tau"] == c["tau"]
        assert json.loads((tmp / "e.json").read_text())["detector"] == "ball_v2.pt"
        assert len(said) == 1 and "candidates" not in said[0]
        assert str(ev_out["candidates"]) not in said[0] and "ece" not in said[0].lower()
        for out in (bh.calib_path(ckpt), sc):
            try:
                run_evaluate([sc], ckpt, out, cal_of=lambda m: cal, log=quiet)
                raise AssertionError(f"--evaluate would write over {out}")
            except ValueError:
                pass
    print(f"  calibrate_ball_heatmap: labels at the rule edges, Platt recovers "
          f"(a, b) = ({a0}, {b0}) as ({c['a']:.3f}, {c['b']:.3f}), ECE {c['ece']:.3f}, "
          f"tau {c['tau']} = brute count, floor / top-8 inexact, guards, motion "
          f"scales, files, --evaluate counts only: ok")


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _check()
    else:
        main()
