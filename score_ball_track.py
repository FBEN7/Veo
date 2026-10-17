"""Protocol v4 (v4-r1): a judged pair sitting scored -- tracked v2 against
raw v2 at the pair frames (`ball_tiles.py --pairs`).

    sha256sum S_U/judged.txt > S_U/judged.sha256      # before the key is read
    python score_ball_track.py --judged S_U/judged.txt \\
        --judged-sha S_U/judged.sha256 --key S_U_key/key.json \\
        [--key-passes S_U_key/key_passes.json] [--scores S_U.json] \\
        [--vnew-passes 108] [--json OUT]

It refuses to run unless the sha256 of judged.txt equals `--judged-sha`
(a hex digest, or a file whose first word is one, as sha256sum writes),
checked before the key is opened; unless every tile of the key is judged
and every judged tile is in the key; and unless key_passes.json agrees
with the key (each detector's tile at a frame is a tile of that window and
frame credited to it, and every pass has its q1 there).

A tile is the ball when judged "B" (the centre of the match ball in play
inside the 24 px box); anything else (N, unsure, B?, B*) is not, and the
labels' counts are printed. A detector without a tile at a pair frame (a
miss) is not the ball.

- **H5** (primary on V-new; G-a on U-test): per pass, d = (pair frames
  where only the tracked pick is the ball) - (those where only raw's is);
  the passes with d > 0, d < 0 and d = 0; the exact one-sided sign test
  over the passes with d != 0. H5 holds when p < 0.025 and d > 0 in more
  passes than d < 0.
- **H6** (with `--extra NAME@qK` tiles, V-new's coco@q1; it counts only if
  H5 holds): tracked against NAME at frame K of every pass, exact
  one-sided McNemar.
- **q1 McNemar**, tracked against raw (secondary), exact one-sided.
- **Decomposition** of the pair frames (only tracked / only raw / both /
  neither) by tracked source, by source and parity, by parity, by match
  and by tercile of local px/m (far / mid / near: at the pass segment's
  point at s, the cut points printed; "unknown" where px/m is not known).
- **Decoys** judged the ball (random, corridor, offset), and re-judged old
  tiles that agree with their old judgement (>= 11 of 12).
- **Sources**: the tracked picks at every pass frame by source, and the
  share interpolated or fallback.
- **Gate** (U-test; all must hold): G-a passes with d > 0 outnumber those
  with d < 0; G-b tracked placebo hits <= raw's + 1, pre-kick and
  post-reception separately, on the placebo frames both measured (from
  `--scores`, the sealed eval file; without it G-b is not computed and the
  gate is not passed); G-c random + corridor decoys judged the ball <= 2
  of 20 and offset decoys <= 1 of 6 (another number of decoys than
  registered fails). G-a..G-c are printed as pass / fail. G-d is eval's
  code check: printed apart, from eval's metadata in `--scores` (per
  window: stride 2, no extra frames, longest fill <= 6 frames, no
  "detected" / "fallback" pick off the stride) and the "detected" /
  "fallback" picks at odd pass frames in key_passes.json.
- **Recorded, not gated**: b and c (frames and passes) and the implied
  power of H5 on V-new (`--vnew-passes` passes, at this sitting's
  informative share and split; also at a 70/30 split); with `--scores`,
  FF-auto's agreement with the judge on the interpolated tracked tiles
  (kappa), confirmed false FF-auto hits at the pair frames (as
  `validate_ff_auto.sealed_check`: an auto hit whose top-1 is more than
  12 px from a tile of another detector at that frame judged the ball),
  and eval's FF-auto paired difference with whether it counts as a measure
  (kappa >= 0.6 and tracked's false hits <= raw's + 2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from validate_ff_auto import (CONFIRM_PX, REJUDGE_MIN, agreement, binom_upper,
                              mcnemar_one_sided)

ALPHA = 0.025
PAIR_Q = 1                                      # ball_tiles.PAIR_Q: q1 always paired
G_B_SLACK = 1                                   # tracked placebo hits <= raw's + 1
G_C = {"random + corridor": (("decoy", "corridor-decoy"), 2, 20),
       "offset": (("offset-decoy",), 1, 6)}     # judged the ball <= k of n
G_D_STRIDE, G_D_GAP = 2, 6                      # eval's code check, from its metadata
# The registered detector and calibration (protocol v4): every window's sidecar.
G_D_CKPT = "31f9a5eb567bfc26ed71967b993503159fabc8577257efcbc09782133d86c965"
G_D_CALIB = "77799641091fcca587e9be1abed99629b4ba0e1d0f4c2019ff7e5caf49ab331c"
FF_KAPPA_MIN, FF_FALSE_SLACK = 0.6, 2
VNEW_PASSES = 108
REF_SPLIT = 0.7
CATS = ("only tracked", "only raw", "both", "neither")
DECOYS = ("decoy", "corridor-decoy", "offset-decoy")


# ---- Inputs -----------------------------------------------------------------------

def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recorded_sha(arg: str) -> str:
    """`--judged-sha`: a hex digest, or a file whose first word is one."""
    p = Path(arg)
    words = p.read_text().split() if p.is_file() else [arg.strip()]
    text = (words or [""])[0].lower()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"--judged-sha: {arg!r} is not a sha256 (64 hex digits) or a "
                         f"file starting with one")
    return text


def check_sha(judged_path, want: str) -> str:
    got = sha256_of(judged_path)
    if got != want:
        raise SystemExit(f"{judged_path}: sha256 {got} is not the recorded {want}; "
                         f"refusing to read the key")
    return got


def read_judged_file(path) -> dict:
    """{tile: label} from a judged.txt ("N LABEL" lines, # notes); a tile
    judged twice is refused."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            n, j = line.split()[:2]
            if int(n) in out:
                raise ValueError(f"{path}: tile {n} judged twice")
            out[int(n)] = j
    return out


def check_complete(judged: dict, key: list):
    tiles = {k["tile"] for k in key}
    missing, stray = sorted(tiles - set(judged)), sorted(set(judged) - tiles)
    if missing or stray:
        raise ValueError(f"judged.txt and the key differ: {len(missing)} tiles not judged "
                         f"(e.g. {missing[:5]}), {len(stray)} judged tiles not in the key "
                         f"(e.g. {stray[:5]})")


def sitting_of(passes: list) -> tuple[str, str, dict]:
    """(tracked, raw, extra {name: q}) of a pair sitting's key_passes.json."""
    if not passes:
        raise ValueError("key_passes.json lists no pass")
    if any("pair frames" not in p for p in passes):
        raise ValueError("key_passes.json is not a pair sitting's (ball_tiles.py --pairs)")
    T, R = passes[0]["tracked"], passes[0]["raw"]
    extra = passes[0].get("extra") or {}
    if any((p["tracked"], p["raw"], p.get("extra") or {}) != (T, R, extra) for p in passes):
        raise ValueError("key_passes.json mixes detectors")
    return T, R, extra


def check_tiles(passes: list, by_tile: dict):
    """key_passes.json against the key: every tile it names is a key tile
    of that window and frame credited to that detector, a frame is its
    pass's frame q, and no (pass, q) is listed twice."""
    for p in passes:
        seen = set()
        for fr in p["pair frames"]:
            q, f = int(fr["q"]), int(fr["frame"])
            if q in seen or int(p["frames"][q]) != f:
                raise ValueError(f"{p['window']} pass {p['pass']}: frame {f} at q{q} is "
                                 f"not its pass frame, or q{q} is listed twice")
            seen.add(q)
            for n, t in fr["tiles"].items():
                if t is None:
                    continue
                k = by_tile.get(t)
                if (k is None or (k["window"], int(k["frame"])) != (p["window"], f)
                        or n not in k.get("sources", [k["source"]])):
                    raise ValueError(f"{p['window']} pass {p['pass']} q{q}: {n}'s tile {t} "
                                     f"is not a key tile of that frame credited to {n}")


# ---- The measures -----------------------------------------------------------------

def power(n: int, split: float, alpha: float = ALPHA) -> float:
    """Power of the exact one-sided sign test on n informative passes when
    a share `split` of them favour the tracker."""
    if n <= 0:
        return 0.0
    k = next((k for k in range(n + 1) if binom_upper(k, n) < alpha), None)
    if k is None:
        return 0.0
    return float(sum(math.comb(n, i) * split ** i * (1 - split) ** (n - i)
                     for i in range(k, n + 1)))


def _cat(t: bool, r: bool) -> str:
    return CATS[0] if t and not r else CATS[1] if r and not t else CATS[2] if t else CATS[3]


def _tally(rows, by) -> dict:
    out = {}
    for x in rows:
        d = out.setdefault(by(x), {c: 0 for c in CATS})
        d[x["cat"]] += 1
    return {str(k): out[k] for k in sorted(out, key=str)}


def frame_rows(passes: list, ball: dict, T: str, R: str) -> list:
    """One row per listed frame of every pass: its pass, q, frame, parity,
    pair (in d or not), tracked source, px/m, tiles, which detectors' tile
    is the ball, and (tracked, raw)'s category."""
    out = []
    for p in passes:
        for fr in p["pair frames"]:
            tiles = fr["tiles"]
            x = {"window": p["window"], "match": str(p["match"]), "pass": p["pass"],
                 "q": int(fr["q"]), "frame": int(fr["frame"]), "parity": fr["parity"],
                 "pair": bool(fr["pair"]), "source": fr.get("track source") or "unknown",
                 "px/m": fr.get("px/m"), "tiles": tiles,
                 "ball": {n: t is not None and ball[t] for n, t in tiles.items()}}
            x["cat"] = _cat(x["ball"].get(T, False), x["ball"].get(R, False))
            out.append(x)
    return out


def mcnemar_at(frames: list, a: str, b: str, q: int, n_passes: int) -> dict:
    """Exact one-sided McNemar, a against b, at frame q of every pass."""
    at = [x for x in frames if x["q"] == q]
    if len(at) != n_passes or any(a not in x["tiles"] or b not in x["tiles"] for x in at):
        raise ValueError(f"frame q{q} with {a} and {b} tiles is listed for {len(at)} of "
                         f"{n_passes} passes")
    ba = sum(x["ball"][a] and not x["ball"][b] for x in at)
    bb = sum(x["ball"][b] and not x["ball"][a] for x in at)
    return {"q": q, "passes": len(at), f"{a} only": ba, f"{b} only": bb,
            "p (exact McNemar, one-sided)": round(mcnemar_one_sided(ba, bb), 5)}


def score(judged: dict, key: list, passes: list, scores: dict | None = None,
          vnew_passes: int = VNEW_PASSES) -> dict:
    """Every number of the module docstring from a sitting's judgements,
    key and key_passes (and the sealed eval file's blob, `scores`)."""
    by_tile = {k["tile"]: k for k in key}
    ball = {t: judged.get(t) == "B" for t in by_tile}
    T, R, extra = sitting_of(passes)
    check_tiles(passes, by_tile)
    frames = frame_rows(passes, ball, T, R)
    pf = [x for x in frames if x["pair"]]
    d = {(p["window"], p["pass"]): 0 for p in passes}
    for x in pf:
        d[(x["window"], x["pass"])] += int(x["cat"] == CATS[0]) - int(x["cat"] == CATS[1])
    pos, neg = sum(v > 0 for v in d.values()), sum(v < 0 for v in d.values())
    out = {"tracked": T, "raw": R, "extra": extra, "passes": len(passes),
           "pair frames": len(pf), "tiles": len(key),
           "labels": dict(sorted(Counter(judged.values()).items()))}
    p5 = binom_upper(pos, pos + neg)
    out["H5"] = {"d > 0": pos, "d < 0": neg, "d = 0": len(d) - pos - neg,
                 "p (exact sign test, one-sided)": round(p5, 5),
                 "holds": bool(pos > neg and p5 < ALPHA),
                 "d per pass": {f"{w}:{i}": v for (w, i), v in d.items()}}
    out["frames"] = {c: sum(x["cat"] == c for x in pf) for c in CATS}
    out["q1 McNemar"] = mcnemar_at(frames, T, R, PAIR_Q, len(passes))
    out["H6"] = {n: dict(mcnemar_at(frames, T, n, int(q), len(passes)),
                         counts=out["H5"]["holds"]) for n, q in extra.items()}
    # Decomposition of the pair frames.
    pxm = np.array([x["px/m"] for x in pf if x["px/m"] is not None], float)
    cuts = [float(c) for c in np.quantile(pxm, [1 / 3, 2 / 3])] if len(pxm) else []

    def tercile(x):
        v = x["px/m"]
        if v is None or not cuts:
            return "unknown"
        return "far" if v <= cuts[0] else "mid" if v <= cuts[1] else "near"

    out["by"] = {"source": _tally(pf, lambda x: x["source"]),
                 "source x parity": _tally(pf, lambda x: f"{x['source']} {x['parity']}"),
                 "parity": _tally(pf, lambda x: x["parity"]),
                 "match": _tally(pf, lambda x: x["match"]),
                 "px/m tercile": _tally(pf, tercile)}
    out["px/m cuts"] = [round(c, 2) for c in cuts]
    # Decoys and re-judged tiles.
    out["decoys"] = {}
    for src in DECOYS:
        ks = [k for k in key if k["source"] == src]
        out["decoys"][src] = [int(sum(ball[k["tile"]] for k in ks)), len(ks)]
    rej = [k for k in key if k["source"] == "rejudge"]
    if rej:
        old, agree = {}, 0
        for k in rej:
            path = Path(k["old_key"]).parent / "judged.txt"
            if path not in old:
                old[path] = read_judged_file(path)
            agree += (old[path].get(k["old_tile"]) == "B") == ball[k["tile"]]
        out["rejudged agree"] = [int(agree), len(rej),
                                 bool(agree / len(rej) >= REJUDGE_MIN - 1e-9)]
    # The tracked picks at every pass frame, by source.
    srcs = Counter(s or "unknown" for p in passes for s in p["track source"])
    n_src = sum(n for s, n in srcs.items() if s in ("detected", "interpolated", "fallback"))
    out["track sources at pass frames"] = dict(sorted(srcs.items()))
    out["interpolated or fallback share"] = round(
        (srcs["interpolated"] + srcs["fallback"]) / n_src, 3) if n_src else None
    # The gate: G-a..G-c here, G-d eval's (its metadata, apart).
    gate = {"G-a": {"pass": pos > neg, "d > 0": pos, "d < 0": neg}}
    gate["G-b"] = placebo_gate(scores, passes, T, R) if scores else {
        "pass": None, "note": "not computed: --scores (the sealed eval file) needed"}
    gc = {"pass": True}
    for name, (srcs_, k, n) in G_C.items():
        hit = sum(out["decoys"][s][0] for s in srcs_)
        tot = sum(out["decoys"][s][1] for s in srcs_)
        ok = hit <= k and tot == n
        gc[name] = {"judged ball": hit, "of": tot, "max": k, "registered n": n,
                    "pass": ok}
        gc["pass"] = gc["pass"] and ok
    gate["G-c"] = gc
    gate["G-a..G-c"] = all(gate[g]["pass"] is True for g in ("G-a", "G-b", "G-c"))
    gate["G-d"] = g_d(passes, scores, T)
    gate["all"] = gate["G-a..G-c"] and gate["G-d"]["pass"] is True
    out["gate"] = gate
    # Recorded, not gated.
    inf = pos + neg
    share = inf / len(passes)
    split = pos / inf if inf else None
    n_v = int(round(vnew_passes * share))
    out["recorded"] = {
        "b frames (only tracked)": out["frames"]["only tracked"],
        "c frames (only raw)": out["frames"]["only raw"],
        "b passes (d > 0)": pos, "c passes (d < 0)": neg,
        "informative share": round(share, 3),
        "split": None if split is None else round(split, 3),
        "V-new informative passes": n_v,
        "V-new power": None if split is None else round(power(n_v, split), 3),
        f"V-new power at {REF_SPLIT:.0%}": round(power(n_v, REF_SPLIT), 3)}
    if scores:
        out["recorded"].update(ff_validity(scores, pf, by_tile, ball, T, R))
    return out


def _results(scores: dict, passes) -> dict:
    want = {p["window"] for p in passes}
    res = {r["window"]: r for r in scores.get("results", []) if r["window"] in want}
    missing = sorted(want - set(res))
    if missing:
        raise ValueError(f"the scores file has no results for {missing}")
    return res


def placebo_gate(scores: dict, passes, T: str, R: str) -> dict:
    """G-b on the placebo frames both detectors measured."""
    res = _results(scores, passes)
    out = {"pass": True}
    for kind in ("pre", "post"):
        t = r_ = n = 0
        for r in res.values():
            for a, b in zip(r[T][f"placebo {kind} hits"], r[R][f"placebo {kind} hits"]):
                if a is not None and b is not None:
                    t, r_, n = t + a, r_ + b, n + 1
        ok = t <= r_ + G_B_SLACK
        out[kind] = {"tracked hits": t, "raw hits": r_, "frames": n, "pass": ok}
        out["pass"] = out["pass"] and ok
    return out


def g_d(passes, scores: dict | None, T: str) -> dict:
    """G-d as eval reports it: per window of the sitting, the tracked
    detector's sidecar stride and extra frames, checkpoint and calibration
    sha256 (the registered ones), its longest fill and its "detected" /
    "fallback" rows off the stride (eval's metadata); and the
    "detected" / "fallback" picks at odd pass frames (key_passes.json).
    `pass` None without the metadata."""
    odd = sum(int(f) % 2 == 1 and s in ("detected", "fallback")
              for p in passes for f, s in zip(p["frames"], p["track source"]))
    out = {"detected or fallback at odd pass frames": odd}
    wins = (((scores or {}).get("detectors") or {}).get(T) or {}).get("windows") or {}
    if not wins:
        out.update({"pass": None, "note": "eval's metadata for the tracked detector not "
                                          "given (--scores)"})
        return out
    per = {}
    for w in sorted({p["window"] for p in passes}):
        side = wins.get(w)
        if not isinstance(side, dict):
            per[w] = None
            continue
        t = side.get("track") or {}
        ex = side.get("extra_frames")
        per[w] = {"stride": side.get("stride"),
                  "extra frames": None if ex is None else len(ex),
                  "registered ckpt": side.get("ckpt sha256") == G_D_CKPT,
                  "registered calib": (side.get("calib") or {}).get("sha256") == G_D_CALIB,
                  "longest fill": (t.get("fill") or {}).get("longest fill"),
                  "detected or fallback off the stride":
                      t.get("detected or fallback off the stride")}
    ok = odd == 0 and all(
        v is not None and v["stride"] == G_D_STRIDE and v["extra frames"] == 0
        and v["registered ckpt"] and v["registered calib"]
        and v["longest fill"] is not None and v["longest fill"] <= G_D_GAP
        and v["detected or fallback off the stride"] == 0 for v in per.values())
    out.update({"windows": per, "pass": bool(ok)})
    return out


def ff_validity(scores: dict, pf, by_tile, ball, T: str, R: str) -> dict:
    """FF-auto against the judge on the interpolated tracked tiles at the
    pair frames, the confirmed false FF-auto hits there (as
    `validate_ff_auto.sealed_check`), and eval's paired FF-auto difference
    with whether it counts."""
    res = {r["window"]: r for r in scores.get("results", [])}
    auto, seen = [], []
    false = {T: 0, R: 0}
    for x in pf:
        r = res[x["window"]]
        a = {n: r[n]["ff frames"][x["pass"]][x["q"]] for n in (T, R)}
        t = x["tiles"].get(T)
        if x["source"] == "interpolated" and t is not None and a[T] is not None:
            auto.append(bool(a[T]))
            seen.append(ball[t])
        for n in (T, R):
            top = r[n]["candidates"].get(str(x["frame"])) or []
            balls = [by_tile[t] for m, t in x["tiles"].items()
                     if m != n and t is not None and ball[t]]
            false[n] += bool(a[n] and top and any(
                np.hypot(top[0][0] - b["u"], top[0][1] - b["v"]) > CONFIRM_PX
                for b in balls))
    ag = agreement(auto, seen)
    counts = bool(ag["kappa"] is not None and ag["kappa"] >= FF_KAPPA_MIN
                  and false[T] <= false[R] + FF_FALSE_SLACK)
    paired = (((scores.get("summary") or {}).get("ff-auto paired") or {})
              .get(f"{T} vs {R}"))
    return {"FF-auto on interpolated tracked tiles": dict(ag, tiles=len(auto)),
            "confirmed false FF-auto hits": dict(false),
            "FF-auto paired counts as a measure": counts,
            "FF-auto paired (eval)": None if paired is None else {
                k: paired[k] for k in ("passes", T, R, "difference", "2.5%", "97.5%",
                                       "blocks") if k in paired}}


# ---- Output -----------------------------------------------------------------------

def _pf(ok) -> str:
    return "PASS" if ok is True else "FAIL" if ok is False else "not computed"


def report(out: dict, log=print):
    T, R = out["tracked"], out["raw"]
    log(f"Pair sitting: {T} (tracked) against {R} (raw), {out['passes']} passes, "
        f"{out['pair frames']} pair frames, {out['tiles']} tiles; labels "
        + ", ".join(f"{k} {v}" for k, v in out["labels"].items()) + " (only B is the ball)")
    h = out["H5"]
    log(f"  H5: passes with d > 0 {h['d > 0']}, d < 0 {h['d < 0']}, d = 0 {h['d = 0']}; "
        f"exact sign test, one-sided p = {h['p (exact sign test, one-sided)']:.4f} "
        f"(< {ALPHA}, more d > 0: {'holds' if h['holds'] else 'does not hold'})")
    f = out["frames"]
    log("  pair frames: " + ", ".join(f"{c} {f[c]}" for c in CATS))
    for title, m, n in [("q1 McNemar, tracked vs raw", out["q1 McNemar"], R)] + [
            (f"H6, tracked vs {n} at q{m['q']}", m, n) for n, m in out["H6"].items()]:
        log(f"  {title}: {m['passes']} passes, {T} only {m[f'{T} only']}, {n} only "
            f"{m[f'{n} only']}, one-sided p = {m['p (exact McNemar, one-sided)']:.4f}"
            + ("" if m.get("counts", True) else " (H5 does not hold: H6 does not count)"))
    for by, tab in out["by"].items():
        extra = f" (cuts {out['px/m cuts']})" if by == "px/m tercile" else ""
        log(f"  by {by}{extra}:")
        for k, v in tab.items():
            log(f"    {k}: " + ", ".join(f"{c} {v[c]}" for c in CATS))
    log("  decoys judged the ball: " + ", ".join(f"{s} {k} of {n}" for s, (k, n)
                                                in out["decoys"].items()))
    if "rejudged agree" in out:
        a, n, ok = out["rejudged agree"]
        log(f"  re-judged old tiles agree: {a} of {n} (>= {REJUDGE_MIN:.3f} of them: "
            f"{'yes' if ok else 'NO'})")
    log("  tracked picks at pass frames by source: " + ", ".join(
        f"{s} {n}" for s, n in out["track sources at pass frames"].items())
        + f"; interpolated or fallback {out['interpolated or fallback share']}")
    g = out["gate"]
    log("Gate (U-test):")
    log(f"  G-a: d > 0 in {g['G-a']['d > 0']} passes, d < 0 in {g['G-a']['d < 0']}: "
        f"{_pf(g['G-a']['pass'])}")
    gb = g["G-b"]
    if gb["pass"] is None:
        log(f"  G-b: {gb['note']}: {_pf(None)}")
    else:
        log("  G-b: " + "; ".join(
            f"{k} tracked {gb[k]['tracked hits']} vs raw {gb[k]['raw hits']} + "
            f"{G_B_SLACK} on {gb[k]['frames']} frames: {_pf(gb[k]['pass'])}"
            for k in ("pre", "post")) + f" => {_pf(gb['pass'])}")
    gc = g["G-c"]
    log("  G-c: " + "; ".join(
        f"{n} judged the ball {gc[n]['judged ball']} of {gc[n]['of']} (<= {gc[n]['max']} "
        f"of {gc[n]['registered n']}): {_pf(gc[n]['pass'])}" for n in G_C)
        + f" => {_pf(gc['pass'])}")
    log(f"  G-a..G-c: {_pf(g['G-a..G-c'])}")
    gd = g["G-d"]
    if gd["pass"] is None:
        log(f"  G-d (eval's code check): {gd['note']}; detected or fallback at odd pass "
            f"frames {gd['detected or fallback at odd pass frames']}")
    else:
        log("  G-d (eval's code check, its metadata): " + "; ".join(
            f"{w} " + ("not in eval's metadata" if v is None else ", ".join(
                f"{k} {x}" for k, x in v.items())) for w, v in gd["windows"].items())
            + f"; detected or fallback at odd pass frames "
              f"{gd['detected or fallback at odd pass frames']}: {_pf(gd['pass'])}")
    log(f"  gate G-a..G-d: {'PASS' if g['all'] else 'FAIL'}")
    log("Recorded, not gated:")
    log("  " + ", ".join(f"{k} {v}" for k, v in out["recorded"].items()
                         if not isinstance(v, dict)))
    for k, v in out["recorded"].items():
        if isinstance(v, dict):
            log(f"  {k}: " + ", ".join(f"{a} {b}" for a, b in v.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--judged", required=True, help="the sitting's judged.txt")
    ap.add_argument("--judged-sha", required=True,
                    help="its sha256 recorded before the key was read (hex or a file)")
    ap.add_argument("--key", required=True, help="ball_tiles.py --pairs's key.json")
    ap.add_argument("--key-passes", help="default: key_passes.json next to --key")
    ap.add_argument("--scores", help="the sealed eval file (G-b, G-d, FF-auto)")
    ap.add_argument("--vnew-passes", type=int, default=VNEW_PASSES)
    ap.add_argument("--json", help="write every number here too")
    args = ap.parse_args()
    try:
        want = recorded_sha(args.judged_sha)
    except ValueError as e:
        ap.error(str(e))
    got = check_sha(args.judged, want)                # before the key is opened
    print(f"  judged.txt sha256 {got}: as recorded")
    judged = read_judged_file(args.judged)
    key = json.loads(Path(args.key).read_text())
    check_complete(judged, key)
    kp = Path(args.key_passes) if args.key_passes else Path(args.key).with_name(
        "key_passes.json")
    passes = json.loads(kp.read_text())
    scores = json.loads(Path(args.scores).read_text()) if args.scores else None
    out = score(judged, key, passes, scores, args.vnew_passes)
    out["judged sha256"] = got
    report(out)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


# ---- Self-check -------------------------------------------------------------------

# Six passes in two matches. Per pass: (window, local pass id, match, frames,
# the tracked source at its four frames, [(q, px/m, T, R, coco)]) with T / R /
# coco "B" (judged the ball), "N" (not), None (no tile: a miss), "=" (R's pick
# within 6 px of T's: one tile) or ... (coco: no tile asked, off q1).
_LAYOUT = [
    ("w1", 0, "M1", [10, 11, 12, 13],
     ["detected", "interpolated", "fallback", "interpolated"],
     [(1, 10, "B", "N", "B"), (2, 11, "B", None, ...)]),                     # d +2
    ("w1", 1, "M1", [20, 22, 23, 24], ["detected", "detected", "fallback", "detected"],
     [(1, 12, "B", "=", None)]),                                             # d 0
    ("w1", 2, "M1", [30, 31, 32, 34], ["detected", "interpolated", "detected", "detected"],
     [(1, 13, "N", "B", "B"), (3, 14, "B", "N", ...)]),                      # d 0
    ("w2", 0, "M2", [40, 41, 42, 44], ["fallback", "interpolated", "detected", "none"],
     [(1, 20, "B", "N", "N")]),                                              # d +1
    ("w2", 1, "M2", [50, 51, 52, 54], ["none", "interpolated", "detected", "detected"],
     [(0, 21, None, "B", ...), (1, None, "N", "N", "B")]),                   # d -1
    ("w2", 2, "M2", [60, 62, 63, 64], ["interpolated", "detected", "interpolated",
                                       "fallback"],
     [(1, 23, "B", "N", "N"), (2, 24, "B", "B", ...)]),                      # d +1
]
_U = {"T": 100.0, "R": 130.0, "coco": 160.0}         # each detector's tile: u (v 50)


def _sitting():
    """`_LAYOUT` as a key, judgements and key_passes, with 10 random + 10
    corridor decoys (one each judged the ball), 6 offset decoys (one) and
    the scores file's FF-auto frames, candidates, placebo hits and
    metadata."""
    T, R, C = "T", "R", "coco"
    key, judged, passes = [], {}, []

    def tile(w, f, u, src, label):
        n = len(key) + 1
        key.append({"tile": n, "window": w, "frame": f, "u": u, "v": 50.0,
                    "source": src, "sources": src.split("+"), "kind": "pair"})
        judged[n] = label
        return n

    cands = {}
    for w, i, m, fs, srcs, frs in _LAYOUT:
        rec = {"window": w, "match": m, "pass": i, "frames": fs,
               "s": [0.25, 0.417, 0.583, 0.75], "tracked": T, "raw": R,
               "extra": {C: 1}, "track source": srcs, "pair frames": []}
        for q, pxm, t, r, c in frs:
            f = fs[q]
            tiles = {}
            cw = cands.setdefault(w, {T: {}, R: {}})
            if r == "=":
                tiles[T] = tiles[R] = tile(w, f, _U[T], f"{T}+{R}", t)
                cw[T][str(f)], cw[R][str(f)] = [[_U[T], 50, .9, 4, 2]], [[_U[T] + 3, 50,
                                                                          .9, 4, 2]]
            else:
                tiles[T] = None if t is None else tile(w, f, _U[T], T, t)
                tiles[R] = None if r is None else tile(w, f, _U[R], R, r)
                cw[T][str(f)] = [] if t is None else [[_U[T], 50, .9, 4, 2]]
                cw[R][str(f)] = [] if r is None else [[_U[R], 50, .5, 4, 0]]
            if c is not ...:
                tiles[C] = None if c is None else tile(w, f, _U[C], C, c)
            rec["pair frames"].append({
                "q": q, "frame": f, "parity": "odd" if f % 2 else "even", "why": "q1",
                "pair": True, "track source": srcs[q], "px/m": pxm, "tiles": tiles})
        passes.append(rec)
    for src, n, hits in (("decoy", 10, 1), ("corridor-decoy", 10, 1),
                         ("offset-decoy", 6, 1)):
        for j in range(n):
            n_ = len(key) + 1
            key.append({"tile": n_, "window": "w1", "frame": 5, "u": 1.0, "v": 1.0,
                        "source": src, "sources": [src], "kind": src})
            judged[n_] = "B" if j < hits else "N"
    # FF-auto hits ((window, pass, q) -> 1): T at w1 0/q1, w1 2/q1, w2 0/q1,
    # w2 2/q2; R at w1 0/q1.
    hits = {"T": {("w1", 0, 1), ("w1", 2, 1), ("w2", 0, 1), ("w2", 2, 2)},
            "R": {("w1", 0, 1)}}
    res = []
    for w in ("w1", "w2"):
        pre = ([1, 0, None, 1], [0, 0, 1, 1]) if w == "w1" else ([], [])
        post = ([1, 1, 1], [0, 0, 1]) if w == "w1" else ([], [])
        res.append({"window": w, **{n: {
            "ff frames": [[int((w, i, q) in hits[n]) for q in range(4)] for i in range(3)],
            "candidates": cands[w][n], "placebo pre hits": pre[k],
            "placebo post hits": post[k]} for k, n in enumerate((T, R))}})
    side = {"stride": 2, "extra_frames": [], "processed": 100, "ckpt sha256": G_D_CKPT,
            "calib": {"sha256": G_D_CALIB}, "track": {
        "fill": {"longest fill": 6}, "detected or fallback off the stride": 0}}
    scores = {"results": res,
              "detectors": {T: {"kind": "v2-track", "windows": {
                  "w1": side, "w2": json.loads(json.dumps(side))}}},
              "summary": {"ff-auto paired": {"T vs R": {
                  "passes": 6, "T": 0.5, "R": 0.4, "difference": 0.1, "2.5%": -0.1,
                  "97.5%": 0.3, "blocks": 2, "matches": {}}}}}
    return key, judged, passes, scores


def _check():
    """The synthetic sitting of `_LAYOUT`, every expected number worked out
    by hand: H5's d per pass (+2, 0, 0, +1, -1, +1: three for, one against,
    p = 5/16), the q1 McNemar (3 vs 1, p = 5/16) and H6 against COCO (3 vs
    2, p = 1/2), the decomposition (source, parity, match, px/m terciles at
    cuts 12.67 / 20.33), decoys, re-judged tiles, G-a..G-c (G-b failing on
    post placebos, then passing), G-d from eval's metadata, FF-auto
    validity (kappa 0.545 on 5 tiles; false hits T 3, R 1), the power, and
    the refusals (a changed judged.txt, a bad digest, an unjudged tile,
    key_passes against the key, a missing q1); then `_check_format`."""
    import contextlib
    import copy
    import io
    import tempfile

    key, judged, passes, scores = _sitting()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        old = tmp / "old"
        old.mkdir()
        (old / "key.json").write_text("[]")
        (old / "judged.txt").write_text("# old\n1 B\n2 N\n")
        for ot, lab in ((1, "B"), (2, "B")):                 # agrees, differs
            n = len(key) + 1
            key.append({"tile": n, "window": "w1", "frame": 7, "u": 0, "v": 0,
                        "source": "rejudge", "sources": ["rejudge"], "kind": "rejudge",
                        "old_key": str(old / "key.json"), "old_tile": ot})
            judged[n] = lab
        jpath = tmp / "judged.txt"
        jpath.write_text("# synthetic\n" + "\n".join(
            f"{n} {judged[n]}" for n in sorted(judged)) + "\n")
        sha = sha256_of(jpath)
        (tmp / "judged.sha256").write_text(f"{sha}  judged.txt\n")
        assert recorded_sha(str(tmp / "judged.sha256")) == sha == recorded_sha(sha.upper())
        assert check_sha(jpath, sha) == sha and read_judged_file(jpath) == judged
        (tmp / "twice.txt").write_text("1 B\n1 N\n")
        for bad in (lambda: check_sha(jpath, "0" * 64), lambda: recorded_sha("abc"),
                    lambda: recorded_sha(str(tmp / "twice.txt")),
                    lambda: read_judged_file(tmp / "twice.txt"),
                    lambda: check_complete({1: "B"}, key),
                    lambda: check_complete({**judged, 999: "B"}, key)):
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    bad()
                raise AssertionError("accepted")
            except (SystemExit, ValueError):
                pass
        check_complete(judged, key)

        out = score(judged, key, passes, scores)
        h = out["H5"]
        assert (h["d > 0"], h["d < 0"], h["d = 0"]) == (3, 1, 2), h
        assert h["d per pass"] == {"w1:0": 2, "w1:1": 0, "w1:2": 0, "w2:0": 1, "w2:1": -1,
                                   "w2:2": 1}
        assert h["p (exact sign test, one-sided)"] == round(5 / 16, 5) and not h["holds"]
        assert out["frames"] == {"only tracked": 5, "only raw": 2, "both": 2,
                                 "neither": 1}, out["frames"]
        # 22 pair tiles (13 B), 26 decoys (3 B), 2 re-judged (both B).
        assert out["pair frames"] == 10 and out["tiles"] == 50
        assert out["labels"] == {"B": 18, "N": 32}
        m = out["q1 McNemar"]
        assert (m["passes"], m["T only"], m["R only"]) == (6, 3, 1)
        assert m["p (exact McNemar, one-sided)"] == round(5 / 16, 5)
        c = out["H6"]["coco"]
        assert (c["passes"], c["T only"], c["coco only"], c["q"], c["counts"]) == (
            6, 3, 2, 1, False)
        assert c["p (exact McNemar, one-sided)"] == 0.5
        by = out["by"]

        def cat(t, r, b, n):
            return {"only tracked": t, "only raw": r, "both": b, "neither": n}

        assert by["source"] == {"detected": cat(2, 0, 1, 0), "fallback": cat(1, 0, 0, 0),
                                "interpolated": cat(2, 1, 1, 1), "none": cat(0, 1, 0, 0)}
        assert by["parity"] == {"even": cat(3, 1, 1, 0), "odd": cat(2, 1, 1, 1)}
        assert by["source x parity"]["interpolated odd"] == cat(2, 1, 1, 1)
        assert by["source x parity"]["detected even"] == cat(2, 0, 1, 0)
        assert by["match"] == {"M1": cat(3, 1, 1, 0), "M2": cat(2, 1, 1, 1)}
        assert out["px/m cuts"] == [12.67, 20.33]
        assert by["px/m tercile"] == {"far": cat(2, 0, 1, 0), "mid": cat(2, 1, 0, 0),
                                      "near": cat(1, 1, 1, 0), "unknown": cat(0, 0, 0, 1)}
        assert out["decoys"] == {"decoy": [1, 10], "corridor-decoy": [1, 10],
                                 "offset-decoy": [1, 6]}
        assert out["rejudged agree"] == [1, 2, False]
        assert out["track sources at pass frames"] == {"detected": 11, "fallback": 4,
                                                       "interpolated": 7, "none": 2}
        assert out["interpolated or fallback share"] == round(11 / 22, 3)
        g = out["gate"]
        assert g["G-a"]["pass"] and g["G-c"]["pass"]
        assert g["G-b"]["pre"] == {"tracked hits": 2, "raw hits": 1, "frames": 3,
                                   "pass": True}
        assert g["G-b"]["post"] == {"tracked hits": 3, "raw hits": 1, "frames": 3,
                                    "pass": False}
        assert g["G-a..G-c"] is False and g["all"] is False
        # G-d: one fallback at an odd pass frame (w1 pass 1, frame 23).
        assert g["G-d"]["detected or fallback at odd pass frames"] == 1
        assert g["G-d"]["pass"] is False and g["G-d"]["windows"]["w2"]["longest fill"] == 6
        rec = out["recorded"]
        assert (rec["b frames (only tracked)"], rec["c frames (only raw)"]) == (5, 2)
        assert (rec["b passes (d > 0)"], rec["c passes (d < 0)"]) == (3, 1)
        assert rec["V-new informative passes"] == 72 and rec["split"] == 0.75
        assert rec["informative share"] == round(4 / 6, 3)
        assert rec["V-new power"] == round(power(72, 0.75), 3)
        # Interpolated tracked tiles: w1 0/q1 (auto, B), w1 2/q1 (auto, N),
        # w2 0/q1 (auto, B), w2 1/q1 (no auto, N), w2 2/q2 (auto, B).
        ffv = rec["FF-auto on interpolated tracked tiles"]
        assert (ffv["tp"], ffv["fp"], ffv["fn"], ffv["tn"], ffv["tiles"]) == (3, 1, 0, 1, 5)
        assert ffv["kappa"] == round((0.8 - 0.56) / 0.44, 3)
        # T: auto hits 30 / 60 / 30 px from a ball tile of another detector at
        # frames 11 (coco), 31 (R, coco), 63 (R); 41 none. R: at 11 (T, coco).
        assert rec["confirmed false FF-auto hits"] == {"T": 3, "R": 1}
        assert rec["FF-auto paired counts as a measure"] is False      # kappa < 0.6
        assert rec["FF-auto paired (eval)"]["difference"] == 0.1
        report(out, log=lambda *a: None)
        # G-b passes when post is even; G-d when the odd fallback goes; then
        # a fill of 8 frames, an extra frame, stride 1, another checkpoint or
        # calibration, or a window missing fail G-d.
        sc2 = copy.deepcopy(scores)
        sc2["results"][0]["T"]["placebo post hits"] = [0, 1, 1]
        ps2 = copy.deepcopy(passes)
        ps2[1]["track source"][2] = "interpolated"
        out2 = score(judged, key, ps2, sc2)
        assert out2["gate"]["G-b"]["pass"] and out2["gate"]["G-a..G-c"]
        assert out2["gate"]["G-d"]["pass"] and out2["gate"]["all"]
        for edit in (lambda s: s["w1"]["track"]["fill"].update({"longest fill": 8}),
                     lambda s: s["w2"].update({"extra_frames": [41]}),
                     lambda s: s["w2"].update({"stride": 1}),
                     lambda s: s["w1"].update({"ckpt sha256": "aa"}),
                     lambda s: s["w2"]["calib"].update({"sha256": "bb"}),
                     lambda s: s.pop("w2")):
            sc3 = copy.deepcopy(sc2)
            edit(sc3["detectors"]["T"]["windows"])
            g3 = score(judged, key, ps2, sc3)["gate"]
            assert g3["G-d"]["pass"] is False and g3["G-a..G-c"] and not g3["all"]
        out3 = score(judged, key, passes)
        assert out3["gate"]["G-b"]["pass"] is None and not out3["gate"]["G-a..G-c"]
        assert out3["gate"]["G-d"]["pass"] is None and "FF-auto paired (eval)" not in (
            out3["recorded"])
        # The FF-auto measure counts with T's auto miss at w1 2/q1 (kappa 1,
        # false hits T 2 <= R 1 + 2).
        sc4 = copy.deepcopy(sc2)
        sc4["results"][0]["T"]["ff frames"][2][1] = 0
        r4 = score(judged, key, ps2, sc4)["recorded"]
        assert r4["FF-auto on interpolated tracked tiles"]["kappa"] == 1.0
        assert r4["confirmed false FF-auto hits"] == {"T": 2, "R": 1}
        assert r4["FF-auto paired counts as a measure"] is True
        # G-c: a 21st random decoy (not the registered 20), or 2 offset hits.
        k2 = key + [{"tile": 999, "window": "w1", "frame": 5, "u": 1, "v": 1,
                     "source": "decoy", "kind": "decoy"}]
        assert not score({**judged, 999: "N"}, k2, passes)["gate"]["G-c"]["pass"]
        off = [k["tile"] for k in key if k["source"] == "offset-decoy"]
        assert not score({**judged, off[-1]: "B"}, key, passes)["gate"]["G-c"]["pass"]
        # "B?" / "unsure" are not the ball: w1 pass 0's q1 T tile -> d +1.
        t1 = passes[0]["pair frames"][0]["tiles"]["T"]
        assert score({**judged, t1: "B?"}, key, passes)["H5"]["d per pass"]["w1:0"] == 1
        # key_passes against the key: a tile credited to another detector, a
        # tile of another frame, a frame off its q, a pass without q1.
        tR = passes[0]["pair frames"][0]["tiles"]["R"]
        for edit in (lambda ps: ps[0]["pair frames"][0]["tiles"].update({"T": tR}),
                     lambda ps: ps[0]["pair frames"][1]["tiles"].update({"T": t1}),
                     lambda ps: ps[0]["pair frames"][1].update({"q": 3}),
                     lambda ps: ps[3]["pair frames"].pop(0),
                     lambda ps: ps[2].update({"raw": "X"}),
                     lambda ps: [p.pop("pair frames") for p in ps]):
            bad = copy.deepcopy(passes)
            edit(bad)
            try:
                score(judged, key, bad)
                raise AssertionError("an inconsistent key_passes.json was accepted")
            except ValueError:
                pass
    # The power: 45 informative passes at 70/30 about 0.75 (the protocol's).
    assert abs(power(45, 0.7) - 0.75) < 0.05 and power(0, 0.7) == 0.0
    assert power(20, 0.5) < 0.025
    _check_format()
    print("  score_ball_track: H5 sign test, q1 and H6 McNemar, decomposition, decoys, "
          "re-judged tiles, gate G-a..G-c, G-d from eval's metadata, FF-auto validity, "
          "power, sha256, completeness and key_passes refusals, ball_tiles' key format: ok")


def _check_format():
    """`ball_tiles.py --pairs`'s key and key_passes, built as its main
    builds them (pair picks, 6 px dedupe, shuffled numbering), scored: two
    passes, every tracked tile judged the ball and every other not."""
    import ball_tiles as tiles

    class Fake:
        def get(self, w):
            return {"info": {"match": "M"}, "frames": {"passes": [
                {"i": i, "frames": [10 * (i + 1) + q for q in range(4)],
                 "s": [0.25, 0.417, 0.583, 0.75], "from": [0, 0], "to": [10, 0]}
                for i in range(2)]}}

    T = {"10": [[100, 50, .9, 4]], "11": [[200, 50, .9, 4]], "12": [[100, 100, .9, 4]],
         "13": [], "20": [[10, 10, .9, 4]], "21": [], "22": [[60, 60, .9, 4]],
         "23": [[70, 70, .9, 4]]}
    R = {"10": [[103, 54, .5, 4]], "11": [[202, 50, .5, 4]], "12": [], "13": [],
         "20": [[30, 10, .5, 4]], "21": [[50, 50, .5, 4]], "22": [[60, 60, .5, 4]],
         "23": [[70, 77, .5, 4]]}
    results = {"mw": {"match": "M", "T": {"candidates": T}, "R": {"candidates": R},
                      "coco": {"candidates": {"11": [[300, 50, .4, 5]], "21": []}}}}
    picks, passes = tiles.pair_picks(results, Fake(), "T", "R", [("coco", 1)],
                                     {("mw", 11): "interpolated"}, None)
    picks, where = tiles.dedupe(picks, tiles.PAIR_PX)
    order = np.random.default_rng(0).permutation(len(picks))
    key = [{"tile": k + 1, **picks[int(j)]} for k, j in enumerate(order)]
    passes = json.loads(json.dumps(tiles.renumber(
        passes, where, {int(j): k + 1 for k, j in enumerate(order)})))
    judged = {k["tile"]: "B" if "T" in k["sources"] else "N" for k in key}
    check_complete(judged, key)
    out = score(judged, key, passes)
    # Pass 0: q1 one tile for both (B), q2 only T -> d +1. Pass 1: q0 and q3
    # T (B) apart from R (N), q1 R alone (N) -> d +2.
    assert out["H5"]["d per pass"] == {"mw:0": 1, "mw:1": 2} and out["pair frames"] == 5
    assert out["frames"] == {"only tracked": 3, "only raw": 0, "both": 1, "neither": 1}
    m, c = out["q1 McNemar"], out["H6"]["coco"]
    assert (m["T only"], m["R only"], c["T only"], c["coco only"]) == (0, 0, 1, 0)
    assert out["by"]["source"]["interpolated"]["both"] == 1
    assert out["by"]["px/m tercile"] == {"unknown": {"only tracked": 3, "only raw": 0,
                                                     "both": 1, "neither": 1}}


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _check()
    else:
        main()
