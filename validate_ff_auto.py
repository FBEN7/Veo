"""Is FF-auto a measure of the ball in flight? Its agreement with the blind
judge, and the table it gives for the v1 heatmap and COCO.

FF-auto (`eval_ball_detector.ff_hit`) counts a pass frame as hit when the
top-1 candidate's ground point is within 4 m of the passer-receiver
segment, at a projection t in [0, 1] within 0.30 of the frame's fraction
s of the pass, and 1 m or more from every person. It was chosen on the T
judgements of four blind sittings (`ball_tiles.py`; `judged.txt` next to
each `key.json`), before v2 existed; this script redoes the numbers
registered with protocol v3:

- **agreement** on the judged detector tiles at pass frames (T: tiles_T,
  tiles_ff; V-old: tiles_ffV, tiles_E4; a tile judged in two sittings
  keeps the free-flight sitting's judgement): Cohen's kappa of "FF-auto
  hit" against "judged the match ball in play" (B), PPV (auto hits
  confirmed), recall (confirmed balls caught), and per detector the auto
  rate next to the judged rate;
- **decoys** (random pitch points): judged B, and counted as hits by
  FF-auto at pass frames;
- **chance**: random pitch points (uniform on the pitch, 300 per pass at
  its second frame, seed 1) on the T windows -- the registered 1.9% is the
  corridor without the |t - s| guard, printed next to FF-auto's;
- **the FF-auto table** of v1 and COCO from the saved candidates
  (heatmap_T.json + coco_T2.json; V.json): per window, per match, pooled,
  and the paired difference with the 1-minute block bootstrap and, as
  registered, the per-pass one.

Each recorded value is printed next to the recomputed one.

    python validate_ff_auto.py --dir SCRATCH/ball

After a sitting is judged, `--sealed-check S.json --judged DIR` checks a
sealed eval file (`eval_ball_detector.py --sealed`) against the sitting
(`ball_tiles.py --pass-frame`): per detector, judged-B rate over the
sampled passes, FF-auto rate on the same frames, kappa / PPV / recall on
its tiles, the differential bias ((auto - judged) of the first detector
minus that of the second, with a one-sided 90% bootstrap upper bound),
FF-auto hits more than 12 px from a ball the judge confirmed in the other
detector's tile at that frame (confirmed false hits), the exact McNemar
test on the judged passes, decoys judged B, confident picks judged B,
and re-judged old tiles that agree -- with protocol v3's S1 / S2 / S3
conditions marked.

    python validate_ff_auto.py --sealed-check V_new.json --judged S2 \\
        --detectors ball_v2.4000.pt coco
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import eval_ball_detector as ev
from src import soccertrack_v2 as st
from src.ball_heatmap import BALL_M, local_scale
from src.paths import DATA_DIR

V1 = "ball_heatmap.3000.pt"
SETS = {"T": ("tiles_T", "tiles_ff"), "V-old": ("tiles_ffV", "tiles_E4")}
PREFER = ("tiles_ff", "tiles_ffV")          # free-flight sittings' judgement first
SCORES = {"T": ("heatmap_T.json", "coco_T2.json"), "V-old": ("V.json",)}
SIZES = ("heatmap_T.json", "coco_T2.json", "V.json", "ffV.json", "ff_heatmap.json",
         "ff_coco.json")
CHANCE_N, CHANCE_SEED = 300, 1
CONFIRM_PX = 12.0
# Protocol v3 conditions: validity on S2, the blind part of G (S1), drift (S3).
KAPPA_MIN, PPV_MIN, BIAS_MAX, BIAS_UB_MAX = 0.6, 0.85, 0.05, 0.10
DECOYS_MAX = 2
REJUDGE_MIN = 11 / 12
# Recorded at the protocol v3 freeze (ball_protocol_v3.json).
RECORDED = {
    "counts": {"T": (69, 0, 9, 34), "V-old": (55, 3, 6, 33)},
    "kappa": {"T": 0.82, "V-old": 0.81},
    "rates": {("T", V1): (0.62, 0.74), ("T", "coco"): (0.61, 0.65),
              ("V-old", V1): (0.58, 0.60), ("V-old", "coco"): (0.62, 0.67)},
    "decoys": (0, 65), "chance": 0.019,
    "table": {"T": (0.651, 0.500, 0.151, 0.089, 0.214),
              "V-old": (0.533, 0.500, 0.033, -0.072, 0.138)},
    "windows": {"st2_117093_1st_f015000": (0.45, 0.36),
                "st2_132877_1st_f015000": (0.62, 0.65),
                "st2_118578_2nd_f015000": (0.59, 0.38),
                "st2_118576_2nd_f015000": (0.58, 0.43),
                "st2_128058_2nd_f015000": (0.74, 0.63)},
}


def read_judged(d: Path) -> dict:
    """{tile: label} from a sitting's judged.txt ("N LABEL" lines, # notes)."""
    out = {}
    for line in (Path(d) / "judged.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            n, j = line.split()[:2]
            out[int(n)] = j
    return out


def agreement(auto, judged) -> dict:
    """Counts, Cohen's kappa, PPV and recall of `auto` against `judged`
    (booleans)."""
    a, b = np.asarray(auto, bool), np.asarray(judged, bool)
    tp, fp = int((a & b).sum()), int((a & ~b).sum())
    fn, tn = int((~a & b).sum()), int((~a & ~b).sum())
    n = max(len(a), 1)
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n / n
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "kappa": round((po - pe) / (1 - pe), 3) if len(a) and pe < 1 else None,
            "ppv": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None}


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p for b and c discordant pairs."""
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def binom_upper(k: int, n: int) -> float:
    """P(X >= k), X ~ Binomial(n, 1/2): the exact one-sided p of a sign
    test with k of n on the predicted side (protocol v4's H5 over passes
    with d != 0), and of McNemar with b = k predicted-side and n - k other
    discordant pairs (`mcnemar_one_sided`)."""
    if not n:
        return 1.0
    return sum(math.comb(n, i) for i in range(max(k, 0), n + 1)) / 2 ** n


def mcnemar_one_sided(b: int, c: int) -> float:
    """Exact one-sided McNemar p that b (the predicted side's discordant
    pairs) exceeds c."""
    return binom_upper(b, b + c)


class Window:
    """A window's calibration, passes (frames_of), pass frame -> (pass, q),
    and people, read once."""

    _cache: dict = {}

    @classmethod
    def get(cls, w: str) -> dict:
        if w not in cls._cache:
            info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
            sets = ev.frames_of(w)
            gt = pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{w}_gt.parquet")
            at = {}
            for p in sets["passes"]:
                for q, f in enumerate(p["frames"]):
                    at[f] = (p["i"], q)
            cls._cache[w] = {"info": info, "cal": st.calibration(info["match"]),
                             "sets": sets, "at": at,
                             "people": {f: g[["x", "y"]].to_numpy(float)
                                        for f, g in gt.groupby("frame")}}
        return cls._cache[w]


def tile_table(root: Path) -> pd.DataFrame:
    """One row per judged tile of the four sittings at a pass frame: its
    set, sitting, source, judgement and FF-auto. A detector tile's ground
    point uses the candidate's saved size; a decoy's (or a candidate not
    found) 0.22 m at the local scale along the pitch."""
    sizes = {}
    for s in SIZES:
        for r in json.loads((root / s).read_text())["results"]:
            for name, v in r.items():
                if isinstance(v, dict) and "candidates" in v:
                    for f, rows in v["candidates"].items():
                        sizes.setdefault((r["window"], int(f), name), rows)
    out = []
    for set_, dirs in SETS.items():
        for tdir in dirs:
            judged = read_judged(root / tdir)
            for k in json.loads((root / tdir / "key.json").read_text()):
                w, f, src = k["window"], int(k["frame"]), k["source"]
                d = Window.get(w)
                if f not in d["at"]:
                    continue
                i, q = d["at"][f]
                size = next((row[3] for row in sizes.get((w, f, src), [])
                             if abs(row[0] - k["u"]) < 0.5
                             and abs(row[1] - k["v"]) < 0.5), np.nan)
                if not np.isfinite(size):
                    g0 = st.image_to_pitch([[k["u"], k["v"]]], d["cal"])[0]
                    size = (BALL_M * local_scale(g0, d["cal"])[0]
                            if np.isfinite(g0).all() else 0)
                g = st.image_to_pitch([[k["u"], k["v"] + size / 2]], d["cal"])[0]
                auto = ev.ff_hit(g if np.isfinite(g).all() else None,
                                 d["sets"]["passes"][i], q, d["people"].get(f))
                out.append({"set": set_, "tdir": tdir, "tile": k["tile"], "src": src,
                            "judged": judged.get(k["tile"]), "w": w, "f": f, "pass": i,
                            "q": q, "auto": auto})
    df = pd.DataFrame(out)
    df["B"] = df.judged == "B"
    return df


def detector_tiles(df: pd.DataFrame) -> pd.DataFrame:
    """Detector tiles, one per (window, frame, source): a tile judged in two
    sittings keeps the free-flight sitting's judgement."""
    det = df[df.src != "decoy"].copy()
    det["pref"] = ~det.tdir.isin(PREFER)
    det = det.sort_values(["pref"], kind="stable")
    return det.drop_duplicates(["w", "f", "src"]).sort_index()


def chance(windows, n: int = CHANCE_N, seed: int = CHANCE_SEED) -> dict:
    """Share of random pitch points (uniform on 105 x 68 m, `n` per pass at
    its second frame) counted as hits: FF-auto, and its corridor without
    the |t - s| guard."""
    rng = np.random.default_rng(seed)
    ff, wide = [], []
    for w in windows:
        d = Window.get(w)
        for p in d["sets"]["passes"]:
            f = p["frames"][1]
            G = np.c_[rng.uniform(-52.5, 52.5, n), rng.uniform(-34, 34, n)]
            A, B = np.asarray(p["from"], float), np.asarray(p["to"], float)
            ab = B - A
            t = (G - A) @ ab / (ab @ ab)
            perp = np.hypot(*(G - (A + np.clip(t, 0, 1)[:, None] * ab)).T)
            P = d["people"][f]
            near = np.hypot(*(G[:, None] - P[None]).transpose(2, 0, 1)).min(axis=1)
            base = (perp <= ev.FF_W) & (t >= 0) & (t <= 1) & (near >= ev.FF_R)
            wide.append(base)
            ff.append(base & (np.abs(t - p["s"][1]) <= ev.FF_DT))
    return {"ff-auto": float(np.concatenate(ff).mean()),
            "corridor without |t - s|": float(np.concatenate(wide).mean()),
            "points": int(sum(map(len, ff)))}


def _flag(ok: bool) -> str:
    return "ok" if ok else "DIFFERS"


def _near(x, y, tol=0.005) -> bool:
    return abs(x - y) <= tol + 1e-9


def validate(root: Path):
    df = tile_table(root)
    det = detector_tiles(df)
    print("Agreement of FF-auto with the judge, detector tiles at pass frames")
    for s in SETS:
        x = det[det.set == s]
        a = agreement(x.auto, x.B)
        got, rec = (a["tp"], a["fp"], a["fn"], a["tn"]), RECORDED["counts"][s]
        print(f"  {s}: {len(x)} tiles; auto hit & B {a['tp']}, auto hit & not B "
              f"{a['fp']}, B missed {a['fn']}, neither {a['tn']}; kappa {a['kappa']:.2f}, "
              f"PPV {a['ppv']:.2f}, recall {a['recall']:.2f}  [recorded {rec}, kappa "
              f"{RECORDED['kappa'][s]}: {_flag(got == rec)}]")
        for src, y in x.groupby("src"):
            ra, rj = float(y.auto.mean()), float(y.B.mean())
            r = RECORDED["rates"].get((s, src))
            note = (f"  [recorded {r[0]:.2f} / {r[1]:.2f}: "
                    f"{_flag(_near(ra, r[0]) and _near(rj, r[1]))}]") if r else ""
            print(f"    {src}: {len(y)} tiles, auto rate {ra:.2f}, judged rate {rj:.2f}"
                  + note)
    keys = {tdir: (read_judged(root / tdir), json.loads((root / tdir / "key.json")
                                                         .read_text()))
            for dirs in SETS.values() for tdir in dirs}
    decoys = [j.get(k["tile"]) == "B" for j, key in keys.values() for k in key
              if k["source"] == "decoy"]
    dec = df[df.src == "decoy"]
    got = (int(dec.auto.sum()), len(dec))
    print(f"Decoys: judged B {sum(decoys)} of {len(decoys)}; at pass frames, FF-auto hits "
          f"{got[0]} of {got[1]}  [recorded {RECORDED['decoys'][0]} of "
          f"{RECORDED['decoys'][1]}: {_flag(got == RECORDED['decoys'])}]")
    ch = chance(["st2_118576_2nd_f015000", "st2_118578_2nd_f015000",
                 "st2_128058_2nd_f015000"])
    wide = ch["corridor without |t - s|"]
    print(f"Chance (random pitch points, T windows, {ch['points']} points): FF-auto "
          f"{ch['ff-auto']:.4f}; the corridor without |t - s| {wide:.4f}  [recorded "
          f"{RECORDED['chance']}: {_flag(_near(wide, RECORDED['chance'], 0.0005))}]")
    print("FF-auto, v1 heatmap and COCO, from the saved candidates")
    for s, files in SCORES.items():
        results, _ = ev.reuse([root / f for f in files], log=lambda *a: None)
        for r in results:
            rec = RECORDED["windows"].get(r["window"])
            a, b = r[V1]["ff-auto"], r["coco"]["ff-auto"]
            note = (f"  [recorded {rec[0]:.2f} / {rec[1]:.2f}: "
                    f"{_flag(_near(a, rec[0]) and _near(b, rec[1]))}]") if rec else ""
            print(f"  {s} {r['window']}: {r['passes']} passes, v1 {a:.3f}, "
                  f"COCO {b:.3f}" + note)
        c = ev.compare_ff(results, V1, "coco")
        p = ev.compare_ff(results, V1, "coco", blocks=False)
        rec = RECORDED["table"][s]
        got = (c[V1], c["coco"], c["difference"], p["2.5%"], p["97.5%"])
        print(f"  {s} pooled: {c['passes']} passes, v1 {c[V1]:.3f}, COCO "
              f"{c['coco']:.3f}, difference {c['difference']:+.3f}; 1-minute blocks "
              f"({c['blocks']}) [{c['2.5%']:+.3f}, {c['97.5%']:+.3f}]; single passes "
              f"[{p['2.5%']:+.3f}, {p['97.5%']:+.3f}]")
        print(f"    [recorded {rec[0]:.3f} / {rec[1]:.3f}, {rec[2]:+.3f} [{rec[3]:+.3f}, "
              f"{rec[4]:+.3f}] over single passes: "
              f"{_flag(all(_near(x, y, 0.001) for x, y in zip(got, rec)))}]")
        for m, v in c["matches"].items():
            print(f"    match {m}: {v['passes']} passes, v1 {v[V1]:.3f}, COCO "
                  f"{v['coco']:.3f}, difference {v['difference']:+.3f}")


# ---- A judged sitting against a sealed eval file -----------------------------------

def sealed_check(scores, judged_dir, detectors=None, draws: int = ev.DRAWS,
                 seed: int = 0) -> dict:
    """Protocol v3's conditions for one sitting (see the module docstring).
    `detectors`: the pass-tile detectors, the first compared against the
    second; default every detector in key_passes.json."""
    judged_dir = Path(judged_dir)
    res = {r["window"]: r for r in json.loads(Path(scores).read_text())["results"]}
    judged = read_judged(judged_dir)
    key = json.loads((judged_dir / "key.json").read_text())
    by_tile = {k["tile"]: k for k in key}
    kp = judged_dir / "key_passes.json"
    passes = json.loads(kp.read_text()) if kp.exists() else []
    names = list(detectors or dict.fromkeys(n for p in passes for n in p["tiles"]))
    isB = {t: judged.get(t) == "B" for t in by_tile}
    # Per detector over the passes (no tile: a miss, auto 0 as well).
    auto = {n: np.array([bool(res[p["window"]][n]["ff frames"][p["pass"]][p["q"]])
                         for p in passes], bool) for n in names}
    seen = {n: np.array([p["tiles"].get(n) is not None and isB[p["tiles"][n]]
                         for p in passes], bool) for n in names}
    out = {"detectors": {}}
    for n in names:
        has = np.array([p["tiles"].get(n) is not None for p in passes], bool)
        conf = [isB[k["tile"]] for k in key if k.get("kind") == "confident"
                and n in k.get("sources", [k["source"]])]
        # Confirmed false FF-auto hits: another detector's tile at the frame
        # is the ball, and this detector's top-1 is more than 12 px from it.
        false = 0
        for i, p in enumerate(passes):
            top = res[p["window"]][n]["candidates"].get(str(p["frame"])) or []
            balls = [by_tile[t] for m, t in p["tiles"].items()
                     if m != n and t is not None and isB[t]]
            false += bool(auto[n][i] and top and any(
                np.hypot(top[0][0] - b["u"], top[0][1] - b["v"]) > CONFIRM_PX
                for b in balls))
        out["detectors"][n] = {
            "passes": len(passes),
            "judged rate": round(float(seen[n].mean()), 3) if len(passes) else None,
            "auto rate": round(float(auto[n].mean()), 3) if len(passes) else None,
            "tiles": int(has.sum()), **agreement(auto[n][has], seen[n][has]),
            "confirmed false FF-auto hits": false,
            "confident judged B": [int(sum(conf)), len(conf)]}
    if len(names) >= 2 and passes:
        a, b = names[:2]
        d = ((auto[a].astype(float) - seen[a]) - (auto[b].astype(float) - seen[b]))
        rng = np.random.default_rng(seed)
        boots = d[rng.integers(0, len(d), (draws, len(d)))].mean(axis=1)
        only_a, only_b = int((seen[a] & ~seen[b]).sum()), int((seen[b] & ~seen[a]).sum())
        out["differential bias"] = {"pair": [a, b], "bias": round(float(d.mean()), 3),
                                    "90% upper bound":
                                        round(float(np.percentile(boots, 90)), 3)}
        out["blind McNemar"] = {"pair": [a, b], f"{a} only": only_a, f"{b} only": only_b,
                                "p (exact, two-sided)": round(mcnemar(only_a, only_b), 4)}
    for kind in ("decoy", "corridor-decoy"):
        ks = [k for k in key if k["source"] == kind]
        out[f"{kind}s judged B"] = [int(sum(isB[k["tile"]] for k in ks)), len(ks)]
    rej = [k for k in key if k["source"] == "rejudge"]
    if rej:
        old = [read_judged(Path(k["old_key"]).parent).get(k["old_tile"]) == "B"
               for k in rej]
        out["rejudged agree"] = [sum(o == isB[k["tile"]] for o, k in zip(old, rej)),
                                 len(rej)]
    return out


def print_sealed(out: dict):
    for n, s in out["detectors"].items():
        print(f"  {n}: {s['passes']} passes, judged B {s['judged rate']}, FF-auto "
              f"{s['auto rate']}; {s['tiles']} tiles: kappa {s['kappa']} (>= {KAPPA_MIN}: "
              f"{_ok(s['kappa'], KAPPA_MIN)}), PPV {s['ppv']} (>= {PPV_MIN}: "
              f"{_ok(s['ppv'], PPV_MIN)}), recall {s['recall']}")
        print(f"    confirmed false FF-auto hits {s['confirmed false FF-auto hits']}; "
              f"confident picks judged B {s['confident judged B'][0]} of "
              f"{s['confident judged B'][1]}")
    if "differential bias" in out:
        b = out["differential bias"]
        ub = b["90% upper bound"]
        print(f"  differential bias ({b['pair'][0]} - {b['pair'][1]}): {b['bias']:+.3f} "
              f"(<= +{BIAS_MAX}: {_ok(-b['bias'], -BIAS_MAX)}); one-sided 90% upper "
              f"bound {ub:+.3f} (<= +{BIAS_UB_MAX}: {_ok(-ub, -BIAS_UB_MAX)})")
        m = out["blind McNemar"]
        print("  blind McNemar: " + ", ".join(f"{k} {v}" for k, v in m.items()
                                               if k != "pair"))
    for kind in ("decoy", "corridor-decoy"):
        k, n = out[f"{kind}s judged B"]
        print(f"  {kind}s judged B: {k} of {n}")
    k, n = (out[f"{x}s judged B"] for x in ("decoy", "corridor-decoy"))
    print(f"  all decoys judged B: {k[0] + n[0]} of {k[1] + n[1]} (<= {DECOYS_MAX}: "
          f"{'yes' if k[0] + n[0] <= DECOYS_MAX else 'NO'})")
    if "rejudged agree" in out:
        a, n = out["rejudged agree"]
        print(f"  re-judged old tiles agree: {a} of {n} (>= {REJUDGE_MIN * n:.0f}: "
              f"{'yes' if n and a / n >= REJUDGE_MIN - 1e-9 else 'NO'})")


def _ok(x, lo) -> str:
    return "yes" if x is not None and x >= lo else "NO"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="the scratchpad's ball folder: tiles_* and score files")
    ap.add_argument("--sealed-check", metavar="S.json")
    ap.add_argument("--judged", metavar="DIR", help="the sitting: key.json, judged.txt")
    ap.add_argument("--detectors", nargs="*")
    args = ap.parse_args()
    if args.sealed_check:
        if not args.judged:
            ap.error("--sealed-check needs --judged DIR")
        print_sealed(sealed_check(args.sealed_check, args.judged, args.detectors))
    elif args.dir:
        validate(Path(args.dir))
    else:
        ap.error("--dir or --sealed-check")


# ---- Self-check -------------------------------------------------------------------

def _check():
    """Kappa on the registered T counts; McNemar against direct sums; a
    synthetic sitting (two detectors, three passes, decoys, a confident
    pick, a re-judged tile) against a synthetic sealed file."""
    import tempfile

    a = agreement([1] * 69 + [0] * 9 + [0] * 34, [1] * 69 + [1] * 9 + [0] * 34)
    assert (a["tp"], a["fn"], a["tn"], a["kappa"]) == (69, 9, 34, 0.823), a
    assert a["ppv"] == 1.0 and a["recall"] == round(69 / 78, 3)
    assert mcnemar(0, 0) == 1.0 and abs(mcnemar(10, 2) - 2 * (1 + 12 + 66) / 4096) < 1e-12
    assert mcnemar_one_sided(0, 0) == 1.0 and binom_upper(0, 5) == 1.0
    assert abs(mcnemar_one_sided(10, 2) - (1 + 12 + 66) / 4096) < 1e-12
    assert abs(mcnemar_one_sided(2, 10) - (4096 - 1 - 12) / 4096) < 1e-12
    assert binom_upper(15, 20) < 0.025 < binom_upper(14, 20)     # 15 of 20 needed
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        old = tmp / "old"
        old.mkdir()
        (old / "key.json").write_text(json.dumps([{"tile": 1, "window": "w", "frame": 9,
                                                   "u": 0, "v": 0, "source": "A"}]))
        (old / "judged.txt").write_text("# old\n1 B\n")
        # Pass 0: A on the ball (B, auto hit), B 20 px off it with an auto hit
        # (a confirmed false hit); pass 1: both miss, A judged N; pass 2: B
        # has no candidate, A judged B without an auto hit.
        frames = {"passes": [{"i": i, "frames": [10 * i, 0, 0, 0]} for i in range(3)]}
        r = {"window": "w", "match": "m", "frames": frames,
             "A": {"ff frames": [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                   "candidates": {"0": [[100, 50, 0.9, 4]], "10": [[5, 5, .5, 4]],
                                  "20": [[7, 7, .5, 4]]}},
             "B": {"ff frames": [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                   "candidates": {"0": [[120, 50, 0.4, 4]], "10": [[9, 9, .3, 4]],
                                  "20": []}}}
        (tmp / "S.json").write_text(json.dumps({"results": [r]}))
        sit = tmp / "sit"
        sit.mkdir()
        key = [{"tile": 1, "window": "w", "frame": 0, "u": 100, "v": 50, "source": "A",
                "sources": ["A"], "kind": "pass", "pass": 0, "q": 0},
               {"tile": 2, "window": "w", "frame": 0, "u": 120, "v": 50, "source": "B",
                "sources": ["B"], "kind": "pass", "pass": 0, "q": 0},
               {"tile": 3, "window": "w", "frame": 10, "u": 5, "v": 5, "source": "A+B",
                "sources": ["A", "B"], "kind": "pass", "pass": 1, "q": 0},
               {"tile": 4, "window": "w", "frame": 20, "u": 7, "v": 7, "source": "A",
                "sources": ["A"], "kind": "pass", "pass": 2, "q": 0},
               {"tile": 5, "window": "w", "frame": 3, "u": 1, "v": 1, "source": "decoy"},
               {"tile": 6, "window": "w", "frame": 0, "u": 1, "v": 1,
                "source": "corridor-decoy"},
               {"tile": 7, "window": "w", "frame": 4, "u": 1, "v": 1, "source": "A",
                "sources": ["A"], "kind": "confident"},
               {"tile": 8, "window": "w", "frame": 9, "u": 0, "v": 0, "source": "rejudge",
                "old_key": str(old / "key.json"), "old_tile": 1}]
        (sit / "key.json").write_text(json.dumps(key))
        (sit / "key_passes.json").write_text(json.dumps([
            {"window": "w", "pass": 0, "q": 0, "frame": 0, "tiles": {"A": 1, "B": 2}},
            {"window": "w", "pass": 1, "q": 0, "frame": 10, "tiles": {"A": 3, "B": 3}},
            {"window": "w", "pass": 2, "q": 0, "frame": 20,
             "tiles": {"A": 4, "B": None}}]))
        (sit / "judged.txt").write_text("1 B\n2 N\n3 N\n4 B\n5 N\n6 B*\n7 B\n8 B\n")
        out = sealed_check(tmp / "S.json", sit)
        A, B = out["detectors"]["A"], out["detectors"]["B"]
        assert (A["judged rate"], A["auto rate"], A["tiles"]) == (round(2 / 3, 3),
                                                                  round(1 / 3, 3), 3)
        assert (A["tp"], A["fp"], A["fn"], A["tn"]) == (1, 0, 1, 1)
        assert (B["judged rate"], B["auto rate"], B["tiles"]) == (0.0, round(1 / 3, 3), 2)
        assert B["confirmed false FF-auto hits"] == 1
        assert A["confirmed false FF-auto hits"] == 0
        assert A["confident judged B"] == [1, 1] and B["confident judged B"] == [0, 0]
        # (auto - judged): A (0, 0, -1), B (1, 0, 0): bias -2/3.
        assert out["differential bias"]["bias"] == round(-2 / 3, 3)
        assert out["blind McNemar"]["A only"] == 2 and out["blind McNemar"]["B only"] == 0
        assert out["decoys judged B"] == out["corridor-decoys judged B"] == [0, 1]
        assert out["rejudged agree"] == [1, 1]
        print_sealed(out)
    print("  validate_ff_auto: kappa, McNemar, a synthetic sitting against a sealed "
          "file: ok")


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _check()
    else:
        main()
