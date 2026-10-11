"""Blind visual check of ball detectors: tiles around detectors' top
candidates and around random pitch points (decoys), shuffled, numbered,
on montages; the key (which tile is which) goes to a separate file, to be
read only once every tile is judged.

Judge each tile: `ball` if the match ball's centre is inside the box the
four ticks mark (24 x 24 px in the frame), else `not ball` (`unsure`
counts as not ball).

    python ball_tiles.py --scores heatmap.json coco.json \\
        --take ball_heatmap.3000.pt=40 coco=40 \\
        --tau 0.2 --decoys 20 --out DIR

Protocol v3's sittings add (all optional; without them the tiles, their
order and numbers are as before):

- `--pass-frame q`: instead of random measured frames, the top-1 of each
  `--take` detector (names only; any p, none = no tile) at frame q of the
  four of every pass, or of `--passes-per-match N` passes sampled per
  match. q is 0-based, as the key's "q": 0-3 is s = 0.25 / 0.417 / 0.583
  / 0.75 along the pass; the s used is printed. Protocol v3's sittings
  use q = 1 (s = 0.417): the frame of the registered free-flight sittings
  (tiles_ff, tiles_ffV) and of FF-auto's validation;
- `--dedupe-px 6`: detectors' picks at one frame within 6 px become one
  tile, credited to each (the key's "sources");
- `--confident NAME=TAU:N`: N top-1 picks of NAME at p >= TAU on random
  frames (p_cal when the rows carry a logit: the calibration from the
  scores file, `<ckpt>.calib.json` or `--calib NAME=JSON`);
- `--corridor-decoys N`: grass on a pass's segment where the ball is not:
  at t = s +- 0.40 (within [0.05, 0.95]) of a pass frame, offset across
  the segment by U(-1.5, 1.5) m, 1 m or more from every person and 30 px
  or more from every detector's candidates -- decoys where FF-auto looks;
- `--rejudge KEY:N`: N old detector tiles of another sitting, drawn
  again (the old judgement is not read here), for the judge's drift.

Every key record has "tile", "window", "frame", "u", "v", "source" (a
detector, "decoy", "corridor-decoy" or "rejudge"; detectors joined by "+"
when deduped), "sources", "kind" and, at a pass frame, "pass" and "q"
(0-based). With `--pass-frame`, `key_passes.json` lists for every pass
the tile of each detector (None: no candidate, a miss). Both are key:
read them only after judging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.paths import DATA_DIR

HALF, ZOOM, BOX = 36, 3, 12
PER_SHEET, COLS = 30, 5
CORRIDOR_DT = 0.40          # t = s +- this
CORRIDOR_T = (0.05, 0.95)
CORRIDOR_OFF = 1.5          # m across the segment, uniform
CORRIDOR_PEOPLE = 1.0       # m from every person
CORRIDOR_PX = 30.0          # px from every detector's candidate
TRIES = 50


def tile(image, u, v, number):
    H, W = image.shape[:2]
    pad = cv2.copyMakeBorder(image, HALF, HALF, HALF, HALF, cv2.BORDER_CONSTANT)
    u, v = int(round(u)) + HALF, int(round(v)) + HALF
    crop = pad[v - HALF:v + HALF, u - HALF:u + HALF]
    t = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
    c, b = HALF * ZOOM, BOX * ZOOM
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        x, y = c + sx * b, c + sy * b
        cv2.line(t, (x, y), (x + sx * 10, y), (0, 0, 255), 1)
        cv2.line(t, (x, y), (x, y + sy * 10), (0, 0, 255), 1)
    strip = np.full((22, t.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, str(number), (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 1)
    return np.vstack([strip, t])


class Windows:
    """What the new picks need of a window, read once: its json, passes
    and random frames (from the scores file, else `frames_of`), people
    and calibration."""

    def __init__(self, results: dict):
        self.results, self.cache = results, {}

    def get(self, w: str) -> dict:
        if w not in self.cache:
            from eval_ball_detector import frame_sets, frames_detail
            from src import soccertrack_v2 as st

            info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
            fr = self.results.get(w, {}).get("frames")
            if fr is None:
                lab = pd.read_csv(DATA_DIR / "soccertrack_v2" / f"{w}_events.csv")
                gt = pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{w}_gt.parquet",
                                     columns=["frame"])
                fr = frames_detail(frame_sets(lab, gt.frame.nunique(), w))
            gt = pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{w}_gt.parquet")
            self.cache[w] = {"info": info, "frames": fr,
                             "cal": st.calibration(info["match"]),
                             "people": {f: g[["x", "y"]].to_numpy(float)
                                        for f, g in gt.groupby("frame")},
                             "at": {f: (p["i"], q) for p in fr["passes"]
                                    for q, f in enumerate(p["frames"])}}
        return self.cache[w]


def pick(w, f, u, v, source, kind, **extra) -> dict:
    return {"window": w, "frame": int(f), "u": float(u), "v": float(v), "source": source,
            "sources": [source], "kind": kind, **extra}


def pass_picks(results, wins, names, q, per_match, rng):
    """Top-1 of each detector at frame q (0-based, of the 4) of every
    (sampled) pass; and every pass's tiles, with its s there, to fill in
    key_passes.json."""
    from eval_ball_detector import FF_S

    by_match = {}
    for w in results:
        d = wins.get(w)
        for p in d["frames"]["passes"]:
            by_match.setdefault(str(d["info"]["match"]), []).append((w, p))
    chosen = []
    for m, ps in by_match.items():
        if per_match is not None and per_match < len(ps):
            keep = sorted(rng.choice(len(ps), per_match, replace=False))
            ps = [ps[i] for i in keep]
        chosen += ps
    picks, passes = [], []
    for w, p in chosen:
        f = p["frames"][q]
        s = float(p["s"][q] if "s" in p else FF_S[q])
        rec = {"window": w, "pass": p["i"], "q": q, "s": round(s, 3), "frame": f,
               "tiles": {}}
        for name in names:
            rows = results[w][name]["candidates"].get(str(f))
            if rows is None:
                raise ValueError(f"{name} was not measured at {w} frame {f}")
            if rows:
                picks.append(pick(w, f, rows[0][0], rows[0][1], name, "pass", **{
                    "pass": p["i"], "q": q, "size": rows[0][3]}))
                rec["tiles"][name] = len(picks) - 1          # pick index for now
            else:
                rec["tiles"][name] = None
        passes.append(rec)
    return picks, passes


def calibration_of(name, meta, calib_args):
    from src.ball_heatmap import load_calibration

    if name in calib_args:
        return load_calibration(calib_args[name])
    m = meta.get(name, {})
    if m.get("calib"):
        return m["calib"]
    return load_calibration(m["path"]) if m.get("path") else None


def confident_picks(results, wins, spec, meta, calib_args, rng):
    """N top-1 picks of a detector at p >= TAU on random frames."""
    from src.ball_heatmap import p_cal

    name, rest = spec.split("=", 1)
    tau, n = rest.split(":")
    tau, n = float(tau), int(n)
    cal = calibration_of(name, meta, calib_args)
    pool = []
    for w, r in results.items():
        if name not in r:
            continue
        for f in wins.get(w)["frames"]["random"]:
            rows = r[name]["candidates"].get(str(f)) or []
            if not rows:
                continue
            if len(rows[0]) > 4:
                if cal is None:
                    raise ValueError(f"{name}: rows carry logits but there is no "
                                     f"calibration; pass --calib {name}=JSON")
                score = float(p_cal(rows[0][4], cal))
            else:
                score = rows[0][2]
            if score >= tau:
                pool.append(pick(w, f, rows[0][0], rows[0][1], name, "confident",
                                 size=rows[0][3]))
    return [pool[i] for i in rng.choice(len(pool), min(n, len(pool)), replace=False)]


def corridor_decoys(results, wins, n, q, rng):
    """Grass on a pass segment away from the ball: see the module docstring."""
    from src import soccertrack_v2 as st
    from src.ball_heatmap import BALL_M, local_scale, pitch_to_image

    allp = [(w, p) for w in results for p in wins.get(w)["frames"]["passes"]]
    out = []
    while len(out) < n and allp:
        for _ in range(TRIES):
            w, p = allp[rng.integers(len(allp))]
            d = wins.get(w)
            qq = q if q is not None else int(rng.integers(4))
            f, s = p["frames"][qq], p["s"][qq]
            ts = [t for t in (s + CORRIDOR_DT, s - CORRIDOR_DT)
                  if CORRIDOR_T[0] <= t <= CORRIDOR_T[1]]
            t = ts[rng.integers(len(ts))] if ts else float(np.clip(s, *CORRIDOR_T))
            a, b = np.asarray(p["from"], float), np.asarray(p["to"], float)
            ab = b - a
            nrm = np.array([-ab[1], ab[0]]) / max(np.hypot(*ab), 1e-9)
            g = a + t * ab + rng.uniform(-CORRIDOR_OFF, CORRIDOR_OFF) * nrm
            people = d["people"].get(f, np.zeros((0, 2)))
            if len(people) and np.hypot(*(people - g).T).min() < CORRIDOR_PEOPLE:
                continue
            ug, vg = pitch_to_image([g], d["cal"])[0]
            size = BALL_M * max(local_scale(g, d["cal"]))
            u, v = float(ug), float(vg - size / 2)
            back = st.image_to_pitch([[u, v + size / 2]], d["cal"])[0]
            poly = np.asarray(d["info"]["pitch"], np.float32).reshape(-1, 1, 2)
            if (not np.isfinite(back).all()
                    or cv2.pointPolygonTest(poly, (u, v), False) < 0):
                continue
            cands = [row for k, val in results[w].items()
                     if isinstance(val, dict) and "candidates" in val
                     for row in val["candidates"].get(str(f), [])]
            if any(np.hypot(row[0] - u, row[1] - v) < CORRIDOR_PX for row in cands):
                continue
            out.append(pick(w, f, u, v, "corridor-decoy", "corridor-decoy",
                            **{"pass": p["i"], "q": qq, "size": size, "t": round(t, 3)}))
            break
        else:
            raise RuntimeError(f"no corridor decoy found in {TRIES} tries")
    return out


def rejudge_picks(spec, rng):
    path, n = spec.rsplit(":", 1)
    old = [k for k in json.loads(Path(path).read_text())
           if "decoy" not in k.get("source", "")]
    return [pick(k["window"], k["frame"], k["u"], k["v"], "rejudge", "rejudge",
                 old_key=str(path), old_tile=k["tile"], old_source=k["source"])
            for k in (old[i] for i in rng.choice(len(old), min(int(n), len(old)),
                                                 replace=False))]


def dedupe(picks, px: float):
    """Detector picks at one (window, frame) within `px` of each other as
    one tile credited to each; returns the picks and old index -> new."""
    out, where = [], {}
    for i, p in enumerate(picks):
        j = None
        if px > 0 and p["kind"] in ("pass", "random", "confident"):
            j = next((k for k, o in enumerate(out)
                      if o["kind"] == p["kind"] and (o["window"], o["frame"]) ==
                      (p["window"], p["frame"]) and p["source"] not in o["sources"]
                      and np.hypot(o["u"] - p["u"], o["v"] - p["v"]) <= px), None)
        if j is None:
            out.append(dict(p))
            j = len(out) - 1
        else:
            out[j]["sources"] = out[j]["sources"] + p["sources"]
            out[j]["source"] = "+".join(out[j]["sources"])
        where[i] = j
    return out, where


def pass_frame_of(ap, args) -> int | None:
    """`--pass-frame` as an index into a pass's 4 frames (0-based, as the
    key's "q"; the protocol's sittings use 1, s = 0.417), or None."""
    q = args.pass_frame
    if q is not None and (not 0 <= q < 4 or any("=" in t for t in args.take)):
        ap.error("--pass-frame takes Q in 0-3 (0-based; the protocol's sittings "
                 "use 1) and --take detector names")
    return q


def arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True,
                    help="eval_ball_detector.py --save files (same windows)")
    ap.add_argument("--take", nargs="+", default=[],
                    help="detector=count, top-1 candidates sampled per detector "
                         "(with --pass-frame: detector names)")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="heatmap candidates below are not taken (COCO: 0.02)")
    ap.add_argument("--decoys", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pass-frame", type=int, metavar="Q",
                    help="one tile per pass and detector at frame Q (0-based, 0-3: "
                         "s = 0.25/0.417/0.583/0.75); the protocol's sittings use 1")
    ap.add_argument("--passes-per-match", type=int,
                    help="with --pass-frame: sample this many passes per match")
    ap.add_argument("--dedupe-px", type=float, default=0.0)
    ap.add_argument("--confident", nargs="*", default=[], metavar="NAME=TAU:N")
    ap.add_argument("--calib", nargs="*", default=[], metavar="NAME=JSON")
    ap.add_argument("--corridor-decoys", type=int, default=0)
    ap.add_argument("--rejudge", nargs="*", default=[], metavar="KEY:N")
    return ap


def main():
    ap = arg_parser()
    args = ap.parse_args()
    q = pass_frame_of(ap, args)
    from eval_ball_detector import load_scores

    rng = np.random.default_rng(args.seed)
    merged, meta = load_scores(args.scores)
    results = list(merged.values())
    wins = Windows(merged)
    picks, passes = [], []
    if q is not None:
        picks, passes = pass_picks(merged, wins, args.take, q, args.passes_per_match, rng)
        print(f"  pass frame q = {q} (0-based): s = "
              f"{', '.join(f'{s:.3f}' for s in sorted({r['s'] for r in passes}))}, "
              f"{len(passes)} passes")
    else:
        for spec in args.take:
            name, n = spec.rsplit("=", 1)
            pool = []
            for r in results:
                floor = 0.02 if name == "coco" else args.tau
                for f, rows in r[name]["candidates"].items():
                    if rows and rows[0][2] >= floor:
                        pool.append((r["window"], int(f), rows[0][0], rows[0][1], name,
                                     rows[0][3]))
            for i in rng.choice(len(pool), min(int(n), len(pool)), replace=False):
                w, f, u, v, name, size = pool[i]
                picks.append(pick(w, f, u, v, name, "random", size=size))
    frames = [(r["window"], int(f)) for r in results
              for f in next(iter(v for k, v in r.items()
                                 if isinstance(v, dict) and "candidates" in v))
              ["candidates"]]
    for i in rng.choice(len(frames), min(args.decoys, len(frames)), replace=False):
        w, f = frames[i]
        info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
        poly = np.asarray(info["pitch"], np.float32)
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        while True:
            u, v = rng.uniform(lo, hi)
            if cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(u), float(v)),
                                    False) > 0:
                break
        picks.append(pick(w, f, float(u), float(v), "decoy", "decoy"))
    for spec in args.confident:
        picks += confident_picks(merged, wins, spec, meta,
                                 dict(c.split("=", 1) for c in args.calib), rng)
    if args.corridor_decoys:
        picks += corridor_decoys(merged, wins, args.corridor_decoys, q, rng)
    for spec in args.rejudge:
        picks += rejudge_picks(spec, rng)
    picks, where = dedupe(picks, args.dedupe_px)
    order = rng.permutation(len(picks))
    picks = [picks[i] for i in order]
    number = {int(j): k + 1 for k, j in enumerate(order)}     # pick index -> tile
    tiles = [None] * len(picks)
    for w in sorted({p["window"] for p in picks}):
        want = {}
        for k, p in enumerate(picks):
            if p["window"] == w:
                want.setdefault(p["frame"], []).append(k)
        cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{w}.mp4"))
        idx = 0
        while idx <= max(want):
            ok, image = cap.read()
            if not ok:
                break
            for k in want.get(idx, []):
                tiles[k] = tile(image, picks[k]["u"], picks[k]["v"], k + 1)
            idx += 1
        cap.release()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blank = np.full_like(next(t for t in tiles if t is not None), 255)
    tiles = [t if t is not None else blank for t in tiles]
    for s in range(0, len(tiles), PER_SHEET):
        sheet = tiles[s:s + PER_SHEET]
        sheet += [blank] * (-len(sheet) % COLS)
        rows = [np.hstack(sheet[i:i + COLS]) for i in range(0, len(sheet), COLS)]
        cv2.imwrite(str(out / f"sheet_{s // PER_SHEET + 1}.jpg"), np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    key = []
    for k, p in enumerate(picks):
        rec = {"tile": k + 1, "window": p["window"], "frame": p["frame"], "u": p["u"],
               "v": p["v"], "source": p["source"]}
        rec.update({k2: v2 for k2, v2 in p.items() if k2 not in rec})
        at = None if "pass" in p else wins.get(p["window"])["at"].get(p["frame"])
        if at:
            rec.update({"pass": at[0], "q": at[1]})
        key.append(rec)
    (out / "key.json").write_text(json.dumps(key, indent=1))
    if passes:
        for rec in passes:
            rec["tiles"] = {n: None if i is None else number[where[i]]
                            for n, i in rec["tiles"].items()}
        (out / "key_passes.json").write_text(json.dumps(passes, indent=1))
    print(f"  {len(picks)} tiles on {-(-len(picks) // PER_SHEET)} sheets in {out}")


def _check():
    """The new picks without video: one tile per pass and detector (none for
    a miss), per-match sampling, `--pass-frame` 0-based (1: s = 0.417),
    deduping within 6 px credited to both, confident picks on p_cal, and
    key_passes' tile numbers."""
    s = [0.25, 0.25 + 0.5 / 3, 0.25 + 1 / 3, 0.75]

    class Fake:
        def get(self, w):
            ps = [{"i": i, "frames": [10 * i + q for q in range(4)], "s": s}
                  for i in range(1, 4)]
            return {"info": {"match": w[0]}, "frames": {"passes": ps, "random": [5, 6]}}

    results = {"mw": {"A": {"candidates": {"10": [[100, 50, .9, 4]],
                                           "20": [[300, 50, .9, 4]],
                                           "30": [[9, 9, .2, 4]], "5": [[1, 1, .99, 4]],
                                           "6": []}},
                      "B": {"candidates": {"10": [[103, 54, .5, 4, .1]], "20": [],
                                           "30": [[9, 30, .2, 4, 0.0]],
                                           "5": [[2, 2, .5, 4, 3.0]],
                                           "6": [[3, 3, .9, 4, -1.0]]}}}}
    rng = np.random.default_rng(0)
    picks, passes = pass_picks(results, Fake(), ["A", "B"], 0, None, rng)
    assert [p["frame"] for p in picks] == [10, 10, 20, 30, 30]
    assert passes[1]["tiles"] == {"A": 2, "B": None}
    out, where = dedupe(picks, 6.0)
    assert len(out) == 4 and out[0]["sources"] == ["A", "B"] and out[0]["source"] == "A+B"
    assert where == {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}         # 21 px apart: two tiles
    assert len(dedupe(picks, 0)[0]) == 5
    two, _ = pass_picks(results, Fake(), ["A"], 0, 2, np.random.default_rng(0))
    assert len(two) == 2
    # --pass-frame is 0-based: 1 is the registered frame (s = 0.417), the
    # 2nd of the 4; 4 and NAME=N are refused.
    import contextlib
    import io

    ap = arg_parser()
    base = ["--scores", "x.json", "--out", "o"]
    args = ap.parse_args(base + ["--pass-frame", "1", "--take", "A"])
    q = pass_frame_of(ap, args)
    for i in range(1, 4):
        results["mw"]["A"]["candidates"][str(10 * i + 1)] = [[50, 50, .5, 4]]
    picks1, passes1 = pass_picks(results, Fake(), ["A"], q, None, rng)
    assert q == 1 and [p["frame"] for p in picks1] == [11, 21, 31]
    assert {r["s"] for r in passes1} == {0.417} and {r["q"] for r in passes1} == {1}
    assert passes[0]["s"] == 0.25
    for bad in (["--pass-frame", "4", "--take", "A"],
                ["--pass-frame", "1", "--take", "A=3"]):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                pass_frame_of(ap, ap.parse_args(base + bad))
            raise AssertionError(f"{bad} was accepted")
        except SystemExit:
            pass
    assert pass_frame_of(ap, ap.parse_args(base + ["--take", "A=3"])) is None
    # Confident on p_cal (logit 3.0 at a = 1, b = 0 is 0.95; -1.0 is 0.27).
    cal = {"B": {"calib": {"a": 1.0, "b": 0.0}}}
    conf = confident_picks(results, Fake(), "B=0.9:5", cal, {}, rng)
    assert [(c["frame"], c["source"]) for c in conf] == [(5, "B")]
    try:
        confident_picks(results, Fake(), "B=0.9:5", {}, {}, rng)
        raise AssertionError("logits without a calibration were taken")
    except ValueError:
        pass
    print("  ball_tiles: pass tiles and misses, sampling, 0-based pass frame, dedupe, "
          "confident picks: ok")


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _check()
    else:
        main()
