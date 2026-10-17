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

Protocol v4 (v4-r1) sittings, tracked v2 against raw v2 (`score_ball_track.py`
scores them):

    python ball_tiles.py --scores S_U.json --pairs T ball_v2.4000.pt \\
        [--extra coco@q1] --decoys 10 --corridor-decoys 10 \\
        --offset-decoys 6 [--rejudge OLD/key.json:12] --seed 4 --out DIR

- `--pairs TRACKED RAW`: the pair frames of every pass -- each of its four
  frames (q0-q3) where the two top-1 picks are more than 6 px apart or
  exactly one is empty, and q1 (s = 0.417) always -- with a tile for each
  pick there (none for a miss), deduped within 6 px (one tile credited to
  each; kind "pair"). `--dedupe-px` does not apply: 6 px is the protocol's.
  A pass frame that either detector did not measure is refused;
- `--extra NAME@qK` (e.g. coco@q1): NAME's top-1 at frame K of every pass
  too, deduped with the pair tiles there (V-new's COCO, for H6); where
  frame K is not a pair frame (K other than 1), TRACKED's pick there is
  tiled as well, outside d;
- `--offset-decoys N`: N of the pair sitting's TRACKED picks, each moved
  18-20 px in a uniformly random direction at the same frame, kept only
  13 px or more from every candidate in the scores file at that frame
  (every detector's: v2's top-8, the tracked pick, COCO's when there) and
  inside the picture. With the tracked pick on the ball, the ball sits
  just outside the 24 px box: a lenient judge says "ball";
- `--decoys N`, `--corridor-decoys N` and `--rejudge KEY:N` as above.

The key goes to `--key-out DIR` (with `--pairs`, default `<out>_key`, a
directory apart from the sheets). Each pair tile's record adds "parity"
(of the local frame: "even" / "odd"), "track source" (the tracked
detector's source at that frame, from its "track source" in the scores
file that `eval_ball_detector.py --track` writes: "detected",
"interpolated", "fallback" or "none"; None when the file does not say)
and "px/m" (local pixels per metre, the larger of
`ball_heatmap.local_scale`'s two, at the pick's ground point). Offset
decoys record their base pick and offset. `key_passes.json` lists per
pass its window, match, frames, every frame's track source, and its pair
frames: q, frame, parity, why ("apart", "one empty", "q1"; None for an
`--extra`-only frame), "pair" (in d or not), the track source, "px/m" at
the pass segment's point at s (the same for every pick at that frame) and
each detector's tile (None: a miss). `--dry-run` prints the counts and
writes nothing (no video).
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
# Protocol v4 (v4-r1): pair sittings.
PAIR_PX = 6.0               # tracked and raw more than this apart: a pair frame
PAIR_Q = 1                  # every pass's q1 (s = 0.417) is a pair frame
OFFSET_PX = (18.0, 20.0)    # offset decoys: a tracked pick moved this far
OFFSET_CLEAR = 13.0         # px from every candidate at the frame


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


# ---- Protocol v4: tracked against raw at the pair frames ---------------------------

def track_sources(results: dict, name: str) -> dict:
    """{(window, frame): source} of the tracked detector `name` from the
    merged scores (`eval_ball_detector.load_scores`): its per-window result
    field "track source" ({frame: "detected" / "interpolated" / "fallback"
    / "none"}, written by `eval_ball_detector.py --track`). Empty where the
    files carry none (the key then records None)."""
    out = {}
    for w, r in results.items():
        v = r.get(name)
        src = v.get("track source") if isinstance(v, dict) else None
        if isinstance(src, dict):
            out.update({(w, int(f)): s for f, s in src.items()})
    return out


class Scale:
    """Local pixels per metre (the larger of `local_scale`'s two) at a
    pick's ground point, or at a pass segment's point at the fraction s of
    its frame q; None above the horizon."""

    def __init__(self, wins):
        self.wins = wins

    def _at(self, w, g):
        from src.ball_heatmap import local_scale

        g = np.asarray(g, float)
        if not np.isfinite(g).all():
            return None
        s = max(local_scale(g, self.wins.get(w)["cal"]))
        return round(float(s), 2) if np.isfinite(s) else None

    def point(self, w, u, v, size):
        from src import soccertrack_v2 as st

        size = 0.0 if size is None or not np.isfinite(size) else float(size)
        return self._at(w, st.image_to_pitch([[u, v + size / 2]],
                                             self.wins.get(w)["cal"])[0])

    def segment(self, w, p, q):
        a, b = np.asarray(p["from"], float), np.asarray(p["to"], float)
        return self._at(w, a + float(p["s"][q]) * (b - a))


def _top(results, w, name, f):
    """`name`'s top-1 row at window w's frame f, None when it has none;
    a frame it did not measure is refused."""
    if name not in results[w]:
        raise ValueError(f"{name} is not in the scores file for {w}")
    rows = results[w][name]["candidates"].get(str(f))
    if rows is None:
        raise ValueError(f"{name} was not measured at {w} frame {f}")
    return rows[0] if rows else None


def parse_extra(specs) -> list:
    """`--extra NAME@qK ...` as [(name, K)] (K 0-based, 0-3; "q" optional)."""
    out = []
    for spec in specs:
        name, sep, q = spec.rpartition("@")
        q = q[1:] if q.lower().startswith("q") else q
        if not sep or not name or not q.isdigit() or not 0 <= int(q) < 4:
            raise ValueError(f"--extra takes NAME@qK with K in 0-3, not {spec!r}")
        out.append((name, int(q)))
    return out


def pair_picks(results, wins, tracked: str, raw: str, extra=(), sources=None,
               scale=None):
    """Protocol v4's pair tiles (kind "pair"): at every pass frame where
    `tracked` and `raw` (top-1) are more than PAIR_PX apart or exactly one
    is empty, and at q = PAIR_Q always, one pick each (none for a miss);
    `extra` [(name, q)]: that detector's top-1 at q of every pass too.
    Every pick records pass, q, parity, the tracked source (`sources`
    {(window, frame): source}) and px/m (`scale`: a `Scale`). Returns the
    picks and one record per pass for key_passes.json (tiles as pick
    indices for now)."""
    sources = sources or {}
    names = [tracked, raw] + [n for n, _ in extra]
    if len(set(names)) < len(names):
        raise ValueError(f"--pairs / --extra: a detector twice in {names}")
    picks, passes = [], []
    for w in results:
        d = wins.get(w)
        match = str(results[w].get("match", d["info"]["match"]))
        for p in d["frames"]["passes"]:
            fs = [int(f) for f in p["frames"]]
            rec = {"window": w, "match": match, "pass": p["i"], "frames": fs,
                   "s": [round(float(s), 3) for s in p["s"]], "tracked": tracked,
                   "raw": raw, "extra": {n: q for n, q in extra},
                   "track source": [sources.get((w, f)) for f in fs], "pair frames": []}
            for q, f in enumerate(fs):
                t, r = _top(results, w, tracked, f), _top(results, w, raw, f)
                ex = [n for n, qq in extra if qq == q]
                if (t is None) != (r is None):
                    why = "one empty"
                elif t is not None and np.hypot(t[0] - r[0], t[1] - r[1]) > PAIR_PX:
                    why = "apart"
                elif q == PAIR_Q:
                    why = "q1"
                else:
                    why = None
                if why is None and not ex:
                    continue
                parity = "odd" if f % 2 else "even"
                fr = {"q": q, "frame": f, "parity": parity, "why": why,
                      "pair": why is not None, "track source": sources.get((w, f)),
                      "px/m": scale.segment(w, p, q) if scale else None, "tiles": {}}
                # An --extra-only frame (not a pair frame) tiles the tracked
                # pick too: H6 compares the two there.
                for name in ([tracked, raw] if why else [tracked]) + ex:
                    row = (t if name == tracked else r if name == raw
                           else _top(results, w, name, f))
                    if row is None:
                        fr["tiles"][name] = None
                        continue
                    size = row[3] if len(row) > 3 else None
                    picks.append(pick(w, f, row[0], row[1], name, "pair", **{
                        "pass": p["i"], "q": q, "size": size, "parity": parity,
                        "track source": sources.get((w, f)),
                        "px/m": scale.point(w, row[0], row[1], size) if scale else None}))
                    fr["tiles"][name] = len(picks) - 1
                rec["pair frames"].append(fr)
            passes.append(rec)
    return picks, passes


def offset_decoys(results, wins, picks, tracked: str, n: int, rng) -> list:
    """N offset decoys (see the module docstring) from the pair picks of
    `tracked`, one per base pick, the bases drawn without replacement."""
    base = [p for p in picks if p["kind"] == "pair" and p["source"] == tracked]
    out = []
    for i in (rng.permutation(len(base)) if base else []):
        if len(out) >= n:
            break
        b = base[int(i)]
        w, f = b["window"], b["frame"]
        info = wins.get(w)["info"]
        W, H = info.get("width", np.inf), info.get("height", np.inf)
        cands = np.asarray([row[:2] for val in results[w].values()
                            if isinstance(val, dict) and "candidates" in val
                            for row in val["candidates"].get(str(f)) or []],
                           float).reshape(-1, 2)
        for _ in range(TRIES):
            ang, r = rng.uniform(0, 2 * np.pi), rng.uniform(*OFFSET_PX)
            u, v = b["u"] + r * np.cos(ang), b["v"] + r * np.sin(ang)
            if not (0 <= u < W and 0 <= v < H):
                continue
            if len(cands) and np.hypot(*(cands - (u, v)).T).min() < OFFSET_CLEAR:
                continue
            out.append(pick(w, f, u, v, "offset-decoy", "offset-decoy", **{
                "pass": b["pass"], "q": b["q"], "parity": b["parity"],
                "track source": b["track source"], "size": b.get("size"),
                "base": [b["u"], b["v"]], "offset px": round(float(r), 2),
                "offset deg": round(float(np.degrees(ang)), 1)}))
            break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} of {n} offset decoys: too few {tracked} "
                           f"picks with a clear spot {OFFSET_PX[0]:g}-{OFFSET_PX[1]:g} px "
                           f"away")
    return out


def renumber(passes, where, number):
    """key_passes.json's pick indices as tile numbers (both layouts)."""
    for rec in passes:
        for t in rec["pair frames"] if "pair frames" in rec else [rec]:
            t["tiles"] = {n: None if i is None else number[where[i]]
                          for n, i in t["tiles"].items()}
    return passes


def dedupe(picks, px: float):
    """Detector picks at one (window, frame) within `px` of each other as
    one tile credited to each; returns the picks and old index -> new."""
    out, where = [], {}
    for i, p in enumerate(picks):
        j = None
        if px > 0 and p["kind"] in ("pass", "random", "confident", "pair"):
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
    ap.add_argument("--pairs", nargs=2, metavar=("TRACKED", "RAW"),
                    help="protocol v4: tiles of both at every pass's pair frames")
    ap.add_argument("--extra", nargs="*", default=[], metavar="NAME@qK",
                    help="with --pairs: NAME's top-1 at frame K of every pass too "
                         "(V-new: coco@q1)")
    ap.add_argument("--offset-decoys", type=int, default=0,
                    help="with --pairs: TRACKED picks moved 18-20 px")
    ap.add_argument("--key-out", metavar="DIR",
                    help="where key.json / key_passes.json go (default --out; with "
                         "--pairs, <out>_key)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the tile counts only: no video, nothing written")
    return ap


def pairs_of(ap, args):
    """`--pairs` / `--extra` / `--offset-decoys` checked: (tracked, raw,
    extra) or None."""
    if not args.pairs:
        if args.extra or args.offset_decoys:
            ap.error("--extra and --offset-decoys need --pairs")
        return None
    if args.take or args.pass_frame is not None or args.passes_per_match or args.confident:
        ap.error("--pairs takes no --take, --pass-frame, --passes-per-match or "
                 "--confident")
    if args.dedupe_px not in (0.0, PAIR_PX):
        ap.error(f"--pairs dedupes within the protocol's {PAIR_PX:g} px")
    try:
        extra = parse_extra(args.extra)
    except ValueError as e:
        ap.error(str(e))
    return args.pairs[0], args.pairs[1], extra


def main():
    ap = arg_parser()
    args = ap.parse_args()
    q = pass_frame_of(ap, args)
    pairs = pairs_of(ap, args)
    from eval_ball_detector import load_scores

    rng = np.random.default_rng(args.seed)
    merged, meta = load_scores(args.scores)
    results = list(merged.values())
    wins = Windows(merged)
    picks, passes = [], []
    if pairs:
        tracked, raw, extra = pairs
        sources = track_sources(merged, tracked)
        picks, passes = pair_picks(merged, wins, tracked, raw, extra, sources,
                                   Scale(wins))
        pf = [fr for r in passes for fr in r["pair frames"] if fr["pair"]]
        print(f"  pairs {tracked} / {raw}: {len(passes)} passes, {len(pf)} pair frames "
              f"(track source known at {sum(fr['track source'] is not None for fr in pf)})"
              + (f"; {', '.join(f'{n} at q{k}' for n, k in extra)}" if extra else ""))
    elif q is not None:
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
    if pairs and args.offset_decoys:
        picks += offset_decoys(merged, wins, picks, pairs[0], args.offset_decoys, rng)
    for spec in args.rejudge:
        picks += rejudge_picks(spec, rng)
    picks, where = dedupe(picks, PAIR_PX if pairs else args.dedupe_px)
    order = rng.permutation(len(picks))
    picks = [picks[i] for i in order]
    number = {int(j): k + 1 for k, j in enumerate(order)}     # pick index -> tile
    if args.dry_run:
        kinds = pd.Series([p["kind"] for p in picks]).value_counts()
        print(f"  dry run: {len(picks)} tiles (" + ", ".join(
            f"{k} {n}" for k, n in kinds.items()) + "); nothing written")
        return
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
    if pairs and any(t is None for t in tiles):
        raise SystemExit(f"--pairs: {sum(t is None for t in tiles)} tiles at frames "
                         f"the video does not have (a blank tile would be judged "
                         f"'not the ball'); nothing written")
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
    kdir = Path(args.key_out) if args.key_out else (
        out.parent / f"{out.name}_key" if pairs else out)
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "key.json").write_text(json.dumps(key, indent=1))
    if passes:
        renumber(passes, where, number)
        (kdir / "key_passes.json").write_text(json.dumps(passes, indent=1))
    print(f"  {len(picks)} tiles on {-(-len(picks) // PER_SHEET)} sheets in {out}"
          + (f"; the key in {kdir}" if kdir != out else ""))


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
    _check_pairs()
    print("  ball_tiles: pass tiles and misses, sampling, 0-based pass frame, dedupe, "
          "confident picks, protocol v4 pair frames, extra, offset decoys, track "
          "sources, key numbering: ok")


def _check_pairs():
    """Protocol v4's pair sitting on two synthetic passes: which frames are
    pair frames (apart > 6 px, one empty, q1 always; not both empty, not
    within 6 px off q1), the 6 px dedupe credited to each, COCO at q1, the
    key's parity / track source / px/m, key_passes' numbering, the offset
    decoys' geometry, the track sources' two layouts, and the refusals."""
    import contextlib
    import io
    import tempfile

    class Fake:
        def get(self, w):
            ps = [{"i": i, "frames": [10 * (i + 1) + q for q in range(4)],
                   "s": [0.25, 0.417, 0.583, 0.75], "from": [0, 0], "to": [10, 0]}
                  for i in range(2)]
            return {"info": {"match": "M", "width": 4096, "height": 1080},
                    "frames": {"passes": ps, "random": [5]}}

    class Sc:
        def point(self, w, u, v, size):
            return u / 100

        def segment(self, w, p, q):
            return 10.0 * p["i"] + q

    T = {"10": [[100, 50, .9, 4]], "11": [[200, 50, .9, 4]], "12": [[100, 100, .9, 4]],
         "13": [], "20": [[10, 10, .9, 4]], "21": [], "22": [[60, 60, .9, 4]],
         "23": [[70, 70, .9, 4]], "5": []}
    R = {"10": [[103, 54, .5, 4]], "11": [[202, 50, .5, 4], [400, 40, .3, 4]],
         "12": [], "13": [], "20": [[30, 10, .5, 4]], "21": [[50, 50, .5, 4]],
         "22": [[60, 60, .5, 4]], "23": [[70, 77, .5, 4]], "5": []}
    C = {"11": [[300, 50, .4, 5]], "21": []}
    results = {"mw": {"match": "M", "T": {"candidates": T}, "R": {"candidates": R},
                      "coco": {"candidates": C}}}
    src = {("mw", 10): "detected", ("mw", 11): "interpolated", ("mw", 12): "fallback",
           ("mw", 20): "detected", ("mw", 21): "none", ("mw", 23): "interpolated"}
    picks, passes = pair_picks(results, Fake(), "T", "R", [("coco", 1)], src, Sc())
    pf = [(r["pass"], fr["q"], fr["why"]) for r in passes for fr in r["pair frames"]]
    assert pf == [(0, 1, "q1"), (0, 2, "one empty"), (1, 0, "apart"), (1, 1, "one empty"),
                  (1, 3, "apart")], pf
    assert [(p["frame"], p["source"]) for p in picks] == [
        (11, "T"), (11, "R"), (11, "coco"), (12, "T"), (20, "T"), (20, "R"), (21, "R"),
        (23, "T"), (23, "R")]
    assert passes[1]["pair frames"][1]["tiles"] == {"T": None, "R": 6, "coco": None}
    assert passes[0]["track source"] == ["detected", "interpolated", "fallback", None]
    p23 = picks[7]
    assert (p23["parity"], p23["track source"], p23["px/m"], p23["q"], p23["pass"]) == (
        "odd", "interpolated", 0.7, 3, 1)
    assert picks[3]["parity"] == "even" and picks[3]["track source"] == "fallback"
    assert passes[1]["pair frames"][2]["px/m"] == 13.0                   # pass 1, q3
    assert passes[0]["pair frames"][0]["pair"] and passes[0]["extra"] == {"coco": 1}
    out, where = dedupe(picks, PAIR_PX)
    assert len(out) == 8 and out[0]["sources"] == ["T", "R"] and where[1] == 0
    assert out[1]["sources"] == ["coco"] and where[2] == 1          # 98 px: its own
    # Numbering: shuffled tiles, key_passes points at each detector's tile.
    order = np.random.default_rng(1).permutation(len(out))
    shuffled = [out[i] for i in order]
    number = {int(j): k + 1 for k, j in enumerate(order)}
    renumber(passes, where, number)
    for r in passes:
        for fr in r["pair frames"]:
            for n, t in fr["tiles"].items():
                assert t is None or (n in shuffled[t - 1]["sources"]
                                     and shuffled[t - 1]["frame"] == fr["frame"])
    t11 = passes[0]["pair frames"][0]["tiles"]
    assert t11["T"] == t11["R"] != t11["coco"]
    # Offset decoys: 18-20 px from a tracked pick, 13 px or more from every
    # candidate at that frame, one per base; more than the bases: refused.
    off = offset_decoys(results, Fake(), picks, "T", 4, np.random.default_rng(2))
    assert sorted(o["frame"] for o in off) == [11, 12, 20, 23]
    for o in off:
        d = np.hypot(o["u"] - o["base"][0], o["v"] - o["base"][1])
        assert 18 - 1e-9 <= d <= 20 + 1e-9 and o["source"] == "offset-decoy"
        rows = [row for v in results["mw"].values() if isinstance(v, dict)
                for row in v["candidates"].get(str(o["frame"]), [])]
        assert min(np.hypot(r[0] - o["u"], r[1] - o["v"]) for r in rows) >= OFFSET_CLEAR
    assert {o["frame"]: o["track source"] for o in off}[12] == "fallback"
    try:
        offset_decoys(results, Fake(), picks, "T", 5, np.random.default_rng(2))
        raise AssertionError("more offset decoys than tracked picks")
    except RuntimeError:
        pass
    # --extra at a frame that is not a pair frame (pass 1's q2, frame 22:
    # T and R the same pick): the tracked pick is tiled there as well, R not,
    # outside d; COCO 1.4 px from it shares its tile.
    C.update({"12": [], "22": [[61, 61, .4, 5]]})
    pk2, ps2 = pair_picks(results, Fake(), "T", "R", [("coco", 2)], src, Sc())
    fr22 = next(fr for fr in ps2[1]["pair frames"] if fr["frame"] == 22)
    assert (fr22["why"], fr22["pair"], sorted(fr22["tiles"])) == (None, False,
                                                                 ["T", "coco"])
    assert fr22["tiles"]["T"] is not None and fr22["tiles"]["coco"] is not None
    assert [fr["q"] for fr in ps2[1]["pair frames"]] == [0, 1, 2, 3]
    fr12 = next(fr for fr in ps2[0]["pair frames"] if fr["frame"] == 12)
    assert fr12["pair"] and fr12["tiles"]["coco"] is None and sorted(fr12["tiles"]) == [
        "R", "T", "coco"]
    o2, w2 = dedupe(pk2, PAIR_PX)
    assert o2[w2[fr22["tiles"]["T"]]]["sources"] == ["T", "coco"]
    # Refusals: a frame not measured, a detector twice, a bad --extra.
    del results["mw"]["R"]["candidates"]["13"]
    for bad in (lambda: pair_picks(results, Fake(), "T", "R"),
                lambda: pair_picks(results, Fake(), "T", "T"),
                lambda: parse_extra(["coco@q4"]), lambda: parse_extra(["coco"])):
        try:
            bad()
            raise AssertionError("accepted")
        except ValueError:
            pass
    assert parse_extra(["coco@q1", "x@2"]) == [("coco", 1), ("x", 2)]
    ap = arg_parser()
    base = ["--scores", "x.json", "--out", "o"]
    assert pairs_of(ap, ap.parse_args(base + ["--pairs", "T", "R", "--extra", "coco@q1"])
                    ) == ("T", "R", [("coco", 1)])
    assert pairs_of(ap, ap.parse_args(base)) is None
    for bad in (["--pairs", "T", "R", "--take", "A"], ["--offset-decoys", "3"],
                ["--pairs", "T", "R", "--dedupe-px", "4"], ["--extra", "coco@q1"]):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                pairs_of(ap, ap.parse_args(base + bad))
            raise AssertionError(f"{bad} was accepted")
        except SystemExit:
            pass
    # Track sources: eval's result field "track source", as `load_scores`
    # merges it over two files; a detector without it -> empty.
    from eval_ball_detector import load_scores

    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.json", Path(tmp) / "b.json"
        a.write_text(json.dumps({"results": [{"window": "w1", "T": {
            "candidates": {"4": [], "6": []},
            "track source": {"4": "interpolated", "6": "detected"}},
            "R": {"candidates": {"4": []}}}]}))
        b.write_text(json.dumps({"results": [
            {"window": "w2", "T": {"candidates": {}, "track source": {"8": "fallback"}}},
            {"window": "w1", "T": {"candidates": {}, "track source": {"7": "none"}}}]}))
        merged, _ = load_scores([a, b])
        assert track_sources(merged, "T") == {
            ("w1", 4): "interpolated", ("w1", 6): "detected", ("w1", 7): "none",
            ("w2", 8): "fallback"}
        assert track_sources(merged, "R") == {} == track_sources(merged, "X")


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _check()
    else:
        main()
