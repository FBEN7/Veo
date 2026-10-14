"""Verified ball labels, step 1: candidate tiles for blind judging.

At every measured frame of seen SoccerTrack v2 windows (`eval_ball_detector`'s
frames: four per pass, placebo, actions, random), every source's candidates
are gathered -- v2's top two, v1's and COCO's top one, the tracker's pick
where a dense v2 run exists, and, at a pass frame where none of them lies
within 3 m of the pass segment on the ground, the segment's point at
t_hat = s + 0.4 s (1 - s) (where a decelerating ball should be) -- and
merged within 6 px. Each frame is one row of tiles (`ball_tiles.tile`:
the 24 px box marked by four corner ticks), its candidates in random
order; rows are shuffled across windows and cut into chunks, each judged
blind by several judges (`label_ball_verify.py` merges the judgements).
Decoy rows (one tile on random grass, 30 px or more from every candidate
and 1 m or more from every person) are mixed in to watch the judges.

    python build_ball_verify.py --scores U_a.json U_b.json ... --out DIR \\
        [--key-out KEYDIR] [--rows-per-chunk 120] [--decoy-rows 4] [--seed 5]

`--scores`: `eval_ball_detector.py --save` / `--sealed` files (merged by
`load_scores`; a tracked detector's rows too). Sources by name: `v2`
ball_v2.4000.pt, `v1` ball_heatmap.3000.pt, `coco`, `T` (the tracker);
a window lacking one simply has fewer. Test matches are refused.

Output, per chunk `DIR/chunk_NN/`: `sheet_K.jpg` (5 rows of up to 6
tiles, each numbered in its header; empty cells blank) and `view_K_P.png`
(rows 1-3 / 4-5 of sheet K at 1.3x) and `layout.json` (tile -> sheet,
row, column: for zooming by number; not the key). The key goes to `KEYDIR` (default
`DIR_key`): `chunk_NN.json` (per tile: chunk, tile, row, window, frame, u,
v, size, sources, kind -- "candidate" / "line" / "decoy" -- and the
frame's kinds: pass and q, action, random, placebo) and `rows.json`. The
judges read only `DIR`; the key is for the merge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.paths import DATA_DIR

SOURCES = {"v2": "ball_v2.4000.pt", "v1": "ball_heatmap.3000.pt", "coco": "coco", "T": "T"}
TOP = {"v2": 2, "v1": 1, "coco": 1, "T": 1}
MERGE_PX = 6.0
LINE_M = 3.0                 # a candidate this near the pass segment: no line point
DECOY_PX, DECOY_PEOPLE = 30.0, 1.0
COLS, ROWS_PER_SHEET = 6, 5
TILE_W, TILE_H = 216, 238
VIEW_ROWS, VIEW_SCALE = 3, 1.3
TEST_MATCHES = {"128057", "132831"}


def t_hat(s: float) -> float:
    return s + 0.4 * s * (1 - s)


def seg_dist(g, a, b) -> float:
    """Ground distance from point g to segment a-b (projection clipped to it)."""
    a, b, g = (np.asarray(x, float) for x in (a, b, g))
    ab = b - a
    t = float(np.clip(np.dot(g - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
    return float(np.hypot(*(a + t * ab - g)))


def merge(cands: list[dict], px: float = MERGE_PX) -> list[dict]:
    """Candidates of one frame within `px` of an earlier one become one
    tile credited to each source (first one's position)."""
    out = []
    for c in cands:
        j = next((o for o in out if np.hypot(o["u"] - c["u"], o["v"] - c["v"]) <= px), None)
        if j is None:
            out.append(dict(c, sources=list(c["sources"])))
        else:
            j["sources"] += [s for s in c["sources"] if s not in j["sources"]]
    return out


def frame_kinds(d: dict) -> dict:
    """{frame: [kind, ...]} of a window's measured frames (frames detail)."""
    kinds = {}
    for p in d["passes"]:
        for q, f in enumerate(p["frames"]):
            kinds.setdefault(int(f), []).append({"kind": "pass", "pass": p["i"], "q": q,
                                                 "s": float(p["s"][q])})
    for a in d["actions"]:
        kinds.setdefault(int(a["frame"]), []).append({"kind": "action", "label": a["label"]})
    for f in d["random"]:
        kinds.setdefault(int(f), []).append({"kind": "random"})
    for x in d["placebo_pre"]:
        kinds.setdefault(int(x["frame"]), []).append({"kind": "placebo_pre"})
    for x in d["placebo_post"]:
        kinds.setdefault(int(x["frame"]), []).append({"kind": "placebo_post"})
    return kinds


def ground(u, v, size, cal):
    from src import soccertrack_v2 as st

    size = 0.0 if size is None or not np.isfinite(size) else float(size)
    return st.image_to_pitch([[u, v + size / 2]], cal)[0]


def window_rows(w: str, res: dict, wins, rng) -> list[dict]:
    """One row per measured frame of window w: its merged candidates."""
    from src.ball_heatmap import BALL_M, local_scale, pitch_to_image

    d = wins.get(w)
    cal = d["cal"]
    kinds = frame_kinds(res["frames"])
    passes = {p["i"]: p for p in res["frames"]["passes"]}
    have = {k: res[name]["candidates"] for k, name in SOURCES.items() if name in res}
    frames = sorted(set().union(*[set(map(int, c)) for c in have.values()])) if have else []
    rows = []
    for f in frames:
        cands = []
        for k, c in have.items():
            for r in (c.get(str(f)) or [])[:TOP[k]]:
                cands.append({"u": float(r[0]), "v": float(r[1]),
                              "size": float(r[3]) if len(r) > 3 else float("nan"),
                              "sources": [k], "kind": "candidate"})
        for kd in kinds.get(f, []):
            if kd["kind"] != "pass":
                continue
            p = passes[kd["pass"]]
            near = [c for c in cands
                    if np.isfinite(g := ground(c["u"], c["v"], c["size"], cal)).all()
                    and seg_dist(g, p["from"], p["to"]) <= LINE_M]
            if near:
                continue
            a, b = np.asarray(p["from"], float), np.asarray(p["to"], float)
            g = a + t_hat(kd["s"]) * (b - a)
            ug, vg = pitch_to_image([g], cal)[0]
            size = BALL_M * max(local_scale(g, cal))
            if np.isfinite([ug, vg, size]).all():
                cands.append({"u": float(ug), "v": float(vg - size / 2), "size": float(size),
                              "sources": ["line"], "kind": "line"})
        cands = merge(cands)
        rng.shuffle(cands)
        if cands:
            rows.append({"window": w, "frame": f, "kinds": kinds.get(f, []),
                         "tiles": cands, "decoy": False})
    return rows


def decoy_row(w: str, res: dict, wins, rng, tries: int = 200) -> dict | None:
    """One tile on random grass inside the pitch at a random measured frame."""
    from src import soccertrack_v2 as st

    d = wins.get(w)
    poly = np.asarray(d["info"]["pitch"], np.float32).reshape(-1, 1, 2)
    x0, y0 = poly[:, 0].min(0)
    x1, y1 = poly[:, 0].max(0)
    frames = sorted(set().union(*[set(map(int, val["candidates"])) for val in res.values()
                                  if isinstance(val, dict) and "candidates" in val]))
    for _ in range(tries):
        f = frames[rng.integers(len(frames))]
        u, v = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
        if cv2.pointPolygonTest(poly, (u, v), False) < 0:
            continue
        cands = [r for k, val in res.items() if isinstance(val, dict) and "candidates" in val
                 for r in (val["candidates"].get(str(f)) or [])]
        if any(np.hypot(r[0] - u, r[1] - v) < DECOY_PX for r in cands):
            continue
        g = st.image_to_pitch([[u, v]], d["cal"])[0]
        people = d["people"].get(f, np.zeros((0, 2)))
        if not np.isfinite(g).all() or (len(people) and
                                         np.hypot(*(people - g).T).min() < DECOY_PEOPLE):
            continue
        return {"window": w, "frame": f, "kinds": [], "decoy": True,
                "tiles": [{"u": u, "v": v, "size": float("nan"), "sources": ["decoy"],
                           "kind": "decoy"}]}
    return None


def render_tiles(rows: list[dict], tmp: Path):
    """Every tile's image (`ball_tiles.tile`, numbered later) to tmp/<id>.png,
    one decode per window; returns {id: path}."""
    from ball_tiles import tile

    tmp.mkdir(parents=True, exist_ok=True)
    by_w = {}
    for r in rows:
        for t in r["tiles"]:
            by_w.setdefault(r["window"], {}).setdefault(r["frame"], []).append(t)
    paths = {}
    for w, want in by_w.items():
        cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{w}.mp4"))
        idx, last = 0, max(want)
        while idx <= last:
            ok, image = cap.read()
            if not ok:
                break
            for t in want.get(idx, []):
                p = tmp / f"{t['id']}.png"
                cv2.imwrite(str(p), tile(image, t["u"], t["v"], 0)[22:])   # no header
                paths[t["id"]] = p
            idx += 1
        cap.release()
    missing = [t["id"] for r in rows for t in r["tiles"] if t["id"] not in paths]
    if missing:
        raise SystemExit(f"{len(missing)} tiles at frames the videos lack; nothing written")
    return paths


def header(img, text):
    strip = np.full((22, img.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return np.vstack([strip, img])


def write_chunk(rows: list[dict], paths: dict, out: Path):
    """Sheets (5 rows of up to 6 numbered tiles) and 1.3x views of a chunk."""
    out.mkdir(parents=True, exist_ok=True)
    layout = {t["tile"]: [i // ROWS_PER_SHEET + 1, i % ROWS_PER_SHEET, c]
              for i, r in enumerate(rows) for c, t in enumerate(r["tiles"])}
    (out / "layout.json").write_text(json.dumps(layout))     # tile -> sheet, row, column
    blank = np.full((TILE_H, TILE_W, 3), 255, np.uint8)
    for s in range(0, len(rows), ROWS_PER_SHEET):
        lines = []
        for r in rows[s:s + ROWS_PER_SHEET]:
            cells = [header(cv2.imread(str(paths[t["id"]])), str(t["tile"])) for t in r["tiles"]]
            cells += [blank] * (COLS - len(cells))
            lines.append(np.hstack(cells))
        sheet = np.vstack(lines)
        k = s // ROWS_PER_SHEET + 1
        cv2.imwrite(str(out / f"sheet_{k}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        for p, r0 in enumerate(range(0, len(lines), VIEW_ROWS)):
            part = sheet[r0 * TILE_H:(r0 + VIEW_ROWS) * TILE_H]
            cv2.imwrite(str(out / f"view_{k}_{p + 1}.png"),
                        cv2.resize(part, None, fx=VIEW_SCALE, fy=VIEW_SCALE,
                                   interpolation=cv2.INTER_CUBIC))


def candidate_table(rows: list[dict]) -> "pd.DataFrame":
    """Every candidate tile of `rows` as a table (no sheets, no decoys): for
    `train_tile_classifier.py crops` / `apply`."""
    import pandas as pd

    return pd.DataFrame([{"window": r["window"], "frame": r["frame"], "u": t["u"], "v": t["v"],
                          "size": t["size"], "sources": "+".join(t["sources"]), "kind": t["kind"],
                          "frame_kinds": "+".join(sorted({k["kind"] for k in r["kinds"]}))}
                         for r in rows for t in r["tiles"]])


def build(scores, out: Path, key_out: Path, rows_per_chunk: int, decoy_rows: int,
          seed: int, windows=None, log=print, tiles_out=None) -> dict:
    from ball_tiles import Windows
    from eval_ball_detector import load_scores

    merged, _ = load_scores(scores)
    ws = [w for w in sorted(merged) if windows is None or w in windows]
    for w in ws:
        if any(f"_{m}_" in f"_{w}_" for m in TEST_MATCHES):
            raise SystemExit(f"{w}: a test match is never opened")
    rng = np.random.default_rng(seed)
    wins = Windows(merged)
    rows = []
    for w in ws:
        rows += window_rows(w, merged[w], wins, rng)
    if tiles_out:
        tab = candidate_table(rows)
        tab.to_parquet(tiles_out)
        log(f"  {len(rows)} frames, {len(tab)} candidate tiles -> {tiles_out}")
        return {"rows": len(rows), "tiles": len(tab)}
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    n_chunks = max(1, int(np.ceil(len(rows) / rows_per_chunk)))
    chunks = [rows[i::n_chunks] for i in range(n_chunks)]
    for c in chunks:                                   # decoy rows at random places
        for _ in range(decoy_rows):
            w = ws[rng.integers(len(ws))]
            d = decoy_row(w, merged[w], wins, rng)
            if d is not None:
                c.insert(int(rng.integers(len(c) + 1)), d)
    uid = 0
    for ci, c in enumerate(chunks):
        n = 0
        for ri, r in enumerate(c):
            for t in r["tiles"]:
                n += 1
                uid += 1
                t.update(id=uid, tile=n, row=ri + 1, chunk=ci + 1)
    paths = render_tiles([r for c in chunks for r in c], out / "_tiles")
    key_out.mkdir(parents=True, exist_ok=True)
    allrows = []
    for ci, c in enumerate(chunks):
        name = f"chunk_{ci + 1:02d}"
        write_chunk(c, paths, out / name)
        recs = [{"chunk": ci + 1, "tile": t["tile"], "row": t["row"], "window": r["window"],
                 "frame": r["frame"], "u": round(t["u"], 2), "v": round(t["v"], 2),
                 "size": None if not np.isfinite(t["size"]) else round(t["size"], 2),
                 "sources": t["sources"], "kind": t["kind"], "kinds": r["kinds"]}
                for r in c for t in r["tiles"]]
        (key_out / f"{name}.json").write_text(json.dumps(recs))
        allrows += [{"chunk": ci + 1, "row": ri + 1, "window": r["window"], "frame": r["frame"],
                     "decoy": r["decoy"], "tiles": [t["tile"] for t in r["tiles"]]}
                    for ri, r in enumerate(c)]
    (key_out / "rows.json").write_text(json.dumps(allrows))
    import shutil

    shutil.rmtree(out / "_tiles")
    n_tiles = sum(len(r["tiles"]) for c in chunks for r in c)
    summary = {"windows": len(ws), "rows": sum(len(c) for c in chunks), "tiles": n_tiles,
               "chunks": n_chunks, "decoy rows": sum(r["decoy"] for c in chunks for r in c)}
    log(f"  {summary}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--out")
    ap.add_argument("--key-out")
    ap.add_argument("--tiles-out", help="write the candidate tiles as a parquet and stop "
                                        "(no sheets): the tile classifier's input")
    ap.add_argument("--rows-per-chunk", type=int, default=120)
    ap.add_argument("--decoy-rows", type=int, default=4)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--windows", nargs="*")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.out and not args.tiles_out:
        ap.error("--out or --tiles-out")
    out = Path(args.out or ".")
    build(args.scores, out, Path(args.key_out or f"{out}_key"), args.rows_per_chunk,
          args.decoy_rows, args.seed, args.windows, tiles_out=args.tiles_out)


def _check():
    """merge(), seg_dist(), t_hat() and frame_kinds() on hand-made cases."""
    m = merge([{"u": 0, "v": 0, "sources": ["v2"]}, {"u": 4, "v": 3, "sources": ["coco"]},
               {"u": 20, "v": 0, "sources": ["v1"]}, {"u": 21, "v": 0, "sources": ["v1"]}])
    assert [x["sources"] for x in m] == [["v2", "coco"], ["v1"]]
    assert abs(seg_dist((5, 3), (0, 0), (10, 0)) - 3) < 1e-9
    assert abs(seg_dist((-4, 3), (0, 0), (10, 0)) - 5) < 1e-9
    assert abs(t_hat(0.5) - 0.6) < 1e-9 and t_hat(0) == 0 and t_hat(1) == 1
    d = {"passes": [{"i": 0, "frames": [10, 12, 14, 16], "s": [0.25, 0.417, 0.583, 0.75]}],
         "actions": [{"frame": 12, "label": "PASS"}], "random": [3],
         "placebo_pre": [{"frame": 1}], "placebo_post": []}
    k = frame_kinds(d)
    assert [x["kind"] for x in k[12]] == ["pass", "action"] and k[3] == [{"kind": "random"}]
    print("  build_ball_verify: merge, segment distance, t_hat, frame kinds: ok")


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _check()
    else:
        main()
