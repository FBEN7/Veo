"""Training strips for the ball detector v3 from the verified labels of the
seen windows (protocol v5, part 1; `.cache/soccertrack_v2/ball_verified`).

    python build_verified_strips.py cut      # strips of each window (restart-proof)
    python build_verified_strips.py label    # codes per 4 x 4 cell
    python build_verified_strips.py combine  # + v2's stream strips -> ball_strips_v3
    python build_verified_strips.py --check

The windows are the 14 of `B/` (M2, M3, M4, M5, M8, M10). M6 (118578, the
selection match), M1 (117092) and the test matches are refused.

Each candidate tile gets a class:
- ball: judged B, or auto "B" (the tile classifier `clf/model_AB.pt`);
- hard: judged N (the judges cleared its whole 24 px box);
- spare: judged S / B* (a ball not in play);
- ignore: unjudged and auto "judge", or auto "not" with p > 0.1 (not
  confident), or static (2 or more other frames of the window have a tile
  within 6 px: spare balls, marks) or proposed by COCO (spare balls);
- neg: the other auto "not" tiles.
A frame is located when it has ball tiles, all within 24 px of each other
on both axes (the ball: their mean); ball tiles further apart: the frame is
left out.

Strips (`build_ball_strips.cut` over the window mp4, background from the
window as at inference): per located frame, 3 crops placed around the ball
(`place`, own seed per crop: kind "verified"); per frame, at most one crop
centred on a hard tile (or, in a located frame, a neg tile) that keeps the
ball's 40 px box outside it (kind "verified_neg").

Codes of a strip (cell centre (x0 + 4j + 2, y0 + 4i + 2)):
- base -1 in a located frame (the match ball is known: no other cell is
  it), 0 in an unlocated one;
- 0 within 16 px of an ignore or spare tile;
- -3 on a hard tile's 24 px box;
- around the ball: 0 within 20 px on both axes, 1 within 12 px (the bag,
  pooled as v2's flight bags: a peak, not a glow).

`combine` writes `ball_strips_v3/labels.parquet`, `labels_codes.npy` and
`labels_meta.json`: v2's labelled stream strips (`ball_strips`, less those
whose frames fall in a B window: their weak codes could contradict the
verified ones) and the verified strips, `file` relative to the new
directory. Held out (monitor only): v2's 300, and every strip of the
window `MONITOR` (inside v2's V2_EXCLUDE: v2 never saw it). `draw_w`: the
verified strips' draw weights, so a match's share goes as the square root
of its strips (M2 and M10 hold two thirds of the located frames).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.ball_heatmap import STRIDE, pitch_box
from src.paths import DATA_DIR

V = Path(".cache/soccertrack_v2/ball_verified")
OUT = Path(".cache/soccertrack_v2/ball_strips_verified")
V2_STRIPS = Path(".cache/soccertrack_v2/ball_strips")
V3 = Path(".cache/soccertrack_v2/ball_strips_v3")
REFUSED = {"118578", "117092", "128057", "132831"}     # M6 (selection), M1, M7, M9
MONITOR = "st2_128058_2nd_f015000"
SIZE = 320
CELLS = SIZE // STRIDE
BALL_PX, RING_PX = 12.0, 20.0
SAME_BALL_PX = 24.0
IGNORE_PX = 16.0
HARD_PX = 12.0
STATIC_PX, STATIC_FRAMES = 6.0, 2
P_CONFIDENT = 0.1
PLACEMENTS = 3
NEG_MIN_PX = 200.0
TRIES = 10
SEED = 0
KINDS = ("verified", "verified_neg")


def _uid(d: pd.DataFrame) -> pd.Series:
    return (d.window.astype(str) + "/" + d.frame.astype(int).astype(str) + "/"
            + d.u.astype(float).round(1).astype(str) + "/" + d.v.astype(float).round(1).astype(str))


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def info_of(w: str) -> dict:
    info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
    if str(info["match"]) in REFUSED:
        raise SystemExit(f"{w}: match {info['match']} is never trained on")
    return info


def inputs() -> dict:
    """The label files, by name (their sha256 go into labels_meta.json)."""
    return {"auto": V / "B" / "auto_labels_AB.parquet",
            **{f"judged_{k}": V / "B" / f"judged_{k}.parquet" for k in ("spot", "train", "spot2")}}


def static_flags(t: pd.DataFrame) -> np.ndarray:
    """A tile is static when STATIC_FRAMES or more other frames of its window
    have a tile within STATIC_PX."""
    out = np.zeros(len(t), bool)
    for _, g in t.groupby("window"):
        uv, fr = g[["u", "v"]].to_numpy(float), g.frame.to_numpy()
        d = np.hypot(uv[:, None, 0] - uv[None, :, 0], uv[:, None, 1] - uv[None, :, 1])
        near = (d <= STATIC_PX) & (fr[:, None] != fr[None, :])
        n = np.array([len(set(fr[near[i]])) for i in range(len(g))])
        out[t.index.get_indexer(g.index)] = n >= STATIC_FRAMES
    return out


def tile_classes(auto: pd.DataFrame, judged: pd.DataFrame) -> pd.DataFrame:
    """Every candidate tile of the B windows with its class (see the module)."""
    t = auto.reset_index(drop=True).copy()
    j = judged.assign(uid=_uid(judged)).drop_duplicates("uid", keep="last").set_index("uid")
    uid = _uid(t)
    t["judged"] = uid.map(j.label).to_numpy()
    t["static"] = static_flags(t)
    coco = t.sources.astype(str).str.split("+").apply(lambda s: "coco" in s)
    cls = np.full(len(t), "neg", object)
    a = t.auto.to_numpy()
    cls[(a == "judge") | ((a == "not") & (t.p.to_numpy() > P_CONFIDENT))] = "ignore"
    cls[(a == "not") & (t.static.to_numpy() | coco.to_numpy())] = "ignore"
    cls[a == "B"] = "ball"
    jl = t.judged.to_numpy()
    cls[jl == "B"] = "ball"
    cls[jl == "N"] = "hard"
    cls[np.isin(jl, ["S", "B*"])] = "spare"
    t["cls"] = cls
    return t


def frame_balls(t: pd.DataFrame) -> pd.DataFrame:
    """Per (window, frame): located (with u, v), unlocated or conflict."""
    rows = []
    for (w, f), g in t.groupby(["window", "frame"]):
        b = g[g.cls == "ball"][["u", "v"]].to_numpy(float)
        if not len(b):
            rows.append({"window": w, "frame": f, "status": "unlocated"})
        elif len(b) > 1 and np.abs(b[:, None] - b[None]).max() > SAME_BALL_PX:
            rows.append({"window": w, "frame": f, "status": "conflict"})
        else:
            rows.append({"window": w, "frame": f, "status": "located",
                         "u": float(b[:, 0].mean()), "v": float(b[:, 1].mean())})
    return pd.DataFrame(rows)


def _rng(w: str, frame: int, kind: int, j: int):
    return np.random.default_rng([SEED, zlib.crc32(w.encode()), int(frame), kind, j])


def place(bbox, rng, box) -> tuple[int, int]:
    from build_ball_strips import place as bs_place

    return bs_place(bbox, rng, box)


def targets_of(w: str, t: pd.DataFrame, fb: pd.DataFrame, box) -> list[dict]:
    """The strips of a window: rows for `build_ball_strips.cut` (absolute
    frames), deduplicated by name."""
    from build_ball_strips import _row

    start = int(info_of(w)["start_frame"])
    out = {}
    tw, fw = t[t.window == w], fb[fb.window == w]
    for r in fw.itertuples():
        g = tw[tw.frame == r.frame]
        if r.status == "conflict":
            continue
        if r.status == "located":
            for j in range(PLACEMENTS):
                x0, y0 = place((r.u - BALL_PX, r.v - BALL_PX, r.u + BALL_PX, r.v + BALL_PX),
                               _rng(w, r.frame, 0, j), box)
                out[("verified", x0, y0, r.frame)] = _row("verified", start + r.frame, x0, y0,
                                                           category="located")
        cand = g[g.cls == "hard"]
        if r.status == "located":
            far = np.hypot(g.u - r.u, g.v - r.v) > NEG_MIN_PX
            cand = pd.concat([cand[far[cand.index]], g[(g.cls == "neg") & far]])
        if not len(cand):
            continue
        rng = _rng(w, r.frame, 1, 0)
        c = cand.iloc[0] if (cand.cls == "hard").any() and cand.iloc[0].cls == "hard" \
            else cand.iloc[int(rng.integers(len(cand)))]
        for _ in range(TRIES):
            x0, y0 = place((c.u - HARD_PX, c.v - HARD_PX, c.u + HARD_PX, c.v + HARD_PX), rng, box)
            if r.status != "located" or not (
                    x0 - RING_PX <= r.u <= x0 + SIZE + RING_PX
                    and y0 - RING_PX <= r.v <= y0 + SIZE + RING_PX):
                out[("verified_neg", x0, y0, r.frame)] = _row(
                    "verified_neg", start + r.frame, x0, y0,
                    category="located" if r.status == "located" else "unlocated")
                break
    return list(out.values())


def codes_of(x0: int, y0: int, g: pd.DataFrame, ball) -> np.ndarray:
    """The 80 x 80 codes of a crop at (x0, y0) of a frame whose tiles are `g`
    (with cls) and whose ball is `ball` ((u, v) or None)."""
    c = np.full((CELLS, CELLS), -1 if ball is not None else 0, np.int8)
    cx = x0 + (np.arange(CELLS) + 0.5) * STRIDE
    cy = y0 + (np.arange(CELLS) + 0.5) * STRIDE
    X, Y = np.meshgrid(cx, cy)
    for r in g.itertuples():
        if r.cls in ("ignore", "spare"):
            c[np.hypot(X - r.u, Y - r.v) <= IGNORE_PX] = 0
    for r in g[g.cls == "hard"].itertuples():
        c[(np.abs(X - r.u) <= HARD_PX) & (np.abs(Y - r.v) <= HARD_PX)] = -3
    if ball is not None:
        dx, dy = np.abs(X - ball[0]), np.abs(Y - ball[1])
        c[(dx <= RING_PX) & (dy <= RING_PX)] = 0
        c[(dx <= BALL_PX) & (dy <= BALL_PX)] = 1
    return c


def load_tables():
    paths = inputs()
    auto = pd.read_parquet(paths["auto"])
    judged = pd.concat([pd.read_parquet(paths[k]) for k in paths if k.startswith("judged")],
                       ignore_index=True)
    for w in sorted(set(auto.window)):
        info_of(w)
    t = tile_classes(auto, judged)
    return t, frame_balls(t)


def cmd_cut(args):
    from build_ball_strips import (check_source, merge, partial_path, resume_cut,
                                   window_frames)

    t, fb = load_tables()
    OUT.mkdir(parents=True, exist_ok=True)
    for w in args.windows or sorted(set(t.window)):
        info = info_of(w)
        m, h, start = str(info["match"]), int(info["half"]), int(info["start_frame"])
        first, last = start, start + int(info["frames"]) - 1
        box = pitch_box((int(info["height"]), int(info["width"])), info["pitch"])
        part = partial_path(OUT, m, h, w, first, last)
        if (OUT / "strips.parquet").exists() and not part.exists():
            done = pd.read_parquet(OUT / "strips.parquet", columns=["source"])
            if (done.source.astype(str) == w).any():
                print(f"  {w}: done", flush=True)
                continue
        check_source(OUT, w)
        targets = targets_of(w, t, fb, box)
        rows, kept = resume_cut(
            OUT, targets,
            lambda k0: window_frames(DATA_DIR / "soccertrack_v2" / f"{w}.mp4", start, k0, last),
            first, last, box, m, h,
            extra=dict(match=m, half=h, width=int(info["width"]), height=int(info["height"]),
                       source=w, allow_any=True, window=w))
        merge(OUT, rows, m, h, first, last, kinds=KINDS, source=w)
        part.unlink(missing_ok=True)
        print(f"  {w}: {len(rows)} strips ({kept} kept from an earlier run)", flush=True)


def cmd_label(args):
    t, fb = load_tables()
    s = pd.read_parquet(OUT / "strips.parquet")
    fb = fb.set_index(["window", "frame"])
    codes = np.zeros((len(s), CELLS, CELLS), np.int8)
    tg = {k: g for k, g in t.groupby(["window", "frame"])}
    for i, r in enumerate(s.itertuples()):
        f = int(r.frame) - int(info_of(r.window)["start_frame"])
        st = fb.loc[(r.window, f)]
        ball = (st.u, st.v) if st.status == "located" else None
        codes[i] = codes_of(int(r.x0), int(r.y0), tg[(r.window, f)], ball)
    bag = (codes == 1).reshape(len(s), -1).sum(1)
    if ((s.kind == "verified").to_numpy() & (bag == 0)).any():
        raise SystemExit("a verified strip without its ball")
    if ((s.kind == "verified_neg").to_numpy() & (bag > 0)).any():
        raise SystemExit("a negative strip with the ball")
    np.save(OUT / "labels_codes.tmp.npy", codes)
    os.replace(OUT / "labels_codes.tmp.npy", OUT / "verified_codes.npy")
    s.to_parquet(OUT / "verified_labels.tmp.parquet")
    os.replace(OUT / "verified_labels.tmp.parquet", OUT / "verified_labels.parquet")
    print(f"  {len(s)} strips: {s.kind.value_counts().to_dict()}; bag cells "
          f"{np.median(bag[bag > 0]):.0f} median; -3 cells {int((codes == -3).sum())}", flush=True)


def _in_b_window(v2: pd.DataFrame, windows) -> np.ndarray:
    hit = np.zeros(len(v2), bool)
    for w in windows:
        info = info_of(w)
        a, b = int(info["start_frame"]) - 2, int(info["start_frame"]) + int(info["frames"]) + 1
        hit |= ((v2.match.astype(str) == str(info["match"])).to_numpy()
                & (v2.half.astype(int) == int(info["half"])).to_numpy()
                & v2.frame.between(a, b).to_numpy())
    return hit


def cmd_combine(args):
    v2 = pd.read_parquet(V2_STRIPS / "labels.parquet")
    v2c = np.load(V2_STRIPS / "labels_codes.npy", mmap_mode="r")
    ver = pd.read_parquet(OUT / "verified_labels.parquet")
    verc = np.load(OUT / "verified_codes.npy")
    drop = _in_b_window(v2, sorted(set(ver.window)))
    v2 = v2[~drop].assign(file=lambda d: "../ball_strips/" + d.file.astype(str), draw_w=1.0)
    keep = np.flatnonzero(~drop)
    n_match = ver.groupby("match").size()
    ver = ver.assign(file="../" + OUT.name + "/" + ver.file.astype(str),
                     positive_only=False, weight=1.0, narrow=False, coco_balls=0,
                     holdout=(ver.window == MONITOR).to_numpy(),
                     draw_w=1.0 / np.sqrt(ver.match.map(n_match).to_numpy(float)))
    cols = list(v2.columns) + ["window"]
    table = pd.concat([v2.assign(window=None), ver[[c for c in cols if c in ver]]],
                      ignore_index=True)
    V3.mkdir(parents=True, exist_ok=True)
    codes = np.lib.format.open_memmap(V3 / "labels_codes.tmp.npy", "w+", np.int8,
                                      (len(table), CELLS, CELLS))
    for a in range(0, len(keep), 2000):
        part = keep[a:a + 2000]
        codes[a:a + len(part)] = v2c[part]
    codes[len(keep):] = verc
    codes.flush()
    del codes
    os.replace(V3 / "labels_codes.tmp.npy", V3 / "labels_codes.npy")
    table.to_parquet(V3 / "labels.tmp.parquet")
    os.replace(V3 / "labels.tmp.parquet", V3 / "labels.parquet")
    v2meta = json.loads((V2_STRIPS / "labels_meta.json").read_text())
    meta = {"constants": v2meta["constants"], "coco": v2meta.get("coco"), "seed": v2meta.get("seed"),
            "v3": {"script sha256": sha256(__file__),
                   "inputs sha256": {k: sha256(p) for k, p in inputs().items()},
                   "classifier": {"model_AB.pt": sha256(V / "clf" / "model_AB.pt"),
                                  "trained on": "M6 judged tiles (2,403) + 500 judged tiles of the 6 other seen matches"},
                   "constants": {k: globals()[k] for k in (
                       "BALL_PX", "RING_PX", "SAME_BALL_PX", "IGNORE_PX", "HARD_PX", "STATIC_PX",
                       "STATIC_FRAMES", "P_CONFIDENT", "PLACEMENTS", "NEG_MIN_PX", "TRIES", "SEED",
                       "MONITOR")},
                   "v2 strips dropped (frames in a B window)": int(drop.sum()),
                   "strips": table.kind.value_counts().to_dict(),
                   "held out": int(table.holdout.sum())}}
    (V3 / "labels_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {len(table)} strips ({len(keep)} v2, {len(ver)} verified; {int(drop.sum())} v2 strips "
          f"in B windows dropped), {int(table.holdout.sum())} held out -> {V3}", flush=True)


def _check():
    g = pd.DataFrame({"u": [100.0, 200.0, 300.0], "v": [100.0, 100.0, 100.0],
                      "cls": ["ball", "hard", "ignore"]})
    c = codes_of(0, 0, g, (100.0, 100.0))
    assert c[25, 25] == 1 and c[25, 20] == 0 and c[25, 15] == -1
    assert c[25, 50] == -3 and c[25, 75] == 0 and c[0, 79] == -1
    assert (c == 1).sum() == 36
    c = codes_of(0, 0, g, None)
    assert c[25, 25] == 0 and c[25, 50] == -3 and c[60, 60] == 0
    t = pd.DataFrame({"window": ["w"] * 4, "frame": [1, 2, 3, 4], "u": [10.0, 11.0, 12.0, 90.0],
                      "v": [5.0] * 4})
    assert static_flags(t).tolist() == [True, True, True, False]
    print("  build_verified_strips: codes, static tiles: ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("cut", "label", "combine"))
    ap.add_argument("--windows", nargs="*")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        _check()
        return
    {"cut": cmd_cut, "label": cmd_label, "combine": cmd_combine}[args.cmd](args)


if __name__ == "__main__":
    main()
