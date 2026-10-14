"""Spot check of the tile classifier's automatic labels: a random sample of
auto-labelled tiles rendered for one blind judge, and the agreement once
judged.

    python spot_check_tiles.py sheets --labels auto_labels.parquet --n 100 \\
        --out DIR [--key-out KEYDIR] [--seed 0]
    python spot_check_tiles.py score --key KEYDIR/key.json --judged judged.txt

`sheets`: half the sample from tiles labelled "B", half from "not" (each
spread over windows by sampling within window strata), shuffled and
numbered; sheets of 5 rows x 6 independent tiles (`ball_tiles.tile`, the
24 px box marked by four corner ticks), 1.3x views (rows 1-3 / 4-5) and
layout.json (tile -> sheet, row, column) in DIR; the key (tile -> window,
frame, u, v, auto label, p) in KEYDIR (default DIR_key). `score`: the
judged labels ("TILE LABEL" lines; only "B" is the ball) against the
automatic ones -- precision of "B", share of "not" that are the ball,
per match.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.paths import DATA_DIR

COLS, ROWS = 6, 5
TILE_W, TILE_H = 216, 238
VIEW_ROWS, VIEW_SCALE = 3, 1.3


def sample(lab: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for auto, k in (("B", n // 2), ("not", n - n // 2)):
        pool = lab[lab.auto == auto]
        ws = sorted(pool.window.unique())
        per = {w: k // len(ws) + (i < k % len(ws)) for i, w in enumerate(ws)}
        for w in ws:
            g = pool[pool.window == w]
            m = min(per[w], len(g))
            out.append(g.iloc[rng.choice(len(g), m, replace=False)])
    s = pd.concat(out).sample(frac=1, random_state=seed).reset_index(drop=True)
    s = s.rename(columns={"tile": "source_tile"}) if "tile" in s else s
    s.insert(0, "tile", np.arange(1, len(s) + 1))
    return s


def render(s: pd.DataFrame, out: Path):
    from ball_tiles import tile

    out.mkdir(parents=True, exist_ok=True)
    img = {}
    for w, g in s.groupby("window"):
        want = {}
        for r in g.itertuples():
            want.setdefault(int(r.frame), []).append(r)
        cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{w}.mp4"))
        idx = 0
        while idx <= max(want):
            ok, im = cap.read()
            if not ok:
                break
            for r in want.get(idx, []):
                img[r.tile] = tile(im, r.u, r.v, r.tile)
            idx += 1
        cap.release()
    if len(img) < len(s):
        raise SystemExit(f"{len(s) - len(img)} tiles at frames the videos lack")
    blank = np.full((TILE_H, TILE_W, 3), 255, np.uint8)
    tiles = [img[t] for t in s.tile]
    layout = {}
    per = COLS * ROWS
    for k in range(0, len(tiles), per):
        part = tiles[k:k + per]
        part += [blank] * (-len(part) % COLS)
        lines = [np.hstack(part[i:i + COLS]) for i in range(0, len(part), COLS)]
        sheet = np.vstack(lines)
        sh = k // per + 1
        cv2.imwrite(str(out / f"sheet_{sh}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        for p, r0 in enumerate(range(0, len(lines), VIEW_ROWS)):
            cv2.imwrite(str(out / f"view_{sh}_{p + 1}.png"),
                        cv2.resize(sheet[r0 * TILE_H:(r0 + VIEW_ROWS) * TILE_H], None,
                                   fx=VIEW_SCALE, fy=VIEW_SCALE, interpolation=cv2.INTER_CUBIC))
        for i in range(len(tiles[k:k + per])):
            layout[k + i + 1] = [sh, i // COLS, i % COLS]
    (out / "layout.json").write_text(json.dumps(layout))


def score(key: pd.DataFrame, judged: dict) -> dict:
    key = key.assign(judge=key.tile.map(judged))
    if key.judge.isna().any():
        raise SystemExit(f"{int(key.judge.isna().sum())} tiles not judged")
    key["ball"] = key.judge == "B"
    b, n = key[key.auto == "B"], key[key.auto == "not"]
    return {"tiles": int(len(key)),
            "auto B judged ball (precision)": [int(b.ball.sum()), int(len(b))],
            "auto not judged ball (misses)": [int(n.ball.sum()), int(len(n))],
            "agree": round(float((key.ball == (key.auto == "B")).mean()), 3),
            "per match": {m: {"B prec": [int(g[g.auto == "B"].ball.sum()), int((g.auto == "B").sum())],
                              "not misses": [int(g[g.auto == "not"].ball.sum()),
                                             int((g.auto == "not").sum())]}
                          for m, g in key.groupby(key.window.str.split("_").str[1])},
            "judge labels": key.judge.value_counts().to_dict()}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("sheets")
    c.add_argument("--labels", required=True)
    c.add_argument("--n", type=int, default=100)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    c.add_argument("--key-out")
    c = sub.add_parser("score")
    c.add_argument("--key", required=True)
    c.add_argument("--judged", required=True)
    args = ap.parse_args()
    if args.cmd == "sheets":
        s = sample(pd.read_parquet(args.labels), args.n, args.seed)
        out = Path(args.out)
        render(s, out)
        kd = Path(args.key_out or f"{out}_key")
        kd.mkdir(parents=True, exist_ok=True)
        s[["tile", "window", "frame", "u", "v", "sources", "p", "auto"]].to_json(
            kd / "key.json", orient="records")
        print(f"  {len(s)} tiles ({(s.auto == 'B').sum()} auto B) -> {out}, key {kd}")
    else:
        key = pd.read_json(args.key)
        judged = {int(a): b for a, b in (l.split()[:2] for l in Path(args.judged).read_text()
                                         .splitlines() if l.strip() and not l.startswith("#"))}
        print(json.dumps(score(key, judged), indent=1))


if __name__ == "__main__":
    main()
