"""Spot check of the tile classifier's automatic labels: a random sample of
auto-labelled tiles rendered for one blind judge, and the agreement once
judged.

    python spot_check_tiles.py sheets --labels auto_labels.parquet --n 100 \\
        --out DIR [--key-out KEYDIR] [--seed 0] [--exclude KEY.json ...]
    python spot_check_tiles.py score --key KEYDIR/key.json --judged judged.txt
    python spot_check_tiles.py pick --labels auto_labels.parquet --random 300 \\
        --hard 100 --chunk 100 --out DIR [--key-out KEYDIR] [--exclude KEY.json ...]
    python spot_check_tiles.py collect --key KEY.json --judged JUDGDIR --out judged.parquet

`sheets`: half the sample from tiles labelled "B", half from "not" (each
spread over windows by sampling within window strata), shuffled and
numbered; sheets of 5 rows x 6 independent tiles (`ball_tiles.tile`, the
24 px box marked by four corner ticks), 1.3x views (rows 1-3 / 4-5) and
layout.json (tile -> sheet, row, column) in DIR; the key (tile -> window,
frame, u, v, auto label, p) in KEYDIR (default DIR_key). `score`: the
judged labels ("TILE LABEL" lines; only "B" is the ball) against the
automatic ones -- precision of "B", share of "not" that are the ball,
per match. `--exclude`: tiles of earlier keys are not drawn again.

`pick`: tiles to judge for training the classifier on more matches --
`--hard` among those it left to a judge, then `--random` among all the
others, each split evenly over matches; shuffled, in chunks of `--chunk`
tiles rendered as by `sheets` (DIR/chunk_NN, tiles numbered from 1 in each)
and one key (KEYDIR/key.json, with each tile's chunk and stratum).
`collect`: a key's judgements (JUDGDIR/[chunk_NN_]{j1,j2,adj}.txt, "TILE
LABEL" lines; j2 and adj may be partial or missing) merged as by
`label_ball_verify.decide` into one row per tile.
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


def _uid(d: pd.DataFrame) -> pd.Series:
    """A tile's identity: window, frame and position to 0.1 px (keys go
    through JSON)."""
    return (d.window.astype(str) + "/" + d.frame.astype(int).astype(str) + "/"
            + d.u.astype(float).round(1).astype(str) + "/" + d.v.astype(float).round(1).astype(str))


def _drop(lab: pd.DataFrame, exclude: list[str] | None) -> pd.DataFrame:
    if not exclude:
        return lab
    seen = set(pd.concat([_uid(pd.read_json(k)) for k in exclude]))
    return lab[~_uid(lab).isin(seen)]


def _even(pool: pd.DataFrame, k: int, by: str, rng) -> pd.DataFrame:
    groups = sorted(pool[by].unique())
    per = {g: k // len(groups) + (i < k % len(groups)) for i, g in enumerate(groups)}
    out = []
    for g in groups:
        d = pool[pool[by] == g]
        out.append(d.iloc[rng.choice(len(d), min(per[g], len(d)), replace=False)])
    return pd.concat(out)


def pick(lab: pd.DataFrame, n_random: int, n_hard: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lab = lab.assign(match=lab.window.str.split("_").str[1])
    hard = _even(lab[lab.auto == "judge"], n_hard, "match", rng).assign(stratum="hard")
    rand = _even(lab.drop(index=hard.index), n_random, "match", rng).assign(stratum="random")
    s = pd.concat([hard, rand]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return s.rename(columns={"tile": "source_tile"}) if "tile" in s else s


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


def grab(s: pd.DataFrame) -> dict:
    """Each row's tile image (numbered by its `tile`), keyed by row index;
    each window's video decoded once."""
    from ball_tiles import tile

    img = {}
    for w, g in s.groupby("window"):
        want = {}
        for i, r in zip(g.index, g.itertuples()):
            want.setdefault(int(r.frame), []).append((i, r))
        cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{w}.mp4"))
        idx = 0
        while idx <= max(want):
            ok, im = cap.read()
            if not ok:
                break
            for i, r in want.get(idx, []):
                img[i] = tile(im, r.u, r.v, r.tile)
            idx += 1
        cap.release()
    if len(img) < len(s):
        raise SystemExit(f"{len(s) - len(img)} tiles at frames the videos lack")
    return img


def render(s: pd.DataFrame, out: Path, img: dict | None = None):
    """Sheets, views and layout.json of the rows of s, in order."""
    img = grab(s) if img is None else img
    out.mkdir(parents=True, exist_ok=True)
    blank = np.full((TILE_H, TILE_W, 3), 255, np.uint8)
    tiles = [img[i] for i in s.index]
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


def read_labels(path: Path) -> dict:
    if not path.exists():
        return {}
    return {int(a): b for a, b in (l.split()[:2] for l in path.read_text().splitlines()
                                   if l.strip() and not l.startswith("#"))}


def collect(key: pd.DataFrame, jdir: Path) -> pd.DataFrame:
    from label_ball_verify import decide

    key = key.reset_index(drop=True)
    judged, rows = {}, []
    for r in key.itertuples():
        c = getattr(r, "chunk", None)
        pre = "" if c is None or pd.isna(c) else f"chunk_{int(c):02d}_"
        if pre not in judged:
            judged[pre] = [read_labels(jdir / f"{pre}{j}.txt") for j in ("j1", "j2", "adj")]
        j1, j2, adj = (j.get(r.tile) for j in judged[pre])
        if j1 is None:
            raise SystemExit(f"{pre}tile {r.tile}: no first judgement")
        lab, how = decide(j1, j2, adj)
        rows.append({"j1": j1, "j2": j2, "adj": adj, "label": lab, "ball": lab == "B",
                     "decided": how})
    return pd.concat([key, pd.DataFrame(rows)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("sheets")
    c.add_argument("--labels", required=True)
    c.add_argument("--n", type=int, default=100)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    c.add_argument("--key-out")
    c.add_argument("--exclude", nargs="*")
    c = sub.add_parser("score")
    c.add_argument("--key", required=True)
    c.add_argument("--judged", required=True)
    c = sub.add_parser("pick")
    c.add_argument("--labels", required=True)
    c.add_argument("--random", type=int, default=300)
    c.add_argument("--hard", type=int, default=100)
    c.add_argument("--chunk", type=int, default=100)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    c.add_argument("--key-out")
    c.add_argument("--exclude", nargs="*")
    c = sub.add_parser("collect")
    c.add_argument("--key", required=True)
    c.add_argument("--judged", required=True)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    cols = ["tile", "window", "frame", "u", "v", "sources", "p", "auto"]
    if args.cmd in ("sheets", "pick"):
        lab = _drop(pd.read_parquet(args.labels), args.exclude)
        out = Path(args.out)
        kd = Path(args.key_out or f"{out}_key")
        if args.cmd == "sheets":
            s = sample(lab, args.n, args.seed)
            render(s, out)
        else:
            s = pick(lab, args.random, args.hard, args.seed)
            s.insert(0, "chunk", np.arange(len(s)) // args.chunk + 1)
            s.insert(1, "tile", np.arange(len(s)) % args.chunk + 1)
            img = grab(s)
            for c, g in s.groupby("chunk"):
                render(g, out / f"chunk_{c:02d}", img)
            cols = ["chunk", "stratum"] + cols
        kd.mkdir(parents=True, exist_ok=True)
        s[cols].to_json(kd / "key.json", orient="records")
        print(f"  {len(s)} tiles ({(s.auto == 'B').sum()} auto B, "
              f"{(s.auto == 'judge').sum()} left to a judge) -> {out}, key {kd}")
    elif args.cmd == "score":
        key = pd.read_json(args.key)
        print(json.dumps(score(key, read_labels(Path(args.judged))), indent=1))
    else:
        t = collect(pd.read_json(args.key), Path(args.judged))
        t.to_parquet(args.out)
        print(f"  {len(t)} tiles: {t.label.value_counts().to_dict()}, decided "
              f"{t.decided.value_counts().to_dict()} -> {args.out}")


if __name__ == "__main__":
    main()
