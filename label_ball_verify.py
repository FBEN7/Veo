"""Verified ball labels, step 1: merge the blind judgements of
`build_ball_verify.py`'s chunks into labels.

Per tile, two judges' labels (B the match ball in play inside the box,
B* a ball in or by a goal, S a ball off the pitch or not in play, N
anything else) and, where they disagree on B / not B, a third judge's;
the final label is B when two judges say B, else the most common other
label (ties: N). Per frame (a sheet row), the verified ball:

- "ball": one B tile (or B tiles within 12 px of each other: their mean);
- "conflict": B tiles further apart (two balls judged in play): unused;
- "unknown": no candidate is the ball -- the ball was not proposed, is
  hidden or out of play. Not a negative: the frame says nothing of where
  the ball is not, beyond its tiles;
- decoy rows are kept apart (their B rate is the judges' false-alarm rate).

    python label_ball_verify.py --key KEYDIR --results judging.json \\
        --out .cache/soccertrack_v2/ball_verified/A [--sample 100]

`--results`: the judging workflow's result (a list of {chunk, j1, j2,
adj}: {tile: label}); several files merge (each its own chunk set, keyed
by its key dir: --key and --results in the same order).
Writes `tiles.parquet` (every tile: window, match, half, local and
absolute frame, u, v, size, sources, kind, the frame's kinds, each
judge's label, final label, ball flag, how it was decided), `frames.parquet`
(per frame: status, ball u / v / size, the sources that hit it) and
`summary.json`: counts, inter-judge agreement and Cohen's kappa on B /
not B, decoys judged B per judge, per source the share of its tiles that
are the ball and of frames whose ball it found. `--sample N`: N tiles
(half final B, half not; decoys excluded; seed 0) listed in
`spot_check.json` for a human check.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import DATA_DIR

SAME_BALL_PX = 12.0


def final_label(labels: list[str]) -> str:
    """B when at least two labels are B; else the most common non-B label
    (ties and none: N)."""
    if sum(lab == "B" for lab in labels) >= 2:
        return "B"
    rest = Counter(lab for lab in labels if lab and lab != "B")
    if not rest:
        return "N"
    top = rest.most_common()
    return top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else "N"


def decide(j1, j2, adj) -> tuple[str, str]:
    """The final label of a tile and how: "unanimous" (both judges agree on
    B / not B), "adjudicated" (the third judge broke the tie) or
    "unresolved" (a disagreement without a third label: N)."""
    if j1 is not None and j2 is not None and (j1 == "B") == (j2 == "B"):
        return final_label([j1, j2, j1 if j1 == j2 else None]), "unanimous"
    if adj is None:
        return "N", "unresolved"
    return final_label([j1, j2, adj]), "adjudicated"


def kappa(a: list[bool], b: list[bool]) -> float | None:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not len(a):
        return None
    po = float(np.mean(a == b))
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def frame_status(g: pd.DataFrame) -> dict:
    """A row's verified ball from its tiles' final labels."""
    hit = g[g.ball]
    if g.kind.eq("decoy").all():
        return {"status": "decoy"}
    if hit.empty:
        return {"status": "unknown"}
    uv = hit[["u", "v"]].to_numpy(float)
    if len(hit) > 1 and np.hypot(*(uv[:, None] - uv[None]).transpose(2, 0, 1)).max() > SAME_BALL_PX:
        return {"status": "conflict"}
    src = sorted({s for ss in hit.sources for s in ss.split("+")})
    return {"status": "ball", "u": float(uv[:, 0].mean()), "v": float(uv[:, 1].mean()),
            "size": float(np.nanmean(hit["size"].astype(float))) if hit["size"].notna().any()
            else float("nan"), "sources_hit": "+".join(src)}


def merge(keys: list[Path], results: list[list[dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    recs = []
    infos = {}
    for kdir, res in zip(keys, results):
        by_chunk = {int(str(r["chunk"]).split("_")[-1]): r for r in res if r}
        for kf in sorted(kdir.glob("chunk_*.json")):
            c = int(kf.stem.split("_")[-1])
            r = by_chunk.get(c)
            if r is None:
                raise SystemExit(f"{kdir.name}/{kf.name}: no judgements")
            j1, j2, adj = ({int(k): v for k, v in (r.get(x) or {}).items()}
                           for x in ("j1", "j2", "adj"))
            for t in json.loads(kf.read_text()):
                w = t["window"]
                if w not in infos:
                    infos[w] = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
                info = infos[w]
                lab, how = decide(j1.get(t["tile"]), j2.get(t["tile"]), adj.get(t["tile"]))
                recs.append({"batch": kdir.name, "chunk": c, "tile": t["tile"], "row": t["row"],
                             "window": w, "match": str(info["match"]), "half": int(info["half"]),
                             "frame": int(t["frame"]),
                             "abs_frame": int(info["start_frame"]) + int(t["frame"]),
                             "u": t["u"], "v": t["v"], "size": t["size"],
                             "sources": "+".join(t["sources"]), "kind": t["kind"],
                             "frame_kinds": "+".join(sorted({k["kind"] for k in t["kinds"]})),
                             "j1": j1.get(t["tile"]), "j2": j2.get(t["tile"]),
                             "adj": adj.get(t["tile"]), "label": lab, "ball": lab == "B",
                             "decided": how})
    tiles = pd.DataFrame(recs)
    frames = []
    for (b, c, row), g in tiles.groupby(["batch", "chunk", "row"], sort=False):
        f = g.iloc[0]
        frames.append({"batch": b, "chunk": c, "row": row, "window": f.window, "match": f.match,
                       "half": f.half, "frame": f.frame, "abs_frame": f.abs_frame,
                       "frame_kinds": f.frame_kinds, "n_tiles": len(g), **frame_status(g)})
    return tiles, pd.DataFrame(frames)


def summarise(tiles: pd.DataFrame, frames: pd.DataFrame) -> dict:
    real = tiles[tiles.kind != "decoy"]
    both = real.dropna(subset=["j1", "j2"])
    a, b = (both.j1 == "B").tolist(), (both.j2 == "B").tolist()
    dec = tiles[tiles.kind == "decoy"]
    fr = frames[frames.status != "decoy"]
    per_source = {}
    for s in ("v2", "v1", "coco", "T", "line"):
        mine = real[real.sources.str.split("+").apply(lambda x: s in x)]
        balls = fr[fr.status == "ball"]
        per_source[s] = {"tiles": int(len(mine)), "judged ball": int(mine.ball.sum()),
                         "share of verified balls found":
                         round(float(balls.sources_hit.fillna("").str.split("+")
                                     .apply(lambda x: s in x).mean()), 3)
                         if len(balls) else None}
    return {"tiles": int(len(real)), "frames": int(len(fr)),
            "frames by status": fr.status.value_counts().to_dict(),
            "labels": real.label.value_counts().to_dict(),
            "decided": real.decided.value_counts().to_dict(),
            "judges agree on B / not B": round(float(np.mean(np.array(a) == np.array(b))), 4)
            if a else None,
            "kappa B / not B": kappa(a, b),
            "decoys": {"rows": int(len(dec)), "j1 B": int((dec.j1 == "B").sum()),
                       "j2 B": int((dec.j2 == "B").sum()), "final B": int(dec.ball.sum())},
            "per source": per_source,
            "per match frames with a verified ball": {
                m: round(float((g.status == "ball").mean()), 3) for m, g in fr.groupby("match")}}


def spot_sample(tiles: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    real = tiles[tiles.kind != "decoy"]
    rng = np.random.default_rng(seed)
    pos, neg = real[real.ball], real[~real.ball]
    k = min(n // 2, len(pos))
    take = [pos.iloc[rng.choice(len(pos), k, replace=False)],
            neg.iloc[rng.choice(len(neg), min(n - k, len(neg)), replace=False)]]
    s = pd.concat(take).sample(frac=1, random_state=seed).reset_index(drop=True)
    s.insert(0, "spot", np.arange(1, len(s) + 1))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", nargs="+", required=True)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()
    if len(args.key) != len(args.results):
        ap.error("one --results per --key")
    tiles, frames = merge([Path(k) for k in args.key],
                          [json.loads(Path(r).read_text()) for r in args.results])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tiles.to_parquet(out / "tiles.parquet")
    frames.to_parquet(out / "frames.parquet")
    summary = summarise(tiles, frames)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if args.sample:
        s = spot_sample(tiles, args.sample)
        (out / "spot_check.json").write_text(s.to_json(orient="records"))
        print(f"  spot check: {len(s)} tiles ({int(s.ball.sum())} judged ball)")


def _check():
    assert final_label(["B", "B", None]) == "B" and final_label(["B", "N", "B"]) == "B"
    assert final_label(["S", "S", None]) == "S" and final_label(["S", "N", None]) == "N"
    assert decide("B", "B", None) == ("B", "unanimous")
    assert decide("S", "N", None) == ("N", "unanimous")
    assert decide("B", "N", None) == ("N", "unresolved")
    assert decide("B", "N", "B") == ("B", "adjudicated")
    assert decide("B", "S", "S") == ("S", "adjudicated")
    assert kappa([True, False, True, False], [True, False, True, False]) == 1.0
    g = pd.DataFrame({"u": [0.0, 5.0, 40.0], "v": [0.0, 0.0, 0.0], "size": [6.0, 6.0, 6.0],
                      "ball": [True, True, False], "sources": ["v2", "coco", "v1"],
                      "kind": ["candidate"] * 3})
    st = frame_status(g)
    assert st["status"] == "ball" and st["u"] == 2.5 and st["sources_hit"] == "coco+v2"
    g.loc[2, "ball"] = True
    assert frame_status(g)["status"] == "conflict"
    g["ball"] = False
    assert frame_status(g)["status"] == "unknown"
    print("  label_ball_verify: final label, decisions, kappa, frame status: ok")


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _check()
    else:
        main()
