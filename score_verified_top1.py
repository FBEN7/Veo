"""Score ball detectors' top-1 against the verified ball of M6 (protocol v5,
v3 selection; `label_ball_verify.py`'s A labels: two blind judges and a
third on disagreements).

    python score_verified_top1.py --scores U_a.json U_b.json [V3.json ...] \\
        --detectors ball_v2.4000.pt ball_v3.500.pt ... --out SCORES.json

Per detector, over M6's measured frames (`A/frames.parquet`; decoy and
conflict frames left out):

- primary: on the 546 frames with a verified ball, the share whose top-1
  is the ball -- within HIT_PX (12) on both axes of the verified centre,
  the judges' 24 px box. An empty candidate list is a miss; a frame with
  no candidates saved at all stops the scorer (every frame must be
  measured). Split by whether v2 proposed the verified ball (the tiles
  were seeded by v2's top 2, so the measure favours v2-like detectors).
- hit@4: any of the top 4 is the ball (the tracker reads the top 4).
- on the 213 frames with no verified ball: the top-1 within MERGE_PX (6)
  of a judged tile (N, S or B*: confirmed false), elsewhere (unverifiable:
  no tile there) or none. Description only: v2's top-1 is always on a
  tile of its own.
- placebos (FF-auto's frames before the kick and after the reception,
  `eval_ball_detector.placebo_summary` over `--scores`): the guard.
- per window, and a paired difference against `--reference` with a block
  bootstrap (1-minute blocks, 4,000 draws, seed 0) for information only.

Candidates are the saved rows (u, v, p_raw, ...), best first, already cut
at the detector's tau (v2-type: p_raw >= 0.05) and top 8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HIT_PX = 12.0
MERGE_PX = 6.0
TOP_K_TRACK = 4
BLOCK = 1500
DRAWS, SEED = 4000, 0
A = Path(".cache/soccertrack_v2/ball_verified/A")
SEALED = {"128057", "132831"}


def hit(rows, u: float, v: float, k: int = 1) -> bool:
    return any(abs(r[0] - u) <= HIT_PX and abs(r[1] - v) <= HIT_PX for r in rows[:k])


def score(merged: dict, name: str, frames: pd.DataFrame, tiles: pd.DataFrame,
          split: str = "v2") -> dict:
    """Per-frame outcomes of one detector, and their summary."""
    recs = []
    judged = tiles[tiles.kind != "decoy"]
    for f in frames.itertuples():
        if f.status not in ("ball", "unknown"):
            continue
        r = merged.get(f.window, {}).get(name)
        if r is None or str(f.frame) not in r["candidates"]:
            raise SystemExit(f"{name}: {f.window} frame {f.frame} not measured")
        rows = r["candidates"][str(f.frame)]
        rec = {"window": f.window, "frame": int(f.frame), "status": f.status}
        if f.status == "ball":
            rec |= {"hit": hit(rows, f.u, f.v), "hit@4": hit(rows, f.u, f.v, TOP_K_TRACK),
                    "v2 proposed": split in str(f.sources_hit).split("+"), "no candidates": not rows}
        else:
            if not rows:
                rec["unknown"] = "none"
            else:
                t = judged[(judged.window == f.window) & (judged.frame == f.frame)]
                d = np.hypot(t.u.to_numpy() - rows[0][0], t.v.to_numpy() - rows[0][1])
                rec["unknown"] = "confirmed false" if (d <= MERGE_PX).any() else "unverifiable"
        recs.append(rec)
    d = pd.DataFrame(recs)
    b, u = d[d.status == "ball"].copy(), d[d.status == "unknown"]
    for c in ("hit", "hit@4", "v2 proposed", "no candidates"):
        b[c] = b[c].astype(bool)

    def rate(x):
        return round(float(x.mean()), 4) if len(x) else None

    return {"frames": b,
            "summary": {
                "ball frames": int(len(b)), "primary (top-1 hit)": rate(b.hit),
                "hit@4": rate(b["hit@4"]), "empty top-1 on ball frames": int(b["no candidates"].sum()),
                "primary, ball proposed by v2": rate(b[b["v2 proposed"]].hit),
                "primary, ball not proposed by v2": rate(b[~b["v2 proposed"]].hit),
                "ball frames not proposed by v2": int((~b["v2 proposed"]).sum()),
                "per window": {w: rate(g.hit) for w, g in b.groupby("window")},
                "unknown frames": int(len(u)),
                "unknown: top-1": u.unknown.value_counts().to_dict()}}


def paired(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """a minus b on the ball frames, block bootstrap (information only)."""
    m = a[a.status == "ball"].merge(b[b.status == "ball"], on=["window", "frame"],
                                    suffixes=("_a", "_b"))
    m["block"] = m.window + ":" + (m.frame // BLOCK).astype(str)
    diff = (m.hit_a.astype(float) - m.hit_b.astype(float))
    blocks = m.block.unique()
    rng = np.random.default_rng(SEED)
    sums = diff.groupby(m.block).sum().reindex(blocks).to_numpy()
    ns = m.groupby("block").size().reindex(blocks).to_numpy()
    boot = []
    for _ in range(DRAWS):
        i = rng.integers(0, len(blocks), len(blocks))
        boot.append(sums[i].sum() / ns[i].sum())
    return {"difference": round(float(diff.mean()), 4),
            "interval 95%": [round(float(np.percentile(boot, 2.5)), 4),
                             round(float(np.percentile(boot, 97.5)), 4)],
            "a only": int((m.hit_a & ~m.hit_b).sum()), "b only": int((~m.hit_a & m.hit_b).sum()),
            "frames": int(len(m)), "blocks": int(len(blocks))}


def cluster_test(a: pd.DataFrame, b: pd.DataFrame, block: int = 250, draws: int = 100_000,
                 seed: int = SEED) -> dict:
    """Protocol v5 part 2's primary test: a minus b on the ball frames, one-
    sided, by sign-flipping the per-block sums of hit differences (10-second
    blocks of a window: frames of one pass or action stay together). Also
    the frame-level exact McNemar (one-sided) for information."""
    from scipy.stats import binomtest

    m = a[a.status == "ball"].merge(b[b.status == "ball"], on=["window", "frame"],
                                    suffixes=("_a", "_b"))
    d = m.hit_a.astype(int) - m.hit_b.astype(int)
    sums = d.groupby(m.window + ":" + (m.frame // block).astype(str)).sum().to_numpy(float)
    obs = float(sums.sum())
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(draws, len(sums))) @ sums
    p = float((np.sum(flips >= obs) + 1) / (draws + 1))
    ao, bo = int((m.hit_a & ~m.hit_b).sum()), int((~m.hit_a & m.hit_b).sum())
    return {"frames": int(len(m)), "blocks": int(len(sums)), "a hit": round(float(m.hit_a.mean()), 4),
            "b hit": round(float(m.hit_b.mean()), 4), "difference": round(obs / max(len(m), 1), 4),
            "a only": ao, "b only": bo, "sign-flip p (one-sided)": round(p, 5),
            "McNemar exact p (one-sided)": round(float(binomtest(ao, ao + bo, 0.5, "greater").pvalue), 5)
            if ao + bo else None}


def placebos(paths, names, windows=None) -> dict:
    import eval_ball_detector as ev

    results, _ = ev.reuse(paths, windows, log=lambda *a: None)
    return {n: {k: v for k, v in ev.placebo_summary(results, n).items() if k != "windows"}
            for n in names}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import eval_ball_detector as ev

    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--detectors", nargs="+", required=True)
    ap.add_argument("--reference", default="ball_v2.4000.pt")
    ap.add_argument("--windows", nargs="+", help="score these windows only")
    ap.add_argument("--labels", default=str(A), help="label_ball_verify.py's output dir")
    ap.add_argument("--split-source", default="v2", help="split the hit by this source")
    ap.add_argument("--test", nargs=2, metavar=("A", "B"), help="the cluster test of A > B")
    ap.add_argument("--no-placebo", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    labels = Path(args.labels)
    frames = pd.read_parquet(labels / "frames.parquet")
    tiles = pd.read_parquet(labels / "tiles.parquet")
    if args.windows:
        frames = frames[frames.window.isin(args.windows)]
    from src.unseal import unsealed

    if any(not unsealed(m, "score") for m in set(frames.match.astype(str)) & SEALED):
        raise SystemExit("sealed match in the labels")
    merged, meta = ev.load_scores(args.scores)
    if any(not unsealed(m, "score") for m in {str(r.get("match")) for r in merged.values()} & SEALED):
        raise SystemExit("sealed match in the scores")
    out, per = {"windows": sorted(set(frames.window)), "labels": str(labels),
                "labels sha256": {f: sha256(labels / f) for f in ("frames.parquet", "tiles.parquet")},
                "scorer sha256": sha256(__file__), "scores": [str(p) for p in args.scores],
                "constants": {"HIT_PX": HIT_PX, "MERGE_PX": MERGE_PX, "TOP_K_TRACK": TOP_K_TRACK,
                              "BLOCK": BLOCK, "DRAWS": DRAWS, "SEED": SEED},
                "detectors": {}}, {}
    for n in args.detectors:
        s = score(merged, n, frames, tiles, args.split_source)
        per[n] = s["frames"]
        out["detectors"][n] = s["summary"] | {"ckpt": meta.get(n, {}).get("ckpt sha256")}
    if not args.no_placebo:
        for n, p in placebos(args.scores, args.detectors, args.windows).items():
            out["detectors"][n]["placebo"] = p
    if args.reference in per:
        out["paired against " + args.reference] = {
            n: paired(per[n], per[args.reference]) for n in args.detectors if n != args.reference}
    if args.test:
        a, b = args.test
        out["registered test"] = {"a": a, "b": b, **cluster_test(per[a], per[b])}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
