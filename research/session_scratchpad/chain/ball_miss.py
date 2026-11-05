"""Exploratory (not registered), after chain.json: for the actions where our
ball is not within 3 m (chain S1), is there no ball row near the frame, or
a ball row elsewhere? And how far is it?

    VEO_UNSEAL=... python ball_miss.py
"""
import glob
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from chain import CLIPS, TREE, SEALED, LO, HI, BALL_DF  # noqa: E402


def unit(w):
    m = pd.read_parquet(TREE / w / f"output_{w}" / "metric_v3s_link.parquet")
    ball = m[(m.cls == "ball") & m.x.notna()].assign(x=lambda d: d.x - 52.5, y=lambda d: d.y - 34.0)
    ev = pd.read_csv(CLIPS / f"{w}_events.csv")
    ev = ev[(ev.frame >= LO) & (ev.frame <= HI) & ev.player_id.notna() & ev.x.notna()]
    bf = ball.frame.to_numpy()
    out = []
    for r in ev.itertuples():
        f = int(r.frame) - int(r.frame) % 2
        k = np.flatnonzero(np.abs(bf - f) <= BALL_DF)
        d = np.nan
        if len(k):
            d = float(np.linalg.norm(ball.iloc[k][["x", "y"]].to_numpy(float) - [r.x, r.y], axis=1).min())
        gap = np.abs(bf - f).min() / 25.0 if len(bf) else np.nan
        out.append({"unit": w, "kind": r.kind, "dist m": d, "nearest ball row s": gap})
    return out


def main():
    from src.unseal import unsealed
    for m in sorted(SEALED):
        if not unsealed(m, "chain exploratory: ball misses (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    units = sorted(os.path.basename(p) for p in glob.glob(str(TREE / "st2_*")) if "_117092_" not in p)
    with Pool(os.cpu_count()) as pool:
        df = pd.DataFrame([r for rs in pool.imap_unordered(unit, units) for r in rs])
    n = len(df)
    out = {"status": "exploratory, not registered; after chain.json", "actions": n,
           "ball row within +-4 frames and <= 3 m": round(float((df["dist m"] <= 3).mean()), 4),
           "ball row within +-4 frames but > 3 m": round(float((df["dist m"] > 3).mean()), 4),
           "no ball row within +-4 frames": round(float(df["dist m"].isna().mean()), 4),
           "when > 3 m, distance quartiles m": [round(float(q), 1) for q in df.loc[df["dist m"] > 3, "dist m"].quantile([.25, .5, .75])],
           "when no row, gap to the nearest ball row, quartiles s": [round(float(q), 2) for q in df.loc[df["dist m"].isna(), "nearest ball row s"].quantile([.25, .5, .75])]}
    (HERE / "reading_ball_miss.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
