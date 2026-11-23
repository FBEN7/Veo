"""v2's stream strips kept in v4's batches (flight 450, action 300, free
450; none of v2's 300 held-out strips), so that 600 steps on clicked shot
balls do not unlearn passes and empty pitch. v3b was trained on all of
them already: nothing new is seen.

Writes stage_stream/labels.parquet, labels_codes.npy (aligned) and
files.txt (the JPEGs to `git lfs pull` and upload with them).

    python stream_subset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
V2 = ROOT / ".cache/soccertrack_v2/ball_strips"
TAKE = {"flight": 450, "action": 300, "free": 450}
SEED = 0


def main():
    t = pd.read_parquet(V2 / "labels.parquet")
    c = np.load(V2 / "labels_codes.npy", mmap_mode="r")
    rng = np.random.default_rng(SEED)
    keep = []
    for kind, n in TAKE.items():
        pool = np.flatnonzero((t.kind == kind).to_numpy() & ~t.holdout.to_numpy().astype(bool))
        keep += sorted(rng.choice(pool, n, replace=False).tolist())
    keep = np.array(sorted(keep))
    out = HERE / "stage_stream"
    out.mkdir(exist_ok=True)
    s = t.iloc[keep].reset_index(drop=True)
    s.to_parquet(out / "labels.parquet")
    np.save(out / "labels_codes.npy", np.asarray(c[keep]))
    (out / "files.txt").write_text("\n".join(s.file) + "\n")
    print(len(s), s.kind.value_counts().to_dict(), s.groupby(["match", "half"]).size().to_dict())


if __name__ == "__main__":
    main()
