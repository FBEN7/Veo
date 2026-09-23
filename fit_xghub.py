"""Fit the expected-goals model on the xGHub dataset.

One command, run once, producing a coefficients file the pipeline can then
apply without ever touching the dataset again.

    python fit_xghub.py --data /path/to/xGHub --out models/xg_xghub.json

The dataset is from github.com/mguti97/xGHub (CVIU, "Once Upon a Goal:
Towards orientation-based shot metrics in football"), distributed through
HuggingFace. Fetch it wherever you have access -- huggingface.co is blocked
by this container's egress proxy -- and point `--data` at the directory
holding `labels/`.

What comes out is a reduced model. xGHub's contribution is body orientation,
and this pipeline has no pose estimation, so the six orientation fields,
`body_part` and `technique` are all unavailable. The fit uses the classical
geometry only: distance and the angle the goal mouth subtends. Say so
wherever the number is shown.

The distance convention is inferred from the data rather than assumed --
`src/xg.py` explains why that matters and how the dataset's own redundancy
settles it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import xg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="directory containing the xGHub labels/ tree")
    ap.add_argument("--out", default="models/xg_xghub.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Reading {args.data} ...")
    raw = xg.load_xghub(Path(args.data))
    goals = int(raw.is_goal.astype(bool).sum())
    print(f"  {len(raw)} shots, {goals} goals "
          f"({goals / max(len(raw), 1):.1%} conversion)")
    if "match_code" in raw.columns:
        print(f"  {raw.match_code.nunique()} matches")

    print("\nWhich reading of distance_from_goal reproduces angle_to_posts?")
    convention, _ = xg.infer_distance_convention(raw)

    features = xg.features_from_xghub(raw, convention)
    print(f"\nFitting on {', '.join(xg.DEFAULT_FEATURES)} "
          f"(held out by match):\n")
    model = xg.fit(features, seed=args.seed,
                   source="xGHub (mguti97), CVIU 2026",
                   convention=convention)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"\nwritten to {out}")
    print("\nThis is the classical part of a model whose published form also "
          "uses body\norientation. Report it as a reduced model, not as the "
          "paper's.")


if __name__ == "__main__":
    main()
