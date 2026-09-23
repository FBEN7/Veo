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
import json
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

    # The dataset ships a split, and it is the looser of the two available:
    # 687 of its 704 matches have shots on both sides, so a model can see the
    # teams, pitch and camera it will be tested on. Holding out whole matches
    # is stricter. Both are reported -- if the official split scores better,
    # that gap is the leakage rather than the model.
    split_path = Path(args.data) / "clip_split.json"
    if split_path.exists():
        official = json.loads(split_path.read_text()).get("test", [])
        print(f"\nFitting on the dataset's own split "
              f"({len(official)} held out, matches span both sides):\n")
        xg.fit(features, seed=args.seed, test_ids=official,
               source="xGHub (mguti97), CVIU", convention=convention)

    print(f"\nFitting with whole matches held out (the stricter split):\n")
    model = xg.fit(features, seed=args.seed,
                   source="xGHub (mguti97), CVIU",
                   convention=convention)

    if "play_pattern" in raw.columns:
        model.play_patterns = sorted(raw.play_pattern.dropna().unique())
        print(f"\nTrained on play patterns: "
              f"{raw.play_pattern.value_counts().to_dict()}")

    print("\nWhat it says about positions you can judge by eye:\n")
    print(f"  {'situation':<30s} {'dist':>6s} {'angle':>7s} {'xG':>7s}")
    for _, r in xg.reference_predictions(model).iterrows():
        print(f"  {r.situation:<30s} {r.distance_m:6.1f} "
              f"{r.angle_deg:7.1f} {r.xg:7.3f}")

    if not any("penalt" in str(p).lower() for p in model.play_patterns):
        near = raw[(raw.distance_from_goal.between(10.5, 11.5))
                   & (raw.angle_to_goal.abs() < 8)]
        rate = near.is_goal.mean() if len(near) else float("nan")
        print(f"\n  NO PENALTIES IN THIS DATASET. The model has never seen "
              f"one, so the\n  penalty-spot figure above is an open-play shot "
              f"from 11 m, not a penalty.\n  It matches the {len(near)} "
              f"dataset shots from that distance and angle,\n  which convert "
              f"at {rate:.3f}. A real penalty converts around 0.76 and needs\n"
              f"  its own number.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"\nwritten to {out}")
    print("\nThis is the classical part of a model whose published form also "
          "uses body\norientation. Report it as a reduced model, not as the "
          "paper's.")


if __name__ == "__main__":
    main()
