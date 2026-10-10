"""Write a clip's touch-based actions (`src/touch_events.py`) as a table.

One row per action in the SPADL-style format of `src/event_schema.py`:
type, start and end frame and second, team, player_from, player_to,
start and end in pitch metres, length, result. Needs a clip the full
pipeline has processed (`tracks_merged.parquet`); with `identities.json`
beside it (`identify_players.py`), players are identities, not tracks.

    python detect_touch_events.py output_stoke_7001 ... [--csv actions.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.touch_events import for_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", nargs="+")
    ap.add_argument("--csv", help="write every clip's actions to this file")
    args = ap.parse_args()
    rows = []
    for out_dir in map(Path, args.out_dirs):
        acts = for_clip(out_dir)
        for a in acts:
            rows.append({"clip": out_dir.name.replace("output_", ""),
                         **a.row()})
        kinds = pd.Series([a.type for a in acts]).value_counts()
        print(f"  {out_dir.name}: " + ", ".join(f"{n} {k}"
                                                for k, n in kinds.items()))
    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"  wrote {args.csv}")


if __name__ == "__main__":
    main()
