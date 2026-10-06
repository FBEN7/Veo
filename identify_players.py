"""Join a clip's player tracks into people and read their numbers.

Runs `src/player_identity.py` on clips the full pipeline has processed
(`tracks_merged.parquet`, written by `score_soccernet.run_pipeline`) and
writes `identities.json` beside the tracks: per track, the person it was
joined to, that person's team and number, and how sure the vote is.

    python identify_players.py output_stoke_7001 ... \\
        --reader .cache/jersey_reader.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src import jersey_reader as jr
from src import player_identity as pid


def identify(out_dir: Path, model) -> dict:
    info = json.loads((out_dir / "clip.json").read_text())
    merged = pd.read_parquet(out_dir / "tracks_merged.parquet")
    tracks = pid.tracks_of(merged)
    pid.read_numbers(info["path"], merged, tracks, model)
    idents = pid.identities(tracks)
    blob = {"tracks": {str(t.track_id): {
                "identity": ident.key, "team": ident.team,
                "number": ident.number, "share": round(ident.share, 3),
                "own_number": t.number, "own_share": round(t.share, 3)}
                for ident in idents for t in ident.tracks},
            "identities": [{"key": i.key, "team": i.team, "number": i.number,
                            "share": round(i.share, 3), "crops": i.crops,
                            "tracks": [t.track_id for t in i.tracks],
                            "frames": int(len(i.frames))} for i in idents]}
    (out_dir / "identities.json").write_text(json.dumps(blob, indent=1))
    return blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", nargs="+")
    ap.add_argument("--reader", required=True,
                    help="weights from train_jersey_reader.py")
    args = ap.parse_args()
    import torch

    model = jr.build()
    model.load_state_dict(torch.load(args.reader))
    model.eval()
    for out_dir in map(Path, args.out_dirs):
        t0 = time.time()
        blob = identify(out_dir, model)
        idents = blob["identities"]
        named = [i for i in idents if i["number"] is not None]
        tracks_named = sum(len(i["tracks"]) for i in named)
        print(f"  {out_dir.name}: {len(blob['tracks'])} tracks -> "
              f"{len(idents)} identities, {len(named)} with a number "
              f"(covering {tracks_named} tracks); "
              + ", ".join(sorted(f"{i['team']}#{i['number']}"
                                 for i in named))
              + f"  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
