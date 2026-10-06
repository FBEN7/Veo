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
import pickle
import time
from pathlib import Path

import pandas as pd

from src import jersey_reader as jr
from src import player_identity as pid


def read_tracks(out_dir: Path, model, embedder=None) -> dict:
    """Each track's number votes and look, cached beside the tracks so the
    joining can be re-run without reading the video again."""
    cache = out_dir / "identity_tracks.pkl"
    if cache.exists():
        tracks = pickle.loads(cache.read_bytes())
        if embedder is None or all(t.look is not None or not len(t.visible)
                                   for t in tracks.values()):
            return tracks
    info = json.loads((out_dir / "clip.json").read_text())
    merged = pd.read_parquet(out_dir / "tracks_merged.parquet")
    tracks = pid.tracks_of(merged)
    pid.read_numbers(info["path"], merged, tracks, model, embedder=embedder)
    cache.write_bytes(pickle.dumps(tracks))
    return tracks


def identify(out_dir: Path, model, embedder=None,
             look_threshold: float | None = None,
             use_numbers: bool = False) -> dict:
    import copy

    tracks = copy.deepcopy(read_tracks(out_dir, model, embedder))
    idents = pid.identities(tracks, look_threshold, use_numbers)
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
    ap.add_argument("--reid", help="weights from train_player_reid.py: join "
                                   "tracks by appearance too")
    ap.add_argument("--look-threshold", type=float, default=None,
                    help="cosine similarity to join two identities by look")
    ap.add_argument("--numbers", action="store_true",
                    help="join tracks by read shirt numbers too (off: not "
                         "yet measured to be right on broadcast footage)")
    args = ap.parse_args()
    import torch

    model = jr.build()
    model.load_state_dict(torch.load(args.reader))
    model.eval()
    embedder = None
    if args.reid:
        from train_player_reid import build

        embedder = build()
        embedder.load_state_dict(torch.load(args.reid))
        embedder.eval()
    for out_dir in map(Path, args.out_dirs):
        t0 = time.time()
        blob = identify(out_dir, model, embedder, args.look_threshold,
                        args.numbers)
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
