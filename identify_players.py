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


def _look(ident):
    """An identity's mean look, frame-weighted, as a list (or None): what
    joins it to identities of other clips (match_identity.py)."""
    import numpy as np

    vs = [t.look * len(t.frames) for t in ident.tracks if t.look is not None]
    if not vs:
        return None
    v = np.sum(vs, axis=0)
    v = v / max(float(np.linalg.norm(v)), 1e-9)
    return [round(float(x), 5) for x in v]


def identify(out_dir: Path, model, embedder=None,
             look_threshold: float | None = None,
             use_numbers: bool = False, split: bool = False,
             track_numbers: bool = False) -> dict:
    import copy

    if split:
        # Cut tracks where they jump to another person, then rejoin the
        # pieces by look (src/track_split.py).
        from src import track_split as ts

        info = json.loads((out_dir / "clip.json").read_text())
        merged = pd.read_parquet(out_dir / "tracks_merged.parquet")
        cache = out_dir / "split_looks.pkl"
        if cache.exists():
            sampled = pickle.loads(cache.read_bytes())
        else:
            sampled = ts.looks(info["path"], merged, embedder)
            cache.write_bytes(pickle.dumps(sampled))
        df, look = ts.split(merged, sampled)
        df.to_parquet(out_dir / "tracks_split.parquet")
        tracks = pid.tracks_of(df, min_rows=1)
        for tid, t in tracks.items():
            t.look = look.get(tid)
        if track_numbers:
            # PARSeq's reads (read_numbers_parseq.py): tracks join by
            # number and team, and two numbers never join by look.
            pid.parseq_numbers(tracks, json.loads(
                (out_dir / "track_numbers.json").read_text()))
        idents = pid.identities(tracks, ts.JOIN if look_threshold is None
                                else look_threshold,
                                use_numbers=track_numbers)
    else:
        tracks = copy.deepcopy(read_tracks(out_dir, model, embedder))
        idents = pid.identities(tracks, look_threshold, use_numbers)
    blob = {"tracks": {str(t.track_id): {
                "identity": ident.key, "team": ident.team,
                "number": ident.number, "share": round(ident.share, 3),
                "own_number": t.number, "own_share": round(t.share, 3)}
                for ident in idents for t in ident.tracks},
            "identities": [{"key": i.key, "team": i.team, "number": i.number,
                            "share": round(i.share, 3), "crops": i.crops,
                            "look": _look(i),
                            "tracks": [t.track_id for t in i.tracks],
                            "frames": int(len(i.frames))} for i in idents]}
    (out_dir / "identities.json").write_text(json.dumps(blob, indent=1))
    return blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", nargs="+")
    ap.add_argument("--reader",
                    help="weights from train_jersey_reader.py (not needed "
                         "with --split)")
    ap.add_argument("--reid", help="weights from train_player_reid.py: join "
                                   "tracks by appearance too")
    ap.add_argument("--look-threshold", type=float, default=None,
                    help="cosine similarity to join two identities by look")
    ap.add_argument("--numbers", action="store_true",
                    help="join tracks by read shirt numbers too (off: not "
                         "yet measured to be right on broadcast footage)")
    ap.add_argument("--track-numbers", action="store_true",
                    help="with --split: numbers from track_numbers.json "
                         "(read_numbers_parseq.py) join tracks, and keep "
                         "two numbers apart")
    ap.add_argument("--split", action="store_true",
                    help="cut tracks where they jump to another person, "
                         "then rejoin by look (needs --reid); writes "
                         "tracks_split.parquet")
    args = ap.parse_args()
    import torch

    model = None
    if args.reader:
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
                        args.numbers, args.split, args.track_numbers)
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
