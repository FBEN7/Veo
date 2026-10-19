"""Cut tracks where they jump from one person to another.

Joining tracks into people (`player_identity`) cannot fix a track that is
two people: on ten SoccerNet game-state clips the tracker's tracks are 87%
pure -- when players cross, a track carries on with the wrong one -- and
joining raised IDF1 only from 0.617 to 0.645.

Such a switch shows in appearance: the appearance embedding
(`train_player_reid.py`), sampled along a track, is steady for one person
and jumps when the track moves to another. This samples every `EVERY`
frames, compares the mean look of the `WINDOW` samples before each point
with the `WINDOW` after, and cuts where they are less alike than
`threshold`, the deepest dip first, keeping pieces at least
`MIN_PIECE` samples long. The pieces get new track ids; joining them back
into people is `player_identity`'s job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVERY = 5
WINDOW = 3
MIN_PIECE = 3
# Chosen on five game-state clips with rejoining at 0.7 (JOIN), tested on
# five others: IDF1 0.638 as tracked, 0.656 joined by look alone, 0.683 cut
# here and then joined.
THRESHOLD = 0.4
JOIN = 0.7


def looks(video_path: str, merged: pd.DataFrame, embedder,
          every: int = EVERY) -> dict:
    """track id -> (frames sampled, unit embeddings)."""
    from .player_identity import crop, embed
    from .video_frames import frames as read_frames

    players = merged[merged.cls == "player"]
    x = "px_raw" if "px_raw" in players else "px"
    y = "py_raw" if "py_raw" in players else "py"
    wanted = {}
    for tid, rows in players.groupby("track_id"):
        rows = rows.sort_values("frame")
        for r in rows.iloc[::every].itertuples():
            wanted.setdefault(int(r.frame), []).append(
                (int(tid), float(getattr(r, x)), float(getattr(r, y)),
                 float(r.crop_h)))
    crops, owners, at = [], [], []
    for index, frame in read_frames(video_path, wanted):
        for tid, px, py, h in wanted[index]:
            c = crop(frame, px, py, h)
            if c is not None:
                crops.append(c.copy())
                owners.append(tid)
                at.append(index)
    vecs = embed(embedder, crops)
    out = {}
    owners, at = np.array(owners), np.array(at)
    for tid in np.unique(owners):
        k = np.flatnonzero(owners == tid)
        order = np.argsort(at[k])
        out[int(tid)] = (at[k][order], vecs[k][order])
    return out


def cut_points(frames, vecs, threshold: float = THRESHOLD,
               window: int = WINDOW, min_piece: int = MIN_PIECE):
    """Frames to cut at: where the look before and after differ most."""
    n = len(vecs)
    if n < 2 * min_piece:
        return []
    sims = np.full(n, np.inf)
    for i in range(min_piece, n - min_piece + 1):
        a = vecs[max(i - window, 0):i].mean(axis=0)
        b = vecs[i:i + window].mean(axis=0)
        sims[i] = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b),
                                    1e-9))
    cuts = []
    for i in np.argsort(sims):
        if sims[i] >= threshold:
            break
        if all(abs(i - j) >= min_piece for j in cuts):
            cuts.append(int(i))
    return sorted(frames[i] for i in cuts)


def split(merged: pd.DataFrame, sampled: dict,
          threshold: float = THRESHOLD):
    """(tracks with each switch the start of a new track id, the mean unit
    look of every track after splitting)."""
    out = merged.copy()
    out["orig_track_id"] = out.track_id
    next_id = int(out.track_id.max()) + 1
    is_player = out.cls == "player"
    piece_look = {}
    for tid, (frames, vecs) in sampled.items():
        bounds = [-np.inf] + list(cut_points(frames, vecs, threshold)) + [np.inf]
        for k in range(len(bounds) - 1):
            lo, hi = bounds[k], bounds[k + 1]
            new = tid if k == 0 else next_id
            if k > 0:
                rows = (is_player & (out.track_id == tid)
                        & (out.frame >= lo) & (out.frame < hi))
                out.loc[rows, "track_id"] = new
                next_id += 1
            inside = (frames >= lo) & (frames < hi)
            if inside.any():
                v = vecs[inside].mean(axis=0)
                piece_look[int(new)] = v / max(np.linalg.norm(v), 1e-9)
    return out, piece_look


def player_tracks(out_dir):
    """A clip's player tracks: split at switches (`tracks_split.parquet`,
    from identify_players.py --split) where that has been run, else the
    re-joined tracks."""
    from pathlib import Path

    out_dir = Path(out_dir)
    path = out_dir / "tracks_split.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.read_parquet(out_dir / "tracks_merged.parquet")
