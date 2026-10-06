"""Anonymous players, each one person, named by a number read once.

The tracker's ids are not people. Re-joining fragments (`track_reid`)
still leaves 96-138 player tracks in a 60-90 s window for about 25 people:
a player leaves the picture, is hidden in a crowd or turns into a
team-mate's path, and comes back as someone new. Which tracks are the same
person is what crediting an event to a player needs.

The evidence is the shirt number, wherever it can be read. A track whose
number is read on a few of its crops (`jersey_reader.vote`) is joined to
every other track of the same team with the same number, provided they are
never on screen together -- two tracks seen on the same frame are two
people, whatever is printed on them. An identity's number is then the vote
of all its tracks' crops together, so a track never read clearly inherits
the number of the identity it joined, and a number misread on one track is
outvoted by the rest.

What this cannot do yet: join a track with no number read to anyone. Those
stay their own anonymous identity, with the team they were given.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import jersey_reader as jr

# Crops per track, spread over its frames, and the smallest player worth
# reading: below this a number is a few pixels.
CROPS_PER_TRACK = 40
MIN_HEIGHT_PX = 40.0
# A track's number counts as read when this share of its vote agrees and at
# least this many crops showed a number.
MIN_SHARE = 0.6
MIN_CROPS = 2
# Tracks sharing more frames than this are two people.
MAX_SHARED_FRAMES = 2
# The player's box from the foot point: width as a share of height.
BOX_WIDTH = 0.45


@dataclass
class Track:
    track_id: int
    team: str
    frames: np.ndarray
    visible: np.ndarray = field(default_factory=lambda: np.zeros(0))
    logp: np.ndarray = field(default_factory=lambda: np.zeros((0, 100)))
    number: int | None = None
    share: float = 0.0


@dataclass
class Identity:
    key: int
    team: str
    tracks: list
    number: int | None = None
    share: float = 0.0
    crops: int = 0

    @property
    def frames(self):
        return np.unique(np.concatenate([t.frames for t in self.tracks]))


def tracks_of(merged: pd.DataFrame, min_rows: int = 5) -> dict:
    """The player tracks of a clip, keyed by id, with their team."""
    players = merged[merged.cls == "player"]
    out = {}
    for tid, rows in players.groupby("track_id"):
        if len(rows) < min_rows:
            continue
        team = rows.team.mode()
        out[int(tid)] = Track(int(tid), str(team.iloc[0]) if len(team)
                              else "unknown",
                              np.sort(rows.frame.to_numpy().astype(int)))
    return out


def crop(frame, px, py, h):
    """The player's box from the foot point and height, clipped."""
    w = BOX_WIDTH * h
    H, W = frame.shape[:2]
    x0, x1 = int(max(px - w / 2, 0)), int(min(px + w / 2, W))
    y0, y1 = int(max(py - h, 0)), int(min(py, H))
    if x1 - x0 < 4 or y1 - y0 < 8:
        return None
    return frame[y0:y1, x0:x1]


def read_numbers(video_path: str, merged: pd.DataFrame, tracks: dict,
                 model, per_track: int = CROPS_PER_TRACK) -> None:
    """Read each track's number from crops spread over its frames, in one
    pass through the video. Fills `visible`, `logp`, `number`, `share`."""
    from .video_frames import frames as read_frames

    players = merged[(merged.cls == "player")
                     & merged.track_id.isin(list(tracks))]
    x = "px_raw" if "px_raw" in players.columns else "px"
    y = "py_raw" if "py_raw" in players.columns else "py"
    wanted = {}
    for tid, rows in players.groupby("track_id"):
        rows = rows[rows.crop_h >= MIN_HEIGHT_PX]
        if rows.empty:
            continue
        pick = rows.iloc[np.linspace(0, len(rows) - 1,
                                     min(per_track, len(rows))).astype(int)]
        for r in pick.itertuples():
            wanted.setdefault(int(r.frame), []).append(
                (int(tid), float(getattr(r, x)), float(getattr(r, y)),
                 float(r.crop_h)))
    crops, owners = [], []
    for index, frame in read_frames(video_path, wanted):
        for tid, px, py, h in wanted[index]:
            c = crop(frame, px, py, h)
            if c is not None:
                crops.append(c)
                owners.append(tid)
    visible, logp = jr.read(model, crops)
    owners = np.array(owners)
    for tid, track in tracks.items():
        mine = owners == tid
        track.visible, track.logp = visible[mine], logp[mine]
        number, share, used = jr.vote(track.visible, track.logp)
        if number is not None and share >= MIN_SHARE and used >= MIN_CROPS:
            track.number, track.share = number, share


def _clash(a, b) -> bool:
    return len(np.intersect1d(a, b, assume_unique=True)) > MAX_SHARED_FRAMES


def identities(tracks: dict) -> list:
    """Tracks joined into people by number and team, never two at once."""
    out = []
    numbered = sorted((t for t in tracks.values() if t.number is not None),
                      key=lambda t: -t.share)
    for track in numbered:
        home = None
        for ident in out:
            if (ident.team == track.team and ident.number == track.number
                    and not _clash(ident.frames, track.frames)):
                home = ident
                break
        if home is None:
            out.append(Identity(len(out), track.team, [track], track.number))
        else:
            home.tracks.append(track)
    for track in tracks.values():
        if track.number is None:
            out.append(Identity(len(out), track.team, [track]))
    # Each identity's number: the vote of all its tracks' crops together.
    for ident in out:
        vis = np.concatenate([t.visible for t in ident.tracks])
        logp = np.concatenate([t.logp for t in ident.tracks])
        number, share, used = jr.vote(vis, logp) if len(vis) else (None, 0, 0)
        ident.crops = used
        if number is not None and share >= MIN_SHARE and used >= MIN_CROPS:
            ident.number, ident.share = number, share
        else:
            ident.number, ident.share = None, share
    return out


def lookup(idents) -> dict:
    """track id -> identity."""
    return {t.track_id: ident for ident in idents for t in ident.tracks}


def label(ident) -> str:
    """How an identity is shown: 'team_A #7', or 'team_A player 12'."""
    if ident.number is not None:
        return f"{ident.team} #{ident.number}"
    return f"{ident.team} player {ident.key}"
