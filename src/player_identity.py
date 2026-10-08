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

Tracks with no number read are joined by appearance (`join_by_look`): an
embedding trained so that team-mates in the same kit come out apart
(`train_player_reid.py`), averaged over a track's crops. Identities of the
same team merge most-similar first, never two on screen together and
never two with different numbers, down to a similarity threshold.
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
MIN_CROPS = 5
# PARSeq reads at its confidence bar are rarer and surer: two suffice.
PARSEQ_MIN_CROPS = 2
# Never join by look two identities whose tracks were read (even once,
# confidently) as numbers they do not share.
KEEP_READS_APART = False
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
    look: np.ndarray | None = None
    # PARSeq's reads of the track's crops (`read_numbers_parseq.py`):
    # (digits, confidences); voted instead of `visible` / `logp` when set.
    reads: tuple | None = None
    # The number those reads vote for even from a single confident read --
    # too weak to join on, used only to keep apart (`KEEP_READS_APART`).
    read: int | None = None


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


def parseq_numbers(tracks: dict, reads: dict, min_share: float = MIN_SHARE,
                   min_crops: int = PARSEQ_MIN_CROPS) -> None:
    """Each track's number from PARSeq's reads of it (`track_numbers.json`):
    sets `reads`, and `number` / `share` where `min_share` of at least
    `min_crops` confident reads agree."""
    from . import parseq_reader as pr

    for tid, track in tracks.items():
        r = reads.get(str(tid))
        if not r:
            continue
        track.reads = (r["labels"], np.array(r["confs"]))
        number, share, used = pr.vote(*track.reads)
        if number is not None and share >= min_share:
            track.read = number
        if number is not None and share >= min_share and used >= min_crops:
            track.number, track.share = number, share


def read_numbers(video_path: str, merged: pd.DataFrame, tracks: dict,
                 model, per_track: int = CROPS_PER_TRACK,
                 embedder=None) -> None:
    """Read each track's number from crops spread over its frames, in one
    pass through the video. Fills `visible`, `logp`, `number`, `share`,
    and with `embedder` the track's mean appearance `look`."""
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
    looks = embed(embedder, crops) if embedder is not None else None
    for tid, track in tracks.items():
        mine = owners == tid
        track.visible, track.logp = visible[mine], logp[mine]
        if looks is not None and mine.any():
            v = looks[mine].mean(axis=0)
            track.look = v / max(np.linalg.norm(v), 1e-9)
        number, share, used = jr.vote(track.visible, track.logp)
        if number is not None and share >= MIN_SHARE and used >= MIN_CROPS:
            track.number, track.share = number, share


def embed(model, crops, batch: int = 64):
    """Unit appearance vectors for crops (train_player_reid.build)."""
    import torch

    from train_player_reid import to_tensor

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            out.append(model(to_tensor(crops[i:i + batch])).numpy())
    return np.concatenate(out) if out else np.zeros((0, 128))


def _look(ident):
    vs = [t.look * len(t.frames) for t in ident.tracks if t.look is not None]
    if not vs:
        return None
    v = np.sum(vs, axis=0)
    return v / max(np.linalg.norm(v), 1e-9)


def join_by_look(idents, threshold: float):
    """Merge identities of one team, most similar appearance first, never
    two on screen together or with different numbers, while the cosine
    similarity of their mean looks is at least `threshold`."""
    idents = list(idents)
    while True:
        best, pair = threshold, None
        looks = [_look(i) for i in idents]
        frames = [i.frames for i in idents]
        for a in range(len(idents)):
            if looks[a] is None:
                continue
            for b in range(a + 1, len(idents)):
                A, B = idents[a], idents[b]
                if (looks[b] is None or A.team != B.team
                        or (A.number is not None and B.number is not None
                            and A.number != B.number)):
                    continue
                if KEEP_READS_APART:
                    ra, rb = _reads(A), _reads(B)
                    if ra and rb and not ra & rb:
                        continue
                sim = float(looks[a] @ looks[b])
                if sim > best and not _clash(frames[a], frames[b]):
                    best, pair = sim, (a, b)
        if pair is None:
            return idents
        a, b = pair
        keep, gone = idents[a], idents[b]
        keep.tracks += gone.tracks
        if keep.number is None:
            keep.number, keep.share = gone.number, gone.share
        idents.pop(b)


def _reads(ident) -> set:
    return {t.read for t in ident.tracks if t.read is not None}


def _clash(a, b) -> bool:
    return len(np.intersect1d(a, b, assume_unique=True)) > MAX_SHARED_FRAMES


def identities(tracks: dict, look_threshold: float | None = None,
               use_numbers: bool = False) -> list:
    """Tracks joined into people -- by number and team with `use_numbers`,
    never two at once -- then, with `look_threshold`, by appearance
    (`join_by_look`).

    Numbers are off by default: on a SoccerNet game-state clip with the
    ground truth's own tight boxes, the reader named none of the 9 people
    whose number is legible there right (2 wrong, 7 unread), though it
    reads 90% of the jersey set's held-out tracklets. A wrong number joins
    two people, so it is not trusted to join until it is measured to.
    """
    out = []
    for t in tracks.values():
        if not use_numbers:
            t.number, t.share = None, 0.0
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
    if look_threshold is not None:
        out = join_by_look(out, look_threshold)
        for k, ident in enumerate(out):
            ident.key = k
    # Each identity's number: the vote of all its tracks' crops together.
    for ident in out:
        if any(t.reads is not None for t in ident.tracks):
            from . import parseq_reader as pr

            labels = [g for t in ident.tracks if t.reads for g in t.reads[0]]
            confs = np.array([c for t in ident.tracks if t.reads
                              for c in t.reads[1]])
            number, share, used = pr.vote(labels, confs)
            need = PARSEQ_MIN_CROPS
        else:
            vis = np.concatenate([t.visible for t in ident.tracks])
            logp = np.concatenate([t.logp for t in ident.tracks])
            number, share, used = (jr.vote(vis, logp) if len(vis)
                                   else (None, 0, 0))
            need = MIN_CROPS
        ident.crops = used
        if (use_numbers and number is not None and share >= MIN_SHARE
                and used >= need):
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


# --- Joining by a learned pair score -------------------------------------
#
# Appearance alone cannot decide: on a game-state clip, the same person's
# tracks are 0.71-0.98 alike and different team-mates 0.68 at the median,
# 0.91 at the 90th percentile. What else separates them is where and when
# the tracks are: a person who leaves the picture comes back near where
# they left, soon. `pair_features` puts both in one row per pair of tracks
# and `train_identity_pairs.py` learns from game-state clips which rows are
# one person.

PAIR_FEATURES = ("look", "shared", "gap_s", "jump", "size_ratio", "short")
UNSURE_TEAMS = {"other", "unknown", "None"}


def geometry(merged: pd.DataFrame) -> dict:
    """track id -> (first frame, last frame, start xy, end xy, height), in
    the camera-compensated picture coordinates the re-joining uses."""
    out = {}
    players = merged[merged.cls == "player"]
    for tid, rows in players.groupby("track_id"):
        rows = rows.sort_values("frame")
        h = float(rows.crop_h.median())
        out[int(tid)] = (int(rows.frame.iloc[0]), int(rows.frame.iloc[-1]),
                         rows[["px", "py"]].iloc[0].to_numpy(float),
                         rows[["px", "py"]].iloc[-1].to_numpy(float), h)
    return out


def pair_features(a: Track, b: Track, geo: dict, fps: float) -> dict:
    ga, gb = geo[a.track_id], geo[b.track_id]
    first, second = (ga, gb) if ga[0] <= gb[0] else (gb, ga)
    shared = len(np.intersect1d(a.frames, b.frames, assume_unique=True))
    gap = (second[0] - first[1]) / fps
    h = (first[4] + second[4]) / 2.0
    jump = float(np.linalg.norm(second[2] - first[3])) / max(h, 1.0)
    look = (float(a.look @ b.look) if a.look is not None
            and b.look is not None else np.nan)
    return {"look": look, "shared": float(shared), "gap_s": float(gap),
            "jump": jump, "size_ratio": float(max(ga[4], gb[4])
                                              / max(min(ga[4], gb[4]), 1.0)),
            "short": float(min(len(a.frames), len(b.frames)))}


def join_by_pairs(tracks: dict, geo: dict, fps: float, model,
                  threshold: float = 0.5,
                  max_shared: int = 25) -> list:
    """Average-link clustering of a team's tracks on the pair model's
    probability, merging while the best pair of clusters averages at least
    `threshold` and no two of their tracks share more than `max_shared`
    frames."""
    out = []
    by_team = {}
    # A track the team assignment was unsure of ("other") may join either
    # team: on game-state clips 9 same-person pairs were kept apart only by
    # that label. It is clustered with each team's tracks, and kept by the
    # first cluster that takes it.
    unsure = [t for t in tracks.values() if t.team in UNSURE_TEAMS]
    for t in tracks.values():
        if t.team not in UNSURE_TEAMS:
            by_team.setdefault(t.team, []).append(t)
    if not by_team:
        by_team["unknown"] = []
    taken = set()
    for team, members in by_team.items():
        members = members + [t for t in unsure if t.track_id not in taken]
        n = len(members)
        prob = np.zeros((n, n))
        clash = np.zeros((n, n), dtype=bool)
        rows, idx = [], []
        for i in range(n):
            for j in range(i + 1, n):
                f = pair_features(members[i], members[j], geo, fps)
                clash[i, j] = clash[j, i] = f["shared"] > max_shared
                rows.append([f[k] for k in PAIR_FEATURES])
                idx.append((i, j))
        if rows:
            p = model.predict_proba(np.array(rows))[:, 1]
            for (i, j), v in zip(idx, p):
                prob[i, j] = prob[j, i] = v
        clusters = [[i] for i in range(n)]
        while True:
            best, pair = threshold, None
            for x in range(len(clusters)):
                for y in range(x + 1, len(clusters)):
                    A, B = clusters[x], clusters[y]
                    if clash[np.ix_(A, B)].any():
                        continue
                    score = prob[np.ix_(A, B)].mean()
                    if score > best:
                        best, pair = score, (x, y)
            if pair is None:
                break
            x, y = pair
            clusters[x] += clusters[y]
            clusters.pop(y)
        for c in clusters:
            group = [members[i] for i in c]
            sure = [t for t in group if t.team not in UNSURE_TEAMS]
            if not sure and team != "unknown" and len(by_team) > 1:
                continue          # left for a later team's clustering
            taken.update(t.track_id for t in group)
            out.append(Identity(len(out), team, group))
    for t in unsure:
        if t.track_id not in taken:
            out.append(Identity(len(out), t.team, [t]))
    return out
