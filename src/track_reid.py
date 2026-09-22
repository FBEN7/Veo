"""Rejoin track fragments that belong to the same player.

ByteTrack drops an identity whenever a player is occluded, leaves frame, or is
missed for a few frames, and issues a fresh id when they reappear. On a
two-minute clip that turned 22 players into 163 track ids, which corrupts
everything computed per player: distance covered is split across fragments,
top speed is measured over stubs, and possession changes hands whenever an id
is reissued.

Two fragments are the same player when the second starts close to where the
first was *heading*. The velocity at the end of the first fragment carries it
forward, so the test is against a prediction rather than against the last
seen position, and the radius around that prediction has to cover how far the
prediction drifts -- not how far a footballer can run. Sizing it as a sprint
was the fault that made this module merge the wrong fragments; see
PREDICTION_DRIFT_MS.

Matching is mutual. A fragment pair is joined only when each is the other's
best candidate, which avoids a popular fragment absorbing several distinct
players. Gaps are tried shortest first, so confident local joins are made
before speculative long ones.

Scale: the gap window is a contiguous slice of fragments sorted by start
frame, found by binary search. Comparing every fragment against every other
is quadratic and becomes the bottleneck on a full match; this stays linear in
the number of candidates actually within reach.
"""

from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

# How fast the *prediction* goes wrong, in metres per second of gap.
#
# The name matters. This was called a footballer's sprint, on the reasoning
# that two fragments further apart than a player could run must be different
# people. The distance being tested is not between the two fragments -- it is
# between a fragment's *predicted* position, already carried forward at the
# player's own velocity, and where the candidate actually starts. What the
# radius has to cover is how far that prediction drifts.
#
# Measured, the drift on real ByteTrack fragments is 7.8 to 13.3 m/s at the
# median across four windows, so 8.0 turns out to be about right -- for a
# different reason than it was chosen for.
PREDICTION_DRIFT_MS = 8.0

# Gap lengths to attempt, shortest first.
GAP_SCHEDULE = (0.4, 0.8, 1.5, 2.5)

# Slack on the reachable radius at zero gap, absorbing detection jitter and
# the fact that a player's position is their feet, not their centre of mass.
#
# Raised from 1.5 on `bench_track_continuity.py`, which cuts real tracks into
# pieces and asks the matcher to rejoin them. Scored with the weights real
# dropout lengths have, tuned on three windows and checked on a fourth:
#
#     drift m/s   margin   tune F1   held out
#         8.0        0.5      0.39       0.48
#         8.0        1.5      0.52       0.61     (as shipped)
#         8.0        3.0      0.65       0.78     <- adopted
#        11.0        3.0      0.64       0.75
#         6.0        3.0      0.63       0.75
#
# Margin 3.0 is the best column at every drift tried, and drift 6 to 11 is
# flat, so neither is a fitted peak.
#
# An earlier pass of this sweep recommended the opposite -- a radius three
# times *tighter* -- and was wrong because the benchmark cut tracks cleanly.
# A clean cut leaves the second piece continuing the first's motion exactly,
# giving a median drift of 1.6-2.9 m/s where real fragments give 7.8-13.3,
# because a real identity is dropped when the player is occluded and the
# frames the velocity is read from are the corrupted ones. The benchmark now
# injects noise at each cut, calibrated to reproduce the observed
# distribution. The tight setting scored better on the clean version and
# merged *fewer* fragments in production, which is what caught it.
POSITION_MARGIN_M = 3.0

# Frames at a fragment's edge used to estimate its velocity.
VELOCITY_WINDOW = 5


class _Fragments:
    """Fragment endpoints as parallel arrays, ordered by start frame."""

    def __init__(self, players: pd.DataFrame, px_per_m: float):
        self.ids: list = []
        starts, ends = [], []
        sx, sy, ex, ey = [], [], [], []
        vx, vy = [], []
        teams: list = []

        has_team = "team" in players.columns

        for tid, group in players.groupby("track_id", sort=False):
            g = group.sort_values("frame")
            f = g.frame.to_numpy(dtype=float)
            x = g.px.to_numpy(dtype=float) / px_per_m
            y = g.py.to_numpy(dtype=float) / px_per_m

            self.ids.append(tid)
            teams.append(str(g.team.iloc[0]) if has_team else "")
            starts.append(f[0])
            ends.append(f[-1])
            sx.append(x[0]); sy.append(y[0])
            ex.append(x[-1]); ey.append(y[-1])

            # Velocity over the last few frames, in metres per frame.
            k = min(VELOCITY_WINDOW, len(f) - 1)
            if k >= 1 and (f[-1] - f[-1 - k]) > 0:
                span = f[-1] - f[-1 - k]
                vx.append((x[-1] - x[-1 - k]) / span)
                vy.append((y[-1] - y[-1 - k]) / span)
            else:
                vx.append(0.0); vy.append(0.0)

        order = np.argsort(np.array(starts))
        self.ids = [self.ids[i] for i in order]
        self.start = np.array(starts)[order]
        self.end = np.array(ends)[order]
        self.sx = np.array(sx)[order]
        self.sy = np.array(sy)[order]
        self.ex = np.array(ex)[order]
        self.ey = np.array(ey)[order]
        self.vx = np.array(vx)[order]
        self.vy = np.array(vy)[order]
        self.team = np.array(teams, dtype=object)[order]
        self._start_list = self.start.tolist()

    def __len__(self):
        return len(self.ids)

    def candidates(self, i: int, max_gap_frames: float) -> np.ndarray:
        """Fragments starting after fragment i ends, within the gap window.

        Binary search on the sorted start frames, so this touches only the
        fragments actually in reach rather than all of them.
        """
        lo = bisect.bisect_right(self._start_list, self.end[i])
        hi = bisect.bisect_right(self._start_list, self.end[i] + max_gap_frames)
        return np.arange(lo, hi)


def _best_match(frag: _Fragments, i: int, max_gap_frames: float,
                fps: float) -> tuple[int, float]:
    """Closest reachable successor to fragment i, and its cost."""
    cand = frag.candidates(i, max_gap_frames)
    if cand.size == 0:
        return -1, np.inf

    gap_frames = frag.start[cand] - frag.end[i]
    gap_s = np.maximum(gap_frames, 1.0) / fps

    # Where the player would be, continuing at their last velocity.
    pred_x = frag.ex[i] + frag.vx[i] * gap_frames
    pred_y = frag.ey[i] + frag.vy[i] * gap_frames

    dist = np.hypot(frag.sx[cand] - pred_x, frag.sy[cand] - pred_y)
    reach = PREDICTION_DRIFT_MS * gap_s + POSITION_MARGIN_M

    # Two fragments of one player wear one kit. Joining across kits was
    # relabelling a rejected non-player's fragment with an accepted player's
    # id, which put its rows back into possession after team assignment had
    # excluded them: 17-26% of merged tracks carried more than one team label,
    # and 8 to 15 per window spliced a rejected fragment into an accepted one.
    feasible = (dist <= reach) & (frag.team[cand] == frag.team[i])
    if not feasible.any():
        return -1, np.inf

    # Cost relative to what was reachable, so a long gap is not penalised for
    # being long -- only for being improbable given its length.
    cost = np.where(feasible, dist / reach, np.inf)
    j = int(np.argmin(cost))
    return int(cand[j]), float(cost[j])


def merge_fragments(tracks: pd.DataFrame, px_per_m: float, fps: float = 25.0,
                    verbose: bool = False) -> pd.DataFrame:
    """Relabel player track ids so fragments of one player share an id."""
    if tracks.empty or "cls" not in tracks.columns:
        return tracks.copy()
    if not np.isfinite(px_per_m) or px_per_m <= 0:
        return tracks.copy()

    players = tracks[tracks.cls == "player"]
    if players.empty or players.track_id.nunique() < 2:
        return tracks.copy()

    frag = _Fragments(players, px_per_m)
    before = len(frag)

    # Union-find over fragment indices.
    parent = list(range(before))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    taken_pred = set()   # fragments already claimed as a successor
    taken_succ = set()   # fragments already given a successor

    for gap_s in GAP_SCHEDULE:
        max_gap_frames = gap_s * fps

        forward: dict[int, tuple[int, float]] = {}
        for i in range(before):
            if i in taken_succ:
                continue
            j, cost = _best_match(frag, i, max_gap_frames, fps)
            if j >= 0 and j not in taken_pred:
                forward[i] = (j, cost)

        # Mutual consent: j accepts i only if i is also j's best predecessor.
        best_back: dict[int, tuple[int, float]] = {}
        for i, (j, cost) in forward.items():
            if j not in best_back or cost < best_back[j][1]:
                best_back[j] = (i, cost)

        for j, (i, _) in best_back.items():
            if find(i) == find(j):
                continue
            parent[find(j)] = find(i)
            taken_succ.add(i)
            taken_pred.add(j)

    mapping = {frag.ids[i]: frag.ids[find(i)] for i in range(before)}
    out = tracks.copy()
    is_player = out.cls == "player"
    out.loc[is_player, "track_id"] = out.loc[is_player, "track_id"].map(
        lambda t: mapping.get(t, t))

    after = out.loc[is_player, "track_id"].nunique()
    if verbose:
        print(f"  [re-id] {before} fragments -> {after} tracks")
    return out
