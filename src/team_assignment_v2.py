"""Assign players to teams by kit colour.

The original implementation reached 0.71-0.83 agreement with SoccerNet's own
team labels on long, well-observed tracks. This version fixes three faults in
how the colour was measured:

  * **Hue was averaged linearly.** OpenCV hue is circular, 0 to 179. A red
    kit has pixels near 0 and near 179, and their mean is ~90, which is cyan.
    Any red, pink or magenta kit produced a colour unrelated to itself.

  * **Frames were reached by seeking.** `cap.set()` on compressed video is not
    frame-accurate -- the same fault that made the camera motion probe
    overstate movement by 80%. A mis-seek samples whatever happens to be at
    those coordinates in a different frame.

  * **The crop was averaged whole.** A torso box reconstructed from the
    bounding box contains grass, limbs and neighbouring players. Their mean is
    not the shirt.

**This file previously recorded that fixing them changed nothing** -- 0.775
against the original's 0.778, measured at matched labelled passes. That
measurement was worthless, and the reason is worth keeping. Scoring team
assignment through possession measures

    (did we cluster this track into the right team)
      x (did we attribute the event to the right player)

and the second factor is broken badly enough to swamp the first: 22% of the
passer/receiver slots on matched passes land on a track that is not a player
at all. Nine colour representations and six values of k all scored 0.64 to
0.75 on that metric, which is the signature of a measurement dominated by
something other than what it names.

Read off the kits directly -- 107 tracks in one window and 84 in another,
labelled by eye from crop montages -- the same variants separate cleanly:

    feature                     Reading   Stoke
    raw (the original's)           0.93    0.97
    circular hue                   0.98    0.98
    circular hue, grass removed    0.99    0.98

Two further changes come from the same labels. Roughly a quarter of the
tracks the pipeline calls players are not players: stewards in hi-vis, staff
in dark coats, spectators, and in one case an advertising hoarding. Green
pixels are now discarded before the colour is measured, since a torso box on
a football pitch is part grass whatever else it contains; and a track sitting
further than `RESIDUAL_CUT` from the nearer kit centre is refused a team.

Measured against the hand-read labels on both windows:

                        purity   coverage   kit accuracy
    before, Reading       0.77       1.00           0.93
    now,    Reading       0.96       0.98           0.99
    before, Stoke         0.73       1.00           0.97
    now,    Stoke         0.89       0.97           0.98

Non-players given a team fall from 16 of 25 to 3 of 25 on Reading, and the
share of passer/receiver slots on matched passes that land on one falls from
22% to 15%. It is reduced, not solved: the few that survive are the ones
standing nearest the play, so they are over-represented in events relative to
their number.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Frames sampled per track. The original took five; a track of 200 frames can
# afford far more, and the cost is one sequential read of the video either way.
SAMPLES_PER_TRACK = 24

# Tracks shorter than this are not assigned. They are too brief to be a player
# the pipeline will report on, and they dilute the clustering.
MIN_TRACK_FRAMES = 10

# Share of the crop's pixels, ranked by saturation, kept as "shirt". Grass is
# saturated too, but the torso box is mostly shirt and the median of the top
# half is robust to the rest.
SATURATION_KEEP = 0.5

# Fraction of the bounding-box height treated as torso, measured from the top.
TORSO_TOP, TORSO_BOTTOM = 0.25, 0.55

# Fraction of the bounding-box width kept, centred -- the edges are background
# and arms.
TORSO_WIDTH = 0.5

# Pixels in this hue band, above this saturation, are pitch and are discarded
# before the shirt colour is measured. OpenCV hue is 0-179, so grass sits
# around 30-90. Keeping them measures the pitch as part of the kit, which is
# worth 0.93 -> 0.99 kit accuracy on one window and 0.97 -> 0.98 on another.
GRASS_HUE_LO, GRASS_HUE_HI, GRASS_MIN_SAT = 30, 90, 40

# Clusters to find. The two largest are the teams; anything else is 'other'.
#
# Three, not five, and the reason is a result that did not replicate. On
# hand-read kit labels, k=5 rejects non-players far better on the window it
# was chosen on -- purity 0.84 to 0.99 -- and barely at all on the other,
# 0.83 to 0.85. Run end to end it then *halved* team attribution on a third
# window, 0.80 to 0.55, splitting one side 103 tracks against 38. A rejection
# rule that depends on the two teams being the two biggest groups fails
# whenever a team is fragmented into several colour clusters, and nothing
# here detects that happening.
#
# At k=3 the two largest clusters and "all but the smallest" are the same
# rule, so this keeps the shipped selection untouched and takes only the
# feature change, which improves kit accuracy on both labelled windows
# (0.93 -> 0.99 and 0.97 -> 0.98) and costs nothing anywhere.
#
# Non-players are excluded by distance instead -- see RESIDUAL_CUT.
N_CLUSTERS = 3

# How far a track may sit from the nearer kit centre and still be a player,
# as a multiple of the median distance among tracks inside the two kit
# clusters.
#
# A quarter of the tracks the detector calls players are stewards in hi-vis,
# staff in dark coats, spectators and one advertising hoarding. Counting
# clusters does not remove them -- that assumes the teams are the two biggest
# groups, and it collapsed when a team fragmented. Distance does not assume
# anything about how many other groups exist, and because it is expressed
# relative to the spread of the kits themselves it does not depend on what
# colours they are.
#
# It separates strongly and, unlike everything else tried, it transfers:
# area under the ROC curve 0.97 on one match and 0.91 on the other, against
# 0.48 for how much a track's colour varies and 0.33-0.60 for how isolated it
# is on the pitch.
#
# Swept and validated both directions on hand-read kit labels. Choosing the
# cut on one match and applying it to the other:
#
#     fitted on Reading (cut 2.00) -> Stoke   purity 0.73 -> 0.91
#     fitted on Stoke   (cut 2.50) -> Reading purity 0.77 -> 0.96
#
# 2.5 rather than 2.0 because it keeps coverage at 0.97 and 0.98 on the two
# windows where 2.0 drops a tenth of the real players on one of them, and a
# dropped player loses their events outright.
RESIDUAL_CUT = 2.5


def _torso_feature(frame: np.ndarray, px: float, py: float, h: float):
    """Kit colour as (cos hue, sin hue, saturation, value), or None.

    Hue is returned as a unit vector rather than an angle so that distances
    between colours mean what they should: red at 179 and red at 1 are
    adjacent, not opposite.
    """
    if not np.isfinite(h) or h <= 4:
        return None
    w = h * 0.4
    x1 = int(px - w * TORSO_WIDTH / 2)
    x2 = int(px + w * TORSO_WIDTH / 2)
    y_top = py - h
    y1 = int(y_top + h * TORSO_TOP)
    y2 = int(y_top + h * TORSO_BOTTOM)

    H, W = frame.shape[:2]
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    if len(hsv) < 4:
        return None

    # Drop the pitch. A torso box on a football pitch contains grass between
    # the arms and the body and around the shoulders, and grass is saturated,
    # so the "most saturated half" below does not exclude it.
    not_grass = ~((hsv[:, 0] >= GRASS_HUE_LO) & (hsv[:, 0] <= GRASS_HUE_HI)
                  & (hsv[:, 1] >= GRASS_MIN_SAT))
    if not_grass.sum() >= 4:
        hsv = hsv[not_grass]

    # Keep the more saturated half: a shirt is more saturated than the skin
    # and shadow that share the box.
    keep = max(2, int(len(hsv) * SATURATION_KEEP))
    idx = np.argsort(hsv[:, 1])[-keep:]
    hsv = hsv[idx]

    ang = hsv[:, 0].astype(float) * (2 * np.pi / 180.0)
    return np.array([
        float(np.median(np.cos(ang))),
        float(np.median(np.sin(ang))),
        float(np.median(hsv[:, 1])) / 255.0,
        float(np.median(hsv[:, 2])) / 255.0,
    ])


def _sample_frames(players: pd.DataFrame, per_track: int) -> dict[int, list]:
    """Which rows to sample for each track, spread across its lifetime."""
    wanted: dict[int, list] = {}
    for tid, g in players.groupby("track_id"):
        if len(g) < MIN_TRACK_FRAMES:
            continue
        g = g.sort_values("frame")
        take = g.iloc[np.linspace(0, len(g) - 1,
                                  min(per_track, len(g))).astype(int)]
        for r in take.itertuples():
            wanted.setdefault(int(r.frame), []).append(
                (r.track_id, float(r.px), float(r.py), float(r.crop_h)))
    return wanted


def assign_teams_v2(video_path: str, tracks: pd.DataFrame,
                    n_clusters: int = N_CLUSTERS,
                    verbose: bool = True) -> pd.DataFrame:
    """Cluster tracks into teams by kit colour."""
    players = tracks[tracks.cls == "player"]
    if players.empty:
        out = tracks.copy()
        out["team"] = None
        return out

    wanted = _sample_frames(players, SAMPLES_PER_TRACK)
    by_track: dict[int, list] = {}

    # One sequential pass. Seeking is both slower and, on compressed video,
    # not frame-accurate.
    cap = cv2.VideoCapture(video_path)
    last = max(wanted) if wanted else -1
    idx = 0
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, px, py, h in wanted.get(idx, ()):
            feat = _torso_feature(frame, px, py, h)
            if feat is not None:
                by_track.setdefault(tid, []).append(feat)
        idx += 1
    cap.release()

    tids = [t for t, v in by_track.items() if v]
    if len(tids) < n_clusters:
        out = tracks.copy()
        out["team"] = None
        out.loc[out.cls == "ball", "team"] = "ball"
        return out

    raw_feats = np.array([np.median(by_track[t], axis=0) for t in tids])
    # Standardised so that hue, saturation and brightness contribute on
    # comparable scales; the distances below are otherwise dominated by
    # whichever happens to have the widest raw range.
    scaler = StandardScaler().fit(raw_feats)
    feats = scaler.transform(raw_feats)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(feats)
    labels = km.labels_

    if n_clusters >= 3:
        # The two largest clusters are the teams; everything else is 'other'.
        #
        # The previous rule discarded the *smallest* cluster, which assumes
        # exactly one non-team group exists. Reading the tracks off the video
        # shows there are several -- match officials, two goalkeepers in their
        # own kits, stewards in hi-vis, staff in dark coats -- and roughly a
        # quarter of "player" tracks belong to them. Discarding one cluster
        # leaves the rest inside the two teams.
        counts = np.bincount(labels, minlength=n_clusters)
        biggest = np.argsort(counts)[::-1][:2]
        mapping = {int(biggest[0]): "team_A", int(biggest[1]): "team_B"}
        mapping.update({c: "other" for c in range(n_clusters)
                        if c not in mapping})
    else:
        mapping = {0: "team_A", 1: "team_B"}

    team_map = {t: mapping[l] for t, l in zip(tids, labels)}

    # Reject by distance to the nearer kit centre. A track inside a team
    # cluster can still be a steward standing where k-means had nowhere
    # better to put them; what marks them out is sitting far from both kits.
    team_centres = np.array(
        [km.cluster_centers_[c] for c in sorted(mapping)
         if mapping[c] != "other"])
    if len(team_centres) == 2:
        dist = np.linalg.norm(
            feats[:, None, :] - team_centres[None, :, :], axis=2).min(axis=1)
        inside = np.array([team_map[t] != "other" for t in tids])
        scale = np.median(dist[inside]) if inside.any() else np.median(dist)
        rejected = 0
        if scale > 1e-9:
            for t, d in zip(tids, dist / scale):
                if d > RESIDUAL_CUT and team_map[t] != "other":
                    team_map[t] = "other"
                    rejected += 1
        if verbose:
            print(f"  [teams] {rejected} tracks rejected as non-players "
                  f"(> {RESIDUAL_CUT}x the kit spread from either kit)")

    out = tracks.copy()
    out["team"] = out.track_id.map(team_map)
    out.loc[out.cls == "ball", "team"] = "ball"

    if verbose:
        got = out[out.cls == "player"].groupby("team").track_id.nunique()
        print(f"  [teams] {len(tids)} tracks clustered: {dict(got)}")
    return out
