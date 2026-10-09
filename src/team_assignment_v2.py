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


# A fixed whole-pitch view (run_pipeline(fixed_camera=True) with a pitch
# outline): the kit feature, and how far inside the outline (in player
# heights, median over the track) a track must stay to take part in the
# clustering. Chosen on SoccerTrack v2 training windows; see
# SOCCERTRACK_V2.md. Broadcast never uses them.
FIXED_VIEW_KIT_FEATURE = "chroma"
FIXED_VIEW_CORE_DEPTH = 0.15


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


def track_kit_features(video_path: str, players: pd.DataFrame
                       ) -> dict[int, np.ndarray]:
    """Each track's kit colour: the median over its sampled frames of
    `_torso_feature` (cos hue, sin hue, saturation, value). One sequential
    pass through the video."""
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
    return {t: np.median(v, axis=0) for t, v in by_track.items() if v}


def kit_vector(feat: np.ndarray, kit_feature: str = "hsv4") -> np.ndarray:
    """What is clustered. "hsv4" (the default, broadcast): (cos hue,
    sin hue, saturation, value). "chroma": (s cos hue, s sin hue, value),
    the colour as a point in the HSV cone, so hue counts only as much as
    the kit is saturated.

    Why "chroma" exists: on a fixed SoccerTrack v2 panorama one kit read
    saturation 0.28 -- a dark, nearly grey shirt whose hue is noise -- and
    under "hsv4" hue takes two of four standardised dimensions, so that
    team fragmented into "other" on hue alone; two kits of the same hue
    and different saturation are told apart along the chroma radius.
    """
    if kit_feature == "hsv4":
        return feat
    if kit_feature == "chroma":
        c, s_, sat, val = feat
        return np.array([sat * c, sat * s_, val])
    raise ValueError(f"unknown kit_feature {kit_feature!r}")


def cluster_teams(feats: dict[int, np.ndarray], n_clusters: int = N_CLUSTERS,
                  residual_cut: float = RESIDUAL_CUT,
                  kit_feature: str = "hsv4", fit_tracks=None,
                  verbose: bool = True) -> dict[int, str]:
    """Team per track from kit features: k-means, the two largest clusters
    are the teams, tracks far from both kits are 'other'.

    `fit_tracks` (a fixed view): only these tracks are clustered and can
    form or join a team; every other track is 'other'. On a SoccerTrack v2
    panorama the substitutes and staff at the touchline were as many tracks
    as a team, so "the two largest clusters" named the bench a team; the
    tracks that stay well inside the pitch cannot include them however many
    there are. With fewer than 3 x n_clusters such tracks the outline is
    probably wrong, and all tracks are used, with a warning.
    """
    tids = list(feats)
    if len(tids) < n_clusters:
        return {}
    fit = tids
    if fit_tracks is not None:
        fit = [t for t in tids if t in fit_tracks]
        if len(fit) < 3 * n_clusters:
            print(f"  [teams] WARNING only {len(fit)} tracks inside the pitch "
                  "core; clustering all tracks (is the outline right?)")
            fit = tids

    raw_feats = np.array([kit_vector(feats[t], kit_feature) for t in fit])
    # Standardised so that hue, saturation and brightness contribute on
    # comparable scales; the distances below are otherwise dominated by
    # whichever happens to have the widest raw range.
    scaler = StandardScaler().fit(raw_feats)
    X = scaler.transform(raw_feats)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(X)
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

    team_map = {t: mapping[l] for t, l in zip(fit, labels)}

    # Reject by distance to the nearer kit centre. A track inside a team
    # cluster can still be a steward standing where k-means had nowhere
    # better to put them; what marks them out is sitting far from both kits.
    team_centres = np.array(
        [km.cluster_centers_[c] for c in sorted(mapping)
         if mapping[c] != "other"])
    if len(team_centres) == 2:
        dist = np.linalg.norm(
            X[:, None, :] - team_centres[None, :, :], axis=2).min(axis=1)
        inside = np.array([team_map[t] != "other" for t in fit])
        scale = np.median(dist[inside]) if inside.any() else np.median(dist)
        rejected = 0
        if scale > 1e-9:
            for t, d in zip(fit, dist / scale):
                if d > residual_cut and team_map[t] != "other":
                    team_map[t] = "other"
                    rejected += 1
        if verbose:
            print(f"  [teams] {rejected} tracks rejected as non-players "
                  f"(> {residual_cut}x the kit spread from either kit)")
    for t in tids:
        team_map.setdefault(t, "other")
    if verbose and len(fit) < len(tids):
        print(f"  [teams] {len(tids) - len(fit)} tracks outside the pitch "
              "core put in 'other'")
    return team_map


def assign_teams_v2(video_path: str, tracks: pd.DataFrame,
                    n_clusters: int = N_CLUSTERS,
                    verbose: bool = True,
                    residual_cut: float = RESIDUAL_CUT,
                    kit_feature: str = "hsv4", fit_tracks=None,
                    feats: dict | None = None) -> pd.DataFrame:
    """Cluster tracks into teams by kit colour. `kit_feature` and
    `fit_tracks` are for a fixed view (`cluster_teams`); `feats`, if
    given, are the tracks' kit features already read (`track_kit_features`)."""
    players = tracks[tracks.cls == "player"]
    if players.empty:
        out = tracks.copy()
        out["team"] = None
        return out

    if feats is None:
        feats = track_kit_features(video_path, players)
    tids = list(feats)
    team_map = cluster_teams(feats, n_clusters, residual_cut, kit_feature,
                             fit_tracks, verbose)
    if not team_map:
        out = tracks.copy()
        out["team"] = None
        out.loc[out.cls == "ball", "team"] = "ball"
        return out

    out = tracks.copy()
    out["team"] = out.track_id.map(team_map)
    out.loc[out.cls == "ball", "team"] = "ball"

    if verbose:
        got = out[out.cls == "player"].groupby("team").track_id.nunique()
        print(f"  [teams] {len(tids)} tracks clustered: {dict(got)}")
        # Both teams in one colour cluster shows as a "team" with more than
        # eleven players on at once (warning only: nothing is changed).
        on = out[(out.cls == "player") & out.team.isin(("team_A", "team_B"))]
        per_frame = on.groupby(["team", "frame"]).track_id.nunique()
        for team, n in per_frame.groupby(level=0).median().items():
            if n > 11:
                print(f"  [teams] WARNING {team} has a median of {n:.0f} "
                      "players on at once: probably both teams in one "
                      "colour cluster")
    return out
