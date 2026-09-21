"""Assign players to teams by kit colour, measured properly.

The original implementation reached 0.71-0.83 agreement with SoccerNet's own
team labels on long, well-observed tracks -- so its errors were not caused by
thin evidence. Three faults in how the colour was measured account for that:

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

This version reads the video once in frame order, represents hue as a unit
vector so that circularity is handled, takes the median over the most
saturated pixels so the shirt dominates its background, and samples many more
frames per track.

**It is no more accurate.** Measured against SoccerNet team labels on four
windows across two matches:

    variant              w1     w2     w3   reading    mean
    original           0.84   0.80   0.76   0.71       0.778
    this, k=3          0.81   0.83   0.78   0.68       0.775
    this, k=2          0.78   0.71   0.78   0.71       0.745

The three faults it fixes are real faults, and fixing them changes nothing.
Whatever limits team assignment to roughly 0.78 is not how the colour is
measured. Kept because the negative result is worth more than the file costs,
and because a future attempt should start by knowing this was tried.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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

    # Keep the more saturated half: a shirt is more saturated than the skin,
    # shadow and grass that share the box.
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
                    n_clusters: int = 3, verbose: bool = True) -> pd.DataFrame:
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

    feats = np.array([np.median(by_track[t], axis=0) for t in tids])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(feats)
    labels = km.labels_

    if n_clusters >= 3:
        # The smallest cluster is treated as officials and goalkeepers. This is
        # a guess, and it is wrong whenever one team simply has fewer tracks --
        # which is why n_clusters is a parameter and both settings are measured
        # rather than assumed.
        counts = np.bincount(labels, minlength=n_clusters)
        other = int(np.argmin(counts))
        team_ids = [c for c in range(n_clusters) if c != other]
        mapping = {team_ids[0]: "team_A", team_ids[1]: "team_B"}
        mapping.update({c: "other" for c in range(n_clusters) if c not in mapping})
    else:
        mapping = {0: "team_A", 1: "team_B"}

    team_map = {t: mapping[l] for t, l in zip(tids, labels)}
    out = tracks.copy()
    out["team"] = out.track_id.map(team_map)
    out.loc[out.cls == "ball", "team"] = "ball"

    if verbose:
        got = out[out.cls == "player"].groupby("team").track_id.nunique()
        print(f"  [teams] {len(tids)} tracks clustered: {dict(got)}")
    return out
