"""Give event detection coordinates in metres, with or without a homography.

The original failure in this project was read as corruption: event detection
returned zero events, the homography was singular, so the pitch coordinates it
produced were assumed to be garbage. The shipped fix dropped the `px`/`py`
columns to "fall back to x, y" -- and crashed, because no `x`/`y` columns
existed.

The real fault was units. Event detection compares distances and speeds
against thresholds written in metres and km/h: a pass is a ball travelling
several metres, possession is a player within a metre or so of the ball. The
track table holds pixels. At roughly 27-35 px/m, every real pass measured
hundreds of "metres" and no threshold matched anything, so nothing was ever
emitted. Nothing was corrupted; the numbers were simply in the wrong unit.

So the job here is to hand event detection metres whatever is available:

  * with a usable homography, real pitch coordinates, and the pitch is
    *absolute* -- the goal is at a known place, so goals, shots and
    out-of-play can be decided
  * without one, pixels divided by an estimated px/m, which is metric but
    *relative* -- distances and speeds are right, absolute pitch positions
    are not, and anything depending on where the goal is must stay switched
    off rather than guess
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A homography whose determinant is this small is degenerate: it collapses the
# plane onto a line or a point, and everything it maps lands on top of
# everything else.
MIN_ABS_DETERMINANT = 1e-12

# Even a non-singular matrix is unusable when it is this ill-conditioned --
# tiny input differences produce huge output ones, so pixel noise becomes
# metres of jitter.
MAX_CONDITION_NUMBER = 1e12

# A footballer, for turning bounding-box height into a scale.
PLAYER_HEIGHT_M = 1.75


def is_homography_usable(H) -> tuple[bool, str]:
    """Can this matrix be trusted to map pixels to pitch metres?

    Returns the verdict and the reason, because "no homography" and "a
    homography that silently produces nonsense" need to be distinguishable in
    the logs -- the second one was mistaken for a coordinate bug for weeks.
    """
    if H is None:
        return False, "no homography"

    H = np.asarray(H, dtype=float)
    if H.shape != (3, 3):
        return False, f"wrong shape {H.shape}"
    if not np.all(np.isfinite(H)):
        return False, "contains non-finite values"

    det = float(np.linalg.det(H))
    if abs(det) < MIN_ABS_DETERMINANT:
        return False, f"singular (|det| = {abs(det):.2e})"

    cond = float(np.linalg.cond(H))
    if cond > MAX_CONDITION_NUMBER:
        return False, f"ill-conditioned (cond = {cond:.2e})"

    return True, f"usable (|det| = {abs(det):.2e}, cond = {cond:.2e})"


def estimate_px_per_m(tracks: pd.DataFrame,
                      player_height_m: float = PLAYER_HEIGHT_M) -> float:
    """Pixels per metre, from how tall players appear.

    Players are the one object in frame whose real size is known and roughly
    constant, so their bounding-box height is the scale that is always
    available -- no pitch markings, no calibration, no homography.

    The median is over every player detection in the clip, which makes this a
    single number for the whole frame. That is wrong in a way worth stating:
    under perspective, a player at the far touchline is much shorter in pixels
    than one near the camera, and on the broadcast clip the two ends of the
    frame differ by about 2.7x. Distances near the median depth are right;
    distances at the extremes are out by that factor.
    """
    if tracks is None or tracks.empty or "crop_h" not in tracks.columns:
        return float("nan")

    players = tracks[tracks.cls == "player"] if "cls" in tracks.columns else tracks
    heights = players.crop_h.to_numpy(dtype=float)
    heights = heights[np.isfinite(heights) & (heights > 0)]
    if heights.size == 0:
        return float("nan")

    return float(np.median(heights) / player_height_m)


def to_metric_coords(tracks: pd.DataFrame, px_per_m: float) -> pd.DataFrame:
    """Convert pixel positions to metres using an estimated scale.

    Publishes `x`/`y` alongside `px`/`py` because downstream code reads both
    spellings: leaving `x`/`y` unset is what turned the first attempt at this
    into a KeyError rather than a fix.
    """
    out = tracks.copy()
    if not np.isfinite(px_per_m) or px_per_m <= 0:
        return out

    out["px"] = out.px.to_numpy(dtype=float) / px_per_m
    out["py"] = out.py.to_numpy(dtype=float) / px_per_m
    out["x"] = out["px"]
    out["y"] = out["py"]
    return out


def project_with_homography(tracks: pd.DataFrame, H) -> pd.DataFrame:
    """Map pixel positions onto the pitch plane."""
    out = tracks.copy()
    pts = np.column_stack([
        out.px.to_numpy(dtype=float),
        out.py.to_numpy(dtype=float),
        np.ones(len(out)),
    ])
    projected = (np.asarray(H, dtype=float) @ pts.T).T
    w = projected[:, 2]
    # A point on the horizon divides by ~0; leave those where they are rather
    # than writing infinities into the track table.
    safe = np.abs(w) > 1e-9
    out["px"] = np.where(safe, projected[:, 0] / np.where(safe, w, 1.0), np.nan)
    out["py"] = np.where(safe, projected[:, 1] / np.where(safe, w, 1.0), np.nan)
    out["x"] = out["px"]
    out["y"] = out["py"]
    return out


def prepare_tracks_for_events(tracks: pd.DataFrame, H=None,
                              verbose: bool = True,
                              px_per_m: float | None = None,
                              ) -> tuple[pd.DataFrame, bool]:
    """Return tracks in metres, and whether the pitch frame is absolute.

    The second value is the one that matters for correctness. `True` means
    coordinates sit on a known pitch, so a position can be compared against
    the goal line. `False` means they are metric but float freely -- distances
    and speeds are meaningful, absolute positions are not, and every event
    type that depends on where the goal is must be disabled rather than
    allowed to guess.

    `px_per_m` overrides the scale estimated from player height. It exists
    for `ground_plane.effective_px_per_m`, which measures how much the height
    estimate overstates the scale and hands back a corrected number. Nothing
    else about the coordinate system changes: this is deliberately a scalar
    override rather than a different map, because the map that is
    geometrically right measures worse. See that function.
    """
    usable, reason = is_homography_usable(H)

    if usable:
        if verbose:
            print(f"  [scale] homography {reason}; pitch coordinates absolute")
        return project_with_homography(tracks, H), True

    source = "from player height"
    if px_per_m is None:
        px_per_m = estimate_px_per_m(tracks)
    else:
        source = "calibrated against the ground plane"
    if not np.isfinite(px_per_m) or px_per_m <= 0:
        raise ValueError(
            "no usable homography and no player detections to estimate scale "
            f"from ({reason})"
        )

    if verbose:
        print(f"  [scale] homography {reason}; "
              f"using {px_per_m:.1f} px/m {source}. "
              "Pitch coordinates are relative: goals, shots and out-of-play "
              "stay disabled.")
    return to_metric_coords(tracks, px_per_m), False
