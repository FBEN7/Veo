"""A metric ground plane, from player heights and the camera's own pan.

Every distance and speed this pipeline reports used to divide pixels by a
single `px_per_m` taken from the median player height. That number is correct
at exactly one depth. Measured by image row it varies by 2.1x on broadcast
and 3.1x on the Veo clip, so a run at the far touchline and the same run near
the camera were reported as different distances -- and the single scale also
treats a metre across the view and a metre away from it as the same number of
pixels, which under perspective they are emphatically not.

A ground plane fixes both. For a pinhole camera looking at a plane, three
numbers determine the whole mapping:

    y_horizon   the image row the plane vanishes at
    h_cam       how high the camera is above it
    f           the focal length, in pixels

The first two come from the players themselves. A fixed-height object
standing on the plane projects to a pixel height linear in its *feet* row,
reaching zero at the horizon, with slope H/h_cam:

    crop_h = (H / h_cam) * (y_feet - y_horizon)

so a straight-line fit over every detection gives both. No pitch markings, no
calibration target, no homography -- the same reasoning that made player
height the scale in the first place, used properly.

The third is the one that blocked this for months, because it cannot be
recovered from heights: f does not appear in the equation above. It comes
instead from the camera's own movement. A camera that rotates without
translating relates any two frames by H = K R K^-1, and requiring R to be a
rotation pins f from H alone. A broadcast camera on a gantry and a Veo
virtual camera panning across its panorama both qualify. See `estimate_focal`
for what this does and does not survive.

With all three, image pixels map to ground metres:

    v = y_feet - y_horizon        rows below the horizon
    Z = f * h_cam / v             metres away from the camera
    X = (x - cx) * Z / f          metres across

which is a homography, so it goes through the same `project_with_homography`
path a real pitch homography would.

What this is not: it is not absolute. The mapping is metric and the ground
frame is fixed, but its origin is the camera, not a corner flag, so nothing
here says where the goal is. Goals, shots and out-of-play stay disabled.
Getting those needs pitch markings, which `probe_homography.py` shows this
footage does not offer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# A footballer. The single assumed quantity in the whole construction, and
# everything scales linearly with it: a squad of 1.68 m under-15s shrinks
# every distance by 4%.
PLAYER_HEIGHT_M = 1.75

# --- the height fit ---------------------------------------------------------

# Detections shorter than this are too small for their height to mean
# anything: a 5 px box is two rows of pixels either side of a guess.
MIN_CROP_H_PX = 5.0

# Rounds of 2.5-sigma trimming. Referees, keepers crouching, half-occluded
# players behind the hoardings and the occasional two-players-in-one-box all
# sit off the line, and a least-squares fit chases them.
FIT_TRIM_ROUNDS = 6
FIT_TRIM_SIGMA = 2.5

# Fewer detections than this and the fit is not worth trusting.
MIN_FIT_DETECTIONS = 500

# A camera height outside this range is not a football camera. This is the
# gate that would catch a fit that has gone backwards -- a negative or
# near-zero slope puts the camera underground or in orbit, and the resulting
# map would look metric and be nonsense.
MIN_CAM_HEIGHT_M = 2.0
MAX_CAM_HEIGHT_M = 60.0

# Rows this close to the horizon are unusable: Z goes to infinity there, so a
# pixel of detection jitter becomes hundreds of metres.
MIN_ROWS_BELOW_HORIZON = 10.0

# --- the horizon over time --------------------------------------------------
# A camera that tilts moves the horizon, and depth goes as 1/(row - horizon),
# so an error there is a multiplicative error in every distance. The obvious
# source for it is the camera-motion estimate the pipeline already computes --
# and that is wrong in a way worth recording, because it looks right.
#
# Compensating vertically by the estimated motion assumes a tilt translates
# the image uniformly. It does not: the true mapping is
# y' = f*tan(atan(y/f) + phi), so a row far off-axis moves by sec^2 of its
# angle more than the centre does. The horizon sits 1068 px above centre on
# the Reading window, which is 31 degrees at that clip's focal length, so it
# moves 36% further than the motion measured at the centre -- and that window
# is precisely where the first version of this fell over, split-half speed
# reliability dropping from 0.57 to 0.19 while the other three held.
#
# So the horizon is measured rather than inferred. With the slope fixed at
# its whole-clip value, crop_h = s * (y_centre - y_horizon) leaves the
# horizon as a one-parameter fit per window, which is stable where a free
# two-parameter fit is not: fitting both per window swings the implied camera
# height between 19 and 29 m on one clip, because ten seconds of play spans
# too few image rows to pin a slope.
HORIZON_WINDOW_FRAMES = 125
MIN_HORIZON_WINDOW_DETECTIONS = 40

# --- the focal length -------------------------------------------------------

# Frame separations to measure over. Several, because agreement across them is
# the evidence that the number is real: the projective part of the homography
# grows with the rotation while the noise does not, so a spurious f drifts
# with the baseline and a true one does not.
FOCAL_GAPS = (15, 30, 60, 120)
FOCAL_PAIRS = 24

# ORB is used rather than the Lucas-Kanade the motion estimator runs on,
# for the reason `camera_motion._direct_displacement` gives: LK searches
# locally and collapses at exactly the displacements worth measuring.
FOCAL_ORB_FEATURES = 4000
FOCAL_MIN_MATCHES = 40
FOCAL_MIN_INLIERS = 30
FOCAL_RANSAC_PX = 2.0

# f outside this range, in units of image width, is rejected before it can
# poison the median: 0.3 is a fisheye, 20 is a telescope.
FOCAL_MIN_PER_WIDTH = 0.3
FOCAL_MAX_PER_WIDTH = 20.0

# Interquartile spread, as a fraction of the median, above which the estimate
# is refused. Measured: broadcast clips come in at 0.10 to 0.35 and agree
# across all four baselines; the Veo clip runs 0.5 to 0.8 and does not.
MAX_FOCAL_SPREAD = 0.45

# Baselines that must independently produce an f before one is believed.
MIN_FOCAL_BASELINES = 3


@dataclass
class GroundPlane:
    """The mapping, plus the evidence for it."""

    focal_px: float
    h_cam_m: float
    y_horizon: float
    cx: float
    width: int
    height: int
    fit_r2: float
    fit_n: int
    focal_spread: float
    focal_n: int
    # Horizon row measured per window, as parallel lists of frame and row.
    # Empty means the single fitted `y_horizon` is used for the whole clip.
    horizon_frames: list = None
    horizon_rows: list = None

    def horizon_at(self, frames: np.ndarray) -> np.ndarray:
        """Horizon row for each frame, interpolated between windows."""
        if not self.horizon_frames or not self.horizon_rows:
            return np.full(len(frames), self.y_horizon, dtype=float)
        return np.interp(np.asarray(frames, dtype=float),
                         np.asarray(self.horizon_frames, dtype=float),
                         np.asarray(self.horizon_rows, dtype=float))

    @property
    def hfov_deg(self) -> float:
        return float(2 * np.degrees(np.arctan(self.width / (2 * self.focal_px))))

    def summary(self) -> str:
        return (f"  camera       {self.h_cam_m:.1f} m up, horizon at row "
                f"{self.y_horizon:.0f}\n"
                f"  focal        {self.focal_px:.0f} px "
                f"({self.hfov_deg:.0f} deg across), spread "
                f"{self.focal_spread:.2f} over {self.focal_n} pairs\n"
                f"  height fit   R^2 {self.fit_r2:.2f} over {self.fit_n} "
                f"detections")

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "GroundPlane":
        return cls(**json.loads(Path(path).read_text()))

    @property
    def centre_factor(self) -> float:
        """How much closer to the horizon a player's centre sits than feet."""
        return 1.0 - PLAYER_HEIGHT_M / (2.0 * self.h_cam_m)

    def homography(self) -> np.ndarray:
        """Image (x, y_centre) to ground (X, Z), in metres.

        Written out rather than composed so the algebra is checkable. With
        k the centre factor and w the rows below the horizon,

            w = y - y_horizon
            X = h_cam * k * (x - cx) / w
            Z = f * h_cam * k / w
        """
        a = self.h_cam_m * self.centre_factor
        return np.array([
            [a, 0.0, -a * self.cx],
            [0.0, 0.0, self.focal_px * a],
            [0.0, 1.0, -self.y_horizon],
        ], dtype=float)


def fit_height_model(tracks: pd.DataFrame,
                     player_height_m: float = PLAYER_HEIGHT_M):
    """Camera height and horizon row, from how tall players appear where.

    Returns (h_cam, y_horizon, r2, n), or None when there is not enough to
    fit.

    The fit is against the box *centre* row, not the feet, and that choice is
    load-bearing. The model describes an object standing on the plane, so the
    feet are the natural variable -- but the feet row is `py + crop_h / 2`,
    which puts the noisy quantity being predicted on both sides of the
    regression. A box that comes out a few pixels too tall then moves the
    prediction *and* the predictor the same way, which biases the slope up
    and the camera down. Fitting on the centre avoids it, and costs only a
    known constant: the centre of a player of height H sits at

        v_centre = v_feet * (1 - H / (2 * h_cam))

    so the fitted slope is H / (h_cam - H/2) and the camera height is the
    reciprocal plus H/2. The horizon is unaffected -- a player at the horizon
    has zero height, so centre and feet coincide there.
    """
    if tracks is None or tracks.empty:
        return None
    if not {"py", "crop_h"} <= set(tracks.columns):
        return None

    players = tracks[tracks.cls == "player"] if "cls" in tracks.columns else tracks
    y_centre = players.py.to_numpy(float)
    crop_h = players.crop_h.to_numpy(float)

    ok = np.isfinite(y_centre) & np.isfinite(crop_h) & (crop_h > MIN_CROP_H_PX)
    y_centre, crop_h = y_centre[ok], crop_h[ok]
    if y_centre.size < MIN_FIT_DETECTIONS:
        return None

    slope, intercept = np.polyfit(y_centre, crop_h, 1)
    keep = np.ones(y_centre.size, dtype=bool)
    for _ in range(FIT_TRIM_ROUNDS):
        resid = crop_h - (slope * y_centre + intercept)
        sigma = resid.std()
        if not np.isfinite(sigma) or sigma <= 0:
            break
        keep = np.abs(resid) < FIT_TRIM_SIGMA * sigma
        if keep.sum() < MIN_FIT_DETECTIONS:
            break
        slope, intercept = np.polyfit(y_centre[keep], crop_h[keep], 1)

    if not np.isfinite(slope) or slope <= 0:
        return None

    pred = slope * y_centre + intercept
    denom = np.sum((crop_h - crop_h.mean()) ** 2)
    r2 = float(1 - np.sum((crop_h - pred) ** 2) / denom) if denom > 0 else 0.0

    h_cam = player_height_m / slope + player_height_m / 2.0
    return float(h_cam), float(-intercept / slope), r2, int(keep.sum()), float(slope)


def horizon_schedule(tracks: pd.DataFrame, slope: float,
                     window: int = HORIZON_WINDOW_FRAMES):
    """Horizon row per window, with the slope held at its whole-clip value.

    `crop_h = slope * (y_centre - y_horizon)` rearranges to
    `y_horizon = y_centre - crop_h / slope`, one estimate per detection. The
    median over a window is robust to the detections that sit off the line
    for the usual reasons -- a crouching keeper, two players in one box.
    """
    if slope <= 0 or "frame" not in tracks.columns:
        return None
    players = tracks[tracks.cls == "player"] if "cls" in tracks.columns else tracks
    y_c = players.py.to_numpy(float)
    crop_h = players.crop_h.to_numpy(float)
    frame = players.frame.to_numpy(float)
    ok = np.isfinite(y_c) & np.isfinite(crop_h) & (crop_h > MIN_CROP_H_PX)
    if ok.sum() < MIN_HORIZON_WINDOW_DETECTIONS:
        return None
    y_c, crop_h, frame = y_c[ok], crop_h[ok], frame[ok]

    est = y_c - crop_h / slope
    bins = (frame // window).astype(int)
    frames, rows = [], []
    for b in np.unique(bins):
        sel = bins == b
        if sel.sum() < MIN_HORIZON_WINDOW_DETECTIONS:
            continue
        frames.append(float((b + 0.5) * window))
        rows.append(float(np.median(est[sel])))
    if len(frames) < 3:
        return None
    return frames, rows


def focal_from_homography(H, width: int, height: int) -> float | None:
    """f from a rotation homography, assuming square pixels and a centred
    principal point.

    With K = diag(f, f, 1) and H = K R K^-1, the centred H has

        R's first column  = (h11, h21, f * h31)
        R's third column  = (h13 / f, h23 / f, h33)

    and those two being orthogonal gives f directly. The equal-norm
    constraint gives a second estimate and is not used: measured against
    synthetic rotations of known f it is an order of magnitude noisier, since
    it leans on the difference of two nearly equal quantities.
    """
    if H is None:
        return None
    shift = np.array([[1.0, 0.0, -width / 2.0],
                      [0.0, 1.0, -height / 2.0],
                      [0.0, 0.0, 1.0]])
    Hc = shift @ np.asarray(H, float) @ np.linalg.inv(shift)
    det = np.linalg.det(Hc)
    if not np.isfinite(det) or abs(det) < 1e-12:
        return None
    Hc = Hc / np.cbrt(det)

    (h11, _, h13), (h21, _, h23), (h31, _, h33) = Hc
    denom = h31 * h33
    if abs(denom) < 1e-18:
        return None
    f_sq = -(h11 * h13 + h21 * h23) / denom
    if not np.isfinite(f_sq) or f_sq <= 0:
        return None
    return float(np.sqrt(f_sq))


def estimate_focal(video_path: str, width: int, height: int,
                   background_mask=None, verbose: bool = False):
    """Focal length in pixels, from the camera's own rotation.

    Returns (f, spread, n) or None. `spread` is the interquartile range over
    the median, and it is the number that decides whether the answer is used.

    The assumption is that the camera rotates about its own centre and does
    not travel. A broadcast camera on a gantry and a Veo virtual camera
    cropping a panorama both satisfy it; a handheld or a drone would not, and
    would produce a confident wrong answer rather than a refusal, because
    parallax from translation also puts a projective term in H. There is no
    test here that separates the two. What there is is agreement across
    baselines, which catches noise but not a systematic parallax.
    """
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n_frames < max(FOCAL_GAPS) * 2:
        return None

    orb = cv2.ORB_create(nfeatures=FOCAL_ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    lo, hi = FOCAL_MIN_PER_WIDTH * width, FOCAL_MAX_PER_WIDTH * width

    per_gap, all_f = {}, []
    for gap in FOCAL_GAPS:
        cap = cv2.VideoCapture(video_path)
        starts = np.linspace(0, n_frames - 1 - gap, FOCAL_PAIRS).astype(int)
        found = []
        for start in starts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
            ok, first = cap.read()
            if not ok:
                continue
            # Read forward: seeking compressed video is not frame-accurate,
            # and a mis-seek here reads as a different focal length.
            second = None
            for _ in range(gap):
                ok2, second = cap.read()
                if not ok2:
                    second = None
                    break
            if second is None:
                continue

            mask = None if background_mask is None else background_mask(first)
            kp_a, desc_a = orb.detectAndCompute(
                cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), mask)
            kp_b, desc_b = orb.detectAndCompute(
                cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), None)
            if desc_a is None or desc_b is None:
                continue
            if len(kp_a) < FOCAL_MIN_MATCHES or len(kp_b) < FOCAL_MIN_MATCHES:
                continue

            pairs = matcher.knnMatch(desc_a, desc_b, k=2)
            good = [p[0] for p in pairs
                    if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
            if len(good) < FOCAL_MIN_MATCHES:
                continue

            src = np.float32([kp_a[m.queryIdx].pt for m in good])
            dst = np.float32([kp_b[m.trainIdx].pt for m in good])
            H, inliers = cv2.findHomography(src, dst, cv2.RANSAC,
                                            FOCAL_RANSAC_PX)
            if H is None or inliers is None:
                continue
            if int(inliers.sum()) < FOCAL_MIN_INLIERS:
                continue

            f = focal_from_homography(H, width, height)
            if f is not None and lo < f < hi:
                found.append(f)
        cap.release()

        if len(found) >= 3:
            per_gap[gap] = float(np.median(found))
            all_f.extend(found)

    if len(per_gap) < MIN_FOCAL_BASELINES or len(all_f) < 8:
        if verbose:
            print(f"  [plane] focal length: only {len(per_gap)} baselines "
                  "produced an estimate")
        return None

    arr = np.array(all_f)
    median = float(np.median(arr))
    spread = float((np.percentile(arr, 75) - np.percentile(arr, 25)) / median)
    if verbose:
        by_gap = ", ".join(f"{g}f:{v:.0f}" for g, v in sorted(per_gap.items()))
        print(f"  [plane] focal {median:.0f} px, spread {spread:.2f} "
              f"over {len(arr)} pairs ({by_gap})")
    return median, spread, len(arr)


def background_seed_mask(video_path: str):
    """Where it is safe to look for features fixed to the ground.

    The same two exclusions the camera-motion estimator makes, for the same
    reasons: the pitch moves under the players, and burned-in graphics never
    move at all. Found once per clip and reused for every frame pair.
    """
    from . import camera_motion

    graphics = camera_motion.static_graphics_mask(video_path)

    def mask(frame: np.ndarray) -> np.ndarray:
        base = camera_motion._background_mask(frame)
        return base if graphics is None else cv2.bitwise_and(base, graphics)

    return mask


def build(video_path: str, tracks: pd.DataFrame, width: int, height: int,
          background_mask=None, verbose: bool = True) -> GroundPlane | None:
    """Fit the plane, or return None with the reason printed.

    Refusing is the point. A ground plane built on a bad focal length is
    worse than the single scale it replaces: it is wrong by a factor that
    varies across the frame, so it would corrupt distances in a way that
    looks like real structure rather than like an error.
    """
    fit = fit_height_model(tracks)
    if fit is None:
        if verbose:
            print("  [plane] no ground plane: too few usable player "
                  "detections to fit a height model")
        return None
    h_cam, y_horizon, r2, n_fit, slope = fit

    if not (MIN_CAM_HEIGHT_M <= h_cam <= MAX_CAM_HEIGHT_M):
        if verbose:
            print(f"  [plane] no ground plane: implied camera height "
                  f"{h_cam:.1f} m is not a football camera")
        return None

    focal = estimate_focal(video_path, width, height,
                           background_mask=background_mask, verbose=verbose)
    if focal is None:
        if verbose:
            print("  [plane] no ground plane: focal length could not be "
                  "measured from the camera's motion")
        return None
    f, spread, n_focal = focal

    if spread > MAX_FOCAL_SPREAD:
        if verbose:
            print(f"  [plane] no ground plane: focal length disagrees with "
                  f"itself across baselines (spread {spread:.2f} > "
                  f"{MAX_FOCAL_SPREAD})")
        return None

    schedule = horizon_schedule(tracks, slope)
    plane = GroundPlane(focal_px=f, h_cam_m=h_cam, y_horizon=y_horizon,
                        cx=width / 2.0, width=width, height=height,
                        fit_r2=r2, fit_n=n_fit,
                        focal_spread=spread, focal_n=n_focal,
                        horizon_frames=(schedule[0] if schedule else None),
                        horizon_rows=(schedule[1] if schedule else None))
    if verbose:
        print(f"  [plane] ground plane fitted\n{plane.summary()}")
    return plane


def load_or_build(video_path: str, tracks: pd.DataFrame, width: int,
                  height: int, cache: Path,
                  verbose: bool = True) -> GroundPlane | None:
    """Cached `build`, remembering refusals as well as successes.

    A refusal costs as much to reach as a fit -- both measure the focal
    length over four baselines -- so not caching it makes every run on
    footage that cannot be calibrated pay for finding that out again.
    """
    cache = Path(cache)
    if cache.exists():
        stored = json.loads(cache.read_text())
        if stored.get("refused"):
            if verbose:
                print(f"  [plane] no ground plane ({stored.get('why', 'cached')})")
            return None
        return GroundPlane(**stored)

    plane = build(video_path, tracks, width, height,
                  background_mask=background_seed_mask(video_path),
                  verbose=verbose)
    if plane is None:
        cache.write_text(json.dumps(
            {"refused": True, "why": "focal length not measurable here"},
            indent=2))
        return None
    plane.save(cache)
    return plane


def to_ground_coords(tracks: pd.DataFrame, plane: GroundPlane) -> pd.DataFrame:
    """Map tracks to ground metres: X across, Z away from the camera.

    Depth is read from the box *centre* row, converted to a feet row by the
    constant centre factor, rather than from the box bottom directly. The two
    are the same quantity in principle and not in practice: the box bottom is
    `py + crop_h / 2`, so every frame's error in the box height lands in the
    position, and depth amplifies it -- one pixel near the horizon is metres.
    Reading it off the bottom cost the Reading window's split-half speed
    reliability, which fell from 0.57 to 0.17; the centre row restored the
    other three clips and, on its own, left Reading at 0.19. The rest of that
    was the horizon, and is handled by the schedule above.

    The object is assumed to be standing on the plane. For a player that is
    true. For a ball in flight it is not: a lofted ball maps as though it
    were resting on the pitch further away, so it is placed too far. The
    single scale this replaces had no notion of the ground at all, so it was
    wrong about the ball differently rather than less.
    """
    out = tracks.copy()
    x = out.px.to_numpy(float)
    horizon = (plane.horizon_at(out.frame.to_numpy())
               if "frame" in out.columns
               else np.full(len(out), plane.y_horizon))
    v = (out.py.to_numpy(float) - horizon) / plane.centre_factor

    usable = v > MIN_ROWS_BELOW_HORIZON

    Z = np.where(usable, plane.focal_px * plane.h_cam_m / np.where(usable, v, 1.0),
                 np.nan)
    X = np.where(usable, (x - plane.cx) * Z / plane.focal_px, np.nan)

    out["px"] = X
    out["py"] = Z
    out["x"] = X
    out["y"] = Z
    return out


# Consecutive detections closer together than this are all noise and no
# motion, and their ratio is a ratio of two noises.
MIN_CALIBRATION_STEP_PX = 0.5
MIN_CALIBRATION_PAIRS = 500


def effective_px_per_m(tracks: pd.DataFrame, plane: GroundPlane,
                       verbose: bool = False) -> float | None:
    """The single scale that reproduces the plane's distances.

    This is the part of the ground plane worth shipping, and it is not the
    ground plane. Measured end to end, projecting onto the plane makes every
    instrument that can be checked worse -- pass F1 falls from 0.79 to 0.76
    on the tuning windows with thresholds re-tuned for it, split-half speed
    reliability collapses from 0.57 to 0.19 on one window, distance-rate
    reliability is equal or worse everywhere. Depth goes as 1/(row - horizon),
    so it multiplies vertical position noise, and at these resolutions that
    costs more than the geometry gains.

    What the plane does establish is that the single scale has the wrong
    *value*, and always in the same direction. A scale from player height is
    the scale for motion *across* the view; motion *into* it covers fewer
    pixels per metre, by a factor of roughly (rows below horizon) / f. So one
    number taken from heights overstates pixels per metre, and every distance
    and speed divided by it comes out too small. Measured on four broadcast
    windows the shipped number is 9%, 9%, 17% and 26% too large.

    Correcting it is free in a way replacing the map is not: a uniform
    rescale leaves every correlation exactly where it was, so no reliability
    figure can move. Only the level does.

    The ratio is taken over the displacements players actually made rather
    than derived in closed form, so it is weighted by how they really moved
    -- a clip where play runs across the camera is corrected less than one
    where it runs away from it.
    """
    if plane is None or tracks is None or tracks.empty:
        return None
    if "track_id" not in tracks.columns:
        return None

    ground = to_ground_coords(tracks, plane)
    players = tracks.cls == "player" if "cls" in tracks.columns else slice(None)

    ratios = []
    for _, g in tracks[players].groupby("track_id", sort=False):
        g = g.sort_values("frame")
        step_px = np.hypot(g.px.diff().to_numpy(float),
                           g.py.diff().to_numpy(float))
        h = ground.loc[g.index]
        step_m = np.hypot(h.px.diff().to_numpy(float),
                          h.py.diff().to_numpy(float))
        ok = (np.isfinite(step_px) & np.isfinite(step_m)
              & (step_m > 1e-6) & (step_px > MIN_CALIBRATION_STEP_PX))
        if ok.any():
            ratios.append(step_px[ok] / step_m[ok])

    if not ratios:
        return None
    allr = np.concatenate(ratios)
    if allr.size < MIN_CALIBRATION_PAIRS:
        return None

    value = float(np.median(allr))
    if not np.isfinite(value) or value <= 0:
        return None
    if verbose:
        print(f"  [plane] scale calibrated to {value:.1f} px/m over "
              f"{allr.size} steps")
    return value
