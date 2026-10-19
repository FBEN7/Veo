"""The ball across frames on a fixed, calibrated view (protocol v4-r1): a
detector's per-frame candidates linked over time, short gaps filled on the
pitch, and the raw top candidate where neither gives a position.

A per-frame detector loses the ball for a frame or two in flight (motion
blur, a 5 px ball on the far side, a body in front) and, when it does, its
top candidate is somewhere else: a spot, a sock, a resting spare ball. The
ball cannot jump, so its picks over time should form one plausible path;
`track` keeps the candidates on the cheapest such path and puts the ball
on the straight line between two picks a few frames apart.

    tr = track(rows, cal, tau=0.25, frames=processed)

Input: the rows of one dense run, `detect_ball_heatmap.py --v2 --stride 2`
without `--extra-frames` (frame, px, py, crop_h, confidence = v2's
calibrated p_cal, logit, p_raw), and the frames it processed.
`check_stride_only(sidecar)` refuses a run with extra frames or with a
frame off the stride: frames timed from the labels must never reach the
linker or the fallback, and the gap fill relies on the stride (step 3).
Pixels are px, py: px_raw / py_raw are dropped first, so the linker, the
fill and the fallback read the columns `eval_ball_detector`'s raw
comparator reads.

1. **Candidates**: the rows with confidence at least `tau`, at most
   `top_k` a frame by confidence.
2. **Link** (`link_picks`): `ball_link.link` as committed, with an empty
   player table -- no players and no ground truth, so the static penalty
   falls on every candidate that has stayed within 0.5 m for 2 s, the ball
   at a standing player's feet included -- and the processed frames
   (`frames`) as its detection frames, so that a processed frame without
   any candidate still counts as a skipped one. At most one pick per
   detection frame: source "detected".
3. **Gap fill** (`fill_gaps`): between consecutive picks a (frame fa) and
   b (frame fb) with fb - fa <= `gap` and an implied ground speed
   |gb - ga| / ((fb - fa) / fps) <= `max_fill_speed` (35 m/s, the
   linker's free speed; ga, gb the picks' ground points), every frame
   strictly between gets the ground point interpolated linearly in pitch
   metres and projected back (`ball_heatmap.pitch_to_image`), the size
   linearly, and v = ground v - size / 2: source "interpolated", with the
   weaker end's p, p_raw and logit. No extrapolation before the first pick
   or after the last. `gap` must be below (ball_link.MAX_SKIP + 2) times
   the smallest spacing of the detection frames (14 at stride 2): two
   picks closer than that are always joined by the linker's motion check
   (60 m/s at most), never across a chain restart.
4. **Fallback** (`fallback`, on by default): a detection frame with no
   pick and no fill gets the raw top candidate there at any p (best p_raw,
   ties in row order, as `eval_ball_detector.parquet_candidates` orders a
   frame): source "fallback".
5. Any other frame has no row: a miss. On a stride-2 run an odd frame is
   never a detection frame, so it is interpolated or a miss.

The output has one row per frame with a position, sorted: frame, u, v (the
picture point, as the detector's px, py), size (its crop_h, the ball's
size in px), p (confidence: p_cal for a detection or the fallback, the
weaker end's for an interpolated row), p_raw and logit (the rows' columns
when they have them; else p_raw = sigmoid(logit) and logit = logit(p)),
source. `track(..., info={})` also fills the dict with counts: picks,
pairs within the gap, pairs filled, pairs refused for speed, the longest
fill (fb - fa) and fallback rows.

Why on the pitch: interpolating in the picture puts the ball off the
ground line the two picks lie on. On the U-E dense runs at a gap of 6 or
less the two differ by more than 6 px at 37 of 2243 fill frames, by up to
35-39 px (up to 1.2 m on the ground), all of them fast fills: 12-22 m
jumps at 37-59 m/s, which the speed limit now refuses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ball_link import MAX_SKIP, _logit, link

TRACK_COLUMNS = ["frame", "u", "v", "size", "p", "p_raw", "logit", "source"]
SOURCES = ("detected", "interpolated", "fallback")
# Protocol v4-r1's frozen settings (tau is the calibration's, 0.25).
DEFAULTS = {"top_k": 4, "c_null": 1.0, "c_static": 2.0, "gap": 6, "max_fill_speed": 35.0,
            "fallback": True}


def check_stride_only(meta: dict) -> tuple[list[int], int]:
    """The processed frames and the stride of a dense run's sidecar
    (`detect_ball_heatmap.read_sidecar`). Refuses a run whose sidecar does
    not say it had no extra frames, or that lists a frame not divisible by
    its stride."""
    stride = meta.get("stride")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"the sidecar's stride is {stride!r}, not a positive integer")
    extra = meta.get("extra_frames")
    if extra is None or len(extra):
        raise ValueError(f"the dense run has extra frames ({extra!r}): the tracker takes "
                         f"a stride-only run (no --extra-frames)")
    frames = sorted(int(f) for f in meta["frames"])
    off = [f for f in frames if f % stride]
    if off:
        raise ValueError(f"{len(off)} processed frames are off the stride {stride} "
                         f"(e.g. {off[:5]})")
    return frames, stride


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
        TRACK_COLUMNS, (int, float, float, float, float, float, float, object))})


def _ball_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Ball rows (`cls` "ball" when there is a column), pixels px, py only."""
    if "cls" in rows:
        rows = rows[rows.cls == "ball"]
    return rows.drop(columns=[c for c in ("px_raw", "py_raw") if c in rows])


def candidates(rows: pd.DataFrame, tau: float, top_k: int | None = 4) -> pd.DataFrame:
    """Ball rows with confidence >= tau, at most `top_k` a frame (None:
    all) by confidence, best first."""
    rows = _ball_rows(rows)
    rows = rows[rows.confidence >= tau]
    rows = rows.sort_values("confidence", ascending=False, kind="stable")
    if top_k is not None:
        rows = rows.groupby("frame", sort=False).head(top_k)
    return rows


def _as_track(rows: pd.DataFrame, source: str) -> pd.DataFrame:
    """Candidate rows as track rows."""
    if rows.empty:
        return _empty()
    p = rows.confidence.to_numpy(float)
    z = rows.logit.to_numpy(float) if "logit" in rows else _logit(p)
    p_raw = rows.p_raw.to_numpy(float) if "p_raw" in rows else 1 / (1 + np.exp(-z))
    size = (rows.crop_h.to_numpy(float) if "crop_h" in rows
            else np.full(len(rows), np.nan))
    out = pd.DataFrame({"frame": rows.frame.to_numpy(int), "u": rows.px.to_numpy(float),
                        "v": rows.py.to_numpy(float), "size": size, "p": p,
                        "p_raw": p_raw, "logit": z, "source": source})
    return out.sort_values("frame", kind="stable").reset_index(drop=True)


def _detection_frames(rows: pd.DataFrame, frames) -> np.ndarray:
    if frames is None:
        return np.unique(rows.frame.to_numpy(int)) if len(rows) else np.zeros(0, int)
    return np.unique(np.asarray(list(frames), int))


def link_picks(rows: pd.DataFrame, cal: dict, tau: float, top_k: int | None = 4,
               c_null: float = 1.0, c_static: float = 2.0, fps: float = 25,
               players: pd.DataFrame | None = None, frames=None) -> pd.DataFrame:
    """Steps 1-2: the linked picks as track rows (source "detected"), at
    most one per detection frame. `frames`: the detection frames (default:
    those of `rows`, before the tau filter). `players` None: an empty
    player table (no clearance)."""
    rows = _ball_rows(rows)
    frames = _detection_frames(rows, frames)
    cand = candidates(rows, tau, top_k)
    if "cls" not in cand:
        cand = cand.assign(cls="ball")
    if players is None:
        players = cand.iloc[:0]
    kept = link(cand, cal, players, c_null=c_null, c_static=c_static, fps=fps,
                tau=tau, frames=frames)
    return _as_track(kept, "detected")


def ground_of(tr: pd.DataFrame, cal: dict) -> np.ndarray:
    """(N, 2) pitch metres of track rows' ground points, image_to_pitch(u,
    v + size / 2) (a missing size as 0), as `ball_link.ground_points` puts
    a ball on the pitch; NaN where unmapped."""
    from .soccertrack_v2 import image_to_pitch

    if tr.empty:
        return np.zeros((0, 2))
    size = np.nan_to_num(tr["size"].to_numpy(float), nan=0.0)
    uv = np.c_[tr.u.to_numpy(float), tr.v.to_numpy(float) + size / 2]
    return np.asarray(image_to_pitch(uv, cal), float).reshape(-1, 2)


def fill_gaps(picks: pd.DataFrame, gap: int, cal: dict, max_speed: float = 35.0,
              fps: float = 25, info: dict | None = None) -> pd.DataFrame:
    """Step 3: `picks` (track rows) with every frame strictly between two
    consecutive picks at most `gap` frames apart, whose implied ground
    speed is at most `max_speed` m/s, filled in on the pitch; p, p_raw,
    logit the weaker end's. `info` receives the counts."""
    from .ball_heatmap import pitch_to_image

    picks = picks.sort_values("frame", kind="stable").reset_index(drop=True)
    f = picks.frame.to_numpy(int)
    if len(np.unique(f)) != len(f):
        raise ValueError("picks: more than one row at a frame")
    d = np.diff(f)
    near = np.flatnonzero((d >= 2) & (d <= gap))
    g = ground_of(picks, cal)
    with np.errstate(invalid="ignore"):
        speed = np.hypot(*(g[near + 1] - g[near]).T) / (d[near] / fps)
    pairs = near[speed <= max_speed]          # NaN (unmapped): never filled
    if info is not None:
        info.update({"pairs within gap": int(len(near)), "pairs filled": int(len(pairs)),
                     "pairs refused for speed": int(len(near) - len(pairs)),
                     "longest fill": int(d[pairs].max()) if len(pairs) else 0})
    if not len(pairs):
        return picks[TRACK_COLUMNS].copy()
    a, b = picks.iloc[pairs], picks.iloc[pairs + 1]
    n = d[pairs] - 1                                   # frames strictly between
    rep = np.repeat(np.arange(len(pairs)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 1
    w = k / np.repeat(d[pairs], n)
    ground = g[pairs][rep] + w[:, None] * (g[pairs + 1] - g[pairs])[rep]
    uv = np.asarray(pitch_to_image(ground, cal), float).reshape(-1, 2)
    sa, sb = a["size"].to_numpy(float)[rep], b["size"].to_numpy(float)[rep]
    size = sa + w * (sb - sa)
    fill = {"frame": np.repeat(f[pairs], n) + k, "u": uv[:, 0],
            "v": uv[:, 1] - np.nan_to_num(size, nan=0.0) / 2, "size": size}
    for c in ("p", "p_raw", "logit"):
        fill[c] = np.minimum(a[c].to_numpy(float), b[c].to_numpy(float))[rep]
    fill["source"] = "interpolated"
    out = pd.concat([picks[TRACK_COLUMNS], pd.DataFrame(fill)[TRACK_COLUMNS]],
                    ignore_index=True)
    return out.sort_values("frame", kind="stable").reset_index(drop=True)


def _rank(rows: pd.DataFrame) -> np.ndarray:
    """The order `eval_ball_detector.parquet_candidates` gives a frame's
    rows: p_raw (or sigmoid(logit)) when there is a logit, else the
    confidence."""
    if "logit" in rows:
        if "p_raw" in rows:
            return rows.p_raw.to_numpy(float)
        return 1 / (1 + np.exp(-rows.logit.to_numpy(float)))
    return rows.confidence.to_numpy(float)


def fallback_picks(rows: pd.DataFrame, frames, taken) -> pd.DataFrame:
    """Step 4: at each of `frames` not in `taken`, the raw top row at any p
    (`_rank`, ties in row order), as track rows (source "fallback")."""
    rows = _ball_rows(rows)
    want = set(np.asarray(list(frames), int).tolist()) - set(np.asarray(list(taken),
                                                                        int).tolist())
    r = rows[rows.frame.isin(want)]
    if r.empty:
        return _empty()
    top = r.iloc[np.argsort(-_rank(r), kind="stable")].groupby("frame", sort=False).head(1)
    return _as_track(top, "fallback")


def track(rows: pd.DataFrame, cal: dict, tau: float, top_k: int | None = 4,
          c_null: float = 1.0, c_static: float = 2.0, gap: int = 6,
          max_fill_speed: float = 35.0, fallback: bool = True, fps: float = 25,
          players: pd.DataFrame | None = None, frames=None,
          info: dict | None = None) -> pd.DataFrame:
    """Per-frame ball positions from a detector's candidate rows (frame,
    px, py, crop_h, confidence; logit, p_raw when there): linked
    (`link_picks`), gaps of at most `gap` frames filled on the pitch when
    the ball need not exceed `max_fill_speed` (`fill_gaps`), and, with
    `fallback`, the raw top row at a detection frame still without one
    (`fallback_picks`). `frames`: the detection frames, the processed
    frames of a dense run (default: those of `rows`). See the module
    docstring. Columns: TRACK_COLUMNS. `info` receives the counts."""
    rows = _ball_rows(rows)
    det = _detection_frames(rows, frames)
    grid = np.unique(np.r_[det, rows.frame.to_numpy(int)])
    spacing = int(np.diff(grid).min()) if len(grid) > 1 else None
    if spacing is not None and gap >= (MAX_SKIP + 2) * spacing:
        raise ValueError(f"gap {gap} >= (MAX_SKIP + 2) x {spacing} (the detection frames' "
                         f"spacing): a fill could cross a chain restart")
    picks = link_picks(rows, cal, tau, top_k=top_k, c_null=c_null, c_static=c_static,
                       fps=fps, players=players, frames=det)
    fill_info = {}
    tr = fill_gaps(picks, gap, cal, max_speed=max_fill_speed, fps=fps, info=fill_info)
    fb = fallback_picks(rows, det, tr.frame) if fallback else _empty()
    if len(fb):
        tr = pd.concat([tr, fb], ignore_index=True)
        tr = tr.sort_values("frame", kind="stable").reset_index(drop=True)
    if info is not None:
        info.update({"picks": int(len(picks)), **fill_info, "fallback": int(len(fb))})
    return tr[TRACK_COLUMNS]


# ---- Self-check -------------------------------------------------------------------

def _stand_in(s: float = 20.0, tilt: float = 0.0) -> dict:
    """A fisheye camera with a very long focal length (a pinhole one to
    within 1e-5 px here) and a homography of `s` px/m about the centre
    spot, which stays at (2048, 540): a pitch point (x, y) is at the centre
    plus s (x, y) / (1 + tilt y), so `tilt` < 0 makes the far side (y < 0)
    smaller, a perspective camera."""
    K = np.array([[1e6, 0, 2048], [0, 1e6, 540], [0, 0, 1]])
    Hp = np.array([[s, 0, 2048 - 52.5 * s], [0, s, 540 - 34 * s], [0, tilt, 1.0]])
    Hp[2, 2] = 1 - tilt * 34                 # the centre spot stays at (2048, 540)
    Hp[:2] += np.outer([2048, 540], Hp[2]) - np.outer([2048, 540], [0, 0, 1])
    return {"K": K, "D": np.zeros(4), "Knew": K, "Hinv": np.linalg.inv(Hp)}


def _check():
    """Synthetic stride-2 detections on a stand-in calibration (20 px/m
    everywhere, as `ball_link._check`): a ball rolling across with three
    missed detection frames is bridged when the gap fits (and not when it
    does not), a long gap is not, a brighter static spare ball is not
    chosen with c_static > 0 (and is without it), odd frames between
    stride-2 picks are interpolated at the midpoint, nothing is
    extrapolated, a processed frame without candidates counts as a skip,
    candidates below tau are never picked, the fallback fills only
    detection frames without a pick or a fill, the fill is on the pitch on
    a perspective camera and refused above the speed limit, a gap that
    could cross a restart is refused, and the sidecar check refuses extra
    or off-stride frames."""
    from .ball_heatmap import pitch_to_image
    from .soccertrack_v2 import image_to_pitch

    s = 20.0
    cal = _stand_in(s)
    assert np.allclose(image_to_pitch([[2048.0, 540.0]], cal), 0, atol=1e-6)
    assert np.allclose(image_to_pitch([[2048.0 + s, 540.0]], cal), [[1, 0]], atol=1e-6)

    def px(x, y):
        return 2048 + x * s, 540 + y * s

    def z(p):
        return float(np.log(p / (1 - p)))

    missed = set(range(40, 46, 2)) | set(range(100, 132, 2))    # 3 short, 16 long
    rows = []
    for f in range(0, 200, 2):
        t = f / 25
        if f not in missed and f >= 10:
            # The ball, at 8 m/s along x from (-21, 0) at frame 10; its size
            # grows by 0.02 px a frame, to check the size is interpolated.
            u, v = px(-21 + 8 * (t - 0.4), 0)
            rows.append(dict(frame=f, cls="ball", px=u, py=v - 2, crop_h=4 + 0.02 * f,
                             confidence=0.6, logit=z(0.6) + 0.001 * f, p_raw=0.55))
        # A brighter spare ball resting at (30, 20) the whole time.
        u, v = px(30, 20)
        rows.append(dict(frame=f, cls="ball", px=u, py=v - 2, crop_h=4.0,
                         confidence=0.7, logit=z(0.7), p_raw=0.65))
        # A weak candidate beside the ball's path, below tau.
        u, v = px(-21 + 8 * (t - 0.4), 1.0)
        rows.append(dict(frame=f, cls="ball", px=u, py=v - 2, crop_h=4.0,
                         confidence=0.45, logit=z(0.45), p_raw=0.4))
    rows = pd.DataFrame(rows)
    ball = rows[rows.confidence == 0.6]
    frames = np.arange(0, 200, 2)

    info = {}
    tr = track(rows, cal, tau=0.5, gap=12, fallback=False, frames=frames, info=info)
    assert list(tr.columns) == TRACK_COLUMNS and tr.frame.is_unique
    assert tr.frame.is_monotonic_increasing
    det = tr[tr.source == "detected"]
    # Every ball detection kept; the spare ball and the weak one never.
    assert set(det.frame) == set(ball.frame), sorted(set(ball.frame) ^ set(det.frame))
    assert np.allclose(det.u.to_numpy(), ball.px.to_numpy())
    assert (det.p == 0.6).all() and not (tr.p >= 0.7).any() and not (tr.p == 0.45).any()
    assert info["picks"] == len(det) and info["fallback"] == 0
    assert info["longest fill"] == 8 and info["pairs refused for speed"] == 0
    # No extrapolation: nothing before the first pick or after the last.
    assert tr.frame.min() == ball.frame.min() and tr.frame.max() == ball.frame.max()
    # The short miss (picks at 38 and 46, 8 frames) bridged at gap 12, on
    # the line (the camera is affine here, so on the picture's line too),
    # size interpolated, p / logit the weaker end's.
    mid = tr[(tr.frame > 38) & (tr.frame < 46)]
    assert list(mid.frame) == list(range(39, 46)) and (mid.source == "interpolated").all()
    want_u = np.array([px(-21 + 8 * (f / 25 - 0.4), 0)[0] for f in mid.frame])
    assert np.allclose(mid.u, want_u) and np.allclose(mid.v, px(0, 0)[1] - 2)
    assert np.allclose(mid["size"], 4 + 0.02 * mid.frame)
    assert np.allclose(mid.logit, z(0.6) + 0.001 * 38) and (mid.p == 0.6).all()
    assert np.allclose(mid.p_raw, 0.55)
    # The long miss (98 -> 132) is not.
    assert not ((tr.frame > 98) & (tr.frame < 132)).any()
    # Odd frames between consecutive stride-2 picks: interpolated, at the
    # midpoint.
    odd = tr[tr.frame % 2 == 1]
    assert (odd.source == "interpolated").all()
    covered = [f for f in range(11, 190, 2) if not 38 < f < 46 and not 98 < f < 132]
    assert set(covered) <= set(odd.frame)
    g = tr.set_index("frame")
    assert np.isclose(g.u[61], (g.u[60] + g.u[62]) / 2)
    # gap 6: the 8-frame miss is not bridged either; odd frames still are.
    tr6 = track(rows, cal, tau=0.5, gap=6, fallback=False, frames=frames)
    assert not ((tr6.frame > 38) & (tr6.frame < 46)).any() and 61 in set(tr6.frame)
    assert set(tr6[tr6.source == "detected"].frame) == set(det.frame)
    # gap 1: no fill at all.
    assert (track(rows, cal, tau=0.5, gap=1, fallback=False,
                  frames=frames).source == "detected").all()
    # Without the static penalty the brighter resting ball wins most frames.
    free = track(rows, cal, tau=0.5, c_static=0.0, gap=6, fallback=False, frames=frames)
    assert (free[free.source == "detected"].p == 0.7).mean() > 0.5
    # The fallback (the default): every detection frame without a pick or a
    # fill gets its raw top row at any p -- here the resting spare ball,
    # the best row at the missed frames -- and only those; odd frames never.
    tf = track(rows, cal, tau=0.5, gap=6, frames=frames, info=info)
    fb = tf[tf.source == "fallback"]
    no_pick = sorted(set(frames) - set(tr6.frame))
    assert sorted(fb.frame) == no_pick and info["fallback"] == len(no_pick)
    assert (fb.frame % 2 == 0).all() and (fb.p == 0.7).all() and np.allclose(fb.p_raw, 0.65)
    assert tf[tf.source != "fallback"].reset_index(drop=True).equals(tr6)
    assert set(no_pick) >= {0, 8, 40, 42, 44} and 38 not in no_pick
    # Ties in the fallback go to the first row; the order is p_raw's.
    tie = pd.DataFrame([dict(frame=0, px=1.0, py=2.0, crop_h=4.0, confidence=0.1,
                             logit=-2.0, p_raw=0.12),
                        dict(frame=0, px=5.0, py=2.0, crop_h=4.0, confidence=0.1,
                             logit=-2.0, p_raw=0.12),
                        dict(frame=0, px=9.0, py=2.0, crop_h=4.0, confidence=0.09,
                             logit=-1.9, p_raw=0.13)])
    assert fallback_picks(tie, [0], []).u.tolist() == [9.0]
    assert fallback_picks(tie.iloc[:2], [0], []).u.tolist() == [1.0]
    assert fallback_picks(tie, [0], [0]).empty and fallback_picks(tie, [2], []).empty
    # A processed frame without candidates counts as a skip: a ball seen at
    # frames 0-20, then a candidate 60 m away at 36-56. Over the 7 empty
    # processed frames between (more than ball_link.MAX_SKIP) the chain may
    # restart, so both stretches are kept; told only of the frames with
    # rows, the two look adjacent, the 60 m jump is forbidden, and the
    # cheapest path drops part of the first.
    two = [dict(frame=f, cls="ball", px=px(-30 + 0.1 * f, 0)[0], py=px(0, 0)[1],
                crop_h=4.0, confidence=0.6) for f in range(0, 22, 2)]
    two += [dict(frame=f, cls="ball", px=px(30, 0)[0], py=px(0, 0)[1], crop_h=4.0,
                 confidence=0.6) for f in range(36, 58, 2)]
    two = pd.DataFrame(two)
    known = track(two, cal, tau=0.5, gap=12, frames=np.arange(0, 58, 2))
    assert len(known) == len(two) + 20      # 22 picks + 10 odd frames in each stretch
    assert set(known[known.source == "detected"].frame) == set(two.frame)
    assert not ((known.frame > 20) & (known.frame < 36)).any()     # 16 > gap
    rows_only = track(two, cal, tau=0.5, gap=12, fallback=False)
    assert (rows_only.source == "detected").sum() < len(two)
    # A gap that could cross a restart is refused: (MAX_SKIP + 2) x 2 = 14.
    for bad in (lambda: track(two, cal, tau=0.5, gap=14, frames=np.arange(0, 58, 2)),
                lambda: track(two, cal, tau=0.5, gap=7, frames=np.arange(0, 58))):
        try:
            bad()
            raise AssertionError("a gap reaching a chain restart was accepted")
        except ValueError:
            pass
    # Nothing worth a frame, or no rows: an empty track with the columns.
    for empty in (track(rows.assign(confidence=0.01), cal, tau=0.5, fallback=False),
                  track(rows.iloc[:0], cal, tau=0.5)):
        assert empty.empty and list(empty.columns) == TRACK_COLUMNS
    # Rows without logit / p_raw / cls: from the confidence.
    bare = ball.drop(columns=["logit", "p_raw", "cls"])
    tb = track(bare, cal, tau=0.5, gap=6)
    assert np.allclose(tb.logit[tb.source == "detected"], z(0.6))
    assert np.allclose(tb.p_raw, 0.6)
    # px / py, never px_raw / py_raw, in every step.
    shifted = ball.assign(px_raw=ball.px + 50.0, py_raw=ball.py + 50.0)
    ts = track(shifted, cal, tau=0.5, gap=6, frames=frames)
    assert ts.equals(track(ball, cal, tau=0.5, gap=6, frames=frames))

    # On the pitch: a perspective camera, a ball rolling at 33 m/s from
    # near the camera towards the far side (y from 20 to 12 in 6 frames). The fill's ground
    # point is on the pitch line between the two picks, at the frame's
    # share; in the picture that is not the straight line's midpoint.
    pc = _stand_in(s, tilt=-0.004)
    gp = pitch_to_image([[0.0, 20.0], [0.0, 19.6], [0.0, -20.0]], pc)
    assert np.allclose(image_to_pitch(gp, pc), [[0, 20], [0, 19.6], [0, -20]], atol=1e-6)
    assert np.hypot(*(gp[1] - gp[0])) > 1.3 * np.hypot(*(pitch_to_image(
        [[0.0, -19.6]], pc)[0] - gp[2]))                 # nearer is bigger: perspective
    ends = []
    for f, y in ((0, 20.0), (6, 12.0)):
        u, vg = pitch_to_image([[0.0, y]], pc)[0]
        ends.append(dict(frame=f, u=u, v=vg - (6.0 - f / 3) / 2, size=6.0 - f / 3, p=0.6,
                         p_raw=0.5, logit=0.2, source="detected"))
    info = {}
    fl = fill_gaps(pd.DataFrame(ends), 6, pc, info=info).set_index("frame")
    assert list(fl.index) == list(range(7)) and info["pairs filled"] == 1
    gm = ground_of(fl.reset_index(), pc)
    assert np.allclose(gm, np.c_[np.zeros(7), 20 - 8 / 6 * np.arange(7)], atol=1e-6)
    assert np.allclose(fl["size"], 6 - np.arange(7) / 3)
    lin_v = fl.v[0] + (fl.v[6] - fl.v[0]) / 2
    assert abs(fl.v[3] - lin_v) > 1.0, (fl.v[3], lin_v)    # not the picture's midpoint
    # The speed limit: 6 frames apart, 8.3 m (34.6 m/s) filled, 8.5 m not.
    for dy, ok in ((8.3, True), (8.5, False)):
        e2 = [dict(ends[0]), dict(ends[0])]
        e2[1]["frame"] = 6
        e2[1]["u"], vg = pitch_to_image([[0.0, 20.0 - dy]], pc)[0]
        e2[1]["v"] = vg - 3.0
        e2[1]["size"] = 6.0
        info = {}
        n = len(fill_gaps(pd.DataFrame(e2), 6, pc, max_speed=35.0, info=info))
        assert n == (7 if ok else 2) and info["pairs refused for speed"] == (not ok)
    # The sidecar: stride-only runs pass; extra frames, an unknown list of
    # them, or a frame off the stride are refused.
    good = {"stride": 2, "extra_frames": [], "frames": [4, 0, 2]}
    assert check_stride_only(good) == ([0, 2, 4], 2)
    for bad in (dict(good, extra_frames=["measured"]), dict(good, extra_frames=[5]),
                {k: v for k, v in good.items() if k != "extra_frames"},
                dict(good, frames=[0, 2, 3]), dict(good, stride=None),
                dict(good, stride=0)):
        try:
            check_stride_only(bad)
            raise AssertionError(f"{bad} was accepted")
        except ValueError:
            pass
    print(f"  ball_track: {len(det)} picks + {(tr.source == 'interpolated').sum()} "
          f"interpolated; short miss bridged at gap 12 (not 6), long miss not, "
          f"static spare ball refused, odd frames at the midpoint, skips counted "
          f"on processed frames, fallback at detection frames only, fill on the "
          f"pitch with its speed limit, restart-crossing gaps and non-stride runs "
          f"refused: ok")


if __name__ == "__main__":
    _check()
