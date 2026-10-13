"""Ball candidates from the panorama heatmap (`src/ball_heatmap.py`) on a
SoccerTrack v2 window, as the ball rows of the pipeline's
`tracks.parquet` (`detection_method` "heatmap", `confidence` its
probability, `crop_h` the ball's size there through the calibration).

    python detect_ball_heatmap.py --window W --weights CKPT --tau 0.2 \\
        --stride 2 --out balls.parquet

`--v2`: the motion model (`build(motion=True)`), protocol v3's dense
path. A first pass grabs the background samples (every 50th frame, grey,
cropped to the pitch box: `motion_bg.BackgroundCache`, with the window as
the available range); a second, sequential, decode keeps frames in a
ring buffer and runs `detect_v2` on frame t once t+2 is in (t-2 / t+2
missing at the window's edges). It is the decode of `eval_ball_detector.py
--v2`, so the two give the same candidates at shared frames.

- Rows: `detection_method` "heatmap_v2"; `confidence` p_cal =
  sigmoid(a logit + b) from `--calib` (default `<ckpt>.calib.json`;
  `--no-calib` keeps p_raw); extra columns `logit`, `p_raw` and `n_bg`
  (the frame's background samples, 15 when none is clipped by the
  window); p_raw >= `--tau` (0.05), at most `--top-k` (8) a frame.
- Frames: every `--stride`-th local frame, plus `--extra-frames` (local
  frames, or `measured`: every frame `eval_ball_detector.py` measures in
  the window, so that its `--candidates` can read this file), within
  `--range A B` (local, B excluded; a window split over several runs
  gives the same rows, the background always coming from the whole
  window).
- A checkpoint is refused on a match of its training provenance
  (`ball_heatmap.check_unseen`) unless `--allow-seen`, and the test
  matches are never opened. Only frame counts and timings are printed.

    python detect_ball_heatmap.py --v2 --window W --weights ball_v2.4000.pt \\
        --extra-frames measured --threads 2 --out balls_v2.parquet
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
import pandas as pd

from src import soccertrack_v2 as st
from src.ball_heatmap import build, detect
from src.paths import DATA_DIR

TEST_MATCHES = {"128057", "132831"}     # M7, M9: never opened
V2_COLUMNS = {"frame": int, "time_s": float, "track_id": int, "cls": str,
              "px": float, "py": float, "crop_h": float, "crop_w": float,
              "detection_method": str, "confidence": float, "logit": float,
              "p_raw": float, "n_bg": int}


def v2_frames(n: int, stride: int, extra=(), start: int = 0,
              stop: int | None = None) -> list[int]:
    """Local frames to detect on: every `stride`-th of the window's `n`
    (counted from local 0) and the `extra` ones, within [start, stop)."""
    stop = n if stop is None else min(stop, n)
    keep = set(range(-(-start // stride) * stride, stop, stride))
    keep |= {int(f) for f in extra if start <= int(f) < stop}
    return sorted(keep)


def background_pass(path, first: int, last: int, need, crop):
    """The background samples `need` (absolute frames) of a window video
    whose local frame 0 is `first`: every frame grabbed, the needed ones
    decoded and kept grey, cropped (as `eval_ball_detector.decode`)."""
    from src.motion_bg import BackgroundCache

    cache = BackgroundCache(first, last, crop=crop)
    cap = cv2.VideoCapture(str(path))
    idx, stop = 0, (max(need) - first) if need else -1
    while idx <= stop:
        if not cap.grab():
            break
        if first + idx in need:
            ok, image = cap.retrieve()
            if ok:
                cache.add(first + idx, image)
        idx += 1
    cap.release()
    cache.finish()
    return cache, idx


def dense_v2(path, info: dict, model, frames, calib: dict | None = None,
             cal: dict | None = None, tau: float = 0.05, top_k: int = 8,
             log=print, every: int = 500) -> pd.DataFrame:
    """`detect_v2` at the local `frames` of a window video (`info`: its
    json: start_frame, frames, width, height, pitch), as tracks rows."""
    from src.ball_heatmap import detect_v2, p_cal, pitch_box
    from src.motion_bg import BG_HALF_SPAN, BG_STEP, OFFSET, RingBuffer, bg_sample_frames

    first = int(info["start_frame"])
    last = first + int(info["frames"]) - 1
    pitch = info["pitch"]
    wanted = set(int(f) for f in frames)
    t0 = time.time()
    need = set()
    for t in wanted:
        need |= set(bg_sample_frames(first + t, first, last))
    box = pitch_box((info["height"], info["width"]), pitch)
    cache, grabbed = background_pass(path, first, last, need, box)
    log(f"  background pass: {grabbed} frames grabbed, {len(cache.samples)} "
        f"samples [{time.time() - t0:.0f} s]", flush=True)

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    rows, done = [], set()

    def run(t, ring):
        bg, n = cache.get(first + t)
        prev, cur, nxt = ring.triple(t)
        for u, v, p, size, z in detect_v2(model, prev, cur, nxt, bg, pitch, tau=tau,
                                          top_k=top_k, cal=cal):
            rows.append((t, t / fps, -1, "ball", u, v, size, np.nan, "heatmap_v2",
                         float(p_cal(z, calib)), z, p, n))
        done.add(t)
        cache.drop_before(first + t - BG_HALF_SPAN - BG_STEP)

    ring = RingBuffer(2 * OFFSET + 1)
    idx, end = 0, (max(wanted) + OFFSET) if wanted else -1
    while idx <= end:
        ok, image = cap.read()
        if not ok:
            break
        ring.push(idx, image)
        if idx - OFFSET in wanted:
            run(idx - OFFSET, ring)
        idx += 1
        if idx % every == 0:
            log(f"  frame {idx}: {len(done)} of {len(wanted)} frames detected "
                f"[{time.time() - t0:.0f} s]", flush=True)
    cap.release()
    for t in range(max(idx - OFFSET, 0), idx):       # t + 2 past the end
        if t in wanted:
            run(t, ring)
    if len(done) < len(wanted):
        log(f"  WARNING: {len(wanted) - len(done)} frames past the end of the "
            f"video (decoded {idx})")
    df = pd.DataFrame(rows, columns=list(V2_COLUMNS))
    return df.astype(V2_COLUMNS)


def extra_frames(window: str, specs) -> list[int]:
    """`--extra-frames`: local frames, or `measured` for every frame
    `eval_ball_detector.py` measures in the window."""
    out = []
    for s in specs:
        if s == "measured":
            from eval_ball_detector import frames_of, measured_frames

            out += measured_frames(frames_of(window))
        else:
            out.append(int(s))
    return out


def main_v2(args):
    import torch

    from src import ball_heatmap as bh

    if any(f"_{m}_" in f"_{args.window}_" for m in TEST_MATCHES):
        raise SystemExit(f"{args.window}: a test match is never opened")
    info = json.loads((DATA_DIR / "soccertrack_v2" / f"{args.window}.json").read_text())
    if str(info["match"]) in TEST_MATCHES:
        raise SystemExit(f"{args.window}: test match {info['match']} is never opened")
    try:
        bh.check_unseen(args.weights, info["match"], args.allow_seen)
    except ValueError as e:
        raise SystemExit(f"refused: {e}")
    calib = None
    if not args.no_calib:
        calib = bh.load_calibration(args.calib or args.weights)
        if calib is None:
            where = args.calib or bh.calib_path(args.weights)
            raise SystemExit(f"no calibration {where}: fit it with "
                             f"calibrate_ball_heatmap.py, or --no-calib for p_raw")
        print(f"  calibration: a = {calib['a']}, b = {calib['b']}, tau = "
              f"{calib.get('tau')}")
    torch.set_num_threads(args.threads)
    model = build(motion=True)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    start, stop = args.range or (0, None)
    frames = v2_frames(int(info["frames"]), args.stride,
                       extra_frames(args.window, args.extra_frames), start, stop)
    t0 = time.time()
    df = dense_v2(DATA_DIR / "soccertrack_v2" / f"{args.window}.mp4", info, model,
                  frames, calib=calib, cal=st.calibration(info["match"]),
                  tau=args.tau if args.tau is not None else 0.05,
                  top_k=args.top_k or 8)
    df.to_parquet(args.out)
    print(f"  {len(frames)} frames detected -> {args.out} [{time.time() - t0:.0f} s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tau", type=float, help="v1: required; --v2: p_raw floor, 0.05")
    ap.add_argument("--top-k", type=int, help="default 4 (v1), 8 (--v2)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v2", action="store_true", help="the motion model, dense")
    ap.add_argument("--calib", help="--v2: default <weights>.calib.json")
    ap.add_argument("--no-calib", action="store_true", help="--v2: confidence = p_raw")
    ap.add_argument("--extra-frames", nargs="*", default=[],
                    help="--v2: local frames to add, or 'measured'")
    ap.add_argument("--range", type=int, nargs=2, metavar=("A", "B"),
                    help="--v2: local frames [A, B) only")
    ap.add_argument("--allow-seen", action="store_true",
                    help="--v2: allow a match the checkpoint was trained on")
    args = ap.parse_args()
    if args.v2:
        main_v2(args)
        return
    if args.tau is None:
        ap.error("--tau is required")
    if args.top_k is None:
        args.top_k = 4
    import torch

    torch.set_num_threads(args.threads)
    info = json.loads((DATA_DIR / "soccertrack_v2" / f"{args.window}.json").read_text())
    cal = st.calibration(info["match"])
    model = build()
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{args.window}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    rows, idx, t0 = [], 0, time.time()
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if idx % args.stride == 0:
            for u, v, p, size in detect(model, image, info["pitch"], tau=args.tau,
                                        top_k=args.top_k, cal=cal):
                rows.append({"frame": idx, "time_s": idx / fps, "track_id": -1,
                             "cls": "ball", "px": u, "py": v, "crop_h": size,
                             "crop_w": float("nan"),
                             "detection_method": "heatmap", "confidence": p})
        if idx % 500 == 0:
            print(f"  frame {idx}: {len(rows)} candidates "
                  f"[{time.time() - t0:.0f} s]", flush=True)
        idx += 1
    cap.release()
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"  {len(rows)} candidates on {idx} frames -> {args.out}")


# ---- Self-check -------------------------------------------------------------------

def _check():
    """A synthetic window video (a ball crossing a textured pitch) and a
    small stand-in motion model: the dense run equals `detect_v2` called
    frame by frame with t-2, t+2 and the background built from the whole
    video (edges: no t-2 at 0 / 1, no t+2 at the end, few samples at the
    start); a window split by `--range` gives the same rows; stride and
    extra frames; p_cal in `confidence`."""
    import tempfile
    from pathlib import Path

    import torch

    from src.ball_heatmap import detect_v2, p_cal, pitch_box
    from src.motion_bg import BackgroundCache, bg_sample_frames

    torch.set_num_threads(1)        # tiny tensors: threads only add overhead
    assert v2_frames(10, 2) == [0, 2, 4, 6, 8]
    assert v2_frames(10, 2, extra=[3, 9, 12], start=3, stop=8) == [3, 4, 6]
    assert v2_frames(10, 3, start=4) == [6, 9]

    class Tiny(torch.nn.Module):
        """Logit per 4 x 4 cell: brightness plus the three motion channels."""

        motion = True

        def forward(self, x):
            s = x[:, :3].mean(1, keepdim=True)
            if x.shape[1] == 6:
                s = s + 0.05 * x[:, 3:4] + 0.05 * x[:, 4:5] + 0.02 * x[:, 5:6]
            return torch.nn.functional.avg_pool2d(s, 4)[:, 0] - 1.0

    rng = np.random.default_rng(0)
    H, W, N, first = 120, 320, 460, 15020      # a start off the 50 grid
    pitch = rng.integers(50, 110, (H, W, 3)).astype(np.uint8)
    outline = [[30, 40], [290, 40], [300, 100], [20, 100]]
    info = {"start_frame": first, "frames": N, "width": W, "height": H,
            "pitch": outline}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "w.avi"
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (W, H))
        for t in range(N):
            im = pitch.copy()
            cv2.circle(im, (40 + (3 * t) % 240, 70), 3, (250, 250, 250), -1)
            out.write(im)
        out.release()
        cap, video = cv2.VideoCapture(str(path)), []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            video.append(im)
        cap.release()
        assert len(video) == N
        model = Tiny().eval()
        calib = {"a": 0.5, "b": -0.3}
        frames = v2_frames(N, 2, extra=[1, 3, 401, N - 1])
        quiet = lambda *a, **k: None  # noqa: E731
        df = dense_v2(path, info, model, frames, calib=calib, log=quiet)
        a = dense_v2(path, info, model, frames, calib=calib, log=quiet,
                     every=7)[lambda d: d.frame < 200]
        parts = [dense_v2(path, info, model, v2_frames(N, 2, [1, 3, 401, N - 1], lo, hi),
                          calib=calib, log=quiet) for lo, hi in ((0, 200), (200, N))]
        assert pd.concat(parts, ignore_index=True).equals(df)
        assert a.reset_index(drop=True).equals(parts[0])
        # The reference: every frame decoded, the background from all of them.
        cache = BackgroundCache(first, first + N - 1,
                                crop=pitch_box((H, W), outline))
        for t, im in enumerate(video):
            cache.add(first + t, im)
        cache.finish()
        ref = []
        for t in frames:
            bg, n = cache.get(first + t)
            assert n == len(bg_sample_frames(first + t, first, first + N - 1))
            prev = video[t - 2] if t >= 2 else None
            nxt = video[t + 2] if t + 2 < N else None
            for u, v, p, size, z in detect_v2(model, prev, video[t], nxt, bg, outline):
                ref.append((t, u, v, p, z, n))
        got = df[["frame", "px", "py", "p_raw", "logit", "n_bg"]].to_numpy()
        assert len(ref) == len(df) > 0 and np.array_equal(got, np.array(ref))
        assert set(df.frame) <= set(frames) and {1, 3, 401, N - 1} <= set(df.frame)
        assert np.allclose(df.confidence, p_cal(df.logit, calib))
        assert (df.p_raw >= 0.05).all() and df.groupby("frame").size().max() <= 8
        assert df.n_bg.min() < 8 <= df.n_bg.max() and df.crop_h.isna().all()
        assert (df.detection_method == "heatmap_v2").all() and (df.track_id == -1).all()
        # Motion matters here: the zero-motion run differs at the ball.
        x0, y0, *_ = pitch_box((H, W), outline)
        t = 200
        b = cache.get(first + t)[0]
        m1 = detect_v2(model, video[t - 2], video[t], video[t + 2], b, outline)
        m0 = detect_v2(model, video[t - 2], video[t], video[t + 2], b, outline,
                       motion_zero=True)
        assert not np.array_equal(m1, m0)
        empty = dense_v2(path, info, model, [], log=quiet)
        assert empty.empty and list(empty.columns) == list(V2_COLUMNS)
    print("  detect_ball_heatmap: dense v2 = detect_v2 frame by frame with the "
          "whole-window background, edges, split windows, stride / extra frames, "
          "p_cal: ok")


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _check()
    else:
        main()
