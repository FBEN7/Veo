"""Measure re-identification against fragmentation we create on purpose.

Tracking continuity is the constraint under several other numbers: 98 to 143
tracks per 90-second window for 22 players, and per-player sprint counts that
do not survive a split-half test because a player's minutes are scattered
across several ids.

The obvious way to improve it is to merge more, and the obvious way to
measure it -- counting tracks against a roster of 22 -- rewards exactly that.
A matcher that merged everything into 22 tracks would score perfectly and be
useless.

So the ground truth is manufactured instead. Take tracks the pipeline already
produced, cut them into pieces with a gap at each cut, give every piece a
fresh id, and ask the matcher to put them back. Which pieces belong together
is then known exactly, and both errors are visible:

    recall      pairs of pieces from one source that were rejoined
    precision   pairs that were joined and came from one source

The trajectories are real, so the geometry the matcher sees is the geometry
it faces in use. What this cannot measure is whether the source tracks were
themselves correct; a source that is really two players makes the benchmark
generous about joining two players. It is a relative instrument for tuning
the matcher, not an absolute accuracy.

    python bench_track_continuity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_pass_outcome import WINDOWS
from score_soccernet import run_pipeline
from src import track_reid as reid

# A source track must be at least this long to be worth cutting up.
MIN_SOURCE_FRAMES = 100

# Gaps punched at each cut, in seconds. The schedule the matcher tries tops
# out at 2.5 s, so 3.0 is included to see what falling off the end costs.
GAPS_S = (0.2, 0.4, 0.8, 1.5, 2.5, 3.0)

PIECES = 3
FPS = 25.0

# Positional noise added to the frames at each cut, in metres.
#
# Without it this benchmark is not the task. A clean cut leaves the second
# piece continuing the first's motion exactly, so the prediction lands almost
# on it: measured, synthetic cuts drift 1.6-2.9 m/s at the median. Real
# ByteTrack fragments drift 7.8-13.3 m/s, because a real identity is dropped
# *when the player is occluded or half-detected*, and the frames on either
# side of that are exactly the corrupted ones the velocity is estimated from.
#
# Tuning against clean cuts therefore recommended a radius three times too
# tight, which scored better on the benchmark and merged fewer fragments in
# production -- the instrument was measuring an easier problem than the one
# the matcher faces. The noise below is calibrated so the synthetic drift
# distribution matches the observed one.
CUT_NOISE_M = 0.8


def fragment(players: pd.DataFrame, gap_s: float, seed: int = 0):
    """Cut long tracks into pieces separated by a gap; return truth."""
    rng = np.random.default_rng(seed)
    gap_frames = max(1, int(round(gap_s * FPS)))
    out, truth, next_id = [], {}, 1_000_000

    for tid, g in players.groupby("track_id"):
        g = g.sort_values("frame")
        if len(g) < MIN_SOURCE_FRAMES:
            # Short tracks are carried through unchanged: they are part of the
            # scene the matcher has to work in, and excluding them would make
            # the task easier than it is.
            out.append(g)
            continue

        # Cut points chosen so every piece survives the gap with frames left.
        usable = len(g) - PIECES * gap_frames
        if usable < PIECES * 10:
            out.append(g)
            continue
        cuts = sorted(rng.choice(np.arange(10, len(g) - 10), PIECES - 1,
                                 replace=False))

        start = 0
        for k, cut in enumerate(list(cuts) + [len(g)]):
            piece = g.iloc[start:cut]
            if k < PIECES - 1:
                piece = piece.iloc[:-gap_frames] if len(piece) > gap_frames else piece
            if len(piece) < 5:
                start = cut
                continue
            piece = piece.copy()
            # Corrupt the frames adjacent to the cut, which is what an
            # occlusion does to the detections a velocity is read from.
            if CUT_NOISE_M > 0 and len(piece) > 2:
                edge = min(reid.VELOCITY_WINDOW + 1, len(piece))
                for col in ("px", "py", "x", "y"):
                    if col not in piece.columns:
                        continue
                    vals = piece[col].to_numpy(dtype=float).copy()
                    vals[-edge:] += rng.normal(0.0, CUT_NOISE_M, edge)
                    vals[:edge] += rng.normal(0.0, CUT_NOISE_M, edge)
                    piece[col] = vals
            piece["track_id"] = next_id
            truth[next_id] = int(tid)
            next_id += 1
            out.append(piece)
            start = cut

    return pd.concat(out, ignore_index=True), truth


def pair_scores(merged: pd.DataFrame, truth: dict[int, int]):
    """Pairwise precision and recall over the pieces we created."""
    # Where each synthetic piece ended up after merging.
    landed = (merged[merged.track_id.notna()]
              .groupby("orig_id").track_id.first().to_dict())
    pieces = [p for p in truth if p in landed]
    if len(pieces) < 2:
        return float("nan"), float("nan"), 0

    tp = fp = fn = 0
    for a in range(len(pieces)):
        for b in range(a + 1, len(pieces)):
            pa, pb = pieces[a], pieces[b]
            same_source = truth[pa] == truth[pb]
            same_track = landed[pa] == landed[pb]
            if same_source and same_track:
                tp += 1
            elif same_track:
                fp += 1
            elif same_source:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return precision, recall, tp + fn


def run_window(metric: pd.DataFrame, gap_s: float, seed: int = 0):
    players = metric[metric.cls == "player"].copy()
    frag, truth = fragment(players, gap_s, seed)
    frag["orig_id"] = frag.track_id
    merged = reid.merge_fragments(frag, px_per_m=1.0, fps=FPS, verbose=False)
    merged["orig_id"] = frag["orig_id"].to_numpy()
    p, r, n = pair_scores(merged, truth)
    return p, r, n, merged.track_id.nunique(), len(truth)


def main():
    cache = {}
    for window, out_dir, *_ in WINDOWS:
        clip = json.loads((Path(out_dir) / "clip.json").read_text())["path"]
        _, _, metric = run_pipeline(clip, Path(out_dir), return_tracks=True)
        cache[window] = metric

    print("Pieces cut from real tracks, then handed back to the matcher.\n")
    print(f"  {'gap':>5s} {'precision':>10s} {'recall':>8s} {'F1':>6s} "
          f"{'pairs':>7s}   (mean over four windows)")
    for gap in GAPS_S:
        ps, rs, ns = [], [], 0
        for window, metric in cache.items():
            p, r, n, _, _ = run_window(metric, gap)
            if np.isfinite(p):
                ps.append(p)
            if np.isfinite(r):
                rs.append(r)
            ns += n
        p, r = np.mean(ps), np.mean(rs)
        f1 = 2 * p * r / (p + r) if (p + r) else float("nan")
        print(f"  {gap:4.1f}s {p:10.2f} {r:8.2f} {f1:6.2f} {ns:7d}")

    print("\nThe matcher's gap schedule stops at 2.5 s, so the 3.0 s row is "
          "the cost\nof a dropout longer than anything it attempts.")


if __name__ == "__main__":
    main()


# --- tuning ----------------------------------------------------------------
# Real dropouts are short. Measured inside the merged tracks of the four
# labelled windows (1813 of them), the gaps the matcher actually has to
# bridge are 66% at or under 0.2 s, 82% under 0.8 s and 96% under 2.5 s, with
# a median of two frames. A setting is therefore judged on an average over
# gap lengths weighted by how often each occurs, not on an unweighted mean
# over a set of gaps chosen for convenience -- which was the first objective
# used here and flattered long gaps roughly tenfold.
GAP_WEIGHTS = {0.2: 0.66, 0.4: 0.08, 0.8: 0.08, 1.5: 0.08, 2.5: 0.06}


def patched_best_match(speed_ms: float, margin_m: float):
    """`_best_match` with the reachable radius under our control."""
    def _match(frag, i, max_gap_frames, fps):
        cand = frag.candidates(i, max_gap_frames)
        if cand.size == 0:
            return -1, np.inf
        gap_frames = frag.start[cand] - frag.end[i]
        gap_s = np.maximum(gap_frames, 1.0) / fps
        pred_x = frag.ex[i] + frag.vx[i] * gap_frames
        pred_y = frag.ey[i] + frag.vy[i] * gap_frames
        dist = np.hypot(frag.sx[cand] - pred_x, frag.sy[cand] - pred_y)
        reach = speed_ms * gap_s + margin_m
        feasible = (dist <= reach) & (frag.team[cand] == frag.team[i])
        if not feasible.any():
            return -1, np.inf
        cost = np.where(feasible, dist / reach, np.inf)
        j = int(np.argmin(cost))
        return int(cand[j]), float(cost[j])
    return _match


def weighted_f1(cache, speed, margin):
    """F1 at each gap, averaged with the weights real dropouts have."""
    total, per_gap = 0.0, {}
    for gap, weight in GAP_WEIGHTS.items():
        ps, rs = [], []
        for metric in cache.values():
            p, r, _, _, _ = run_window(metric, gap)
            if np.isfinite(p):
                ps.append(p)
            if np.isfinite(r):
                rs.append(r)
        p, r = (np.mean(ps) if ps else 0.0), (np.mean(rs) if rs else 0.0)
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_gap[gap] = (p, r, f1)
        total += weight * f1
    return total / sum(GAP_WEIGHTS.values()), per_gap


def sweep():
    cache = {}
    for window, out_dir, *_ in WINDOWS:
        clip = json.loads((Path(out_dir) / "clip.json").read_text())["path"]
        _, _, metric = run_pipeline(clip, Path(out_dir), return_tracks=True)
        cache[window] = metric

    tune = {k: v for k, v in cache.items() if "reading" not in k}
    held = {k: v for k, v in cache.items() if "reading" in k}
    original = reid._best_match

    print("\nReachable radius = speed * gap + margin, scored with real gap "
          "weights.\nTuned on three Stoke windows, checked on the Reading "
          "match.\n")
    print(f"  {'speed':>6s} {'margin':>7s} {'tune F1':>8s} {'HELD OUT':>9s}"
          "   P/R at 0.2s      at 0.8s      at 2.5s")
    best = None
    for speed in (2.5, 4.0, 6.0, 8.0, 11.0, 15.0):
        for margin in (0.5, 1.5, 3.0):
            reid._best_match = patched_best_match(speed, margin)
            try:
                t, per = weighted_f1(tune, speed, margin)
                h, _ = weighted_f1(held, speed, margin)
            finally:
                reid._best_match = original
            if best is None or t > best[0]:
                best = (t, h, speed, margin)
            cells = "  ".join(f"{per[g][0]:.2f}/{per[g][1]:.2f}"
                              for g in (0.2, 0.8, 2.5))
            print(f"  {speed:6.1f} {margin:7.2f} {t:8.2f} {h:9.2f}   {cells}")

    print(f"\nbest on the tuning windows: speed {best[2]}, margin {best[3]} "
          f"-> {best[0]:.2f}, held out {best[1]:.2f}")


if __name__ == "__main__":
    main()
