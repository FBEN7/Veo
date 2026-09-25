"""Carrying an anchor step by step instead of in one jump.

Out of play, throw-ins, corners and goal kicks all fail for one reason, and
it is not geometry: on the three labelled crossings in this footage the ball
is detected 43, 74 and 51 times within two seconds, and placed on the pitch
zero times. There is no pitch map on those frames. Anchors come from the
centre circle, the centre circle is at midfield, and the ball goes out at the
edges, so the frames that have an anchor and the frames that contain a
crossing are anti-correlated.

Raising the borrow reach from 6 s to 16 s found one of the three. Raising it
further to 36 s found nothing more, which is the interesting part: if
distance were the limit, 36 s would beat 16 s. It does not, so something
else stops the carry.

## What stops it

The carry matches the anchored frame **directly** against the target frame,
one ORB fit across the whole gap. A camera that has panned across the pitch
in the intervening twenty seconds shows a different part of the ground, the
two frames share few features, and the fit fails however patient the reach
is. Distance in seconds is a proxy; the real limit is overlap.

Consecutive frames always overlap. So the alternative is to carry the map in
steps -- grid frame to grid frame, four frames at a time, composing the
warps -- and to accept whatever error accumulates along the way.

## Which is the question, because chaining has an obvious cost

Every step multiplies in its own error, and they do not cancel. Twenty
seconds at four frames a step is 125 multiplications, and if each one is
worth a millimetre they are worth 125 of them, not one.

This measures that, and it can be measured honestly because the anchors
themselves are the ruler. Take two frames that both carry their own anchor,
derived independently from their own centre circles. Carry the first one's
map to the second, both ways, and compare each against the anchor that is
already there. Neither number comes from the thing being tested.

    python probe_anchor_chain.py [--frames 80] [--clip w3]
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from check_anchor_consistency import SAMPLE_COLS, SAMPLE_ROWS
from detect_shots import ANCHOR_GRID
from fit_pitch_anchor import anchored_frames, plausible_anchor
from probe_pitch_lines import CLIPS
from propagate_anchor import features, warp_between

# Gaps to report, in frames. At 25 fps: 2 s, 8 s, 20 s, 40 s, beyond.
# How many multiplications a map may pass through before it is refused.
MAX_HOPS = 8

GAP_BANDS = ((0, 50), (50, 200), (200, 500), (500, 1000), (1000, 10 ** 9))


def sample_points(width, height):
    """Image points on the lower part of the frame, where the ground is."""
    return np.array([[c * width for c in SAMPLE_COLS for _ in SAMPLE_ROWS],
                     [r * height for _ in SAMPLE_COLS for r in SAMPLE_ROWS],
                     [1.0] * (len(SAMPLE_COLS) * len(SAMPLE_ROWS))])


def gap_metres(carried, truth, points):
    """Median distance, in metres, between two maps' idea of the ground."""
    if carried is None:
        return float("nan")
    here, there = truth @ points, carried @ points
    if np.any(np.abs(here[2]) < 1e-9) or np.any(np.abs(there[2]) < 1e-9):
        return float("nan")
    here, there = here[:2] / here[2], there[:2] / there[2]
    gaps = np.hypot(*(here - there))
    gaps = gaps[np.isfinite(gaps)]
    return float(np.median(gaps)) if gaps.size else float("nan")


def walk_features(path: str, indices):
    """ORB features at the given frames, in one sequential pass.

    Seeking to each frame instead costs more than decoding the clip, and a
    chain wants every grid frame, so the whole clip is decoded once.
    """
    wanted = set(int(i) for i in indices)
    out = {}
    cap = cv2.VideoCapture(path)
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            found = features(frame)
            if found is not None:
                out[index] = found
        index += 1
    cap.release()
    return out


def step_warps(feature_by_frame, grid):
    """The warp from each grid frame to the next, where one can be fitted."""
    warps = {}
    for a, b in zip(grid, grid[1:]):
        warp, _ = warp_between(feature_by_frame.get(a),
                               feature_by_frame.get(b))
        if warp is not None:
            warps[(a, b)] = warp
    return warps


def chained_warp(warps, grid, a: int, b: int):
    """Points in frame `a` to points in frame `b`, composing single steps."""
    if a == b:
        return np.eye(3)
    forward = a < b
    lo, hi = (a, b) if forward else (b, a)
    composite = np.eye(3)
    start = grid.index(lo)
    for k in range(start, grid.index(hi)):
        step = warps.get((grid[k], grid[k + 1]))
        if step is None:
            return None                      # the chain is broken here
        composite = step @ composite
    if forward:
        return composite
    try:
        return np.linalg.inv(composite)
    except np.linalg.LinAlgError:
        return None


def multi_scale_edges(found, scales, total, anchors):
    """Warps between frames at several spacings, over one shared graph.

    A coarse step drifts less because there are fewer of them, and fails
    more often because two frames two seconds apart share less. Holding both
    lets a walk take coarse steps where they fit and drop to fine ones only
    to bridge a break.
    """
    edges, grids = {}, {}
    for step in scales:
        grid = sorted(set(range(0, total, step)) | set(anchors))
        grid = [i for i in grid if i in found]
        grids[step] = grid
        for a, b in zip(grid, grid[1:]):
            if (a, b) in edges:
                continue
            warp, _ = warp_between(found.get(a), found.get(b))
            if warp is not None:
                edges[(a, b)] = warp
    neighbours = {}
    for (a, b), warp in edges.items():
        neighbours.setdefault(a, []).append((b, warp, True))
        neighbours.setdefault(b, []).append((a, warp, False))
    return neighbours, grids


def walk_from(neighbours, source, max_hops):
    """Accumulated warp from one frame to every frame reachable in few hops.

    Breadth-first, so each frame is reached by the fewest multiplications
    available rather than by whichever path is tried first -- and since a
    coarse step covers more ground per hop, fewest hops naturally prefers
    the coarse ones.
    """
    reached = {source: (np.eye(3), 0)}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        warp_so_far, hops = reached[node]
        if hops >= max_hops:
            continue
        for other, warp, forward in neighbours.get(node, ()):
            if other in reached:
                continue
            try:
                step = warp if forward else np.linalg.inv(warp)
            except np.linalg.LinAlgError:
                continue
            reached[other] = (step @ warp_so_far, hops + 1)
            queue.append(other)
    return reached


def carry(anchor, warp):
    """The map a frame inherits, given the warp that reaches it."""
    if warp is None:
        return None
    try:
        return anchor @ np.linalg.inv(warp)
    except np.linalg.LinAlgError:
        return None


def band_of(gap: int) -> int:
    for k, (lo, hi) in enumerate(GAP_BANDS):
        if lo <= gap < hi:
            return k
    return len(GAP_BANDS) - 1


def probe_clip(name: str, out_dir: str, n_frames: int, rng, steps):
    """Direct and chained carries between anchored frames, per step size.

    One anchor pass and one decode serve every step size: fitting the
    anchors is the expensive part and the features for a coarser grid are a
    subset of those for a finer one, up to the frames only the coarse grid
    lands on.
    """
    path = Path(out_dir)
    if not (path / "clip.json").exists():
        return None
    info = json.loads((path / "clip.json").read_text())
    _, owned = anchored_frames(path, n_frames, rng, use_penalty_arc=False)
    if len(owned) < 2:
        return None

    total = int(info["n_frames"])
    anchors = {int(i): h for i, h in owned}
    grids = {step: sorted(set(range(0, total, step)) | set(anchors))
             for step in steps}
    found = walk_features(info["path"],
                          sorted({i for g in grids.values() for i in g}))

    points = sample_points(info["width"], info["height"])
    usable = [i for i in sorted(anchors) if i in found]
    rows, fitted = [], {}
    for step in steps:
        grid = [i for i in grids[step] if i in found]
        grids[step] = grid
        warps = step_warps(found, grid)
        fitted[step] = (len(warps), max(len(grid) - 1, 1))
        for a in usable:
            for b in usable:
                if a >= b:
                    continue
                stepped = carry(anchors[a],
                                chained_warp(warps, grid, a, b))
                rows.append({
                    "step": step, "gap": b - a, "kind": "chain",
                    "fitted": stepped is not None,
                    "metres": gap_metres(stepped, anchors[b], points),
                })

    neighbours, _ = multi_scale_edges(found, steps, total, anchors)
    for a in usable:
        reached = walk_from(neighbours, a, MAX_HOPS)
        for b in usable:
            if b <= a or b not in reached:
                continue
            warp, hops = reached[b]
            stepped = carry(anchors[a], warp)
            rows.append({"step": -1, "gap": b - a, "kind": "multi",
                         "hops": hops, "fitted": stepped is not None,
                         "metres": gap_metres(stepped, anchors[b], points)})

    for a in usable:
        for b in usable:
            if a >= b:
                continue
            direct, _ = warp_between(found[a], found[b])
            straight = carry(anchors[a], direct)
            rows.append({"step": 0, "gap": b - a, "kind": "direct",
                         "fitted": straight is not None,
                         "metres": gap_metres(straight, anchors[b], points)})

    return {"name": name, "rows": rows, "anchors": len(usable),
            "fitted": fitted}


def report(results, steps):
    print("\nCarrying one anchor to a frame that has its own, so the error is "
          "measured\nagainst something the carry never saw.\n")

    pooled = [row for result in results for row in result["rows"]]
    columns = ([("direct", 0)] + [(f"chain/{s}", s) for s in steps]
               + [("multi", -1)])
    head = "".join(f"{label:>16s}" for label, _ in columns)
    print(f"  {'gap':>10s} {'pairs':>6s}{head}")

    for k, (lo, hi) in enumerate(GAP_BANDS):
        band = [row for row in pooled if band_of(row["gap"]) == k]
        if not band:
            continue
        seconds = (f"{lo / 25.0:.0f}-{hi / 25.0:.0f}s" if hi < 10 ** 9
                   else f"{lo / 25.0:.0f}s+")
        pairs = sum(1 for row in band if row["kind"] == "direct")
        cells = ""
        for label, step in columns:
            here = [row for row in band
                    if (row["kind"] == "direct" if step == 0
                        else row["step"] == step)]
            if step == -1:
                # A multi-scale walk that does not reach a frame leaves no
                # row at all, so its coverage is against every pair, not
                # against the rows it produced.
                here = here + [{"fitted": False, "metres": float("nan")}
                               for _ in range(pairs - len(here))]
            if not here:
                cells += f"{'-':>16s}"
                continue
            good = [row["metres"] for row in here
                    if row["fitted"] and np.isfinite(row["metres"])]
            rate = sum(1 for row in here if row["fitted"]) / len(here)
            median = f"{np.median(good):.1f}m" if good else "-"
            cells += f"{f'{rate:.0%} {median}':>16s}"
        print(f"  {seconds:>10s} {pairs:6d}{cells}")

    print("\n  Each cell is how often a warp came back at all, then what it "
          "cost in\n  metres when it did. A chain that fits everywhere and "
          "drifts ten metres is\n  no use, and neither is a direct fit that "
          "is perfect and rare.")

    for result in results:
        parts = ", ".join(f"{step}: {good}/{total}"
                          for step, (good, total) in result["fitted"].items())
        print(f"\n  {result['name']}: {result['anchors']} anchored frames; "
              f"single steps fitted -- {parts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--steps", default="4,12,25",
                    help="grid steps to chain over, in frames")
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    rng = np.random.default_rng(0)
    results = []
    for name, out_dir in CLIPS:
        if args.clip and args.clip not in out_dir:
            continue
        result = probe_clip(name, out_dir, args.frames, rng, steps)
        if result is not None:
            results.append(result)
            print(f"  {name}: {result['anchors']} anchors", flush=True)
    if results:
        report(results, steps)


if __name__ == "__main__":
    main()
