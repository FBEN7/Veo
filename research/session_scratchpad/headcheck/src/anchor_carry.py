"""Carrying a pitch map from the frames that have one to the frames that do not.

An anchor is fitted from the centre circle, so it exists only where a centre
circle is visible -- between a sixth and a half of sampled frames, depending
on the footage. Every event this pipeline measures in metres needs a map on
the frame it happens in, so the rest of the clip has to inherit one.

Two ways to do that, and the difference is the whole point of this module.

**Directly.** Match the anchored frame against the target frame and push the
map through the warp between them. One fit, no accumulated error, and it
fails as soon as the two frames stop overlapping -- which for a panning
camera is a few seconds, not a few minutes. Raising the time limit does not
help, because time was never the limit.

**In steps.** Match each grid frame to the next one and compose. Consecutive
frames always overlap, so a step never fails for lack of common ground; what
it costs is error, multiplied in once per step and never cancelling.

Which is better is measured in `probe_anchor_chain.py` against anchors the
carry never sees, and this module exists so that the answer is implemented
once rather than in each detector that needs it.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from propagate_anchor import features, warp_between

# Grid step, in frames. At 25 fps this is 0.16 s: short enough that
# consecutive grid frames always overlap heavily, long enough that a clip
# costs hundreds of feature fits rather than thousands.
GRID = 4

# How many steps a map may be carried before it is refused outright. At four
# frames a step this is forty seconds in either direction, beyond which drift
# is what the probe measures it to be rather than something to hope about.
MAX_STEPS = 250


def grid_frames(total: int, anchored) -> list[int]:
    """The frames features are computed on: the grid, plus every anchor."""
    return sorted(set(range(0, int(total), GRID)) | {int(i) for i in anchored})


def walk_features(path: str, wanted) -> dict[int, tuple]:
    """ORB features at the given frames, in one sequential pass.

    Seeking to each frame costs more than decoding the whole clip once, and
    a chain wants every grid frame anyway.
    """
    need = {int(i) for i in wanted}
    found, index = {}, 0
    cap = cv2.VideoCapture(path)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index in need:
            feature = features(frame)
            if feature is not None:
                found[index] = feature
        index += 1
    cap.release()
    return found


def step_warps(found: dict[int, tuple], grid) -> dict[tuple[int, int], object]:
    """The warp from each grid frame to the next, where one can be fitted."""
    warps = {}
    for a, b in zip(grid, grid[1:]):
        warp, _ = warp_between(found.get(a), found.get(b))
        if warp is not None:
            warps[(a, b)] = warp
    return warps


def carried(anchor, warp):
    """The map a frame inherits, given the warp that reaches it.

    `warp` takes points in the anchored frame to points in this one, so the
    map that reads this frame is the anchor composed with its inverse.
    """
    if warp is None or anchor is None:
        return None
    try:
        return anchor @ np.linalg.inv(warp)
    except np.linalg.LinAlgError:
        return None


def spread(anchors: dict[int, object], warps, grid, keep=None,
           max_steps: int = MAX_STEPS) -> dict[int, object]:
    """Carry every anchor outward along the grid, nearest anchor winning.

    A breadth-first walk, so each frame is reached by the shortest chain of
    steps available rather than by whichever anchor happens to be tried
    first. `keep` is asked of each carried map and refuses the implausible
    ones, which stops a bad step propagating for another forty seconds.
    """
    where = {frame: k for k, frame in enumerate(grid)}
    maps = dict(anchors)
    steps = {frame: 0 for frame in anchors if frame in where}
    queue = deque(sorted(steps))

    while queue:
        frame = queue.popleft()
        if steps[frame] >= max_steps:
            continue
        k = where[frame]
        for neighbour, warp in ((grid[k + 1] if k + 1 < len(grid) else None,
                                 warps.get((frame, grid[k + 1])
                                           if k + 1 < len(grid) else None)),
                                (grid[k - 1] if k > 0 else None,
                                 warps.get((grid[k - 1], frame)
                                           if k > 0 else None))):
            if neighbour is None or warp is None or neighbour in maps:
                continue
            # Forward, the stored warp already runs this way; backward, it
            # runs the other way, so it is the inverse that carries the map.
            forward = neighbour > frame
            try:
                reach = warp if forward else np.linalg.inv(warp)
            except np.linalg.LinAlgError:
                continue
            inherited = carried(maps[frame], reach)
            if inherited is None:
                continue
            if keep is not None and not keep(inherited):
                continue
            maps[neighbour] = inherited
            steps[neighbour] = steps[frame] + 1
            queue.append(neighbour)
    return maps
