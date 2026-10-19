"""How far an anchor may be carried, and what each distance buys.

A frame with no centre circle inherits its pitch map from the nearest frame
that has one. How far that borrow may reach was raised from 150 frames to
400 on the strength of a sweep whose numbers do not reproduce: it reported
993 of 1811 ball positions placed on the Stoke clip at reach 400, and the
shipping detector reports 749 and 751 on two separate runs of the same
setting. The Reading clip agrees with the sweep exactly, which makes it
worse rather than better -- a difference that appears on one clip and not
the other is not a difference in the reach.

That sweep was written as a shell heredoc and is gone, so it cannot be
audited. This is the same measurement written down, which is the whole
reason for it being a file.

## One pass answers every reach

The borrow takes the nearest anchored frame within the reach. So the nearest
anchored frame *overall* decides every reach at once: if it lies 300 frames
away, this frame is unplaced at reach 150 and placed at reach 400, using
that same source and that same warp. Computing the nearest source once and
comparing its distance against each candidate reach gives the whole curve
for the price of one, and -- more to the point -- gives it from **one set of
anchors**, so the reaches are not being compared across different RANSAC
draws.

    python probe_borrow_reach.py [--frames 80] [--clip w3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import detect_ball_events as ball_events
import detect_shots
from detect_shots import ANCHOR_GRID
from fit_pitch_anchor import anchored_frames, plausible_anchor
from probe_pitch_lines import CLIPS
from propagate_anchor import frame_to_frame

REACHES = (150, 400, 900, 10 ** 6)


def nearest_carries(out_dir: Path, info, frames_wanted, rng, n_frames):
    """For each wanted frame, the nearest anchor's map and how far it came.

    Returns {frame: (map, distance_in_frames)}, plus the anchors themselves
    at distance zero.
    """
    _, owned = anchored_frames(out_dir, n_frames, rng, use_penalty_arc=False)
    if not owned:
        return {}, 0
    carries = {int(i): (h, 0) for i, h in owned}

    cap = cv2.VideoCapture(info["path"])
    sources = {}
    for index, _ in owned:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            sources[int(index)] = frame

    wanted = sorted({int(f) - int(f) % ANCHOR_GRID for f in frames_wanted})
    for index in wanted:
        if index in carries:
            continue
        near = min(sources, key=lambda i: abs(i - index), default=None)
        if near is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        warp, _ = frame_to_frame(sources[near], frame)
        if warp is None:
            continue
        try:
            carried = carries[near][0] @ np.linalg.inv(warp)
        except np.linalg.LinAlgError:
            continue
        if plausible_anchor(carried, info):
            carries[index] = (carried, abs(index - near))
    cap.release()
    return carries, len(owned)


def maps_within(carries, reach: int):
    """The pitch maps a given reach would allow."""
    return {frame: carried for frame, (carried, distance) in carries.items()
            if distance <= reach}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--clip", default=None)
    args = ap.parse_args()

    print("Ball positions placed on the pitch, and labelled crossings seen, "
          "against\nhow far an anchor may be carried. One set of anchors per "
          "clip, so the\nreaches differ only in the reach.\n")
    print(f"  {'clip':>16s} {'anchors':>8s} {'reach':>8s} {'placed':>12s} "
          f"{'crossings':>10s} {'false':>6s}")

    rng = np.random.default_rng(0)
    for name, out_dir in CLIPS:
        if args.clip and args.clip not in out_dir:
            continue
        path = Path(out_dir)
        if not (path / "clip.json").exists():
            continue
        info = json.loads((path / "clip.json").read_text())
        ball = detect_shots.ball_track(path)
        if ball.empty:
            continue
        carries, n_anchors = nearest_carries(path, info, ball.frame.tolist(),
                                             rng, args.frames)
        source = ball_events.CLIP_SOURCES.get(out_dir)
        truth = []
        if source is not None:
            truth = ball_events.labelled("OUT", source[0], source[1],
                                         info["n_frames"] / info["fps"])

        for reach in REACHES:
            maps = maps_within(carries, reach)
            events, placed = ball_events.find_ball_events(ball, maps,
                                                          info["fps"])
            outs = [e for e in events if e["event_type"] == "out_of_play"]
            matched = sum(1 for t in truth
                          if any(abs(e["time_s"] - t)
                                 <= ball_events.TOLERANCE_S for e in outs))
            false = sum(1 for e in outs
                        if not any(abs(e["time_s"] - t)
                                   <= ball_events.TOLERANCE_S for t in truth))
            label = "any" if reach > 10 ** 5 else str(reach)
            seen = (f"{matched}/{len(truth)}" if truth else "-")
            print(f"  {name[-16:]:>16s} {n_anchors:8d} {label:>8s} "
                  f"{f'{placed}/{len(ball)}':>12s} {seen:>10s} {false:6d}",
                  flush=True)

    print("\n  'placed' counts ball positions with a pitch map under them; "
          "'crossings'\n  is how many of that clip's labelled OUT events "
          "were seen, at the +/-2 s\n  tolerance used everywhere else here.")


if __name__ == "__main__":
    main()
