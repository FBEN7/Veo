"""Replays in broadcast footage, found by the wipe that frames them.

The labelled clips are television broadcasts, and a broadcast shows a
replay after most incidents: the same shot or goal again, often from
another camera and in slow motion. A Veo recording has none -- one camera,
live -- so a replay must not count as an event, and its frames must not
be read as play.

On all six labelled clips (EFL Championship) every replay is framed by the
same graphic, a wipe of large white discs with the EFL logo, about 1.5 s
long, before the replay and again after it. Measured on a 160 x 90 copy of
each frame, the share of near-white pixels during a wipe peaks at
0.21-0.27 in two bursts; elsewhere, the 90th percentile is 0.005-0.028 and
the highest short run 0.08. So a wipe is a run of frames above
`WHITE_SHARE`, runs closer than `MERGE_FRAMES` are one wipe, and the
frames from the start of one wipe to the end of the next are a replay. A
wipe with no partner runs the replay to the end of the clip.

Footage without these wipes -- Veo's, or another broadcaster's -- gives no
replay, which for Veo is the truth and for another broadcaster has to be
checked before it is believed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# A pixel is white when all three channels exceed this (0-255).
WHITE_LEVEL = 225
# A frame is part of a wipe when this share of it is white.
WHITE_SHARE = 0.12
# A run shorter than this is a white shirt or a flash, not a wipe.
MIN_RUN = 4
# The two bursts of one wipe are about 20 frames apart.
MERGE_FRAMES = 40

CACHE_FILE = "replays.json"


def white_shares(video_path: str):
    """Share of near-white pixels in each frame of a video."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    shares = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        shares.append(float((small.min(axis=2) > WHITE_LEVEL).mean()))
    cap.release()
    return shares


def wipes(shares):
    """(first, last) frame of each wipe."""
    high = np.asarray(shares) > WHITE_SHARE
    runs, i = [], 0
    while i < len(high):
        if high[i]:
            j = i
            while j + 1 < len(high) and high[j + 1]:
                j += 1
            if j - i + 1 >= MIN_RUN:
                runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= MERGE_FRAMES:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [tuple(r) for r in merged]


def replay_spans(shares):
    """(first, last) frame of each replay, wipes included."""
    found = wipes(shares)
    spans = []
    for k in range(0, len(found), 2):
        start = found[k][0]
        end = found[k + 1][1] if k + 1 < len(found) else len(shares) - 1
        spans.append((int(start), int(end)))
    return spans


def for_clip(out_dir: Path, info: dict):
    """The clip's replay spans, cached next to its tracks."""
    cache = Path(out_dir) / CACHE_FILE
    if cache.exists():
        return [tuple(s) for s in json.loads(cache.read_text())["replays"]]
    spans = replay_spans(white_shares(info["path"]))
    cache.write_text(json.dumps({"replays": spans}))
    return spans


def in_replay(frames, spans):
    """A boolean per frame: is it inside a replay?"""
    frames = np.asarray(frames)
    inside = np.zeros(len(frames), dtype=bool)
    for start, end in spans:
        inside |= (frames >= start) & (frames <= end)
    return inside


def selftest(verbose: bool = True) -> bool:
    """Wipes in a synthetic white-share series, paired into replays."""
    rng = np.random.default_rng(0)
    shares = list(rng.uniform(0.0, 0.02, 2250))

    def wipe(at):
        for f in range(at, at + 12):
            shares[f] = 0.22
        for f in range(at + 28, at + 36):
            shares[f] = 0.26

    wipe(1266); wipe(1421)          # a replay, as on stoke_1302
    wipe(2150)                      # one running to the end, as on stoke_4207
    for f in range(865, 870):       # a short bright moment, not a wipe
        shares[f] = 0.08
    for f in range(300, 302):       # two white frames, a flash
        shares[f] = 0.3
    got = replay_spans(shares)
    want = [(1266, 1456), (2150, 2249)]
    ok = got == want
    if verbose:
        print(f"  replays {got}  {'ok' if ok else 'WRONG, want ' + str(want)}")
    none = replay_spans(list(rng.uniform(0.0, 0.03, 500)))
    ok2 = none == []
    if verbose:
        print(f"  footage without wipes: {none}  {'ok' if ok2 else 'WRONG'}")
    return ok and ok2


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
