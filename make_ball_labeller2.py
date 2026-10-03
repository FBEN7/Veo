"""A second ball-clicking page: denser around events, and where the ball is lost.

The first round (`make_ball_labeller.py`) clicked every 5th frame around
each labelled shot, goal and out. It showed the limit is the detector: it
proposes the ball on 63 of 138 clicked frames, and never twice in flight
between a strike and the goal line. Training a ball detector on this
footage needs many more balls in flight than 138, and interpolating
between the first round's clicks is not good enough to make them -- held
out, a click sits a median 9 px and up to 72 px from the midpoint of its
neighbours ten frames apart, more than a ball's width.

So this page asks for two kinds of frame:

- **around events**, every `STEP` frames from `BEFORE` seconds before to
  `AFTER` after each labelled event, skipping frames the first round has.
  Where first-round clicks lie on both sides within 5 frames, a guess is
  drawn there: Enter accepts it, a click corrects it. These frames are
  shown as a crop around the guess at the footage's own scale, which makes
  a small ball easier to see.
- **where the ball is lost**: `LOST_PER_CLIP` frames per clip away from the
  events where the current ball path has no point, full frame.

Same output format as the first round, normalised to the full frame.
Not published, for the same reason: the frames are under the data
agreement, so the page goes to the person who has the data, as a file.

    python make_ball_labeller2.py --first data/<ball_labels.json>
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from make_ball_labeller import CLIPS, PAGE, labelled_events
from src.paths import DATA_DIR

BEFORE, AFTER = 0.4, 1.6
STEP = 2
# A guess only between two first-round clicks at most this many frames away.
GUESS_REACH = 5
CROP_W, CROP_H = 640, 360
LOST_PER_CLIP = 12
# Lost frames are kept this far, in seconds, from any labelled event.
LOST_CLEARANCE_S = 3.0
# Crops are small and keep more quality; full frames are what fills the
# 29 MB upload limit.
JPEG_QUALITY = 80
FULL_FRAME_QUALITY = 60


def page_html(frames) -> str:
    """The first round's page, with crops and an Enter-to-accept guess."""
    html = PAGE.replace('"ball-click-v1"', '"ball-click-v2"')
    html = html.replace(
        '<div id="loupe"></div>',
        '<div id="loupe"></div><div id="guess"></div>')
    # Crops are 640 px across; shown at the same width as a full frame,
    # which enlarges them.
    html = html.replace("max-width:1000px}", "width:min(1000px,100%)}", 1)
    html = html.replace(
        " #ask{",
        " #guess{position:absolute;width:26px;height:26px;margin:-13px 0 0 "
        "-13px;border:2px dashed #ffea00;border-radius:50%;pointer-events:"
        "none;display:none}\n #ask{")
    html = html.replace(
        "<kbd>U</kbd> goes back one frame.",
        "<kbd>U</kbd> goes back one frame. Where a dashed yellow circle "
        "is drawn, it is a guess: <kbd>Enter</kbd> if it is on the ball, "
        "otherwise click the ball.")
    # Clicks on a crop are mapped back to the full frame.
    html = html.replace(
        "  record({ball:[+u.toFixed(5), +v.toFixed(5)]});",
        "  const f = FRAMES[at];\n"
        "  record({ball:[+((f.ox + u*f.cw)/f.fw).toFixed(5),"
        " +((f.oy + v*f.ch)/f.fh).toFixed(5)]});")
    html = html.replace(
        "  paint();\n}",
        "  paint();\n"
        "  const g = document.getElementById('guess');\n"
        "  if (f.guess){ g.style.display='block';\n"
        "    g.style.left = (100*(f.guess[0]-f.ox)/f.cw) + '%';\n"
        "    g.style.top = (100*(f.guess[1]-f.oy)/f.ch) + '%'; }\n"
        "  else g.style.display='none';\n}")
    html = html.replace(
        '  else if (k==="u") document.getElementById("undo").click();',
        '  else if (k==="u") document.getElementById("undo").click();\n'
        '  else if (k==="enter" && at < FRAMES.length && FRAMES[at].guess){\n'
        '    const f = FRAMES[at];\n'
        '    record({ball:[+(f.guess[0]/f.fw).toFixed(5),'
        ' +(f.guess[1]/f.fh).toFixed(5)], accepted_guess:true}); }')
    return html.replace("__DATA__", json.dumps(frames))


def first_round(path: Path):
    """{(clip, frame): (x, y) px or None} from a first-round label file."""
    out = {}
    for r in json.loads(path.read_text())["frames"]:
        if "unlabelled" in r:
            continue
        out[(r["clip"], int(r["frame"]))] = (
            None if not r.get("ball") else
            (r["ball"][0] * r["width"], r["ball"][1] * r["height"]))
    return out


def guess_at(clip, frame, first):
    """Linear between first-round clicks on both sides, or None."""
    before = after = None
    for d in range(1, GUESS_REACH + 1):
        if before is None and first.get((clip, frame - d)):
            before = (frame - d, first[(clip, frame - d)])
        if after is None and first.get((clip, frame + d)):
            after = (frame + d, first[(clip, frame + d)])
    if before is None or after is None:
        return None
    w = (frame - before[0]) / (after[0] - before[0])
    return tuple((1 - w) * np.array(before[1]) + w * np.array(after[1]))


def lost_frames(clip, info, events, rng):
    """Frames away from the events where the ball path has no point."""
    from src import ball_path

    tracks = pd.read_parquet(Path(f"output_{clip}") / "tracks.parquet")
    have = set(ball_path.track(tracks, float(info["fps"])).frame.astype(int))
    fps = float(info["fps"])
    pool = [f for f in range(0, int(info["n_frames"]), 5)
            if f not in have and all(abs(f / fps - t) > LOST_CLEARANCE_S
                                     for t, _ in events)]
    if len(pool) <= LOST_PER_CLIP:
        return pool
    return sorted(rng.choice(pool, LOST_PER_CLIP, replace=False).tolist())


def encode(image, quality=JPEG_QUALITY):
    ok, buf = cv2.imencode(".jpg", image,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii") if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True,
                    help="ball_labels.json from the first round")
    ap.add_argument("--out", default="ball_labeller2.html")
    args = ap.parse_args()

    first = first_round(Path(args.first))
    rng = np.random.default_rng(0)
    frames = []
    for clip in CLIPS:
        labels = sorted(DATA_DIR.glob(f"*-{clip}_2shots.txt"))
        info_path = Path(f"output_{clip}") / "clip.json"
        if not labels or not info_path.exists():
            print(f"  {clip}: no labels or no clip.json, skipped")
            continue
        info = json.loads(info_path.read_text())
        fps, fw, fh = float(info["fps"]), int(info["width"]), int(info["height"])
        events = labelled_events(labels[0])
        wanted = {}
        for when, kind in events:
            first_f = max(0, int((when - BEFORE) * fps))
            last_f = min(int(info["n_frames"]) - 1, int((when + AFTER) * fps))
            for f in range(first_f, last_f + 1, STEP):
                if (clip, f) not in first:
                    wanted.setdefault(
                        f, f"{kind} at {when // 60}:{when % 60:02d}")
        lost = lost_frames(clip, info, events, rng)
        for f in lost:
            wanted.setdefault(f, "ball lost by the tracker")
        cap = cv2.VideoCapture(info["path"])
        n_guess = 0
        for f in sorted(wanted):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, image = cap.read()
            if not ok:
                continue
            guess = guess_at(clip, f, first)
            if guess is not None:
                n_guess += 1
                ox = int(np.clip(guess[0] - CROP_W / 2, 0, fw - CROP_W))
                oy = int(np.clip(guess[1] - CROP_H / 2, 0, fh - CROP_H))
                cw, ch = CROP_W, CROP_H
                shown = image[oy:oy + ch, ox:ox + cw]
            else:
                ox = oy = 0
                cw, ch = fw, fh
                shown = image
            jpeg = encode(shown, JPEG_QUALITY if guess is not None
                          else FULL_FRAME_QUALITY)
            if jpeg:
                frames.append({"id": f"{clip}:{f}", "clip": clip, "frame": f,
                               "near": wanted[f], "fw": fw, "fh": fh,
                               "ox": ox, "oy": oy, "cw": cw, "ch": ch,
                               "guess": ([round(guess[0], 1), round(guess[1], 1)]
                                         if guess is not None else None),
                               "jpeg": jpeg})
        cap.release()
        print(f"  {clip}: {len(wanted) - len(lost)} around events "
              f"({n_guess} with a guess), {len(lost)} where the ball is lost")

    target = Path(args.out)
    target.write_text(page_html(frames), encoding="utf-8")
    size = target.stat().st_size / 1e6
    print(f"\n  {len(frames)} frames -> {target} ({size:.1f} MB)")
    if size > 29:
        print("  TOO BIG to upload")


if __name__ == "__main__":
    main()
