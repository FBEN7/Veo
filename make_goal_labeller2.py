"""A second goal-labelling page: the frames the goal detector is unsure of.

Shots are now missed upstream of the shot reader. Without a camera pose no
ball can be placed on the pitch, and poses come from the goal detector
(`train_goal_detector.py`), trained on 150 hand-labelled frames. On
stoke_7001 it scores the goal through both labelled shots at 0.09-0.25,
under the 0.15 cut-off, though a pose solved from each of those boxes
fits it to under a pixel -- the goal is seen, the confidence is not there.
More labelled goals at these angles are what it lacks.

Frames come from the six labelled clips, outside broadcast replays
(`src/replays.py`), every `STEP` frames, in three groups per clip:

- **unsure**: the detector's best box scores between `UNSURE` limits --
  most of these are goals it does not trust;
- **around events**: within `EVENT_REACH_S` of a labelled shot, goal or
  out, where the pose matters most;
- **nothing found**: no box above the lower limit, as negatives (or goals
  it misses outright).

The page is the first round's (`make_goal_labeller.py`), with its own
storage key: two clicks, one corner of the goal and then the opposite one,
as before, or N for no goal. Not published: the frames are under the data
agreement.

    python make_goal_labeller2.py --goal-weights <best.pt>
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

from make_ball_labeller import labelled_events
from make_goal_labeller import JPEG_QUALITY, PAGE, SHOW_H, SHOW_W
from src import replays
from src.paths import DATA_DIR

CLIPS = ["stoke_1302", "stoke_4207", "stoke_7001",
         "reading_0737", "reading_1155", "reading_2519"]
STEP = 5
UNSURE = (0.03, 0.25)
EVENT_REACH_S = 2.0
PER_CLIP = {"unsure": 18, "around events": 14, "nothing found": 8}
# Picked frames at least this far apart, so they are not the same picture.
MIN_GAP = 8


def spread(frames, count, rng, taken):
    """Up to `count` of `frames`, at least `MIN_GAP` from each other."""
    out = []
    for f in rng.permutation(frames):
        f = int(f)
        if all(abs(f - g) >= MIN_GAP for g in list(taken) + out):
            out.append(f)
        if len(out) == count:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal-weights", required=True)
    ap.add_argument("--out", default="goal_labeller2.html")
    args = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.goal_weights)
    rng = np.random.default_rng(0)
    frames = []
    for clip in CLIPS:
        out_dir = Path(f"output_{clip}")
        info = json.loads((out_dir / "clip.json").read_text())
        spans = replays.for_clip(out_dir, info)
        fps = float(info["fps"])
        labels = sorted(DATA_DIR.glob(f"*-{clip}_2shots.txt"))
        events = [t for t, _ in labelled_events(labels[0])] if labels else []
        cap = cv2.VideoCapture(info["path"])
        best = {}
        for f in range(0, int(info["n_frames"]), STEP):
            if replays.in_replay([f], spans)[0]:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, image = cap.read()
            if not ok:
                continue
            result = model.predict(image, conf=0.01, verbose=False)[0]
            best[f] = (float(result.boxes.conf.max()) if len(result.boxes)
                       else 0.0)
        groups = {
            "unsure": [f for f, c in best.items()
                       if UNSURE[0] <= c < UNSURE[1]],
            "around events": [f for f in best if any(
                abs(f / fps - t) <= EVENT_REACH_S for t in events)],
            "nothing found": [f for f, c in best.items() if c < UNSURE[0]],
        }
        picked = {}
        for name in ("around events", "unsure", "nothing found"):
            for f in spread(groups[name], PER_CLIP[name], rng, picked):
                picked[f] = name
        for f in sorted(picked):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, image = cap.read()
            if not ok:
                continue
            small = cv2.resize(image, (SHOW_W, SHOW_H))
            ok, buf = cv2.imencode(".jpg", small,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                frames.append({"id": f"{clip}:{f}", "clip": clip, "frame": f,
                               "group": picked[f],
                               "jpeg": base64.b64encode(buf).decode("ascii")})
        cap.release()
        counts = {g: sum(1 for v in picked.values() if v == g)
                  for g in PER_CLIP}
        print(f"  {clip}: " + ", ".join(f"{v} {k}" for k, v in counts.items()),
              flush=True)

    # The first round's boxes turned out to run from the top of the left post
    # to the base of the right one (`goal_pose.BOX_FROM`, `BOX_TO`); asked
    # for in so many words this time, so the two rounds agree.
    hint = ('"Click the <b>top of the left post</b>, then the <b>bottom of '
            'the right post</b>. "\n    + "If only part of the goal is in '
            'the picture, click at the edge of the picture where it is cut. "'
            '\n    + "If no goal is visible press <kbd>N</kbd>."')
    old_hint = ('"Click <b>one corner</b> of the goal, then the <b>opposite "\n'
                '    + "corner</b>. Include the posts and crossbar, not the net '
                'behind. "\n    + "If no goal is visible press <kbd>N</kbd>."')
    assert old_hint in PAGE
    page = (PAGE.replace(old_hint, hint)
                .replace('"goal-labels-v1"', '"goal-labels-v2"')
                .replace("__DATA__", json.dumps(frames))
                .replace("__W__", str(SHOW_W)).replace("__H__", str(SHOW_H)))
    target = Path(args.out)
    target.write_text(page, encoding="utf-8")
    size = target.stat().st_size / 1e6
    print(f"\n  {len(frames)} frames -> {target} ({size:.1f} MB)")
    if size > 29:
        print("  TOO BIG to upload")


if __name__ == "__main__":
    main()
