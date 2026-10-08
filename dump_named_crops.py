"""Crops of the players a person named, as jersey-set tracklets.

For each click from `make_number_labeller.py` (resolved to a track by
`score_player_names.resolve`), the clicked track's crops within
`WINDOW` frames of the click -- where the person could read the number --
are written in the SoccerNet jersey set's layout (`images/<key>/*.jpg`
and `<name>_gt.json`), one set per match, so the reader can be chosen and
tested on this footage with `train_jersey_reader.py --dev / --test`.
Derived from the footage: kept in `.cache`, never committed.

    python dump_named_crops.py player_names.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from score_player_names import resolve
from src.paths import CACHE_DIR
from src.player_identity import crop
from src.track_split import player_tracks
from src.video_frames import frames as read_frames

WINDOW = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--out", default=str(CACHE_DIR))
    args = ap.parse_args()
    clicks, _ = resolve(json.loads(Path(args.labels).read_text())["frames"])
    gt = defaultdict(dict)
    by_clip = defaultdict(list)
    for c in clicks:
        if c["number"] and c["number"].isdigit():
            by_clip[c["clip"]].append(c)
    for clip, cs in by_clip.items():
        match = clip.split("_")[0]
        root = Path(args.out) / f"jersey_named_{match}"
        d = Path(f"output_{clip}")
        info = json.loads((d / "clip.json").read_text())
        t = player_tracks(d)
        t = t[t.cls == "player"]
        wanted = defaultdict(list)
        for c in cs:
            key = f"{clip}_{c['frame']}_{c['track']}"
            gt[match][key] = int(c["number"])
            rows = t[(t.track_id == c["track"])
                     & ((t.frame - c["frame"]).abs() <= WINDOW)]
            for r in rows.itertuples():
                wanted[int(r.frame)].append((key, r.px_raw, r.py_raw, r.crop_h))
        for f, image in read_frames(info["path"], sorted(wanted)):
            for key, x, y, h in wanted[f]:
                c = crop(image, x, y, h)
                if c is not None:
                    out = root / "images" / key
                    out.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out / f"{f:05d}.jpg"), c)
    for match, labels in gt.items():
        root = Path(args.out) / f"jersey_named_{match}"
        (root / f"{match}_gt.json").write_text(json.dumps(labels, indent=1))
        print(f"  {root}: {len(labels)} named players")


if __name__ == "__main__":
    main()
