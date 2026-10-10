"""Read every player track's shirt number with PARSeq (`src/parseq_reader.py`).

For every player track of a clip (`track_split.player_tracks`: the split
pieces when the tracks were cut), up to `PER_TRACK` crops spread over its
frames, players at least `MIN_HEIGHT_PX` tall, are read in one pass
through the video. Each crop's digits and confidence are written to
`track_numbers.json` beside the tracks; `identify_players.py
--track-numbers` votes them (`parseq_reader.vote`), joins tracks with the
same number and team, and never joins two with different numbers.

Read per track, not per identity: on Reading 0737 identities voted as a
whole read 4 of 22 named players wrong, two of them because one identity
held three people (#5, #10 and #26) and read 10 -- a number per track
keeps such tracks apart instead.

    python read_numbers_parseq.py output_stoke_1302 ... \\
        --parseq-repo parseq --weights .cache/parseq_from_reading.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src import parseq_reader as pr
from src.player_identity import crop
from src.track_split import player_tracks
from src.video_frames import frames as read_frames

PER_TRACK = 30
MIN_HEIGHT_PX = 40.0


def read_clip(model, out_dir: Path) -> dict:
    """track id -> {"labels": [...], "confs": [...]}, written to
    `track_numbers.json`."""
    info = json.loads((out_dir / "clip.json").read_text())
    t = player_tracks(out_dir)
    t = t[(t.cls == "player") & (t.crop_h >= MIN_HEIGHT_PX)]
    wanted = defaultdict(list)
    for tid, rows in t.groupby("track_id"):
        pick = rows.iloc[np.linspace(0, len(rows) - 1,
                                     min(PER_TRACK, len(rows))).astype(int)]
        for r in pick.itertuples():
            wanted[int(r.frame)].append((int(tid), r.px_raw, r.py_raw, r.crop_h))
    crops, owners = [], []
    for f, image in read_frames(info["path"], sorted(wanted)):
        for tid, x, y, h in wanted[f]:
            c = crop(image, x, y, h)
            if c is not None:
                c = pr.torso(c)
                if c.shape[0] >= 4 and c.shape[1] >= 4:
                    crops.append(c)
                    owners.append(tid)
    labels, confs = pr.read(model, crops)
    out = defaultdict(lambda: {"labels": [], "confs": []})
    for tid, g, c in zip(owners, labels, confs):
        out[str(tid)]["labels"].append(g)
        out[str(tid)]["confs"].append(round(float(c), 4))
    (out_dir / "track_numbers.json").write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", nargs="+")
    ap.add_argument("--parseq-repo", required=True)
    ap.add_argument("--weights")
    args = ap.parse_args()
    model = pr.load(args.parseq_repo, args.weights)
    for d in args.out_dirs:
        out = read_clip(model, Path(d))
        read = sum(pr.vote(v["labels"], np.array(v["confs"]))[0] is not None
                   for v in out.values())
        print(f"  {d}: {len(out)} tracks read, {read} with a confident "
              f"number", flush=True)


if __name__ == "__main__":
    main()
