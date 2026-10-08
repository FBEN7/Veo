"""Read each identity's shirt number with PARSeq (`src/parseq_reader.py`).

For every identity of a clip (`identities.json`), up to `PER_IDENTITY`
crops spread over its tracks' frames, players at least `MIN_HEIGHT_PX`
tall, are read in one pass through the video and voted
(`parseq_reader.vote`). Written to `numbers.json` beside the identities:
identity key -> number, its share of the vote, crops used. A number counts
when its share is at least `MIN_SHARE` of `MIN_CROPS` or more crops read.

    python read_numbers_parseq.py output_stoke_1302 ... \\
        --parseq-repo parseq --weights .cache/parseq_from_reading.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src import parseq_reader as pr
from src.player_identity import crop
from src.track_split import player_tracks
from src.video_frames import frames as read_frames

PER_IDENTITY = 60
MIN_HEIGHT_PX = 40.0
MIN_SHARE = 0.6
MIN_CROPS = 2


def read_clip(model, out_dir: Path, min_conf: float = pr.MIN_CONF):
    info = json.loads((out_dir / "clip.json").read_text())
    blob = json.loads((out_dir / "identities.json").read_text())
    t = player_tracks(out_dir)
    t = t[(t.cls == "player") & (t.crop_h >= MIN_HEIGHT_PX)]
    by_track = {int(k): g for k, g in t.groupby("track_id")}
    wanted = defaultdict(list)
    for ident in blob["identities"]:
        rows = [by_track[int(k)] for k in ident["tracks"] if int(k) in by_track]
        if not rows:
            continue
        rows = pd.concat(rows).sort_values("frame")
        pick = rows.iloc[np.linspace(0, len(rows) - 1,
                                     min(PER_IDENTITY, len(rows))).astype(int)]
        for r in pick.itertuples():
            wanted[int(r.frame)].append((ident["key"], r.px_raw, r.py_raw,
                                         r.crop_h))
    crops, owners = [], []
    for f, image in read_frames(info["path"], sorted(wanted)):
        for key, x, y, h in wanted[f]:
            c = crop(image, x, y, h)
            if c is not None:
                c = pr.torso(c)
                if c.shape[0] >= 4 and c.shape[1] >= 4:
                    crops.append(c)
                    owners.append(key)
    labels, confs = pr.read(model, crops)
    owners = np.array(owners)
    out = {}
    for key in sorted(set(owners.tolist())):
        mine = owners == key
        number, share, used = pr.vote([labels[i] for i in np.where(mine)[0]],
                                      confs[mine], min_conf)
        ok = number is not None and share >= MIN_SHARE and used >= MIN_CROPS
        out[str(key)] = {"number": number if ok else None,
                         "read": number, "share": round(share, 3),
                         "used": used, "crops": int(mine.sum())}
    (out_dir / "numbers.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", nargs="+")
    ap.add_argument("--parseq-repo", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--min-conf", type=float, default=pr.MIN_CONF)
    args = ap.parse_args()
    model = pr.load(args.parseq_repo, args.weights)
    for d in args.out_dirs:
        out = read_clip(model, Path(d), args.min_conf)
        n = sum(v["number"] is not None for v in out.values())
        print(f"  {d}: {n} of {len(out)} identities numbered", flush=True)


if __name__ == "__main__":
    main()
