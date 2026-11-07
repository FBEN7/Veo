"""Cut the clicked strips (prepare.py's stage/) from the shot clips and join
v2's stream subset (stream_subset.py), into one strips dir for train_v4.py.

Each clip is decoded in order (build_ball_strips.window_frames, as the
detectors read it; the click page read the same frames by seeking, checked
equal on st2_117092_1st_f011381); backgrounds come from the clip alone, as
v3's verified window strips. The stream strips are linked, not copied.

    python cut_v4.py --stage STAGE --clips DIR [DIR ...] --stream DIR --out OUT
    (or --part I --parts N per process, then --merge --stream DIR)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from build_ball_strips import cut, window_frames  # noqa: E402
from src.ball_heatmap import pitch_box  # noqa: E402

CLICKED = {"category": "clicked", "label": "BALL", "positive_only": False, "weight": 1.0,
           "narrow": False, "coco_balls": 0, "holdout": False, "draw_w": 1.0}


def find(clips, name):
    for d in clips:
        hit = list(Path(d).rglob(name))
        if hit:
            return hit[0]
    raise SystemExit(f"{name} not found under {clips}")


def cut_part(args, stage: Path, out: Path):
    t = pd.read_parquet(stage / "targets.parquet")
    t["i"] = np.arange(len(t))
    units = sorted(set(t.unit))[args.part::args.parts]
    rows = []
    for unit in units:
        g = t[t.unit == unit]
        info = json.loads(find(args.clips, f"{unit}.json").read_text())
        mp4 = find(args.clips, f"{unit}.mp4")
        first = int(info["start_frame"])
        last = first + int(info["frames"]) - 1
        box = pitch_box((int(info["height"]), int(info["width"])), info["pitch"])
        done = cut(g.to_dict("records"), window_frames(mp4, first, first, last), first, last, box,
                   out, str(info["match"]), int(info["half"]), log_every=0)
        rows += done
        print(f"  {unit}: {len(done)} of {len(g)} strips", flush=True)
    pd.DataFrame(rows).to_parquet(out / f"clicked_part{args.part}.parquet")


def merge(args, stage: Path, out: Path):
    codes = np.load(stage / "codes.npy")
    c = pd.concat([pd.read_parquet(p) for p in sorted(out.glob("clicked_part*.parquet"))],
                  ignore_index=True).assign(**CLICKED)
    c["source"] = c.unit
    c = c.drop_duplicates("file").reset_index(drop=True)
    cc = codes[c.i.to_numpy()]
    s = pd.read_parquet(Path(args.stream) / "labels.parquet").assign(fold="stream", draw_w=1.0)
    sc = np.load(Path(args.stream) / "labels_codes.npy")
    for f in s.file:
        dst = out / f
        if not dst.exists():
            os.symlink(Path(args.stream).resolve() / f, dst)
    table = pd.concat([c[list(s.columns)], s], ignore_index=True)
    table.to_parquet(out / "labels.parquet")
    np.save(out / "labels_codes.npy", np.concatenate([cc, sc]))
    print("strips", table.groupby(["fold", "kind"]).size().to_dict(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--clips", nargs="+")
    ap.add_argument("--stream")
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="only join the parts and the stream strips")
    args = ap.parse_args()
    stage, out = Path(args.stage), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.merge:
        cut_part(args, stage, out)
    if args.merge or args.parts == 1:
        merge(args, stage, out)


if __name__ == "__main__":
    main()
