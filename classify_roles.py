"""Player, goalkeeper or referee, for every track of a clip.

Applies `train_role_classifier.py`'s model to crops spread over each
track's frames and writes `roles.json` beside the tracks: per track, the
role with the most probability over its crops, and the probabilities.
`src/touch_events.py` then never gives the ball to a referee.

With `--gsr`, also scores the roles against SoccerNet game-state ground
truth for clips run through the pipeline (`output_gsr_<clip>`).

    python classify_roles.py output_stoke_7001 ... \\
        --model .cache/role_classifier.pt [--gsr .cache/gsr]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_role_classifier import NAMES, build, vote

PER_TRACK = 20
MIN_HEIGHT_PX = 30.0


def crops_by_track(video_path, merged):
    from src.player_identity import crop
    from src.video_frames import frames as read_frames

    players = merged[(merged.cls == "player")
                     & (merged.crop_h >= MIN_HEIGHT_PX)]
    x = "px_raw" if "px_raw" in players else "px"
    y = "py_raw" if "py_raw" in players else "py"
    wanted = {}
    for tid, rows in players.groupby("track_id"):
        pick = rows.iloc[np.linspace(0, len(rows) - 1,
                                     min(PER_TRACK, len(rows))).astype(int)]
        for r in pick.itertuples():
            wanted.setdefault(int(r.frame), []).append(
                (int(tid), float(getattr(r, x)), float(getattr(r, y)),
                 float(r.crop_h)))
    out = {}
    for index, frame in read_frames(video_path, wanted):
        for tid, px, py, h in wanted[index]:
            c = crop(frame, px, py, h)
            if c is not None:
                out.setdefault(tid, []).append(c.copy())
    return out


def classify(out_dir: Path, model) -> dict:
    info = json.loads((out_dir / "clip.json").read_text())
    merged = pd.read_parquet(out_dir / "tracks_merged.parquet")
    roles = {}
    for tid, crops in crops_by_track(info["path"], merged).items():
        p = vote(model, crops)
        roles[str(tid)] = {"role": NAMES[int(np.argmax(p))],
                           "p": [round(float(v), 3) for v in p]}
    (out_dir / "roles.json").write_text(json.dumps(roles, indent=1))
    return roles


def score_gsr(out_dir: Path, roles: dict, gsr: Path):
    """Our tracks' roles against the game-state roles of the people they
    follow, frame-weighted."""
    from eval_identity_gsr import match, ours

    name = out_dir.name.replace("output_gsr_", "")
    blob = json.loads((gsr / name / "Labels-GameState.json").read_text())
    frame_of = {im["image_id"]: int(Path(im["file_name"]).stem) - 1
                for im in blob["images"]}
    rows = []
    for a in blob["annotations"]:
        if a.get("category_id") not in (1, 2, 3) or "bbox_image" not in a:
            continue
        b = a["bbox_image"]
        rows.append({"frame": frame_of[a["image_id"]], "person": a["track_id"],
                     "x0": b["x"], "y0": b["y"], "x1": b["x"] + b["w"],
                     "y1": b["y"] + b["h"],
                     "role": NAMES[{1: 0, 2: 1, 3: 2}[a["category_id"]]]})
    gt = pd.DataFrame(rows)
    det = ours(pd.read_parquet(out_dir / "tracks_merged.parquet"))
    person = match(det, gt)
    role_of = gt.groupby("person").role.first()
    true = person.map(role_of)
    pred = det.track_id.map(lambda t: roles.get(str(int(t)), {}).get("role"))
    df = pd.DataFrame({"true": true, "pred": pred}).dropna()
    return pd.crosstab(df.true, df.pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gsr")
    args = ap.parse_args()
    import torch

    model = build()
    model.load_state_dict(torch.load(args.model))
    model.eval()
    total = None
    for out_dir in map(Path, args.out_dirs):
        roles = classify(out_dir, model)
        counts = pd.Series([r["role"] for r in roles.values()]).value_counts()
        print(f"  {out_dir.name}: " + ", ".join(f"{n} {k}"
                                                for k, n in counts.items()))
        if args.gsr:
            t = score_gsr(out_dir, roles, Path(args.gsr))
            total = t if total is None else total.add(t, fill_value=0)
    if total is not None:
        total = total.fillna(0).astype(int)
        print("\n  detections, true role (rows) by our track's role (columns):")
        print(total.to_string())


if __name__ == "__main__":
    main()
