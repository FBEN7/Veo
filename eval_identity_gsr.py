"""Score player identities against SoccerNet game-state ground truth.

The SoccerNet game-state reconstruction set (SN-GSR-2025, public) gives,
for 30-second broadcast clips, every person's box on every frame with one
id for the whole clip, their team, role and shirt number -- exactly what
`src/player_identity.py` is meant to recover. The clips are run through the
pipeline like any other (`score_soccernet.run_pipeline`) and the
identities scored here.

Each detection of ours is matched, frame by frame, to a ground-truth box
by overlap (IoU >= 0.5, one-to-one). Then, for a set of identities:

- **people split**: how many of our identities one real person is spread
  over, frame-weighted -- 1.0 is perfect;
- **purity**: the share of an identity's frames that are its main person;
- **IDF1**: the standard identity score -- identities and people paired
  one to one to maximise shared frames, and the shared frames counted
  against all detections on both sides;
- **numbers**: identities with a number, and how many are the person's.

The scores are given for the tracker's fragments, after re-joining
(`tracks_merged.parquet`), and after `identify_players.py`.

    python eval_identity_gsr.py output_gsr_SNGS-021 ... --gsr .cache/gsr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MIN_IOU = 0.5
BOX_WIDTH = 0.45


def truth(gsr_dir: Path) -> pd.DataFrame:
    blob = json.loads((gsr_dir / "Labels-GameState.json").read_text())
    frame_of = {im["image_id"]: int(Path(im["file_name"]).stem) - 1
                for im in blob["images"]}
    rows = []
    for a in blob["annotations"]:
        if a.get("category_id") not in (1, 2) or "bbox_image" not in a:
            continue                       # players and goalkeepers
        b = a["bbox_image"]
        att = a.get("attributes") or {}
        jersey = att.get("jersey")
        rows.append({"frame": frame_of[a["image_id"]], "person": a["track_id"],
                     "x0": b["x"], "y0": b["y"], "x1": b["x"] + b["w"],
                     "y1": b["y"] + b["h"], "team": att.get("team"),
                     "jersey": int(jersey) if jersey not in (None, "")
                     and str(jersey).isdigit() else None})
    return pd.DataFrame(rows)


def ours(merged: pd.DataFrame) -> pd.DataFrame:
    p = merged[merged.cls == "player"].copy()
    x = p.px_raw if "px_raw" in p else p.px
    y = p.py_raw if "py_raw" in p else p.py
    w = BOX_WIDTH * p.crop_h
    p["x0"], p["x1"] = x - w / 2, x + w / 2
    p["y0"], p["y1"] = y - p.crop_h, y
    return p


def match(det: pd.DataFrame, gt: pd.DataFrame) -> pd.Series:
    """Ground-truth person per detection (index of det), or NaN."""
    out = pd.Series(np.nan, index=det.index)
    gt_by = {f: g for f, g in gt.groupby("frame")}
    for f, d in det.groupby("frame"):
        g = gt_by.get(f)
        if g is None:
            continue
        a = d[["x0", "y0", "x1", "y1"]].to_numpy()[:, None, :]
        b = g[["x0", "y0", "x1", "y1"]].to_numpy()[None, :, :]
        iw = np.clip(np.minimum(a[..., 2], b[..., 2])
                     - np.maximum(a[..., 0], b[..., 0]), 0, None)
        ih = np.clip(np.minimum(a[..., 3], b[..., 3])
                     - np.maximum(a[..., 1], b[..., 1]), 0, None)
        inter = iw * ih
        area = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
        iou = inter / (area(a) + area(b) - inter + 1e-9)
        r, c = linear_sum_assignment(-iou)
        for i, j in zip(r, c):
            if iou[i, j] >= MIN_IOU:
                out.loc[d.index[i]] = g.person.iloc[j]
    return out


def scores(ident: pd.Series, person: pd.Series, n_gt: int) -> dict:
    """ident: our identity per detection; person: matched truth or NaN."""
    df = pd.DataFrame({"ident": ident, "person": person})
    hit = df.dropna()
    table = hit.groupby(["ident", "person"]).size().unstack(fill_value=0)
    split = (hit.groupby("person").ident.nunique()
             * hit.groupby("person").size()).sum() / max(len(hit), 1)
    purity = table.max(axis=1).sum() / max(table.values.sum(), 1)
    r, c = linear_sum_assignment(-table.to_numpy())
    idtp = table.to_numpy()[r, c].sum()
    idf1 = 2 * idtp / (len(df) + n_gt)
    return {"identities": int(df.ident.nunique()),
            "people seen": int(hit.person.nunique()),
            "split": round(float(split), 2),
            "purity": round(float(purity), 3), "IDF1": round(float(idf1), 3),
            "detections matched": f"{len(hit)}/{len(df)}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", nargs="+")
    ap.add_argument("--gsr", default=".cache/gsr")
    args = ap.parse_args()
    for out_dir in map(Path, args.out_dirs):
        name = out_dir.name.replace("output_gsr_", "")
        gt = truth(Path(args.gsr) / name)
        merged = pd.read_parquet(out_dir / "tracks_merged.parquet")
        det = ours(merged)
        person = match(det, gt)
        print(f"\n  {name}: {gt.person.nunique()} people, {len(gt)} boxes; "
              f"{len(det)} player detections of ours")
        print(f"    re-joined tracks : {scores(det.track_id, person, len(gt))}")
        ident_file = out_dir / "identities.json"
        if ident_file.exists():
            blob = json.loads(ident_file.read_text())
            ident = det.track_id.map(lambda t: blob["tracks"].get(
                str(int(t)), {}).get("identity", f"t{t}"))
            print(f"    identities       : {scores(ident, person, len(gt))}")
            jersey = gt.groupby("person").jersey.agg(
                lambda s: s.dropna().mode().iloc[0]
                if s.notna().any() else None)
            df = pd.DataFrame({"ident": ident, "person": person}).dropna()
            main_person = df.groupby("ident").person.agg(
                lambda s: s.mode().iloc[0])
            right = wrong = 0
            for i in blob["identities"]:
                if i["number"] is None or i["key"] not in main_person.index:
                    continue
                true = jersey.get(main_person[i["key"]])
                right += true == i["number"]
                wrong += true != i["number"]
            print(f"    numbers          : {right} right, {wrong} wrong, of "
                  f"{len(blob['identities'])} identities")


if __name__ == "__main__":
    main()
