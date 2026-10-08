"""Run the pipeline on a SoccerTrack v2 window and score it against the
window's ground truth (`fetch_soccertrack_v2.py` cuts the window).

The ground truth gives, per frame, every player's and goalkeeper's image
box, persistent id and side -- but no referees, and boxes that are all
42-43 px tall and usually wider than the player (median ~45 px against
~18 px detected), placed within ~10 px of the feet. So two matchings are
reported, both one-to-one per frame:

- **IoU >= 0.5** against the boxes as released (`eval_identity_gsr.match`):
  the usual criterion, but a correct detection of a thin player in a wide
  box cannot reach it;
- **feet**: the detection's bottom centre within `FOOT_PX` of the box's
  (one player-box height), which is what the boxes can support.

With `--pitch-margin`, the window's pitch outline (from the release's
pitch lines, stored by `fetch_soccertrack_v2.py`) is given to the
pipeline, which drops player detections more than that many pixels outside
it before teams are assigned; the tracker's fragments are scored after the
same cut.

For each: detection recall and precision (a referee detected counts
against precision: the truth has none), then identities scored as in
`eval_identity_gsr.py` (people split, purity, IDF1) for the tracker's
fragments, the re-joined tracks and, where `identify_players.py` has run,
its identities; and the team of each matched detection against the
person's side (our two team labels mapped to the sides one-to-one).

    python eval_soccertrack_v2.py data/soccertrack_v2/st2_117093_1st_f015000.mp4 \\
        --out output_st2_117093_1st_f015000 --run --model yolov8s.pt \
        --stride 2 --pitch-margin 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from eval_identity_gsr import BOX_WIDTH, match as match_iou, scores
from src.player_filter import inside_pitch

FOOT_PX = 43.0


def truth(gt_path: Path) -> pd.DataFrame:
    g = pd.read_parquet(gt_path)
    g = g[g.bx.notna()]
    return pd.DataFrame({
        "frame": g.frame.astype(int), "person": g.track_id.astype(int),
        "x0": g.bx, "y0": g.by, "x1": g.bx + g.bw, "y1": g.by + g.bh,
        "side": g.side, "jersey": g.jersey})


def boxes(tracks: pd.DataFrame) -> pd.DataFrame:
    """Player detections with image boxes: the detector's own width where
    the run stored it (`crop_w`), else BOX_WIDTH x height."""
    p = tracks[tracks.cls == "player"].copy()
    x = p.px_raw if "px_raw" in p else p.px
    y = p.py_raw if "py_raw" in p else p.py
    w = p.crop_w if "crop_w" in p else BOX_WIDTH * p.crop_h
    p["x0"], p["x1"] = x - w / 2, x + w / 2
    p["y0"], p["y1"] = y - p.crop_h, y
    return p


def match_feet(det: pd.DataFrame, gt: pd.DataFrame,
               limit: float = FOOT_PX) -> pd.Series:
    """Ground-truth person per detection by bottom-centre distance."""
    out = pd.Series(np.nan, index=det.index)
    gt_by = {f: g for f, g in gt.groupby("frame")}
    for f, d in det.groupby("frame"):
        g = gt_by.get(f)
        if g is None:
            continue
        a = np.c_[(d.x0 + d.x1) / 2, d.y1]
        b = np.c_[(g.x0 + g.x1) / 2, g.y1]
        dist = np.linalg.norm(a[:, None] - b[None], axis=2)
        r, c = linear_sum_assignment(dist)
        for i, j in zip(r, c):
            if dist[i, j] <= limit:
                out.loc[d.index[i]] = g.person.iloc[j]
    return out


def team_accuracy(team: pd.Series, side: pd.Series) -> tuple[float, int]:
    """Share of matched detections whose team, mapped one-to-one to the
    sides, is the person's side; and how many had a team at all."""
    df = pd.DataFrame({"team": team, "side": side}).dropna()
    df = df[df.team.isin(df.team.value_counts().index[:2])]
    if df.empty:
        return float("nan"), 0
    table = pd.crosstab(df.team, df.side)
    r, c = linear_sum_assignment(-table.to_numpy())
    return float(table.to_numpy()[r, c].sum() / len(df)), len(df)


def score_dir(out_dir: Path, gt: pd.DataFrame, stride: int) -> dict:
    gt = gt[gt.frame % stride == 0]
    side = gt.groupby("person").side.agg(lambda s: s.mode().iloc[0])
    report = {"gt boxes": len(gt), "gt people": int(gt.person.nunique())}
    stages = [("fragments", "tracks.parquet", None),
              ("re-joined", "tracks_merged.parquet", None)]
    if (out_dir / "identities.json").exists():
        stages.append(("identities", "tracks_split.parquet"
                       if (out_dir / "tracks_split.parquet").exists()
                       else "tracks_merged.parquet", "identities.json"))
    for how, matcher in (("iou0.5", match_iou), ("feet", match_feet)):
        rep = {}
        for name, file, ident_file in stages:
            tracks = pd.read_parquet(out_dir / file)
            if name == "fragments" and (out_dir / "pitch.json").exists():
                cut = json.loads((out_dir / "pitch.json").read_text())
                tracks = inside_pitch(tracks, cut["pitch"], cut["margin_px"])
            det = boxes(tracks)
            det = det[det.frame % stride == 0]
            person = matcher(det, gt)
            ident = det.track_id
            if ident_file:
                blob = json.loads((out_dir / ident_file).read_text())
                ident = det.track_id.map(lambda t: blob["tracks"].get(
                    str(int(t)), {}).get("identity", f"t{t}"))
            s = scores(ident, person, len(gt))
            hit = person.notna().sum()
            s["recall"] = round(float(hit / len(gt)), 3)
            s["precision"] = round(float(hit / max(len(det), 1)), 3)
            if name == "re-joined" and "team" in det:
                acc, n = team_accuracy(det.team, person.map(side))
                s["team accuracy"] = round(acc, 3)
                s["team matched"] = n
            rep[name] = s
        report[how] = rep
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="store_true",
                    help="run the pipeline first (fixed-camera mode)")
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--pitch-margin", type=float, default=None,
                    help="drop players this many px outside the pitch "
                         "outline (none: no outline)")
    args = ap.parse_args()
    clip, out_dir = Path(args.clip), Path(args.out)
    if args.run:
        from score_soccernet import run_pipeline

        pitch = None
        if args.pitch_margin is not None:
            pitch = json.loads(clip.with_suffix(".json").read_text())["pitch"]
        try:
            run_pipeline(str(clip), out_dir, fixed_camera=True,
                         model_name=args.model, stride=args.stride,
                         pitch=pitch, pitch_margin_px=args.pitch_margin or 0.0)
        except Exception as e:      # tracks are written before events
            print(f"pipeline stopped after tracking: {type(e).__name__}: {e}")
    gt = truth(clip.with_name(clip.stem + "_gt.parquet"))
    report = score_dir(out_dir, gt, args.stride)
    (out_dir / "soccertrack_v2_scores.json").write_text(
        json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
