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
its identities; and the team of each detection matched to an outfield
player against the person's side (`team_scores`: accuracy, coverage and
end-to-end recall).

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
        "side": g.side, "jersey": g.jersey, "role": g.role})


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
        # Pairs beyond the limit cost more than any set of pairs within it,
        # so the assignment maximises matches first, then closeness.
        r, c = linear_sum_assignment(np.where(dist <= limit, dist, 1e6))
        for i, j in zip(r, c):
            if dist[i, j] <= limit:
                out.loc[d.index[i]] = g.person.iloc[j]
    return out


TEAMS = ("team_A", "team_B")


def team_scores(team: pd.Series, side: pd.Series, n_outfield: int) -> dict:
    """Teams of the detections matched to outfield players (goalkeepers'
    kits differ by design and are left out): `team_A` / `team_B` mapped
    one-to-one to the sides.

    - accuracy: right side, of those given team_A or team_B;
    - coverage: given team_A or team_B, of those matched -- a track put in
      "other" is not wrong but is lost, so accuracy alone can be bought by
      refusing hard tracks;
    - recall: outfield ground-truth boxes detected, kept and on the right
      team -- the end-to-end number."""
    df = pd.DataFrame({"team": team, "side": side}).dropna(subset=["side"])
    named = df[df.team.isin(TEAMS)]
    out = {"team coverage": round(len(named) / max(len(df), 1), 3)}
    if named.empty:
        return {**out, "team accuracy": float("nan"), "team recall": 0.0}
    table = pd.crosstab(named.team, named.side)
    r, c = linear_sum_assignment(-table.to_numpy())
    right = float(table.to_numpy()[r, c].sum())
    return {**out, "team accuracy": round(right / len(named), 3),
            "team recall": round(right / max(n_outfield, 1), 3)}


def score_dir(out_dir: Path, gt: pd.DataFrame, stride: int) -> dict:
    gt = gt[gt.frame % stride == 0]
    side = gt.groupby("person").side.agg(lambda s: s.mode().iloc[0])
    outfield = gt.groupby("person").role.agg(
        lambda s: s.mode().iloc[0]) == "player"
    outfield_side = side[outfield]
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
                s.update(team_scores(det.team, person.map(outfield_side),
                                     int((gt.role == "player").sum())))
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
    ap.add_argument("--calibrated", action="store_true",
                    help="put tracks on the pitch with the match's released "
                         "calibration, and detect events (events.json)")
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
        to_pitch = None
        if args.calibrated:
            from src import soccertrack_v2 as st

            cal = st.calibration(
                json.loads(clip.with_suffix(".json").read_text())["match"])
            to_pitch = lambda uv: st.image_to_pitch(uv, cal)  # noqa: E731
        try:
            run_pipeline(str(clip), out_dir, fixed_camera=True,
                         model_name=args.model, stride=args.stride,
                         pitch=pitch, pitch_margin_px=args.pitch_margin or 0.0,
                         to_pitch=to_pitch)
        except Exception as e:      # tracks are written before events
            print(f"pipeline stopped after tracking: {type(e).__name__}: {e}")
    gt = truth(clip.with_name(clip.stem + "_gt.parquet"))
    report = score_dir(out_dir, gt, args.stride)
    (out_dir / "soccertrack_v2_scores.json").write_text(
        json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
