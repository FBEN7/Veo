"""Team assignment against the ground truth on every GPU-processed unit
(the shots clips and windows): eval_soccertrack_v2's feet matching and
team_scores on output_<w>/events_v1/tracks_merged.parquet, plus the per-track
ceiling (every track given its true majority side) and how often the two
teams are 'not separated' (their mean pitch x within 5 m of each other).

    python team_eval.py [--out team_eval.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import eval_soccertrack_v2 as ev  # noqa: E402

GPU = ROOT / ".cache/gpu"
DATA = [ROOT / "data/soccertrack_v2", ROOT / "data/shot_clips/soccertrack_v2"]


def gt_of(w):
    for d in DATA:
        if (d / f"{w}_gt.parquet").exists():
            return ev.truth(d / f"{w}_gt.parquet")
    return None


def unit(w):
    run = GPU / w / f"output_{w}" / "events_v1"
    gt = gt_of(w)
    if gt is None or not (run / "tracks_merged.parquet").exists():
        return None
    gt = gt[gt.frame % 2 == 0]
    side = gt.groupby("person").side.agg(lambda s: s.mode().iloc[0])
    outfield = gt.groupby("person").role.agg(lambda s: s.mode().iloc[0]) == "player"
    det = ev.boxes(pd.read_parquet(run / "tracks_merged.parquet"))
    det = det[det.frame % 2 == 0]
    person = ev.match_feet(det, gt)
    s = ev.team_scores(det.team, person.map(side[outfield]), int((gt.role == "player").sum()))
    # ceiling: each track its majority true side
    df = pd.DataFrame({"track": det.track_id, "team": det.team, "side": person.map(side[outfield])}).dropna(subset=["side"])
    df = df[df.team.isin(ev.TEAMS)]
    if len(df):
        maj = df.groupby("track").side.agg(lambda s: s.mode().iloc[0])
        s["track ceiling accuracy"] = round(float((df.side == df.track.map(maj)).mean()), 3)
    m = pd.read_parquet(GPU / w / f"output_{w}" / "metric_v3s_link.parquet") if \
        (GPU / w / f"output_{w}" / "metric_v3s_link.parquet").exists() else None
    if m is not None:
        mx = m[(m.cls == "player") & m.team.isin(ev.TEAMS)].groupby("team").x.mean()
        s["team mean x gap (m)"] = round(float(abs(mx.diff().iloc[-1])), 1) if len(mx) == 2 else None
    s["match"] = w.split("_")[1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("team_eval.json")))
    args = ap.parse_args()
    res = {}
    for g in sorted(GPU.iterdir()):
        r = unit(g.name)
        if r:
            res[g.name] = r
            print(g.name, {k: r[k] for k in ("team accuracy", "team coverage")}, flush=True)
    d = pd.DataFrame(res).T
    for c in ("team accuracy", "team coverage", "team recall", "track ceiling accuracy", "team mean x gap (m)"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    by = d.groupby("match")[["team accuracy", "team coverage", "track ceiling accuracy", "team mean x gap (m)"]].median().round(3)
    by["units"] = d.groupby("match").size()
    by["units with accuracy < 0.7"] = d.groupby("match")["team accuracy"].apply(lambda s: int((s < 0.7).sum()))
    print(by.to_string())
    Path(args.out).write_text(json.dumps({"per unit": res, "per match (median)": json.loads(by.to_json(orient="index"))}, indent=1))


if __name__ == "__main__":
    main()
