"""Is the scan's failure on M9 2nd the whole-half clustering? The 16 M9 2nd
clips (spread over the half) at 1 fps: per-detection hsv4 kit features
clustered (a) clip by clip, (b) all clips pooled, as the scan does over a
half, (c) pooled but within 10-minute blocks. Accuracy against the ground
truth (outfield, feet matching), as scan_teams_diag.py.

    python pooled_diag.py --match 132831 --half 2nd
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scan_teams_diag as D  # noqa: E402

ev, ta = D.ev, D.ta
CLIPS = D.ROOT / "data/shot_clips/soccertrack_v2"


def unit(w):
    run = D.GPU / w / f"output_{w}" / "events_v1"
    tt = pd.read_parquet(run / "tracks_teams.parquet")
    p = tt[(tt.cls == "player") & (tt.frame % D.STEP == 0)].copy().reset_index(drop=True)
    info = json.loads((CLIPS / f"{w}.json").read_text())
    span = None
    cap = cv2.VideoCapture(str(CLIPS / f"{w}.mp4"))
    want = {f: g for f, g in p.groupby("frame")}
    feats = {}
    idx, last = 0, max(want)
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for r in want.get(idx, pd.DataFrame()).itertuples():
            f = ta._torso_feature(frame, r.px, r.py, r.crop_h)
            if f is not None:
                feats[r.Index] = f
        idx += 1
    cap.release()
    poly = np.asarray(info["pitch"], np.float32).reshape(-1, 1, 2)
    inside = {i for i, r in p.iterrows()
              if cv2.pointPolygonTest(poly, (float(r.px), float(r.py)), True) >= 0.15 * r.crop_h}
    gt = ev.truth(CLIPS / f"{w}_gt.parquet")
    gt = gt[gt.frame % D.STEP == 0]
    side = gt.groupby("person").side.agg(lambda s: s.mode().iloc[0])
    outfield = gt.groupby("person").role.agg(lambda s: s.mode().iloc[0]) == "player"
    det = ev.boxes(p)
    truth = ev.match_feet(det, gt).map(side[outfield])
    return {"start": int(info["start_frame"]), "feats": feats, "inside": inside, "truth": truth,
            "n_out": int((gt.role == "player").sum())}


def acc(team: dict, units: dict) -> dict:
    """Accuracy with ONE mapping of team_A / team_B to sides for all units
    (as a half-long clustering must keep), and coverage."""
    rows = []
    for w, u in units.items():
        for i, s in u["truth"].dropna().items():
            rows.append((team.get((w, i), "other"), s))
    df = pd.DataFrame(rows, columns=["team", "side"])
    named = df[df.team.isin(ev.TEAMS)]
    tab = pd.crosstab(named.team, named.side)
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-tab.to_numpy())
    return {"accuracy": round(float(tab.to_numpy()[r, c].sum() / max(len(named), 1)), 3),
            "coverage": round(len(named) / max(len(df), 1), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="132831")
    ap.add_argument("--half", default="2nd")
    args = ap.parse_args()
    names = sorted(p.stem for p in CLIPS.glob(f"st2_{args.match}_{args.half}_f*.mp4"))
    units = {w: unit(w) for w in names}
    out = {"clips": len(units)}
    # (a) clip by clip: mapping per clip
    per = {}
    for w, u in units.items():
        t = ta.cluster_teams(u["feats"], fit_tracks=u["inside"], verbose=False)
        per[w] = acc({(w, i): v for i, v in t.items()}, {w: u})["accuracy"]
    out["(a) per clip, median accuracy"] = float(np.median(list(per.values())))
    # (b) pooled over the half
    feats = {(w, i): f for w, u in units.items() for i, f in u["feats"].items()}
    inside = {(w, i) for w, u in units.items() for i in u["inside"]}
    out["(b) pooled over the half"] = acc(ta.cluster_teams(feats, fit_tracks=inside, verbose=False), units)
    # (c) pooled within 10-minute blocks (each block its own mapping)
    blocks = {}
    for w, u in units.items():
        blocks.setdefault(u["start"] // 15000, []).append(w)
    res = []
    for b, ws in sorted(blocks.items()):
        f = {(w, i): v for w in ws for i, v in units[w]["feats"].items()}
        ins = {(w, i) for w in ws for i in units[w]["inside"]}
        res.append(acc(ta.cluster_teams(f, fit_tracks=ins, verbose=False), {w: units[w] for w in ws})["accuracy"])
    out["(c) 10-min blocks, accuracy per block"] = res
    # kit drift: the team centres' value (brightness) over the half
    print(json.dumps(out, indent=1))
    (HERE / f"pooled_diag_{args.match}_{args.half}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
