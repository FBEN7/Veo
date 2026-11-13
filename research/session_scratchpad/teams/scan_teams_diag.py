"""Why the goals scan's teams fail (formation_scan.py clusters each 1-fps
detection on its own, without tracks): its per-detection rule against
variants, on the windows whose video is here, scored against the ground
truth (eval_soccertrack_v2 feet matching, outfield players).

Per window: every 25th frame, the GPU run's detections (events_v1
tracks_teams: feet, box height, the pipeline's per-track team), each
detection's kit feature (team_assignment_v2._torso_feature, plus Lab), then
  - track: the pipeline's per-track team (reference);
  - det_hsv4: cluster_teams on the detections, hsv4 (the scan's rule);
  - det_chroma: the same, chroma;
  - det_lab: the same on (L, a, b) medians of the torso;
  - det_hsv4_k2: hsv4, two clusters only, no 'other';
  - det_smooth: each detection's hsv4 averaged with its own track's
    detections within +-2 s before clustering (what short tracklets in the
    scan would give).
Detections fitted: feet at least 0.15 box heights inside the outline (as
the scan). Accuracy and coverage as eval_soccertrack_v2.team_scores.

    python scan_teams_diag.py [--out scan_teams_diag.json]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import eval_soccertrack_v2 as ev  # noqa: E402
from src import team_assignment_v2 as ta  # noqa: E402

GPU = ROOT / ".cache/gpu"
DATA = ROOT / "data/soccertrack_v2"
if "--clips" in sys.argv:            # the shots clips cut back here (M9: opened through src/unseal.py)
    DATA = ROOT / "data/shot_clips/soccertrack_v2"
STEP = 25


def lab_feature(frame, px, py, h):
    if not np.isfinite(h) or h <= 4:
        return None
    w = h * 0.4
    x1, x2 = int(px - w * ta.TORSO_WIDTH / 2), int(px + w * ta.TORSO_WIDTH / 2)
    y_top = py - h
    y1, y2 = int(y_top + h * ta.TORSO_TOP), int(y_top + h * ta.TORSO_BOTTOM)
    H, W = frame.shape[:2]
    x1, x2, y1, y2 = max(0, x1), min(W, x2), max(0, y1), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    ng = ~((hsv[:, 0] >= ta.GRASS_HUE_LO) & (hsv[:, 0] <= ta.GRASS_HUE_HI) & (hsv[:, 1] >= ta.GRASS_MIN_SAT))
    if ng.sum() >= 4:
        lab = lab[ng]
    return np.median(lab, axis=0) / 255.0


def cluster(feats: dict, fit: set, kit="hsv4", k=None):
    k = k or ta.N_CLUSTERS
    return ta.cluster_teams(feats, n_clusters=k, kit_feature=kit, fit_tracks=fit, verbose=False)


def window(w: str) -> dict:
    run = GPU / w / f"output_{w}" / "events_v1"
    tt = pd.read_parquet(run / "tracks_teams.parquet")
    p = tt[(tt.cls == "player") & (tt.frame % STEP == 0)].copy().reset_index(drop=True)
    info = json.loads((DATA / f"{w}.json").read_text())
    cap = cv2.VideoCapture(str(DATA / f"{w}.mp4"))
    want = {f: g for f, g in p.groupby("frame")}
    hsv, lab = {}, {}
    idx, last = 0, max(want)
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for r in want.get(idx, pd.DataFrame()).itertuples():
            f1 = ta._torso_feature(frame, r.px, r.py, r.crop_h)
            f2 = lab_feature(frame, r.px, r.py, r.crop_h)
            if f1 is not None:
                hsv[r.Index] = f1
            if f2 is not None:
                lab[r.Index] = f2
        idx += 1
    cap.release()
    poly = np.asarray(info["pitch"], np.float32).reshape(-1, 1, 2)
    inside = {i for i, r in p.iterrows()
              if cv2.pointPolygonTest(poly, (float(r.px), float(r.py)), True) >= 0.15 * r.crop_h}
    # smoothed: mean of the same track's features within +-2 s (50 frames)
    sm = {}
    for i, r in p.iterrows():
        if i not in hsv:
            continue
        g = p[(p.track_id == r.track_id) & (abs(p.frame - r.frame) <= 50)].index
        v = [hsv[j] for j in g if j in hsv]
        sm[i] = np.median(v, axis=0)
    variants = {
        "track": {i: t for i, t in p.team.items()},
        "det_hsv4": cluster(hsv, inside),
        "det_chroma": cluster(hsv, inside, "chroma"),
        "det_lab": cluster(lab, inside),
        "det_hsv4_k2": cluster(hsv, inside, k=2),
        "det_smooth": cluster(sm, inside),
    }
    gt = ev.truth(DATA / f"{w}_gt.parquet")
    gt = gt[gt.frame % STEP == 0]
    side = gt.groupby("person").side.agg(lambda s: s.mode().iloc[0])
    outfield = gt.groupby("person").role.agg(lambda s: s.mode().iloc[0]) == "player"
    det = ev.boxes(p)
    person = ev.match_feet(det, gt)
    truth = person.map(side[outfield])
    n_out = int((gt.role == "player").sum())
    out = {}
    for name, team in variants.items():
        s = ev.team_scores(pd.Series(team).reindex(det.index), truth, n_out)
        out[name] = {k: s[k] for k in ("team accuracy", "team coverage")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("scan_teams_diag.json")))
    ap.add_argument("--clips", action="store_true")
    args = ap.parse_args()
    res = {}
    for v in sorted(DATA.glob("st2_*.mp4")):
        w = v.stem
        if v.stat().st_size < 100000 or not (GPU / w).exists():
            continue
        res[w] = window(w)
        print(w, json.dumps(res[w]), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
