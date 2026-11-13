"""Exploratory (not registered), after v7.json: what are the unmatched boxes
at the roster cap (S4) on the action frames, for A (0.20) and A' (each
clip's conf_player), and those only A has.

Each unmatched box: its foot's distance to the nearest truth person;
'duplicate' if within FOOT_PX of a person another box matched, 'near'
within 100 px, else 'far'; its confidence (from the GPU pass's boxes), team
label, height, and whether its foot is inside the outline.

    VEO_UNSEAL=... python extra_boxes.py
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import eval_soccertrack_v2 as ev  # noqa: E402

CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
TRK = ROOT / ".cache/near_ball_trk/trk"
TRK7 = ROOT / ".cache/near_ball_trk7/trk"
DET = ROOT / ".cache/near_ball_det/det"
SEALED = {"128057", "132831"}
NEAR = 100.0


def iou(A, B):
    lt = np.maximum(A[:, None, :2], B[None, :, :2])
    rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
    return inter / (area(A)[:, None] + area(B)[None, :] - inter)


def unit(args):
    w, frames = args
    tr = ev.truth(CLIPS / f"{w}_gt.parquet")
    tr = tr[tr.frame.isin(frames)]
    poly = np.asarray(json.loads((CLIPS / f"{w}.json").read_text())["pitch"], np.float32).reshape(-1, 1, 2)
    det = pd.read_parquet(DET / f"{w}.parquet")
    det = det[det.frame.isin(frames)]
    arms = {}
    for name, f in (("A", TRK / f"{w}_A.parquet"), ("A'", TRK7 / f"{w}_Ap.parquet")):
        t = pd.read_parquet(f)
        t = t[t.in_s4 & t.frame.isin(frames)]
        d = ev.boxes(t.assign(cls="player")).reset_index(drop=True)
        d["person"] = ev.match_feet(d, tr) if len(d) else pd.Series(dtype=float)
        arms[name] = d
    rows = []
    for name, d in arms.items():
        other = arms["A'" if name == "A" else "A"]
        for f, g in d.groupby("frame"):
            u = g[g.person.isna()]
            if u.empty:
                continue
            t = tr[tr.frame == f]
            B = u[["x0", "y0", "x1", "y1"]].to_numpy(float)
            foot = np.c_[(u.x0 + u.x1) / 2, u.y1]
            if len(t):
                tf = np.c_[(t.x0 + t.x1) / 2, t.y1]
                dist = np.linalg.norm(foot[:, None] - tf[None], axis=2)
                j = dist.argmin(axis=1)
                dmin = dist[np.arange(len(u)), j]
                taken = set(g.person.dropna().astype(int))
                near_person = t.person.to_numpy()[j]
                near_role = t.role.to_numpy()[j]
            else:
                dmin = np.full(len(u), np.inf)
                near_person = np.full(len(u), -1)
                near_role = np.full(len(u), None)
                taken = set()
            dg = det[det.frame == f]
            conf = np.full(len(u), np.nan)
            if len(dg):
                m = iou(B, dg[["x0", "y0", "x1", "y1"]].to_numpy(float))
                k = m.argmax(axis=1)
                ok = m[np.arange(len(u)), k] >= 0.99
                conf[ok] = dg.conf.to_numpy()[k[ok]]
            og = other[other.frame == f]
            in_other = (iou(B, og[["x0", "y0", "x1", "y1"]].to_numpy(float)).max(axis=1) >= 0.9) if len(og) else np.zeros(len(u), bool)
            for i in range(len(u)):
                kind = ("duplicate" if dmin[i] <= ev.FOOT_PX and int(near_person[i]) in taken
                        else "near" if dmin[i] <= NEAR else "far")
                inside = cv2.pointPolygonTest(poly, (float(foot[i, 0]), float(foot[i, 1])), False) >= 0
                rows.append({"unit": w, "match": w.split("_")[1], "frame": int(f), "arm": name, "kind": kind,
                             "dist": float(dmin[i]), "role": near_role[i], "conf": float(conf[i]),
                             "team": str(u.team.iloc[i]), "h": float(u.y1.iloc[i] - u.y0.iloc[i]),
                             "inside outline": bool(inside), "in other arm": bool(in_other[i])})
    return rows


def main():
    from src.unseal import unsealed
    for m in sorted(SEALED):
        if not unsealed(m, "near_ball exploratory: the extra unmatched boxes of plan_v7 on the opened M7 / M9 clips"):
            raise SystemExit("VEO_UNSEAL needed")
    v7 = pd.read_parquet(HERE / "v7.parquet")
    frames = v7.groupby("unit").frame.apply(lambda s: sorted(set(s)))
    with Pool(os.cpu_count()) as pool:
        rows = [r for rs in pool.imap_unordered(unit, list(frames.items())) for r in rs]
    df = pd.DataFrame(rows)
    df.to_parquet(HERE / "extra_boxes.parquet")
    n_frames = int(v7[["unit", "frame"]].drop_duplicates().shape[0])
    groups = {"A, all unmatched": df[df.arm == "A"], "A', all unmatched": df[df.arm == "A'"],
              "A only (the extra boxes)": df[(df.arm == "A") & ~df["in other arm"]]}
    out = {"status": "exploratory, not registered; after v7.json", "action frames": n_frames}
    for k, g in groups.items():
        out[k] = {"per action frame": round(len(g) / n_frames, 3),
                  "kind (share)": g.kind.value_counts(normalize=True).round(3).to_dict(),
                  "nearest truth role, duplicate or near (share)": g[g.kind != "far"].role.value_counts(normalize=True).round(3).to_dict(),
                  "team label (share)": g.team.value_counts(normalize=True).round(3).to_dict(),
                  "foot inside the outline": round(float(g["inside outline"].mean()), 3),
                  "height px, median": round(float(g.h.median()), 1),
                  "conf, quartiles": [round(float(q), 3) for q in g.conf.quantile([0.25, 0.5, 0.75])],
                  "conf found": round(float(g.conf.notna().mean()), 3)}
    extra = groups["A only (the extra boxes)"]
    out["extra per action frame, per match"] = (extra.groupby("match").size() / v7[["unit", "frame", "match"]].drop_duplicates().groupby("match").size()).round(2).to_dict()
    (HERE / "reading_extra_boxes.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
