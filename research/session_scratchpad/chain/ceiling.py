"""Exploratory (not registered), after possessor.json: with the action's
true position, the nearest of our detections: its label's team, its truth
team (image matching), whether it has a label; and whether any detection
lies within 1.5 m.

    VEO_UNSEAL=... python ceiling.py
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import eval_soccertrack_v2 as ev  # noqa: E402
from chain import CLIPS, HI, LO, MATCH_M, SEALED, TEAMS, TREE, match  # noqa: E402


def unit(w):
    m = pd.read_parquet(TREE / w / f"output_{w}" / "metric_v3s_link.parquet")
    m = m.assign(x=m.x - 52.5, y=m.y - 34.0)
    pl = m[(m.cls == "player") & m.x.notna()]
    gt = pd.read_parquet(CLIPS / f"{w}_gt.parquet"); gt = gt[gt.x.notna()]
    truth = ev.truth(CLIPS / f"{w}_gt.parquet")
    evs = pd.read_csv(CLIPS / f"{w}_events.csv")
    evs = evs[(evs.frame >= LO) & (evs.frame <= HI) & evs.player_id.notna() & evs.x.notna()]
    pl_f = {f: g for f, g in pl.groupby("frame")}; gt_f = {f: g for f, g in gt.groupby("frame")}
    votes = {}
    for f, g in pl_f.items():
        t = gt_f.get(f)
        if t is None:
            continue
        for i, j in match(g[["x", "y"]].to_numpy(float), t[["x", "y"]].to_numpy(float), MATCH_M).items():
            if g.team.iloc[i] in TEAMS:
                votes.setdefault(g.team.iloc[i], []).append(t.side.iloc[j])
    side_of = {lab: pd.Series(v).mode().iloc[0] for lab, v in votes.items()}
    side_by_person = gt.drop_duplicates("track_id").set_index("track_id").side.to_dict()
    out = []
    for r in evs.itertuples():
        f = int(r.frame) - int(r.frame) % 2
        g, t = pl_f.get(f), gt_f.get(f)
        if g is None or t is None or t[t.player_id.astype(str) == str(int(r.player_id))].empty:
            continue
        d = ev.boxes(g.assign(cls="player")).reset_index(drop=True)
        person = ev.match_feet(d, truth[truth.frame == f])
        P = g[["x", "y"]].to_numpy(float)
        dist = np.linalg.norm(P - [r.x, r.y], axis=1)
        i = int(np.argmin(dist))
        lab = g.team.iloc[i]
        tp = person.iloc[i]
        out.append({"unit": w, "kind": r.kind, "near dist m": float(dist[i]),
                    "label team right": (side_of.get(lab) if lab in TEAMS else None) == r.side,
                    "has label": lab in TEAMS,
                    "truth team right": pd.notna(tp) and side_by_person.get(int(tp)) == r.side,
                    "matched to a truth person": bool(pd.notna(tp))})
    return out


def main():
    from src.unseal import unsealed
    for m in sorted(SEALED):
        if not unsealed(m, "chain exploratory: ceiling near the ball (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    units = sorted(pd.read_parquet(HERE / "chain.parquet").unit.unique())
    with Pool(os.cpu_count()) as pool:
        df = pd.DataFrame([r for rs in pool.imap_unordered(unit, units) for r in rs])
    lab = df[df["has label"]]
    out = {"status": "exploratory, not registered; after possessor.json", "actions": len(df),
           "nearest detection to the true position: within 1.5 m": round(float((df["near dist m"] <= 1.5).mean()), 4),
           "its label's team is right": round(float(df["label team right"].mean()), 4),
           "it has a label": round(float(df["has label"].mean()), 4),
           "label right, among labelled": round(float(lab["label team right"].mean()), 4),
           "its truth team is right (a perfect label)": round(float(df["truth team right"].mean()), 4),
           "it matches a truth person (image)": round(float(df["matched to a truth person"].mean()), 4)}
    (HERE / "reading_ceiling.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
