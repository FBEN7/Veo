"""Exploratory (not registered), after v7.json: the team label of the box
that detects the actor at S4, for A (0.20) and A' (each clip's conf_player).

    VEO_UNSEAL=... python actor_team.py
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import eval_soccertrack_v2 as ev  # noqa: E402
from extra_boxes import CLIPS, SEALED, TRK, TRK7  # noqa: E402


def unit(args):
    w, acts = args
    gt = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
    gt = gt[gt.bx.notna()]
    frames = sorted(set(acts.frame))
    tr = ev.truth(CLIPS / f"{w}_gt.parquet")
    tr = tr[tr.frame.isin(frames)]
    out = []
    for name, f in (("A", TRK / f"{w}_A.parquet"), ("A'", TRK7 / f"{w}_Ap.parquet")):
        t = pd.read_parquet(f)
        t = t[t.in_s4 & t.frame.isin(frames)]
        d = ev.boxes(t.assign(cls="player")).reset_index(drop=True)
        d["person"] = ev.match_feet(d, tr) if len(d) else pd.Series(dtype=float)
        for r in acts.itertuples():
            a = gt[(gt.frame == r.frame) & (gt.player_id.astype(str) == str(int(r.player_id)))]
            if a.empty:
                continue
            hit = d[(d.frame == r.frame) & (d.person == int(a.track_id.iloc[0]))]
            out.append({"unit": w, "match": w.split("_")[1], "frame": int(r.frame), "player_id": int(r.player_id),
                        "arm": name, "team": str(hit.team.iloc[0]) if len(hit) else "not detected"})
    return out


def main():
    from src.unseal import unsealed
    for m in sorted(SEALED):
        if not unsealed(m, "near_ball exploratory: team label of the detected actor in plan_v7 on the opened M7 / M9 clips"):
            raise SystemExit("VEO_UNSEAL needed")
    v7 = pd.read_parquet(HERE / "v7.parquet")
    acts = v7[v7.variant == "A"][["unit", "frame", "player_id"]].drop_duplicates()
    with Pool(os.cpu_count()) as pool:
        rows = [r for rs in pool.imap_unordered(unit, list(acts.groupby("unit"))) for r in rs]
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["unit", "frame", "player_id"], columns="arm", values="team", aggfunc="first")
    team = lambda s: s.isin(["team_A", "team_B"])
    out = {"status": "exploratory, not registered; after v7.json", "actions": len(piv),
           "label of the actor's box (share of actions)": {a: df[df.arm == a].team.value_counts(normalize=True).round(4).to_dict() for a in ("A", "A'")},
           "actor detected with a team label": {a: round(float(team(piv[a]).mean()), 4) for a in ("A", "A'")},
           "gained by A (A' not detected), label in A": piv[(piv["A'"] == "not detected") & (piv.A != "not detected")].A.value_counts().to_dict()}
    (HERE / "reading_actor_team.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
