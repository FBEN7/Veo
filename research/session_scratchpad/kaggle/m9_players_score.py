"""Score the M9 player-detection diagnostic (veo-m9-players output): per
model and setting, recall and precision of single-frame person detections
against the truth (players and goalkeepers, feet matching), M9 vs M7.
Settings: the pipeline's confidence 0.20 with the near-grass filter; no
grass filter; confidence 0.10; confidence 0.05. No tracker here, so the
pipeline's own recall (after ByteTrack) is lower than the first row.

    VEO_UNSEAL=... python m9_players_score.py m9_players.parquet
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import eval_soccertrack_v2 as ev  # noqa: E402
from src.unseal import unsealed  # noqa: E402

CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
SETTINGS = {"conf 0.20 + grass (pipeline)": (0.20, True), "conf 0.20, no grass": (0.20, False),
            "conf 0.10 + grass": (0.10, True), "conf 0.05 + grass": (0.05, True)}


def main():
    for m in ("128057", "132831"):
        if not unsealed(m, "M9 player-detection diagnostic: truth of the sampled frames"):
            raise SystemExit("VEO_UNSEAL needed")
    d = pd.read_parquet(sys.argv[1])
    out = {}
    for (model, clip), g in d.groupby(["model", "clip"]):
        gt = ev.truth(CLIPS / f"{clip}_gt.parquet")
        gt = gt[gt.frame.isin(g.frame.unique()) & gt.role.isin(["player", "goalkeeper"])]
        for name, (c, grass) in SETTINGS.items():
            q = g[(g.conf >= c) & (g.near | (not grass))]
            p = ev.match_feet(q, gt)
            s = out.setdefault(model, {}).setdefault("M9" if "132831" in clip else "M7", {}).setdefault(
                name, {"truth": 0, "dets": 0, "matched": 0})
            s["truth"] += len(gt)
            s["dets"] += len(q)
            s["matched"] += int(p.notna().sum())
    for model, a in out.items():
        for match, b in a.items():
            for name, s in b.items():
                s["recall"] = round(s["matched"] / s["truth"], 3)
                s["precision"] = round(s["matched"] / max(s["dets"], 1), 3)
    print(json.dumps(out, indent=1))
    (Path(__file__).with_name("m9_players.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
