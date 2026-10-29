"""Out-of-fold per-frame probabilities of the shots v4 spotter (learn_v4.py's
nested leave-one-match-out, same units, tracks and features), saved for the
hybrid ball's windows, and how well windows from them cover the shots.

    VEO_UNSEAL=... python oof.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "v2"))
sys.path.insert(0, str(HERE.parent / "v3"))
sys.path.insert(0, str(HERE.parent / "v4"))
import learn_shots as LS  # noqa: E402
import learn_v2 as V  # noqa: E402
import learn_v3 as V3  # noqa: E402
import learn_v4 as V4  # noqa: E402

THRESHOLDS = (0.02, 0.05, 0.1, 0.2)
HALF_S = 2.0


def main():
    from src.unseal import unsealed
    for m in V3.NEW_PLAYERS:
        if not unsealed(m, "shots hybrid: v4 spotter out-of-fold probabilities (M7 / M9 clips as training data)"):
            raise SystemExit("VEO_UNSEAL needed")
    us = V3.new_tree([u for u in V.units() if (LS.GPU / u["name"]).exists()])
    LS.GPU = V.GPU = V3.NEW
    d = {u["name"]: (V4.features_v4(u), LS.labels(u), u["match"]) for u in us}
    names = list(d)
    probs, thr = LS.nested(d, names, LS.cols_of(d))
    res = LS.score(d, names, probs, thr)
    rows = [pd.DataFrame({"unit": n, "match": d[n][2], "frame": d[n][0].index.to_numpy(), "p": probs[n]})
            for n in names]
    p = pd.concat(rows, ignore_index=True)
    p.to_parquet(HERE / "oof_v4.parquet")
    cover = {}
    for t in THRESHOLDS:
        n_fr = n_in = n_lab = lab_in = miss = miss_in = 0
        for n in names:
            fr = d[n][0].index.to_numpy()
            hot = fr[probs[n] >= t]
            inw = np.zeros(len(fr), bool)
            for h in hot:
                inw |= np.abs(fr - h) <= HALF_S * LS.FPS
            n_fr, n_in = n_fr + len(fr), n_in + int(inw.sum())
            pred = [fr[i] / LS.FPS for i in LS.peaks(fr, probs[n], thr[d[n][2]])]
            _, _, missed = LS.sc.match_one_to_one(pred, d[n][1], LS.TOL_S)
            for lab in d[n][1]:
                k = np.abs(fr / LS.FPS - lab) <= 0.5
                n_lab += 1
                lab_in += int(inw[k].any())
            for lab in missed:
                k = np.abs(fr / LS.FPS - lab) <= 0.5
                miss += 1
                miss_in += int(inw[k].any())
        cover[str(t)] = {"frames in windows": round(n_in / n_fr, 3), "shots in windows": f"{lab_in}/{n_lab}",
                         "missed shots in windows": f"{miss_in}/{miss}"}
    out = {"spotter (same as cv_v4.json)": {k: res[k] for k in ("tp", "fp", "fn", "recall", "precision")},
           "thresholds": {str(k): float(v) for k, v in thr.items()}, "windows +-2 s": cover}
    (HERE / "oof_v4.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
