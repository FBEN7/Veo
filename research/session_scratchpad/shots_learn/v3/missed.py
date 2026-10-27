"""Which shots the v3 spotter misses, and what its inputs looked like there
(owner, 2026-10-27). The same nested leave-one-match-out as learn_v3.py on
the relinked tracks, outer folds in parallel; per labelled shot: hit or
missed, the model's best probability within 2 s and the fold's threshold,
and the ball around it (tracked share within 1 s, top speed, distance to
the nearer goal). Writes missed.parquet and missed.json.

    VEO_UNSEAL=... python missed.py
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "v2"))
sys.path.insert(0, str(HERE))
import learn_shots as LS  # noqa: E402
import learn_v2 as V  # noqa: E402
import learn_v3 as V3  # noqa: E402
import score_soccernet as sc  # noqa: E402

D = None


def fold(m):
    names = list(D)
    tr_ = [n for n in names if D[n][2] != m]
    cols = LS.cols_of(D)
    inner = LS.oof(D, tr_, cols)
    thr = LS.rate_threshold(D, tr_, inner)
    clf = LS.fit(D, tr_, cols)
    return m, thr, {n: clf.predict_proba(D[n][0][cols])[:, 1] for n in names if D[n][2] == m}


def main():
    global D
    from src.unseal import unsealed
    for m in V3.NEW_PLAYERS:
        if not unsealed(m, "shots v3: which shots are missed"):
            raise SystemExit("VEO_UNSEAL needed")
    us = V3.new_tree([u for u in V.units() if (LS.GPU / u["name"]).exists()])
    LS.GPU = V.GPU = V3.NEW
    D = {u["name"]: (V.features_v2(u), LS.labels(u), u["match"]) for u in us}
    with Pool(4) as pool:
        res = pool.map(fold, sorted({v[2] for v in D.values()}))
    thr = {m: t for m, t, _ in res}
    probs = {n: p for _, _, ps in res for n, p in ps.items()}
    rows = []
    for n, (f, lab, m) in D.items():
        fr = f.index.to_numpy()
        pred = [fr[i] / LS.FPS for i in LS.peaks(fr, probs[n], thr[m])]
        matched, _, _ = sc.match_one_to_one(pred, lab, LS.TOL_S)
        hit_labels = {lab[ti] for _, ti, _ in matched}
        for t in lab:
            w = np.abs(fr / LS.FPS - t) <= 2.0
            w1 = np.abs(fr / LS.FPS - t) <= 1.0
            g = f[w1]
            rows.append({"unit": n, "match": m, "t": t, "frame": int(round(t * LS.FPS)),
                         "p_max_2s": float(probs[n][w].max()) if w.any() else np.nan, "thr": thr[m],
                         "ball_share_1s": float(g.has_0.mean()) if len(g) else np.nan,
                         "speed_max_1s": float(g.speed_0.max()) if len(g) else np.nan,
                         "goal_dist": float(g.filter(like="g_dist").iloc[:, 0].min()) if len(g) and g.filter(like="g_dist").shape[1] else np.nan,
                         "_hit": t in hit_labels})
    df = pd.DataFrame(rows).rename(columns={"_hit": "hit"})
    miss = df[~df.hit]

    def cause(r):
        if not (r.ball_share_1s >= 0.3):
            return "no ball tracked around the shot"
        if r.p_max_2s >= r.thr:
            return "model fired, but not matched (timing / merged with another peak)"
        if r.p_max_2s >= r.thr - 0.15:
            return "model close to its threshold"
        return "ball tracked, model did not see a shot"

    df["cause"] = [cause(r) if not r.hit else "hit" for r in df.itertuples()]
    out = {"labelled shots": len(df), "hit": int(df.hit.sum()), "missed": len(miss),
           "missed by cause": df[~df.hit].cause.value_counts().to_dict(),
           "hit vs missed medians": df.groupby("hit")[["p_max_2s", "ball_share_1s", "speed_max_1s", "goal_dist"]]
           .median().round(3).to_dict()}
    print(json.dumps(out, indent=1, default=str))
    (HERE / "missed.json").write_text(json.dumps(out, indent=1, default=str))
    df.to_parquet(HERE / "missed.parquet")


if __name__ == "__main__":
    main()
