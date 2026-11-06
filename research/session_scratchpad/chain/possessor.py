"""chain plan v4 (plan_v4.json, hashed first): a learned 'who has the ball'
at the labelled actions, against rules B1-B3, leaving one match out.

    VEO_UNSEAL=... python possessor.py
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import eval_soccertrack_v2 as ev  # noqa: E402
from src import ball_link  # noqa: E402
from chain import CLIPS, HI, LO, MATCH_M, SEALED, TEAMS, TREE, match  # noqa: E402
from cand import cal  # noqa: E402

FEATS = ["d_link", "rank_link", "d_last", "dt_last", "c1", "c2", "c3", "n2", "s2", "rank_c2", "d_top",
         "speed", "n3", "d_opp", "d_mate", "h", "has_team", "n_players"]


def unit(w):
    c = cal(w.split("_")[1])
    run = Path(os.path.realpath(TREE / w / f"output_{w}" / "metric_v3s_link.parquet")).parent
    m = pd.read_parquet(TREE / w / f"output_{w}" / "metric_v3s_link.parquet")
    m = m.assign(x=m.x - 52.5, y=m.y - 34.0)
    pl = m[(m.cls == "player") & m.x.notna()]
    ball = m[(m.cls == "ball") & m.x.notna()]
    gt = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
    gt = gt[gt.x.notna()]
    evs = pd.read_csv(CLIPS / f"{w}_events.csv")
    evs = evs[(evs.frame >= LO) & (evs.frame <= HI) & evs.player_id.notna() & evs.x.notna()]
    pl_f = {f: g for f, g in pl.groupby("frame")}
    gt_f = {f: g for f, g in gt.groupby("frame")}
    votes = {}
    for f, g in pl_f.items():                       # chain.py's team map, unchanged
        t = gt_f.get(f)
        if t is None:
            continue
        mm = match(g[["x", "y"]].to_numpy(float), t[["x", "y"]].to_numpy(float), MATCH_M)
        for i, j in mm.items():
            lab = g.team.iloc[i]
            if lab in TEAMS:
                votes.setdefault(lab, []).append(t.side.iloc[j])
    side_of = {lab: pd.Series(v).mode().iloc[0] for lab, v in votes.items()}
    truth = ev.truth(CLIPS / f"{w}_gt.parquet")
    v = pd.read_parquet(run / "v3s_rows.parquet").reset_index(drop=True)
    vxy = ball_link.ground_points(v, c)
    ok = np.isfinite(vxy).all(1)
    v, vxy = v[ok].reset_index(drop=True), vxy[ok]
    vf, vc = v.frame.to_numpy(int), v.confidence.to_numpy(float)
    bf, bxy = ball.frame.to_numpy(int), ball[["x", "y"]].to_numpy(float)
    trk = {tid: g.set_index("frame")[["x", "y"]] for tid, g in pl.groupby("track_id")}
    rows, acts = [], []
    for a_i, r in enumerate(evs.itertuples()):     # chain.py's actions, unchanged
        f = int(r.frame) - int(r.frame) % 2
        g, t = pl_f.get(f), gt_f.get(f)
        if g is None or t is None:
            continue
        if t[t.player_id.astype(str) == str(int(r.player_id))].empty:
            continue
        a = t[t.player_id.astype(str) == str(int(r.player_id))]
        actor = int(a.track_id.iloc[0])
        d = ev.boxes(g.assign(cls="player")).reset_index(drop=True)
        tr = truth[truth.frame == f]
        person = ev.match_feet(d, tr)
        P = g[["x", "y"]].to_numpy(float)
        n = len(P)
        lab = g.team.astype(str).to_numpy()
        side = np.array([side_of.get(x) if x in TEAMS else None for x in lab], object)
        aid = f"{w}|{a_i}"
        acts.append({"aid": aid, "unit": w, "match": w.split("_")[1], "frame": int(r.frame), "kind": r.kind,
                     "side": r.side, "actor matched": bool((person == actor).any())})
        # linked ball at the nearest linked frame within f +- 4, and the nearest in time within f +- 25
        k4 = np.flatnonzero(np.abs(bf - f) <= 4)
        k25 = np.flatnonzero(np.abs(bf - f) <= 25)
        d_link = np.full(n, np.nan)
        if len(k4):
            kk = k4[np.argmin(np.abs(bf[k4] - f))]
            d_link = np.linalg.norm(P - bxy[kk], axis=1)
        d_last, dt_last = np.full(n, np.nan), np.nan
        if len(k25):
            kk = k25[np.argmin(np.abs(bf[k25] - f))]
            d_last, dt_last = np.linalg.norm(P - bxy[kk], axis=1), abs(int(bf[kk]) - f)
        # raw candidates within f +- 6
        kw = np.flatnonzero(np.abs(vf - f) <= 6)
        D = np.linalg.norm(P[:, None] - vxy[kw][None], axis=2) if len(kw) else np.zeros((n, 0))
        cw = vc[kw]
        cmax = lambda rad: np.where((D <= rad).any(1), np.where(D <= rad, cw[None], 0).max(1) if D.shape[1] else 0, 0.0)
        c1, c2, c3 = cmax(1.0), cmax(2.0), cmax(3.0)
        n2 = ((D <= 2.0) & (cw[None] >= 0.02)).sum(1) if D.shape[1] else np.zeros(n)
        s2 = np.where(D <= 2.0, cw[None], 0).sum(1) if D.shape[1] else np.zeros(n)
        k4c = np.flatnonzero(np.abs(vf - f) <= 4)
        d_top = np.full(n, np.nan)
        if len(k4c):
            fr = vf[k4c][np.argmin(np.abs(vf[k4c] - f))]
            kt = np.flatnonzero(vf == fr)
            d_top = np.linalg.norm(P - vxy[kt[np.argmax(vc[kt])]], axis=1)
        speed = np.full(n, np.nan)
        for i, tid in enumerate(g.track_id.to_numpy()):
            s = trk.get(tid)
            if s is None:
                continue
            w4 = s[(s.index >= f - 4) & (s.index <= f + 4)]
            if len(w4) >= 2 and w4.index[-1] > w4.index[0]:
                speed[i] = np.linalg.norm(w4.iloc[-1].values - w4.iloc[0].values) / ((w4.index[-1] - w4.index[0]) / 25.0)
        DD = np.linalg.norm(P[:, None] - P[None], axis=2)
        np.fill_diagonal(DD, np.inf)
        same = lab[:, None] == lab[None]
        has = np.isin(lab, TEAMS)
        d_opp = np.where(has, np.where(~same & has[None], DD, np.inf).min(1), np.nan)
        d_mate = np.where(has, np.where(same, DD, np.inf).min(1), np.nan)
        rank = lambda x, asc=True: pd.Series(x).rank(ascending=asc, method="min").to_numpy()
        rk_link, rk_c2 = rank(d_link), rank(c2, False)
        for i in range(n):
            rows.append({"aid": aid, "i": i, "y": int(person.iloc[i] == actor) if pd.notna(person.iloc[i]) else 0,
                         "side_i": side[i], "d_link": d_link[i], "rank_link": rk_link[i] if len(k4) else np.nan,
                         "d_last": d_last[i], "dt_last": dt_last, "c1": c1[i], "c2": c2[i], "c3": c3[i], "n2": n2[i],
                         "s2": s2[i], "rank_c2": rk_c2[i], "d_top": d_top[i], "speed": speed[i],
                         "n3": int((DD[i] <= 3.0).sum()), "d_opp": d_opp[i] if np.isfinite(d_opp[i]) else np.nan,
                         "d_mate": d_mate[i] if np.isfinite(d_mate[i]) else np.nan,
                         "h": float(g.crop_h.iloc[i]), "has_team": int(has[i]), "n_players": n})
    return rows, acts


def pick(df, col, asc=True):
    """Per action, the side of the player with the best value in col (NaN-free rows only)."""
    x = df[df[col].notna()]
    if x.empty:
        return pd.Series(dtype=object)
    idx = x.groupby("aid")[col].idxmin() if asc else x.groupby("aid")[col].idxmax()
    return x.loc[idx].set_index("aid").side_i


def main():
    from src.unseal import unsealed
    for m in sorted(SEALED):
        if not unsealed(m, "chain plan v4: learned possessor (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    units = sorted(pd.read_parquet(HERE / "chain.parquet").unit.unique())
    R, A = [], []
    with Pool(os.cpu_count()) as pool:
        for r, a in pool.imap_unordered(unit, units):
            R += r; A += a
    X, acts = pd.DataFrame(R), pd.DataFrame(A).set_index("aid")
    X.to_parquet(HERE / "possessor_rows.parquet"); acts.to_parquet(HERE / "possessor_actions.parquet")
    acts["match"] = acts["match"].astype(str)
    X = X.merge(acts[["match"]], left_on="aid", right_index=True)
    # rules
    b1 = pick(X, "d_link")
    b2 = b1.combine_first(pick(X, "d_last"))
    b3 = pick(X, "d_top")
    for name, s in (("B1", b1), ("B2", b2), ("B3", b3)):
        acts[name] = acts.index.map(s.to_dict()) == acts.side
    # model, leave one match out
    acts["model"], acts["model actor"] = False, False
    trainable = set(acts.index[acts["actor matched"]])
    for mo in sorted(acts.match.unique()):
        tr = X[(X.match != mo) & X.aid.isin(trainable)]
        te = X[X.match == mo].copy()
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
        clf.fit(tr[FEATS], tr.y)
        te["score"] = clf.predict_proba(te[FEATS])[:, 1]
        best = te.loc[te.groupby("aid").score.idxmax()].set_index("aid")
        acts.loc[best.index, "model"] = (best.side_i == acts.loc[best.index, "side"]).to_numpy()
        acts.loc[best.index, "model actor"] = best.y.astype(bool).to_numpy()
    acts["model"] = acts["model"].astype(bool)
    rule_share = {b: float(acts[b].mean()) for b in ("B1", "B2", "B3")}
    best_rule = max(rule_share, key=rule_share.get)
    by = {m: g for m, g in acts.groupby("match")}
    ms = sorted(by)
    val = lambda s: sum(by[m].model.sum() - by[m][best_rule].sum() for m in s) / sum(len(by[m]) for m in s)
    rng = np.random.default_rng(0)
    draws = [val(rng.choice(ms, len(ms))) for _ in range(5000)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    kinds = acts.kind.where(acts.kind.isin(["pass", "carry"]), "other")
    cols = ["B1", "B2", "B3", "model", "model actor"]
    out = {"plan sha256": (HERE / "plan_v4.json.sha256").read_text().split()[0], "actions": len(acts),
           "actions with the actor matched (trained on)": int(acts["actor matched"].sum()),
           "end to end (held out for the model)": {c: round(float(acts[c].mean()), 4) for c in cols},
           "best rule": best_rule,
           "gain: model - best rule": {"value": round(float(val(ms)), 4), "95% over matches": [round(float(lo), 4), round(float(hi), 4)]},
           "bar met (gain >= 0.03, lower bound > 0)": bool(val(ms) >= 0.03 and lo > 0),
           "per kind": {k: {c: round(float(g[c].mean()), 4) for c in cols} | {"actions": len(g)} for k, g in acts.groupby(kinds)},
           "per match": {m: {c: round(float(g[c].mean()), 4) for c in cols} | {"actions": len(g)} for m, g in acts.groupby("match")}}
    acts.to_parquet(HERE / "possessor_actions.parquet")
    (HERE / "possessor.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per match"}, indent=1))


if __name__ == "__main__":
    main()
