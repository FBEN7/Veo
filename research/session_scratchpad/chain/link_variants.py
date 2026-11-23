"""chain plan v3 (plan_v3.json, hashed first): ball linking variants, chosen
leaving one match out, for the team at the ball; guard on the owner's
clicks of the ball after the kick.

    VEO_UNSEAL=... python link_variants.py
"""
import json
import os
import sys
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src import ball_link  # noqa: E402
from chain import BALL_DF, BALL_M, CLIPS, HI, LO, MATCH_M, SEALED, TEAMS, TREE, match  # noqa: E402
from cand import cal  # noqa: E402

VARIANTS = [(fr, tau, b) for fr, tau, b in product(("even", "all"), (0.10, 0.05, 0.03, 0.01), (0, 1, 2))]
NAME = lambda v: f"{v[0]} tau{v[1]:.2f} b{v[2]}"
BASE = NAME(("even", 0.10, 0))
FEET_M, CLICK_PX = 1.5, 40.0
CLICKS = ROOT / "research/session_scratchpad/ball_v4/clicks.parquet"
MANIFEST = ROOT / "research/session_scratchpad/ball_clicks/manifest.json"


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def unit(w):
    c = cal(w.split("_")[1])
    run = Path(os.path.realpath(TREE / w / f"output_{w}" / "metric_v3s_link.parquet")).parent
    m = pd.read_parquet(TREE / w / f"output_{w}" / "metric_v3s_link.parquet")
    players_px = m[m.cls == "player"].assign(px=lambda d: d.px_raw, py=lambda d: d.py_raw)
    stored = m[m.cls == "ball"]
    m = m.assign(x=m.x - 52.5, y=m.y - 34.0)
    pl = m[(m.cls == "player") & m.x.notna()]
    gt = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
    gt = gt[gt.x.notna()]
    ev = pd.read_csv(CLIPS / f"{w}_events.csv")
    ev = ev[(ev.frame >= LO) & (ev.frame <= HI) & ev.player_id.notna() & ev.x.notna()]
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
    acts = []
    for r in ev.itertuples():                       # chain.py's actions, unchanged
        f = int(r.frame) - int(r.frame) % 2
        g, t = pl_f.get(f), gt_f.get(f)
        if g is None or t is None:
            continue
        if t[t.player_id.astype(str) == str(int(r.player_id))].empty:
            continue
        acts.append((r, f, g))
    # candidates, their ground points and the feet test
    v = pd.read_parquet(run / "v3s_rows.parquet").reset_index(drop=True)
    gxy = ball_link.ground_points(v, c)
    pxy = {f: g[["x", "y"]].to_numpy(float) for f, g in pl_f.items()}
    near = np.zeros(len(v), bool)
    for i, (f, xy) in enumerate(zip(v.frame.to_numpy(int), gxy)):
        P = next((pxy[k] for k in (f, f - 1, f + 1) if k in pxy), None)
        if P is not None and np.isfinite(xy).all() and len(P):
            near[i] = np.hypot(*(P - xy).T).min() <= FEET_M
    even = v.frame.isin(set(players_px.frame)).to_numpy()
    sh = pd.read_parquet(CLICKS)
    man = {s["id"]: s for s in json.loads(MANIFEST.read_text())}
    sh = sh[(sh.status == "ball") & sh.shot.map(lambda s: man.get(s, {}).get("unit")) .eq(w)]
    clicks = [(int(r.frame), r.x + man[r.shot]["x0"], r.y + man[r.shot]["y0"]) for r in sh.itertuples()]
    act_rows = [{"unit": w, "match": w.split("_")[1], "frame": int(r.frame), "kind": r.kind, "side": r.side} for r, _, _ in acts]
    click_rows = [{"unit": w, "match": w.split("_")[1], "frame": f} for f, _, _ in clicks]
    repro = {}
    for var in VARIANTS:
        fr, tau, b = var
        sel = v[(even if fr == "even" else np.ones(len(v), bool)) & (v.confidence >= tau).to_numpy()]
        sel = sel.sort_values("confidence", ascending=False, kind="stable").groupby("frame", sort=False).head(4)
        sel = sel.assign(confidence=1 / (1 + np.exp(-(logit(sel.confidence) + b * near[sel.index]))))
        kept = ball_link.link(sel, c, players_px, fps=25, tau=tau, c_null=1.0, c_static=2.0)
        kxy = ball_link.ground_points(kept, c)
        bf = kept.frame.to_numpy(int)
        name = NAME(var)
        if name == BASE:
            a = set(zip(kept.frame, kept.px.round(1), kept.py.round(1)))
            s = set(zip(stored.frame, stored.px_raw.round(1), stored.py_raw.round(1)))
            repro = {"relinked": len(a), "stored": len(s), "common": len(a & s)}
        for row, (r, f, g) in zip(act_rows, acts):
            P = g[["x", "y"]].to_numpy(float)
            pos = np.array([r.x, r.y], float)
            k = np.flatnonzero(np.abs(bf - f) <= BALL_DF)
            bxy = None
            if len(k):
                kk = k[np.argmin(np.abs(bf[k] - f))]
                cand = kxy[bf == bf[kk]]
                cand = cand[np.argmin(np.linalg.norm(cand - pos, axis=1))]
                if np.linalg.norm(cand - pos) <= BALL_M:
                    bxy = cand
            i_ball = int(np.argmin(np.linalg.norm(P - bxy, axis=1))) if bxy is not None and len(P) else None
            lab = g.team.iloc[i_ball] if i_ball is not None else None
            row[f"S1|{name}"] = bxy is not None
            row[f"e2e|{name}"] = (side_of.get(lab) if lab in TEAMS else None) == r.side
        for row, (f, cx, cy) in zip(click_rows, clicks):
            near_k = (np.abs(bf - f) <= 1)
            d = np.hypot(kept.px.to_numpy(float)[near_k] - cx, kept.py.to_numpy(float)[near_k] - cy)
            row[f"hit|{name}"] = bool(len(d) and d.min() <= CLICK_PX)
    return act_rows, click_rows, {w: repro}


def main():
    from src.unseal import unsealed
    for m in sorted(SEALED):
        if not unsealed(m, "chain plan v3: ball linking variants (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    ch = pd.read_parquet(HERE / "chain.parquet")
    units = sorted(ch.unit.unique())
    A, C, R = [], [], {}
    with Pool(os.cpu_count()) as pool:
        for i, (a, c, r) in enumerate(pool.imap_unordered(unit, units)):
            A += a; C += c; R.update(r)
            if i % 25 == 0:
                print(i, flush=True)
    acts, clicks = pd.DataFrame(A), pd.DataFrame(C)
    acts.to_parquet(HERE / "link_variants_actions.parquet"); clicks.to_parquet(HERE / "link_variants_clicks.parquet")
    names = [NAME(v) for v in VARIANTS]
    # reproduction checks
    rep1 = all(r["relinked"] == r["stored"] == r["common"] for r in R.values())
    key = ["unit", "frame", "kind", "side"]
    a2 = acts.assign(k=acts.groupby(key).cumcount()); c2 = ch.assign(k=ch.groupby(key).cumcount())
    j = a2.merge(c2[key + ["k", "end to end"]], on=key + ["k"], how="inner", validate="one_to_one")
    rep2 = bool(len(j) == len(ch) == len(acts) and (j[f"e2e|{BASE}"] == j["end to end"]).all())
    # leave one match out
    matches = sorted(acts.match.unique())
    folds, per_match = {}, []
    for mo in matches:
        tr = acts[acts.match != mo]
        score = {n: tr[f"e2e|{n}"].mean() for n in names}
        best = max(names, key=lambda n: (round(score[n], 12), -names.index(n)))
        folds[mo] = best
        te = acts[acts.match == mo]
        ck = clicks[clicks.match == mo]
        per_match.append({"match": mo, "n": len(te), "chosen": best, "chosen e2e": int(te[f"e2e|{best}"].sum()),
                          "base e2e": int(te[f"e2e|{BASE}"].sum()), "clicks": len(ck),
                          "chosen hits": int(ck[f"hit|{best}"].sum()) if len(ck) else 0,
                          "base hits": int(ck[f"hit|{BASE}"].sum()) if len(ck) else 0})
    pm = pd.DataFrame(per_match)
    gain = (pm["chosen e2e"].sum() - pm["base e2e"].sum()) / pm.n.sum()
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(5000):
        s = pm.iloc[rng.integers(0, len(pm), len(pm))]
        draws.append((s["chosen e2e"].sum() - s["base e2e"].sum()) / s.n.sum())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    hit_ch = pm["chosen hits"].sum() / max(pm.clicks.sum(), 1)
    hit_b = pm["base hits"].sum() / max(pm.clicks.sum(), 1)
    full = {n: {"e2e": round(float(acts[f"e2e|{n}"].mean()), 4), "ball within 3 m": round(float(acts[f"S1|{n}"].mean()), 4),
                "guard hit": round(float(clicks[f"hit|{n}"].mean()), 4)} for n in names}
    chosen_all = max(names, key=lambda n: (round(acts[f"e2e|{n}"].mean(), 12), -names.index(n)))
    guard_ok = bool(hit_ch >= hit_b - 0.02)
    out = {"plan sha256": (HERE / "plan_v3.json.sha256").read_text().split()[0],
           "reproduction: base relinks the stored ball on every unit": rep1,
           "reproduction: base end to end equals chain.parquet's": rep2,
           "actions": len(acts), "clicks (status ball, M1 out)": len(clicks),
           "held out": {"chosen per match": folds,
                        "end to end, chosen": round(float(pm['chosen e2e'].sum() / pm.n.sum()), 4),
                        "end to end, pipeline": round(float(pm['base e2e'].sum() / pm.n.sum()), 4),
                        "gain": round(float(gain), 4), "95% over matches": [round(float(lo), 4), round(float(hi), 4)],
                        "per match": pm.to_dict(orient="records")},
           "guard (ball after the kick)": {"hit, chosen": round(float(hit_ch), 4), "hit, pipeline": round(float(hit_b), 4),
                                           "holds (chosen >= pipeline - 0.02)": guard_ok},
           "bar met": bool(gain >= 0.03 and lo > 0 and guard_ok),
           "variant chosen on all 9 matches": chosen_all,
           "every variant (in sample)": full}
    (HERE / "link_variants.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "every variant (in sample)"}, indent=1))


if __name__ == "__main__":
    main()
