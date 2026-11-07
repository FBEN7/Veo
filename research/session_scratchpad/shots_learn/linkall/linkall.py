"""shots, linkall (plan.json, hashed first): build the relinked tree, then
score the v4 spotter on it with eval_ap's measure.

    VEO_UNSEAL=... python linkall.py
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "eval"))
import eval_ap as E  # noqa: E402
from src import ball_link  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.pixel_scale import to_pitch_metres  # noqa: E402

TREE = ROOT / ".cache/gpu_shots_linkall"
TAU = 0.03


def build(u):
    w = u["name"]
    src = E.V3.NEW / w / f"output_{w}"
    dst = TREE / w / f"output_{w}"
    out = dst / "metric_v3s_link.parquet"
    if out.exists():
        return w, None
    dst.mkdir(parents=True, exist_ok=True)
    cal = st.calibration(u["match"])
    m = pd.read_parquet(src / "metric_v3s_link.parquet")
    v = pd.read_parquet(src / "v3s_rows.parquet")
    players = m[m.cls == "player"].assign(px=lambda d: d.px_raw, py=lambda d: d.py_raw)
    c = v[v.confidence >= TAU].sort_values("confidence", ascending=False, kind="stable").groupby("frame", sort=False).head(4)
    kept = ball_link.link(c, cal, players, fps=25, tau=TAU, c_null=1.0, c_static=2.0)
    b = kept.assign(px_raw=kept.px, py_raw=kept.py, cls="ball", track_id=-1, team="ball", crop_w=np.nan, grass_frac=np.nan)
    b = to_pitch_metres(b, lambda uv: st.image_to_pitch(uv, cal))
    new = pd.concat([m[m.cls != "ball"], b[[c for c in m.columns if c in b.columns]]], ignore_index=True)
    new = new.sort_values(["frame", "cls"], kind="stable").reset_index(drop=True)
    tmp = dst / "metric_v3s_link.tmp.parquet"
    new.to_parquet(tmp)
    os.replace(tmp, out)
    if not (dst / "v3s_rows.parquet").exists():
        os.symlink(os.path.realpath(src / "v3s_rows.parquet"), dst / "v3s_rows.parquet")
    return w, {"old ball rows": int((m.cls == "ball").sum()), "new ball rows": int(len(b))}


def main():
    from src.unseal import unsealed
    for m in sorted(E.V3.NEW_PLAYERS):
        if not unsealed(m, "shots linkall: relinked ball on the M7 / M9 clips (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    us = E.units()
    with Pool(os.cpu_count()) as pool:
        counts = dict(pool.imap_unordered(build, us))
    print("tree built", len(counts), flush=True)
    E.TREES["linkall"] = TREE
    labs = {u["name"]: (E.LS.labels(u), u["match"]) for u in us}
    matches = sorted({u["match"] for u in us})
    ens, res = {}, {}
    for v in ("v4", "linkall"):
        ps = [E.oof(v, s) for s in E.SEEDS]
        print(v, "oof done", flush=True)
        mean_p = ps[0].assign(p=np.mean([p.p.to_numpy() for p in ps], axis=0))
        ens[v] = E.detections(mean_p)
        ap_e, hit_e, n = E.event_ap(ens[v], labs)
        res[v] = {"AP per seed": [round(E.event_ap(E.detections(p), labs)[0], 4) for p in ps],
                  "AP of the seed average": round(ap_e, 4), "hits at the rate threshold": hit_e, "labels": n,
                  "AP per match": {m: round(E.event_ap(ens[v], labs, [m])[0], 3) for m in matches}}
    n_of = {m: sum(len(l) for l, mm in labs.values() if mm == m) for m in matches}
    by = {v: {m: g for m, g in E.flags(ens[v], labs).groupby("match")} for v in ens}

    def ap_draw(v, ms):
        g = [by[v][m] for m in ms if m in by[v]]
        return E.ap_of(np.concatenate([x.score.to_numpy() for x in g]), np.concatenate([x.right.to_numpy() for x in g]),
                       sum(n_of[m] for m in ms))[0]

    rng = np.random.default_rng(0)
    draws = []
    for _ in range(5000):
        ms = rng.choice(matches, len(matches))
        draws.append(ap_draw("linkall", ms) - ap_draw("v4", ms))
    diff = res["linkall"]["AP of the seed average"] - res["v4"]["AP of the seed average"]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    ball = pd.DataFrame(counts).T.dropna()
    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0], "units": len(us),
           "ball rows (all units)": {k: int(ball[k].sum()) for k in ball.columns},
           "variants": res, "AP difference (linkall - v4)": round(float(diff), 4),
           "95% bootstrap over matches": [round(float(lo), 4), round(float(hi), 4)],
           "bar met (>= +0.04, lower bound > 0)": bool(diff >= 0.04 and lo > 0)}
    (HERE / "linkall.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
