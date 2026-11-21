"""Why the foot-point gap is match-specific (plan.json): the labels' timing
(A, B), the gap's shape in the image (C), on the unsealed shots clips; the
picture check (D) is pictures.py.

    python diag.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
import eval_soccertrack_v2 as ev  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402

GPU = ROOT / ".cache/gpu"
CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
SKIP = {"117092", "128057", "132831"}  # M1: boxes in another frame; M7 / M9 sealed
STEP = 10
LAGS = np.arange(-12, 13)


def clip_pairs(w: str) -> pd.DataFrame:
    """Our foot point every 10th frame with its truth person (lag 0), and that
    person's truth bottom centre and pitch position at every lag."""
    m = pd.read_parquet(GPU / w / f"output_{w}" / "metric_v3s_link.parquet")
    p = m[(m.cls == "player") & (m.frame % STEP == 0)].copy()
    truth = ev.truth(CLIPS / f"{w}_gt.parquet")
    p["person"] = ev.match_feet(ev.boxes(p), truth[truth.frame % STEP == 0])
    p = p.dropna(subset=["person"]).reset_index(drop=True)
    g = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
    g = g[g.bx.notna()]
    tb = pd.DataFrame({"frame": g.frame.astype(int), "person": g.track_id.astype(int),
                       "bu": g.bx + g.bw / 2, "bv": g.by + g.bh, "tx": g.x, "ty": g.y}).set_index(["person", "frame"])
    key = pd.MultiIndex.from_arrays([p.person.astype(int), p.frame.astype(int)])
    cols = {"unit": w, "match": w.split("_")[1], "frame": p.frame.to_numpy(), "person": p.person.astype(int).to_numpy(),
            "u": p.px_raw.to_numpy(), "v": p.py_raw.to_numpy()}
    for k in LAGS:
        t = tb.reindex(pd.MultiIndex.from_arrays([key.get_level_values(0), key.get_level_values(1) + k]))
        cols[f"bu{k}"], cols[f"bv{k}"] = t.bu.to_numpy(), t.bv.to_numpy()
        cols[f"tx{k}"], cols[f"ty{k}"] = t.tx.to_numpy(), t.ty.to_numpy()
    # the truth person's horizontal image velocity (px per frame) at lag 0
    a = tb.reindex(pd.MultiIndex.from_arrays([key.get_level_values(0), key.get_level_values(1) + 2])).bu.to_numpy()
    b = tb.reindex(pd.MultiIndex.from_arrays([key.get_level_values(0), key.get_level_values(1) - 2])).bu.to_numpy()
    cols["vel"] = (a - b) / 4
    return pd.DataFrame(cols)


def metres(d: pd.DataFrame, u, v, k) -> np.ndarray:
    xy = st.image_to_pitch(np.c_[u, v], st.calibration(d.match.iloc[0]))
    return np.hypot(xy[:, 0] - d[f"tx{k}"].to_numpy(), xy[:, 1] - d[f"ty{k}"].to_numpy())


def affine(d: pd.DataFrame, du, dv):
    X = np.c_[np.ones(len(d)), d.u / 4096, d.v / 1080]
    ok = np.isfinite(du) & np.isfinite(dv)
    return np.linalg.lstsq(X[ok], du[ok], rcond=None)[0], np.linalg.lstsq(X[ok], dv[ok], rcond=None)[0]


def apply(d: pd.DataFrame, cu, cv):
    X = np.c_[np.ones(len(d)), d.u / 4096, d.v / 1080]
    return d.u.to_numpy() + X @ cu, d.v.to_numpy() + X @ cv


def main():
    units = sorted(p.name for p in GPU.iterdir() if p.name.startswith("st2_") and p.name.split("_")[1] not in SKIP
                   and (GPU / p.name / f"output_{p.name}" / "metric_v3s_link.parquet").exists()
                   and (CLIPS / f"{p.name}_gt.parquet").exists())
    per_clip, rows = {}, []
    for w in units:
        d = clip_pairs(w)
        dist = {int(k): float(np.nanmedian(np.hypot(d[f"bu{k}"] - d.u, d[f"bv{k}"] - d.v))) for k in LAGS}
        k = min(dist, key=dist.get)
        du0, dub = d["bu0"] - d.u, d[f"bu{k}"] - d.u
        ok = np.isfinite(du0) & np.isfinite(d.vel) & (du0.abs() < 20)
        slope = float(np.polyfit(d.vel[ok], du0[ok], 1)[0]) if ok.sum() > 50 else float("nan")
        per_clip[w] = {"match": d.match.iloc[0], "pairs": len(d), "best lag": k,
                       "median px at lag 0": round(dist[0], 2), "median px at best lag": round(dist[k], 2),
                       "median du at lag 0": round(float(np.nanmedian(du0)), 2), "median du at best lag": round(float(np.nanmedian(dub)), 2),
                       "median dv at lag 0": round(float(np.nanmedian(d["bv0"] - d.v)), 2),
                       "slope of du on velocity": round(slope, 2),
                       "median m at lag 0": round(float(np.nanmedian(metres(d, d.u, d.v, 0))), 3),
                       "median m at best lag": round(float(np.nanmedian(metres(d, d.u, d.v, k))), 3)}
        d = d.assign(lag=k, bu=d[f"bu{k}"], bv=d[f"bv{k}"], tx=d[f"tx{k}"], ty=d[f"ty{k}"])
        rows.append(d[["unit", "match", "frame", "person", "u", "v", "bu", "bv", "tx", "ty", "lag"]].assign(
            m0=metres(d, d.u, d.v, 0), mk=metres(d, d.u, d.v, k)))
        print(w, per_clip[w]["best lag"], per_clip[w]["median px at lag 0"], per_clip[w]["median px at best lag"], flush=True)
    df = pd.concat(rows, ignore_index=True)
    pc = pd.DataFrame(per_clip).T

    by_match = {}
    for m, g in pc.groupby("match"):
        du0, dub = g["median du at lag 0"].astype(float), g["median du at best lag"].astype(float)
        dm = df[df.match == m]
        by_match[m] = {"clips": len(g), "best lag: median, share of clips at 0": [float(g["best lag"].astype(int).median()),
                                                                                round(float((g["best lag"].astype(int) == 0).mean()), 2)],
                       "best lags": g["best lag"].astype(int).value_counts().sort_index().to_dict(),
                       "median of clips' du, lag 0 -> best": [round(float(du0.median()), 2), round(float(dub.median()), 2)],
                       "median of clips' dv at lag 0": round(float(g["median dv at lag 0"].astype(float).median()), 2),
                       "median slope of du on velocity": round(float(g["slope of du on velocity"].astype(float).median()), 2),
                       "pooled median m, lag 0 -> best": [round(float(np.nanmedian(dm.m0)), 3), round(float(np.nanmedian(dm.mk)), 3)]}

    # C: after the best lag, an affine gap in (u, v): within a match (half its clips -> the other half) vs from the other matches
    spatial = {}
    for m in sorted(df.match.unique()):
        dm = df[df.match == m]
        us = sorted(dm.unit.unique())
        if len(us) < 4:
            continue
        halves = [set(us[0::2]), set(us[1::2])]
        within = np.full(len(dm), np.nan)
        for a, b in ((0, 1), (1, 0)):
            tr, te = dm[dm.unit.isin(halves[a])], dm.unit.isin(halves[b]).to_numpy()
            cu, cv = affine(tr, (tr.bu - tr.u).to_numpy(), (tr.bv - tr.v).to_numpy())
            q = dm[te]
            uu, vv = apply(q, cu, cv)
            within[te] = metres(q.assign(tx0=q.tx, ty0=q.ty), uu, vv, 0)
        oth = df[df.match != m]
        cu, cv = affine(oth, (oth.bu - oth.u).to_numpy(), (oth.bv - oth.v).to_numpy())
        uu, vv = apply(dm, cu, cv)
        across = metres(dm.assign(tx0=dm.tx, ty0=dm.ty), uu, vv, 0)
        spatial[m] = {"median m: best lag, no correction": round(float(np.nanmedian(dm.mk)), 3),
                      "within the match (other half of its clips)": round(float(np.nanmedian(within)), 3),
                      "from the other matches": round(float(np.nanmedian(across)), 3),
                      "affine from all its clips (du, dv) at u, v = centre": [round(float(c @ [1, 0.5, 0.5]), 2) for c in
                                                                              affine(dm, (dm.bu - dm.u).to_numpy(), (dm.bv - dm.v).to_numpy())]}
    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0], "clips": len(units), "pairs": len(df),
           "sign": "du(k) = truth at frame f + k minus ours at f; labels that trail the video by L frames give a best lag of +L and a slope of du on velocity of -L",
           "pooled median m, lag 0 -> each clip's best lag": [round(float(np.nanmedian(df.m0)), 3), round(float(np.nanmedian(df.mk)), 3)],
           "by match": by_match, "C spatial (after the best lag)": spatial, "per clip": per_clip}
    (HERE / "diag.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("pooled median m, lag 0 -> each clip's best lag", "by match", "C spatial (after the best lag)")}, indent=1, default=str))


if __name__ == "__main__":
    main()
