"""Learned foot-point correction (plan.json): the image offset from our
detection's foot point to the truth box's bottom centre, predicted from the
box, validated leave one match out in pitch metres.

    VEO_UNSEAL=... python footpoint.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import eval_soccertrack_v2 as ev  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.unseal import unsealed  # noqa: E402

GPU = ROOT / ".cache/gpu"
CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
SKIP = {"117092"}                       # M1: truth boxes in another frame
SEALED = {"128057", "132831"}
STEP = 10
FEATS = ["u", "v", "h", "w", "hw", "W"]
if "--box-only" in sys.argv:
    FEATS = ["h", "w", "hw"]


def pairs(w: str) -> pd.DataFrame:
    info = json.loads((CLIPS / f"{w}.json").read_text())
    m = pd.read_parquet(GPU / w / f"output_{w}" / "metric_v3s_link.parquet")
    p = m[(m.cls == "player") & (m.frame % STEP == 0)].copy()
    g = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
    g = g[(g.frame % STEP == 0) & g.bx.notna() & g.x.notna()]
    truth = ev.truth(CLIPS / f"{w}_gt.parquet")
    truth = truth[truth.frame % STEP == 0]
    person = ev.match_feet(ev.boxes(p), truth)
    p = p.assign(person=person).dropna(subset=["person"])
    g = g.assign(person=g.track_id.astype(int))[["frame", "person", "bx", "by", "bw", "bh", "x", "y"]]
    d = p.merge(g, on=["frame", "person"], suffixes=("_det", ""))
    return pd.DataFrame({
        "match": str(info["match"]), "unit": w,
        "u": d.px_raw, "v": d.py_raw, "h": d.crop_h, "w": d.crop_w, "hw": d.crop_h / d.crop_w,
        "W": float(info["width"]),
        "du": d.bx + d.bw / 2 - d.px_raw, "dv": d.by + d.bh - d.py_raw,
        "tx": d.x, "ty": d.y})


def err(df: pd.DataFrame, u, v) -> np.ndarray:
    out = np.full(len(df), np.nan)
    for m, idx in df.groupby("match").indices.items():
        xy = st.image_to_pitch(np.c_[u[idx], v[idx]], st.calibration(m))
        out[idx] = np.hypot(xy[:, 0] - df.tx.to_numpy()[idx], xy[:, 1] - df.ty.to_numpy()[idx])
    return out


def fit(train: pd.DataFrame):
    return {k: HistGradientBoostingRegressor(random_state=0).fit(train[FEATS], train[k]) for k in ("du", "dv")}


def main():
    for m in SEALED:
        if not unsealed(m, "footpoint: truth boxes of the shots clips"):
            raise SystemExit("VEO_UNSEAL needed for M7 / M9")
    units = sorted(p.name for p in GPU.iterdir() if p.name.split("_")[1] not in SKIP
                   and (GPU / p.name / f"output_{p.name}" / "metric_v3s_link.parquet").exists()
                   and (CLIPS / f"{p.name}_gt.parquet").exists())
    df = pd.concat([pairs(w) for w in units], ignore_index=True)
    print(len(units), "units,", len(df), "pairs", flush=True)
    before = err(df, df.u.to_numpy(), df.v.to_numpy())
    after = np.full(len(df), np.nan)
    for m in sorted(df.match.unique()):
        te = (df.match == m).to_numpy()
        mdl = fit(df[~te])
        q = df[te]
        after[te] = err(q.reset_index(drop=True), (q.u + mdl["du"].predict(q[FEATS])).to_numpy(),
                        (q.v + mdl["dv"].predict(q[FEATS])).to_numpy())
    df["before"], df["after"] = before, after

    def stats(g):
        return {"n": len(g), "median before": round(float(np.nanmedian(g.before)), 3),
                "median after": round(float(np.nanmedian(g.after)), 3),
                "p90 before": round(float(np.nanquantile(g.before, 0.9)), 3),
                "p90 after": round(float(np.nanquantile(g.after, 0.9)), 3),
                "median offset px (du, dv)": [round(float(g.du.median()), 1), round(float(g.dv.median()), 1)]}

    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0], "units": len(units),
           "pooled (held out)": stats(df), "per match (held out)": {m: stats(g) for m, g in df.groupby("match")}}
    pm = out["per match (held out)"]
    out["adopt"] = bool(out["pooled (held out)"]["median after"] <= 0.7 * out["pooled (held out)"]["median before"]
                        and all(v["median after"] <= v["median before"] for v in pm.values()))
    print(json.dumps(out, indent=1))
    (HERE / ("footpoint_box_only.json" if "--box-only" in sys.argv else "footpoint.json")).write_text(json.dumps(out, indent=1))
    if "--box-only" in sys.argv:
        return
    final = fit(df)
    (HERE / "model.pkl").write_bytes(pickle.dumps({"models": final, "feats": FEATS, "matches": sorted(df.match.unique())}))


if __name__ == "__main__":
    main()
