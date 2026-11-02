"""Score the detector variants (plan_v2.json) on the action frames: raw
detections inside the pitch outline, matched to the truth by feet; per
variant the actor detected / taken / missed, merged boxes, the control and
unmatched detections per frame; the selection rule.

    VEO_UNSEAL=... python score_det.py DET_DIR     # DET_DIR: the kernel's det/<unit>.parquet
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import eval_soccertrack_v2 as ev  # noqa: E402

CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
SEALED = {"128057", "132831"}
VARIANTS = {"V0": ("s_c10", 0.20), "V1": ("s_iou85", 0.20), "V2": ("s_c10", 0.10),
            "V3": ("m", 0.20), "V4": ("x", 0.20), "V5": ("s_tta", 0.20)}


def unit(w: str, det: pd.DataFrame, acts: pd.DataFrame) -> list:
    info = json.loads((CLIPS / f"{w}.json").read_text())
    poly = np.asarray(info["pitch"], np.float32).reshape(-1, 1, 2)
    gt = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
    gt = gt[gt.bx.notna()]
    truth = ev.truth(CLIPS / f"{w}_gt.parquet")
    out = []
    for v, (ps, conf) in VARIANTS.items():
        d = det[(det["pass"] == ps) & (det.conf >= conf)].copy()
        fu, fv = (d.x0 + d.x1) / 2, d.y1
        d = d[[cv2.pointPolygonTest(poly, (float(u), float(y)), False) >= 0 for u, y in zip(fu, fv)]]
        frames = sorted(set(acts.f))
        tr = truth[truth.frame.isin(frames)]
        d = d[d.frame.isin(frames)].reset_index(drop=True)
        d["person"] = ev.match_feet(d, tr) if len(d) else pd.Series(dtype=float)
        by_f, tr_f = {f: g for f, g in d.groupby("frame")}, {f: g for f, g in tr.groupby("frame")}
        unmatched = {f: int(g.person.isna().sum()) for f, g in by_f.items()}
        for r in acts.itertuples():
            a = gt[(gt.frame == r.f) & (gt.player_id.astype(str) == str(int(r.player_id)))]
            t = tr_f.get(r.f)
            if a.empty or t is None:
                continue
            actor = int(a.track_id.iloc[0])
            g = by_f.get(r.f, d.iloc[:0])
            au, av = float(a.bx.iloc[0] + a.bw.iloc[0] / 2), float(a.by.iloc[0] + a.bh.iloc[0])
            dist = np.hypot(((g.x0 + g.x1) / 2).to_numpy() - au, g.y1.to_numpy() - av)
            hit = g[g.person == actor]
            if len(hit):
                kind, row = "detected", hit.iloc[0]
            elif len(dist) and (dist <= ev.FOOT_PX).any():
                kind, row = "taken", g.iloc[int(dist.argmin())]
            else:
                kind, row = "missed", None
            merged = None
            if row is not None:
                tu, tv = ((t.x0 + t.x1) / 2).to_numpy(), t.y1.to_numpy()
                merged = bool((((tu >= row.x0) & (tu <= row.x1) & (tv >= row.y0) & (tv <= row.y1 + 5))).sum() >= 2)
            others = t[(t.person != actor) & (t.role == "player")]
            ctrl = float(others.person.isin(set(g.person.dropna())).mean()) if len(others) else np.nan
            out.append({"unit": w, "match": w.split("_")[1], "frame": r.f, "variant": v, "kind": kind,
                        "merged": merged, "control": ctrl, "unmatched": unmatched.get(r.f, 0)})
    return out


def main():
    from src.unseal import unsealed
    for m in sorted(SEALED):
        if not unsealed(m, "near_ball plan_v2: scoring detector variants on the opened M7 / M9 clips"):
            raise SystemExit("VEO_UNSEAL needed")
    det_dir = Path(sys.argv[1])
    acts = pd.read_parquet(HERE / "actions.parquet")[["unit", "frame"]]
    lab = pd.concat([pd.read_csv(CLIPS / f"{w}_events.csv").assign(unit=w) for w in sorted(set(acts.unit))])
    acts = acts.merge(lab[["unit", "frame", "player_id"]].dropna(), on=["unit", "frame"]).drop_duplicates(["unit", "frame", "player_id"])
    acts["f"] = acts.frame - acts.frame % 2
    rows, missing = [], []
    for i, (w, a) in enumerate(acts.groupby("unit")):
        if not (det_dir / f"{w}.parquet").exists():
            missing.append(w)
            continue
        rows += unit(w, pd.read_parquet(det_dir / f"{w}.parquet"), a)
        if i % 25 == 0:
            print(i, w, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(HERE / "variants.parquet")
    res = {}
    for v, g in df.groupby("variant"):
        k = g.kind.value_counts(normalize=True)
        fr = g.drop_duplicates(["unit", "frame"])
        res[v] = {"actions": len(g), "detected": round(float(k.get("detected", 0)), 4), "taken": round(float(k.get("taken", 0)), 4),
                  "missed": round(float(k.get("missed", 0)), 4), "merged": round(float(g.merged.dropna().mean()), 4),
                  "control": round(float(g.control.mean()), 4), "unmatched per frame": round(float(fr.unmatched.mean()), 3)}
    guard = 1.2 * res["V0"]["unmatched per frame"]
    ok = [v for v in res if res[v]["unmatched per frame"] <= guard]
    best = max(ok, key=lambda v: res[v]["detected"])
    matches = sorted(df.match.unique())
    per = {m: {v: (g.kind == "detected").sum() for v, g in df[df.match == m].groupby("variant")} for m in matches}
    n = {m: (df[(df.match == m) & (df.variant == "V0")]).shape[0] for m in matches}
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(5000):
        s = rng.choice(matches, len(matches))
        diffs.append(sum(per[m][best] - per[m]["V0"] for m in s) / sum(n[m] for m in s))
    gain = res[best]["detected"] - res["V0"]["detected"]
    out = {"plan_v2 sha256": (HERE / "plan_v2.json.sha256").read_text().split()[0], "clips missing": missing,
           "variants": res, "guard (unmatched per frame <=)": round(guard, 3), "selected": best,
           "gain over V0 (actor detected)": round(gain, 4),
           "95% bootstrap over matches": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
           "goes to the full pipeline": bool(best != "V0" and gain >= 0.03)}
    (HERE / "variants.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
