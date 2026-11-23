"""shots, realistic (plan.json, hashed first): the v4 spotter's false alarms
in 21 min of ordinary play, and what they mean for a whole match.

    VEO_UNSEAL=... python run.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "eval"))
import eval_ap as EA  # noqa: E402

LS, V, V3, V4 = EA.LS, EA.V, EA.V3, EA.V4
WINDOWS = list(LS.TRAIN_WINDOWS)
WIN_TREE = ROOT / ".cache/gpu_shots_windows"
MATCH_MIN = 92.0


def check_plan():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit("plan.json does not match its sha256")
    return got


def window_tree():
    for w in WINDOWS:
        src = ROOT / ".cache/gpu" / w
        d = WIN_TREE / w / f"output_{w}"
        d.mkdir(parents=True, exist_ok=True)
        rows = src / f"output_{w}" / "v3s_rows.parquet"
        rows = rows if rows.exists() else src / "v3s_rows.parquet"
        for name, target in (("metric_v3s_link.parquet", src / f"output_{w}" / "metric_v3s_link_nograss.parquet"),
                             ("v3s_rows.parquet", rows)):
            if not target.exists():
                raise SystemExit(f"missing {target}")
            if not (d / name).exists():
                os.symlink(target, d / name)


def poisson_ci(k, t):
    lo = chi2.ppf(0.025, 2 * k) / 2 if k else 0.0
    hi = chi2.ppf(0.975, 2 * k + 2) / 2
    return lo / t, hi / t


def main():
    sha = check_plan()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from src.unseal import unsealed
    for m in V3.NEW_PLAYERS:
        if not unsealed(m, "shots realistic: M7 / M9 clips as spotter training data (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    # clips: v4 features (cached) and their leave-one-match-out detections (cached by eval_ap)
    us = EA.units()
    dc = EA.data("v4", us)
    cols = LS.cols_of(dc)
    labs_c = {u["name"]: (LS.labels(u), u["match"]) for u in us}
    ps = [EA.oof("v4", s) for s in EA.SEEDS]
    mean_p = ps[0].assign(p=np.mean([p.p.to_numpy() for p in ps], axis=0))
    fc = EA.flags(EA.detections(mean_p), labs_c)
    n_lab = sum(len(l) for l, _ in labs_c.values())
    clip_min = {}
    for n, (f, _, m) in dc.items():
        fr = f.index.to_numpy()
        clip_min[m] = clip_min.get(m, 0.0) + (fr.max() - fr.min()) / LS.FPS / 60
    matches = sorted(clip_min)
    # windows: features on the relinked tree, models fitted on the other matches' clips
    window_tree()
    LS.GPU = V.GPU = WIN_TREE
    wu = [{"name": w, "match": w.split("_")[1], "span": None} for w in WINDOWS]
    dw = {u["name"]: (V4.features_v4(u), LS.labels(u), u["match"]) for u in wu}
    base = LS.L.model()
    rows = []
    for u in wu:
        m, n = u["match"], u["name"]
        tr = [k for k in dc if dc[k][2] != m]
        X = pd.concat([dc[k][0][cols] for k in tr])
        y = pd.concat([LS.target(dc[k][0], dc[k][1]) for k in tr])
        p = []
        for s in EA.SEEDS:
            clf = HistGradientBoostingClassifier(**{**base.get_params(), "random_state": s}).fit(X, y)
            p.append(clf.predict_proba(dw[n][0][cols])[:, 1])
        rows.append(pd.DataFrame({"unit": n, "match": m, "frame": dw[n][0].index.to_numpy(), "p": np.mean(p, axis=0)}))
    pw = pd.concat(rows, ignore_index=True)
    labs_w = {n: (dw[n][1], dw[n][2]) for n in dw}
    fw = EA.flags(EA.detections(pw), labs_w)
    win_min = sum((dw[n][0].index.max() - dw[n][0].index.min()) / LS.FPS / 60 for n in dw)
    n_lab_w = sum(len(l) for l, _ in labs_w.values())
    # thresholds
    sc = np.sort(fc.score.to_numpy())[::-1]
    t_rate = float(sc[min(n_lab, len(sc)) - 1])
    o = fc.sort_values("score", ascending=False)
    cum = np.cumsum(o.right.to_numpy())
    k80 = int(np.argmax(cum >= 0.8 * n_lab)) if (cum >= 0.8 * n_lab).any() else None
    t_80 = float(o.score.iloc[k80]) if k80 is not None else None

    def at(t):
        rc = fc[fc.score >= t]
        rw = fw[fw.score >= t]
        found_c, fa_c = int(rc.right.sum()), int((~rc.right).sum())
        fa_w, hit_w = int((~rw.right).sum()), int(rw.right.sum())
        rate = fa_w / win_min
        rest_min = len(matches) * MATCH_MIN - sum(clip_min.values())
        fa_full = fa_c + rate * rest_min
        lo, hi = poisson_ci(fa_w, win_min)
        prec_full = found_c / max(found_c + fa_full, 1e-9)
        return {"threshold": round(t, 4), "clips: recall": round(found_c / n_lab, 3),
                "clips: precision": round(found_c / max(found_c + fa_c, 1), 3),
                "windows: false alarms": fa_w, "windows: labels found": f"{hit_w} / {n_lab_w}",
                "windows: false alarms per minute": round(rate, 3),
                "95% interval per minute": [round(lo, 3), round(hi, 3)],
                "whole match (extrapolated): shots found per match": round(found_c / len(matches), 1),
                "whole match: false alarms per match": round(fa_full / len(matches), 1),
                "whole match: false alarms per match, 95%": [round((fa_c + lo * rest_min) / len(matches), 1),
                                                             round((fa_c + hi * rest_min) / len(matches), 1)],
                "whole match: precision": round(prec_full, 3),
                "whole match: precision, 95%": [round(found_c / (found_c + fa_c + hi * rest_min), 3),
                                                round(found_c / (found_c + fa_c + lo * rest_min), 3)]}

    curve = [at(t) for t in sorted({round(x, 3) for x in np.quantile(fc.score, np.linspace(0.05, 0.95, 19))})]
    out = {"plan sha256": sha, "clip labels": n_lab, "clip minutes per match": {m: round(v, 1) for m, v in clip_min.items()},
           "window minutes": round(win_min, 1), "window labels": n_lab_w,
           "(a) rate threshold": at(t_rate), "(b) recall 0.80 on the clips": at(t_80) if t_80 is not None else None,
           "curve": curve}
    (HERE / "realistic.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    main()
