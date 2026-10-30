"""Learned shots v3 (plan.json): the v2 spotter on the relinked pitch tracks
(no grass test) against the old ones, same units, nested leave-one-match-out.

    VEO_UNSEAL=... python learn_v3.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "v2"))
import learn_shots as LS  # noqa: E402
import learn_v2 as V  # noqa: E402

ROOT = LS.ROOT
NEW = ROOT / ".cache/gpu_shots3"        # per unit: relinked tracks under the usual names
NEW_PLAYERS = {"128057", "132831"}


def new_tree(us):
    keep = []
    for u in us:
        w = u["name"]
        src = ROOT / (".cache/gpu_nograss" if u["match"] in NEW_PLAYERS else ".cache/gpu") / w / f"output_{w}"
        relinked = src / "metric_v3s_link_nograss.parquet"
        old = ROOT / ".cache/gpu" / w / f"output_{w}" / "metric_v3s_link.parquet"
        if not (relinked.exists() and old.exists()):
            continue
        d = NEW / w / f"output_{w}"
        d.mkdir(parents=True, exist_ok=True)
        for name, target in (("metric_v3s_link.parquet", relinked), ("v3s_rows.parquet", src / "v3s_rows.parquet")):
            if not (d / name).exists():
                os.symlink(target, d / name)
        keep.append(u)
    return keep


def run(us, gpu):
    LS.GPU = V.GPU = gpu
    d = {u["name"]: (V.features_v2(u), LS.labels(u), u["match"]) for u in us}
    names = list(d)
    probs, thr = LS.nested(d, names, LS.cols_of(d))
    return LS.score(d, names, probs, thr)


def main():
    from src.unseal import unsealed
    for m in NEW_PLAYERS:
        if not unsealed(m, "shots v3: M7 / M9 clips as training data"):
            raise SystemExit("VEO_UNSEAL needed")
    us = [u for u in V.units() if (LS.GPU / u["name"]).exists()]
    us = new_tree(us)
    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0], "units": len(us),
           "matches": sorted({u["match"] for u in us})}
    for name, gpu in (("old tracks", ROOT / ".cache/gpu"), ("no grass tests", NEW)):
        out[name] = run(us, gpu)
        print(name, {k: out[name][k] for k in ("tp", "fp", "fn", "recall", "precision")}, flush=True)
    a, b = out["old tracks"]["per match [tp, fp, fn]"], out["no grass tests"]["per match [tp, fp, fn]"]
    ms = sorted(a)
    diff = np.array([b[m][0] - a[m][0] for m in ms], float)
    rng = np.random.default_rng(0)
    boots = [diff[rng.integers(0, len(ms), len(ms))].sum() for _ in range(5000)]
    out["hits new - old"] = {"total": float(diff.sum()), "bootstrap 95% over matches":
                             [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    (HERE / "cv_v3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["hits new - old"]))


if __name__ == "__main__":
    main()
