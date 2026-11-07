"""A steadier measure for the shots spotter (protocol.json): leave-one-match-
out probabilities per seed, event AP of the pooled peaks, and the noise
study over seeds and variants.

    VEO_UNSEAL=... python eval_ap.py run VARIANT SEED     # OOF probabilities, cached
    python eval_ap.py report
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in ("", "v2", "v3", "v4"):
    sys.path.insert(0, str(HERE.parent / p))
import learn_shots as LS  # noqa: E402
import learn_v2 as V  # noqa: E402
import learn_v3 as V3  # noqa: E402
import learn_v4 as V4  # noqa: E402

OUT = ROOT / ".cache/shots_eval"
TREES = {"v4": V3.NEW, "control": ROOT / ".cache/gpu_shots_hybrid_control",
         "hybrid": ROOT / ".cache/gpu_shots_hybrid"}
CAND = {"candB": ROOT / ".cache/shots_cand/B", "candC": ROOT / ".cache/shots_cand/C"}
VARIANTS = ("v4", "control", "hybrid", "candB", "candC")
SEEDS = (0, 1, 2)


def units() -> list[dict]:
    return V3.new_tree([u for u in V.units() if (LS.GPU / u["name"]).exists()])


def data(variant: str, us: list[dict]) -> dict:
    if variant in CAND:
        f = {u["name"]: pd.read_parquet(CAND[variant] / f"{u['name']}.parquet") for u in us}
    else:
        LS.GPU = V.GPU = TREES[variant]
        f = {u["name"]: V4.features_v4(u) for u in us}
    return {u["name"]: (f[u["name"]], LS.labels(u), u["match"]) for u in us}


def oof(variant: str, seed: int) -> pd.DataFrame:
    path = OUT / f"{variant}_s{seed}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from src.unseal import unsealed
    for m in V3.NEW_PLAYERS:
        if not unsealed(m, f"shots eval: {variant} seed {seed} (M7 / M9 clips as spotter training data)"):
            raise SystemExit("VEO_UNSEAL needed")
    base = LS.L.model()
    LS.L.model = lambda: HistGradientBoostingClassifier(**{**base.get_params(), "random_state": seed})
    d = data(variant, units())
    names = list(d)
    probs = LS.oof(d, names, LS.cols_of(d))
    out = pd.concat([pd.DataFrame({"unit": n, "match": d[n][2], "frame": d[n][0].index.to_numpy(), "p": probs[n]})
                     for n in names], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


def detections(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n, m), g in p.groupby(["unit", "match"]):
        fr, pr = g.frame.to_numpy(), g.p.to_numpy()
        s = pd.Series(pr).rolling(3, center=True, min_periods=1).mean().to_numpy()
        for i in LS.peaks(fr, pr, 0.0):
            rows.append((n, m, fr[i] / LS.FPS, s[i]))
    return pd.DataFrame(rows, columns=["unit", "match", "t", "score"])


def flags(det: pd.DataFrame, labs: dict) -> pd.DataFrame:
    """Each detection's right / wrong: in score order, the nearest not-yet-
    matched label of its unit within 2 s. Units are matched apart, so the
    flags do not depend on which other matches are pooled."""
    used = {n: np.zeros(len(l), bool) for n, (l, _) in labs.items()}
    right = []
    det = det.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    for r in det.itertuples():
        lab = np.asarray(labs[r.unit][0])
        ok = False
        if len(lab):
            d = np.abs(lab - r.t)
            d[used[r.unit]] = np.inf
            j = int(np.argmin(d))
            if d[j] <= LS.TOL_S:
                used[r.unit][j] = ok = True
        right.append(ok)
    return det.assign(right=right)


def ap_of(score: np.ndarray, right: np.ndarray, n_lab: int) -> tuple[float, int]:
    """(event AP, right detections among the n_lab best)."""
    o = np.argsort(-score, kind="stable")
    r = right[o]
    if not n_lab:
        return float("nan"), 0
    prec = np.cumsum(r) / np.arange(1, len(r) + 1)
    return float((prec * r).sum() / n_lab), int(r[:n_lab].sum())


def event_ap(det: pd.DataFrame, labs: dict, matches=None) -> tuple[float, int, int]:
    """(AP, right detections at the rate threshold, labels) over `matches`."""
    f = flags(det, labs)
    keep = f.match.isin(matches) if matches is not None else np.ones(len(f), bool)
    n_lab = sum(len(l) for l, m in labs.values() if matches is None or m in set(matches))
    ap, hits = ap_of(f.score.to_numpy()[keep], f.right.to_numpy()[keep], n_lab)
    return ap, hits, n_lab


def report():
    us = units()
    labs = {u["name"]: (LS.labels(u), u["match"]) for u in us}
    matches = sorted({u["match"] for u in us})
    res, ens = {}, {}
    for v in VARIANTS:
        ps = [oof(v, s) for s in SEEDS]
        aps = [event_ap(detections(p), labs) for p in ps]
        mean_p = ps[0].assign(p=np.mean([p.p.to_numpy() for p in ps], axis=0))
        ens[v] = detections(mean_p)
        ap_e, hit_e, n = event_ap(ens[v], labs)
        res[v] = {"AP per seed": [round(a[0], 4) for a in aps], "hits at the rate threshold per seed": [a[1] for a in aps],
                  "AP of the seed average": round(ap_e, 4), "hits of the seed average": hit_e, "labels": n,
                  "AP per match (seed average)": {m: round(event_ap(ens[v], labs, [m])[0], 3) for m in matches}}
    rng = np.random.default_rng(0)
    idx = [rng.choice(matches, len(matches)) for _ in range(5000)]
    n_of = {m: sum(len(l) for l, mm in labs.values() if mm == m) for m in matches}
    by = {v: {m: g for m, g in flags(ens[v], labs).groupby("match")} for v in VARIANTS}

    def ap_draw(v, ms):
        g = [by[v][m] for m in ms if m in by[v]]
        return ap_of(np.concatenate([x.score.to_numpy() for x in g]), np.concatenate([x.right.to_numpy() for x in g]),
                     sum(n_of[m] for m in ms))[0]

    cmp = {}
    for v in VARIANTS[1:]:
        dd = [ap_draw(v, ms) - ap_draw("v4", ms) for ms in idx]
        cmp[f"{v} - v4"] = {"AP difference": round(res[v]["AP of the seed average"] - res["v4"]["AP of the seed average"], 4),
                            "95% bootstrap over matches": [round(float(np.percentile(dd, 2.5)), 4),
                                                           round(float(np.percentile(dd, 97.5)), 4)]}
    seed_sd = {v: round(float(np.std(res[v]["AP per seed"], ddof=1)), 4) for v in VARIANTS}
    out = {"protocol sha256": (HERE / "protocol.json.sha256").read_text().split()[0],
           "variants": res, "against v4 (seed-averaged probabilities)": cmp, "AP spread over seeds (sd)": seed_sd}
    (HERE / "eval_ap.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "variants"}, indent=1))
    print({v: (res[v]["AP per seed"], res[v]["AP of the seed average"], res[v]["hits at the rate threshold per seed"]) for v in VARIANTS})


if __name__ == "__main__":
    if sys.argv[1] == "run":
        p = oof(sys.argv[2], int(sys.argv[3]))
        print(sys.argv[2], sys.argv[3], len(p))
    else:
        report()
