"""identity step 2b what-if (plan2.json, hashed first): the fusion with every
crop's number read at one size band's rates (a finer camera).

    VEO_UNSEAL=... python what_if.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fusion as F

HERE = Path(__file__).resolve().parent


def check():
    out = {}
    for name in ("plan.json", "plan2.json"):
        want = (HERE / f"{name}.sha256").read_text().split()[0]
        got = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"{name} does not match its sha256")
        out[name] = got
    return out


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def run(seg, Z, lineup, rates):
    rng = np.random.default_rng(0)
    res = {m: {"fusion": [], "no cues": [], "numbered": [], "dist fusion": []} for m in sorted(seg.match.unique())}
    for _ in range(F.DRAWS):
        num = F.draw_numbers(seg, rates, lineup, rng)
        gf, g0 = F.assign(seg, Z, num, lineup, True), F.assign(seg, Z, num, lineup, False)
        for m in res:
            idx = seg.index[seg.match == m]
            res[m]["fusion"].append(float((gf.loc[idx] == seg.player_id.loc[idx]).mean()))
            res[m]["no cues"].append(float((g0.loc[idx] == seg.player_id.loc[idx]).mean()))
            res[m]["numbered"].append(float(pd.Series(num, index=seg.index).loc[idx].notna().mean()))
            res[m]["dist fusion"].append(F.distance_ok(seg.loc[idx], gf))
    return {m: {k: round(float(np.mean(v)), 3) for k, v in d.items()} for m, d in res.items()}


def main():
    sha = check()
    from src.unseal import unsealed
    for m in F.J.T.S.V3.NEW_PLAYERS:
        if not unsealed(m, "identity step 2b what-if: 132831 cues (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    seg = pd.concat([F.per_video(k) for k in range(len(F.J.SOURCES))], ignore_index=True)
    Z = F.zscore(seg)
    lineup = {k: sorted(g.player_id.unique().tolist()) for k, g in seg.groupby(["match", "side"])}
    sc = json.loads((F.ROOT / "research/session_scratchpad/jersey_read/score.json").read_text())["part 2 per band (our box height)"]
    obs = {b: (v["correct"], v["read"] - v["correct"]) for b, v in sc.items()}
    lo, hi = wilson(13, 30)
    variants = {"observed": obs,
                "all 60-80 px": {b: obs["60-80"] for b in obs},
                "all >= 80 px": {b: obs[">=80"] for b in obs},
                ">= 80 px, low": {b: (round(lo, 3), obs[">=80"][1]) for b in obs},
                ">= 80 px, high": {b: (round(hi, 3), obs[">=80"][1]) for b in obs}}
    out = {"sha256": sha, "rates (correct, wrong) per band, per variant": variants, "results": {}}
    parts = HERE / "what_if_parts"  # one file per finished variant (a restart skips them)
    parts.mkdir(exist_ok=True)
    for name, rates in variants.items():
        f = parts / f"{name.replace(' ', '_').replace('>=', 'ge')}.json"
        if not f.exists():
            f.write_text(json.dumps(run(seg, Z, lineup, rates)))
        out["results"][name] = json.loads(f.read_text())
        print(name, out["results"][name], flush=True)
    need = [n for n in ("all 60-80 px", "all >= 80 px") if all(out["results"][n][m]["fusion"] >= 0.80 for m in ("117093", "132831"))]
    out["smallest band reaching 0.80 on both"] = need[0] if need else None
    (HERE / "what_if.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
