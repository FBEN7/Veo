"""Distance run per team (plan.json): our tracks' 1-s speeds, per-detection
kit teams, 10 x the team's mean speed each second, against the truth's
outfield path lengths and a constant rate, per match.

    VEO_UNSEAL=... python team_distance.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "research/session_scratchpad/team_passes"))
import kit_v3 as K  # noqa: E402
import team_passes as T  # noqa: E402

KIT = ROOT / ".cache/team_passes_kit/kit"
CACHE = ROOT / ".cache/team_distance"
LO, HI, STEP = 300, 1125, 25
CENTRES = np.arange(LO, HI - STEP + 1, STEP) + 12      # the middle of each 1-s interval
OUTFIELD = 10
SIDES = ("left", "right")


def speeds(p: pd.DataFrame) -> pd.Series:
    """Per player row (even frames): speed from the track's positions smoothed
    over 1 s (13 rows), 12 frames before and after; NaN where they are missing."""
    out = pd.Series(np.nan, index=p.index)
    for _, g in p.groupby("track_id"):
        g = g.drop_duplicates("frame").sort_values("frame")
        fr = g.frame.to_numpy()
        xy = g[["x", "y"]].set_axis(fr)
        full = xy.reindex(np.arange(fr.min(), fr.max() + 1, 2))
        sm = full.rolling(13, center=True, min_periods=13).mean()
        a, b = sm.shift(6), sm.shift(-6)          # 12 frames before / after
        v = np.hypot(b.x - a.x, b.y - a.y) / (24 / 25)
        out.loc[g.index] = v.reindex(fr).to_numpy()
    return out


def ours(w: str) -> tuple:
    """(distance per side with our teams, with the truth's teams, seconds)."""
    cache = CACHE / f"{w}.parquet"
    if cache.exists():
        p = pd.read_parquet(cache)
    else:
        new = K.kit_teams(w, pd.read_parquet(KIT / f"{w}.parquet"))
        tm = K.team_map_of(w, new)
        p = new[(new.cls == "player") & (new.frame >= LO - 20) & (new.frame <= HI + 20)].reset_index()
        p["side"] = p.team.map(tm)
        p["speed"] = speeds(p)
        # the truth's side of each row at the centres: nearest truth person within 3 m
        g = pd.read_parquet(T.CLIPS / f"{w}_gt.parquet")
        g = g[g.x.notna()]
        gb = {f: d for f, d in g.groupby("frame")}
        p["true_side"] = None
        for f, d in p[p.frame.isin(CENTRES - CENTRES % 2)].groupby("frame"):
            t = gb.get(f)
            if t is None:
                continue
            dist = np.hypot(d.x.to_numpy()[:, None] - 52.5 - t.x.to_numpy()[None],
                            d.y.to_numpy()[:, None] - 34.0 - t.y.to_numpy()[None])
            j = dist.argmin(1)
            ok = dist[np.arange(len(d)), j] <= 3.0
            p.loc[d.index[ok], "true_side"] = t.side.to_numpy()[j[ok]]
        p = p[["frame", "track_id", "side", "true_side", "speed"]]
        CACHE.mkdir(parents=True, exist_ok=True)
        p.to_parquet(cache)
    c = p[p.frame.isin(CENTRES - CENTRES % 2) & p.speed.notna()]

    def dist(col):
        out = {}
        for s in SIDES:
            per = c[c[col] == s].groupby("frame").speed.mean().reindex(CENTRES - CENTRES % 2)
            per = per.fillna(per.mean()) if per.notna().any() else per.fillna(0.0)
            out[s] = float(OUTFIELD * per.sum())          # 1 s per centre
        return out
    return dist("side"), dist("true_side"), len(CENTRES)


def truth(w: str) -> dict:
    g = pd.read_parquet(T.CLIPS / f"{w}_gt.parquet")
    g = g[(g.role == "player") & g.x.notna()]
    out = {s: 0.0 for s in SIDES}
    samples = np.arange(LO, HI + 1, STEP)
    for (pid, side), d in g.groupby(["player_id", "side"]):
        d = d.drop_duplicates("frame").set_index("frame").sort_index()[["x", "y"]]
        sm = d.reindex(np.arange(d.index.min(), d.index.max() + 1)).rolling(25, center=True, min_periods=25).mean()
        s = sm.reindex(samples)
        out[side] += float(np.nansum(np.hypot(np.diff(s.x), np.diff(s.y))))
    return out


def main():
    from src.unseal import unsealed
    for m in T.SEALED:
        if not unsealed(m, "team distance: M7 / M9 clips (scored data)"):
            raise SystemExit("VEO_UNSEAL needed")
    units = sorted(p.name for p in T.TREE.iterdir() if p.name.startswith("st2_"))
    rows = []
    for i, w in enumerate(units):
        o, ot, n = ours(w)
        t = truth(w)
        for s in SIDES:
            rows.append((w, w.split("_")[1], s, o[s], ot[s], t[s], n))
        if i % 25 == 0:
            print(i, w, flush=True)
    df = pd.DataFrame(rows, columns=["unit", "match", "side", "ours", "ours_true_teams", "truth", "seconds"])
    df.to_parquet(HERE / "distance.parquet")
    pm = df.groupby(["match", "side"])[["ours", "ours_true_teams", "truth", "seconds"]].sum()
    matches = sorted(df.match.unique())
    rate = {m: pm.drop(index=m, level=0).truth.sum() / pm.drop(index=m, level=0).seconds.sum() for m in matches}
    e_d, e_dt, c_d, e_s, e_st, c_s = [], [], [], [], [], []
    for m in matches:
        r = pm.loc[m]
        rel = lambda col: np.mean([abs(r.loc[s, col] - r.loc[s, "truth"]) / r.loc[s, "truth"] for s in SIDES])  # noqa: E731
        e_d.append(rel("ours"))
        e_dt.append(rel("ours_true_teams"))
        c_d.append(np.mean([abs(rate[m] * r.loc[s, "seconds"] - r.loc[s, "truth"]) / r.loc[s, "truth"] for s in SIDES]))
        share = lambda col: r.loc["left", col] / (r.loc["left", col] + r.loc["right", col])  # noqa: E731
        e_s.append(abs(share("ours") - share("truth")))
        e_st.append(abs(share("ours_true_teams") - share("truth")))
        c_s.append(abs(0.5 - share("truth")))
    e_d, e_dt, c_d, e_s, e_st, c_s = map(np.array, (e_d, e_dt, c_d, e_s, e_st, c_s))
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(matches), len(matches)) for _ in range(5000)]

    def boot(a, b):
        dd = [np.mean(a[i] - b[i]) for i in idx]
        return [round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)]

    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0], "units": len(units),
           "distance per team, mean relative error": {"ours": round(float(e_d.mean()), 4), "constant rate": round(float(c_d.mean()), 4),
                                                      "ours - constant, 95% over matches": boot(e_d, c_d),
                                                      "shown": bool(boot(e_d, c_d)[1] < 0),
                                                      "ours with the truth's teams (descriptive)": round(float(e_dt.mean()), 4)},
           "left side's share of the distance, mean abs error": {"ours": round(float(e_s.mean()), 4), "constant 0.5": round(float(c_s.mean()), 4),
                                                                  "ours - constant, 95% over matches": boot(e_s, c_s),
                                                                  "shown": bool(boot(e_s, c_s)[1] < 0),
                                                                  "ours with the truth's teams (descriptive)": round(float(e_st.mean()), 4)},
           "per match (m)": {m: {s: {k: round(float(pm.loc[(m, s), k]), 1) for k in ("ours", "ours_true_teams", "truth", "seconds")}
                                 for s in SIDES} for m in matches}}
    (HERE / "team_distance.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per match (m)"}, indent=1))


if __name__ == "__main__":
    main()
