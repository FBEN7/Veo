"""Per-detection kit teams on the clips (plan_v3.json): the Kaggle kernel's
kit colours (kit_kernel.py) smoothed along each track and clustered per
clip; then the detections' teams, the ceiling, our events' teams and
plan.json's test with them.

    VEO_UNSEAL=... python kit_v3.py KIT_DIR     # KIT_DIR: the kernel's kit/<unit>.parquet
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import team_passes as T  # noqa: E402
import team_v2 as V  # noqa: E402

L, GROUPS, FPS = T.L, T.GROUPS, T.FPS
from src import team_assignment_v2 as ta  # noqa: E402

SMOOTH = 50                     # frames: +-2 s along the track
CORE = 0.15                     # box heights inside the outline to be fitted


def kit_teams(w: str, kit: pd.DataFrame) -> pd.DataFrame:
    """The clip's metric with each player row's team replaced by its
    per-detection kit team (det_smooth); rows without a kit colour: 'other'."""
    metric = pd.read_parquet(T.TREE / w / f"output_{w}" / "metric_v3s_link.parquet")
    p = metric[metric.cls == "player"].reset_index().merge(kit, on=["frame", "track_id"], how="left").set_index("index")
    p = p[p.h0.notna()]
    feats = {}
    for _, g in p.groupby("track_id"):
        g = g.sort_values("frame")
        fr, X = g.frame.to_numpy(), g[["h0", "h1", "h2", "h3"]].to_numpy()
        lo, hi = np.searchsorted(fr, fr - SMOOTH), np.searchsorted(fr, fr + SMOOTH, side="right")
        for i, a, b in zip(g.index, lo, hi):
            feats[i] = np.median(X[a:b], axis=0)
    info = json.loads((T.CLIPS / f"{w}.json").read_text())
    poly = np.asarray(info["pitch"], np.float32).reshape(-1, 1, 2)
    inside = {i for i, r in p.iterrows()
              if cv2.pointPolygonTest(poly, (float(r.px_raw), float(r.py_raw)), True) >= CORE * r.crop_h}
    team = ta.cluster_teams(feats, kit_feature="hsv4", fit_tracks=inside, verbose=False) if len(feats) >= 3 else {}
    out = metric.copy()
    pl = out.cls == "player"
    out.loc[pl, "team"] = "other"
    idx = [i for i in team if i in out.index]
    out.loc[idx, "team"] = [team[i] for i in idx]
    return out


def team_map_of(w: str, metric: pd.DataFrame) -> dict:
    """plan.json's per-clip team mapping, on a given metric."""
    p = metric[(metric.cls == "player") & metric.team.isin(["team_A", "team_B"]) & (metric.frame % 25 == 0)]
    g = pd.read_parquet(T.CLIPS / f"{w}_gt.parquet")
    g = g[(g.frame % 25 == 0) & g.x.notna()]
    votes = {"team_A": {"left": 0, "right": 0}, "team_B": {"left": 0, "right": 0}}
    gb = {f: d for f, d in g.groupby("frame")}
    for f, d in p.groupby("frame"):
        t = gb.get(f)
        if t is None:
            continue
        dist = np.hypot(d.x.to_numpy()[:, None] - 52.5 - t.x.to_numpy()[None],
                        d.y.to_numpy()[:, None] - 34.0 - t.y.to_numpy()[None])
        j = dist.argmin(1)
        ok = dist[np.arange(len(d)), j] <= 3.0
        for team, side in zip(d.team.to_numpy()[ok], t.side.to_numpy()[j[ok]]):
            votes[team][side] += 1
    return {k: max(v, key=v.get) if sum(v.values()) else None for k, v in votes.items()}


def detections(w: str, metric: pd.DataFrame, tm: dict) -> tuple:
    """(rows within 3 m of a truth person, with a team label, right) at
    every 25th frame of the ball range."""
    p = metric[(metric.cls == "player") & (metric.frame >= T.LO) & (metric.frame <= T.HI) & (metric.frame % 25 == 0)]
    g = pd.read_parquet(T.CLIPS / f"{w}_gt.parquet")
    g = g[(g.frame % 25 == 0) & g.x.notna()]
    gb = {f: d for f, d in g.groupby("frame")}
    n = lab = right = 0
    for f, d in p.groupby("frame"):
        t = gb.get(f)
        if t is None:
            continue
        dist = np.hypot(d.x.to_numpy()[:, None] - 52.5 - t.x.to_numpy()[None],
                        d.y.to_numpy()[:, None] - 34.0 - t.y.to_numpy()[None])
        j = dist.argmin(1)
        ok = dist[np.arange(len(d)), j] <= 3.0
        side = [tm.get(x) for x in d.team.to_numpy()[ok]]
        n += int(ok.sum())
        lab += sum(s is not None for s in side)
        right += sum(s == ts for s, ts in zip(side, t.side.to_numpy()[j[ok]]))
    return n, lab, right


def main():
    from src.unseal import unsealed
    for m in T.SEALED:
        if not unsealed(m, "team passes plan_v3: kit teams on the M7 / M9 clips (scored data)"):
            raise SystemExit("VEO_UNSEAL needed")
    kit_dir = Path(sys.argv[1])
    ev = pd.read_parquet(HERE / "events_v2.parquet")
    units = sorted(p.name for p in T.TREE.iterdir() if p.name.startswith("st2_"))
    missing = [w for w in units if not (kit_dir / f"{w}.parquet").exists()]
    det = {"old": [0, 0, 0], "new": [0, 0, 0]}
    ceil, teamA, rows = [], [], []
    for i, w in enumerate(units):
        old = pd.read_parquet(T.TREE / w / f"output_{w}" / "metric_v3s_link.parquet")
        new = old if w in missing else kit_teams(w, pd.read_parquet(kit_dir / f"{w}.parquet"))
        tm_old, tm_new = team_map_of(w, old), team_map_of(w, new)
        for k, (mt, tm) in {"old": (old, tm_old), "new": (new, tm_new)}.items():
            det[k] = [a + b for a, b in zip(det[k], detections(w, mt, tm))]
        ceil += V.ceiling(w, new, tm_new)
        f = T.features(w)
        e = ev[ev.unit == w].sort_values("frame")
        fr = e.frame.to_numpy()
        pt = f.reindex(fr)
        pl = L._players(new, fr, pt.x_0.to_numpy(), pt.y_0.to_numpy())
        side = [tm_new.get({1.0: "team_A", -1.0: "team_B"}.get(t)) if np.isfinite(t) else None for t in pl.near_team]
        teamA += list(zip(e.index, side))
        rows += T.unit_rows(w, list(e.frame / FPS), side, list(e.cls))
        if i % 25 == 0:
            print(i, w, flush=True)
    ev = ev.join(pd.DataFrame(teamA, columns=["i", "A_kit"]).set_index("i"))
    ev.to_parquet(HERE / "events_v3_teams.parquet")
    hit = n_match = 0
    for w, e in ev.groupby("unit"):
        lab = T.labels(w)
        for c in ("pass", "carry"):
            pe = e[e.cls == c].reset_index(drop=True)
            tl = lab[lab.label.isin(GROUPS[c][0])].reset_index(drop=True)
            for pi, ti, _ in V.sc.match_one_to_one(list(pe.frame / FPS), list(tl.frame / FPS), 1.0)[0]:
                n_match += 1
                hit += pe.A_kit.iloc[pi] == tl.side.iloc[ti]
    df = pd.DataFrame(rows, columns=["unit", "match", "who", "t", "side", "success"])
    df.to_parquet(HERE / "passes_v3.parquet")
    c = np.array(ceil)
    out = {"plan_v3 sha256": (HERE / "plan_v3.json.sha256").read_text().split()[0],
           "clips without kit colours (old labels kept)": missing,
           "detections within 3 m of a truth person (old, new)": {
               k: {"rows": v[0], "with a team label": round(v[1] / v[0], 4), "right among labelled": round(v[2] / max(v[1], 1), 4),
                   "right of all": round(v[2] / v[0], 4)} for k, v in det.items()},
           "ceiling (new labels)": {"with a team label": round(float(c[:, 0].mean()), 4),
                                    "label is the truth's side": round(float(c[:, 1].mean()), 4),
                                    "right among labelled": round(float(c[:, 1].sum() / max(c[:, 0].sum(), 1)), 4)},
           "events' team accuracy, method A with kit teams": round(hit / n_match, 4), "matched events": n_match,
           "test (primary)": T.score(df)}
    (HERE / "kit_v3.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "test (primary)"}, indent=1, default=str))
    print(json.dumps({k: v for k, v in out["test (primary)"].items() if k != "per match"}, indent=1))


if __name__ == "__main__":
    main()
