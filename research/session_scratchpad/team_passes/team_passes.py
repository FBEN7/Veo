"""Passes per team (plan.json, addendum1.json): the learned pass / carry
spotter (events_learn/learn.py) on the shots clips' ball range, nested leave
one match out; per match, each team's share of the passes and its pass
success rate against constants.

    VEO_UNSEAL=... python team_passes.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball/events_learn"))
import learn as L  # noqa: E402
from score_st2_actions import GROUPS  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402

TREE = ROOT / ".cache/gpu_shots3"
CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
CACHE = ROOT / ".cache/team_passes"
LO, HI = 350, 1075              # the clips' ball range (addendum1.json)
FPS = L.FPS
NEXT_S = 10.0
SEALED = {"128057", "132831"}


def labels(w: str) -> pd.DataFrame:
    e = pd.read_csv(CLIPS / f"{w}_events.csv")
    return e[(e.frame >= LO) & (e.frame <= HI)].sort_values("frame").reset_index(drop=True)


def clip_labels(w: str) -> dict:
    e = labels(w)
    return {g: list(e[e.label.isin(GROUPS[g][0])].frame / FPS) for g in L.CLASSES}


def candidates(rows: pd.DataFrame, frames: np.ndarray, cal: dict) -> pd.DataFrame:
    rows = rows[rows.frame.isin(frames)]
    best = rows.sort_values("confidence", ascending=False).drop_duplicates("frame").set_index("frame")
    xy = st.image_to_pitch(np.c_[best.px, best.py + best.crop_h / 2], cal)
    out = pd.DataFrame({"c_max": best.confidence, "c_x": xy[:, 0], "c_y": xy[:, 1]}, index=best.index)
    out["c_n"] = rows[rows.confidence >= 0.10].groupby("frame").size()
    return out.reindex(frames)


def features(w: str) -> pd.DataFrame:
    """learn.features on a clip: its tracks and candidates, its ball range."""
    cache = CACHE / f"{w}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    cal = st.calibration(w.split("_")[1])
    base = TREE / w / f"output_{w}"
    metric = pd.read_parquet(base / "metric_v3s_link.parquet")
    frames = np.sort(metric[metric.cls == "player"].frame.unique())
    frames = frames[(frames >= LO) & (frames <= HI)]
    b = L._ball_series(metric, frames)
    c = candidates(pd.read_parquet(base / "v3s_rows.parquet"), frames, cal)
    px = np.where(b.x.notna(), b.x, c.c_x)
    py = np.where(b.y.notna(), b.y, c.c_y)
    pl = L._players(metric, frames, px, py)
    bs = pd.concat([b[["has", "speed", "confidence"]].rename(columns={"confidence": "b_conf"}),
                    c[["c_max", "c_n"]], pl.drop(columns="near_id")], axis=1)
    bs["x"], bs["y"] = px, py
    bs["dAB"] = bs.dA - bs.dB
    feats = [bs.add_suffix("_0")]
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 2
    for o in L.OFFSETS:
        sh = bs.shift(-o // step)
        feats.append(sh[["has", "speed", "c_max", "d1", "near_team", "n3", "dAB"]].add_suffix(f"_{o:+d}"))
    f = pd.concat(feats, axis=1)
    k = 6 // step
    nid, nt = pl.near_id, pl.near_team
    f["id_change"] = (nid.shift(k) != nid.shift(-k)).astype(float).where(nid.shift(k).notna() & nid.shift(-k).notna())
    f["team_change"] = (nt.shift(k) != nt.shift(-k)).astype(float).where(nt.shift(k).notna() & nt.shift(-k).notna())
    f["disp"] = np.hypot(pd.Series(px, index=frames).shift(-k) - pd.Series(px, index=frames).shift(k),
                         pd.Series(py, index=frames).shift(-k) - pd.Series(py, index=frames).shift(k))
    f["speed_max_next"] = b.speed[::-1].rolling(6, min_periods=1).max()[::-1].to_numpy()
    f["has_frac_1s"] = b.has.rolling(25 // step, center=True, min_periods=1).mean().to_numpy()
    f["near_team_0"] = pl.near_team.to_numpy()
    f.index.name = "frame"
    CACHE.mkdir(parents=True, exist_ok=True)
    f.to_parquet(cache)
    return f


def targets(w: str, frames: np.ndarray) -> pd.DataFrame:
    lab = clip_labels(w)
    t = frames / FPS
    return pd.DataFrame({c: np.array([np.any(np.abs(np.asarray(lab[c]) - x) <= L.POS_S) for x in t], float)
                         for c in L.CLASSES}, index=frames)


def team_map(w: str) -> dict:
    """Our team_A / team_B to the truth's left / right: majority over our
    players within 3 m of a truth person, every 25th frame."""
    m = pd.read_parquet(TREE / w / f"output_{w}" / "metric_v3s_link.parquet")
    p = m[(m.cls == "player") & m.team.isin(["team_A", "team_B"]) & (m.frame % 25 == 0)]
    g = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
    g = g[(g.frame % 25 == 0) & g.x.notna()]
    votes = {"team_A": {"left": 0, "right": 0}, "team_B": {"left": 0, "right": 0}}
    gb = {f: d for f, d in g.groupby("frame")}
    for f, d in p.groupby("frame"):
        t = gb.get(f)
        if t is None:
            continue
        # our pitch metres run from a corner (0..105, 0..68), the truth's from the centre
        dist = np.hypot(d.x.to_numpy()[:, None] - 52.5 - t.x.to_numpy()[None],
                        d.y.to_numpy()[:, None] - 34.0 - t.y.to_numpy()[None])
        j = dist.argmin(1)
        ok = dist[np.arange(len(d)), j] <= 3.0
        for team, side in zip(d.team.to_numpy()[ok], t.side.to_numpy()[j[ok]]):
            votes[team][side] += 1
    return {k: max(v, key=v.get) if sum(v.values()) else None for k, v in votes.items()}


def passes_with_success(times, sides, kinds, out_label=None):
    """(time, side, success or None) for each pass: the next action within
    10 s by the same side (and, for the truth, not OUT)."""
    rows = []
    for i, (t, s, k) in enumerate(zip(times, sides, kinds)):
        if k != "pass":
            continue
        nxt = [j for j in range(i + 1, len(times)) if times[j] > t]
        if not nxt or times[nxt[0]] - t > NEXT_S:
            rows.append((t, s, None))
            continue
        j = nxt[0]
        ok = sides[j] == s and (out_label is None or out_label[j] != "OUT")
        rows.append((t, s, bool(ok)))
    return rows


def main():
    from src.unseal import unsealed
    for m in SEALED:
        if not unsealed(m, "team passes: M7 / M9 clips as spotter training and scored data"):
            raise SystemExit("VEO_UNSEAL needed")
    L.sp.labels_of = clip_labels              # the spotter's rate thresholds read the clips' labels
    units = sorted(p.name for p in TREE.iterdir() if p.name.startswith("st2_"))
    d = {}
    for i, w in enumerate(units):
        f = features(w)
        d[w] = (f, targets(w, f.index.to_numpy()))
        if i % 25 == 0:
            print("features", i, w, flush=True)
    probs, thr_by = L.nested(units, d)
    print("thresholds", thr_by, flush=True)
    rows = []
    for w in units:
        m = w.split("_")[1]
        tm = team_map(w)
        ev = L.events_of(d[w][0], probs[w], thr_by[m])
        ours = passes_with_success([e["timestamp_s"] for e in ev], [tm.get(e["team"]) if e["team"] else None for e in ev],
                                   [e["event_type"] for e in ev])
        lab = labels(w)
        lab = lab[lab.label.isin(set().union(*[g[0] for g in GROUPS.values()]) | {"BALL PLAYER BLOCK", "HEADER", "FREE KICK", "THROW IN"})]
        kinds = ["pass" if l in GROUPS["pass"][0] else "other" for l in lab.label]
        truth = passes_with_success(list(lab.frame / FPS), list(lab.side), kinds, list(lab.label))
        rows += [(w, m, "ours", t, s, ok) for t, s, ok in ours]
        rows += [(w, m, "truth", t, s, ok) for t, s, ok in truth]
    df = pd.DataFrame(rows, columns=["unit", "match", "who", "t", "side", "success"])
    df.to_parquet(HERE / "passes.parquet")

    matches = sorted(df.match.unique())
    per = {}
    for m in matches:
        g = df[df.match == m]
        r = {}
        for who in ("ours", "truth"):
            x = g[g.who == who]
            n = x.side.value_counts()
            r[who] = {"left": int(n.get("left", 0)), "right": int(n.get("right", 0)), "no team": int(x.side.isna().sum()),
                      "success": {s: [int((x[x.side == s].success == True).sum()), int(x[x.side == s].success.notna().sum())]  # noqa: E712
                                  for s in ("left", "right")}}
        per[m] = r

    def share(r):
        n = r["left"] + r["right"]
        return r["left"] / n if n else np.nan

    def rate(r, s):
        a, n = r["success"][s]
        return a / n if n else np.nan

    e_share = np.array([abs(share(per[m]["ours"]) - share(per[m]["truth"])) for m in matches])
    c_share = np.array([abs(0.5 - share(per[m]["truth"])) for m in matches])
    e_succ, c_succ = [], []
    for m in matches:
        a = sum(per[o]["truth"]["success"][s][0] for o in matches if o != m for s in ("left", "right"))
        n = sum(per[o]["truth"]["success"][s][1] for o in matches if o != m for s in ("left", "right"))
        const = a / n
        e_succ.append(np.nanmean([abs(rate(per[m]["ours"], s) - rate(per[m]["truth"], s)) for s in ("left", "right")]))
        c_succ.append(np.nanmean([abs(const - rate(per[m]["truth"], s)) for s in ("left", "right")]))
    e_succ, c_succ = np.array(e_succ), np.array(c_succ)
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(matches), len(matches)) for _ in range(5000)]

    def boot(a, b):
        dd = [np.nanmean(a[i] - b[i]) for i in idx]
        return [round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)]

    cnt_rel, cnt_half = [], []
    for m in matches:
        o, t = per[m]["ours"], per[m]["truth"]
        tot = o["left"] + o["right"]
        for s in ("left", "right"):
            if t[s]:
                cnt_rel.append(abs(o[s] - t[s]) / t[s])
                cnt_half.append(abs(tot / 2 - t[s]) / t[s])
    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0],
           "addendum1 sha256": (HERE / "addendum1.json.sha256").read_text().split()[0],
           "units": len(units), "thresholds per held-out match": thr_by,
           "share of the left side's passes": {"mean abs error, ours": round(float(e_share.mean()), 4),
                                               "constant 0.5": round(float(c_share.mean()), 4),
                                               "ours - constant, 95% over matches": boot(e_share, c_share),
                                               "shown": bool(boot(e_share, c_share)[1] < 0)},
           "pass success rate per team": {"mean abs error, ours": round(float(np.nanmean(e_succ)), 4),
                                          "constant (other matches' rate)": round(float(np.nanmean(c_succ)), 4),
                                          "ours - constant, 95% over matches": boot(e_succ, c_succ),
                                          "shown": bool(boot(e_succ, c_succ)[1] < 0)},
           "pass counts per team (descriptive)": {"median relative error, ours": round(float(np.median(cnt_rel)), 4),
                                                  "our total split 50/50": round(float(np.median(cnt_half)), 4)},
           "per match": per}
    (HERE / "team_passes.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "per match"}, indent=1, default=str))


if __name__ == "__main__":
    main()
