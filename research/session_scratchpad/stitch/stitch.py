"""Offline stitching of player tracklets through the detector's misses
(plan.json; owner's idea, 2026-10-26), tested on M7 and M9.

    VEO_UNSEAL=... python stitch.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/passer"))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/shots_goals"))
from score_goals_test import wilson  # noqa: E402
from src import passer as SP  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.unseal import unsealed  # noqa: E402

GPU = ROOT / ".cache/gpu"
CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
MATCHES = ("128057", "132831")
FPS = 25.0
MAX_GAP_S, SLACK_M, VMAX = 5.0, 2.0, 8.0
NEAR_M = 3.0
EVAL_STEP = 10
NOT_ACTOR = {"OUT", "GOAL"}


def tracks_of(w):
    info = json.loads((CLIPS / f"{w}.json").read_text())
    m = pd.read_parquet(GPU / w / f"output_{w}" / "metric_v3s_link.parquet")
    m[["x", "y"]] = st.image_to_pitch(m[["px_raw", "py_raw"]].to_numpy(), st.calibration(str(info["match"])))
    return info, m[["frame", "track_id", "cls", "team", "x", "y"]].dropna(subset=["x", "y"])


def stitch(players: pd.DataFrame) -> pd.DataFrame:
    """Players with `slot` (stitched id) and `kind` (det / fill)."""
    out, next_slot = [], 0
    frames = np.sort(players.frame.unique())
    for team in ("team_A", "team_B"):
        p = players[players.team == team]
        if p.empty:
            continue
        g = p.sort_values("frame").groupby("track_id")
        tl = pd.DataFrame({"start": g.frame.min(), "end": g.frame.max(),
                           "x0": g.x.first(), "y0": g.y.first(), "x1": g.x.last(), "y1": g.y.last()})
        ids = tl.index.to_numpy()
        n = len(ids)
        gap = (tl.start.to_numpy()[None, :] - tl.end.to_numpy()[:, None]) / FPS
        dist = np.hypot(tl.x0.to_numpy()[None, :] - tl.x1.to_numpy()[:, None],
                        tl.y0.to_numpy()[None, :] - tl.y1.to_numpy()[:, None])
        ok = (gap > 0) & (gap <= MAX_GAP_S) & (dist <= SLACK_M + VMAX * gap)
        cost = np.where(ok, dist + 0.5 * gap, 1e6)
        r, c = linear_sum_assignment(cost)
        nxt = {int(ids[i]): int(ids[j]) for i, j in zip(r, c) if ok[i, j]}
        has_prev = set(nxt.values())
        for t in ids:
            if t in has_prev:
                continue
            chain, cur = [int(t)], int(t)
            while cur in nxt:
                cur = nxt[cur]
                chain.append(cur)
            rows = p[p.track_id.isin(chain)].sort_values("frame").drop_duplicates("frame")
            rows = rows.assign(slot=next_slot, kind="det")
            fill = []
            for a, b in zip(chain[:-1], chain[1:]):
                ea, sb = tl.loc[a], tl.loc[b]
                ff = frames[(frames > ea.end) & (frames < sb.start)]
                if len(ff):
                    s = (ff - ea.end) / (sb.start - ea.end)
                    fill.append(pd.DataFrame({"frame": ff, "track_id": -2, "cls": "player", "team": team,
                                              "x": ea.x1 + s * (sb.x0 - ea.x1), "y": ea.y1 + s * (sb.y0 - ea.y1),
                                              "slot": next_slot, "kind": "fill"}))
            out.append(pd.concat([rows, *fill], ignore_index=True))
            next_slot += 1
    return pd.concat(out, ignore_index=True) if out else players.iloc[:0]


def nearest_person(pos: pd.DataFrame, gt: pd.DataFrame) -> pd.Series:
    """Truth person within NEAR_M of each position (Hungarian per frame)."""
    res = pd.Series(np.nan, index=pos.index)
    gby = {f: g for f, g in gt.groupby("frame")}
    for f, d in pos.groupby("frame"):
        g = gby.get(f)
        if g is None:
            continue
        dist = np.hypot(d.x.to_numpy()[:, None] - g.x.to_numpy()[None], d.y.to_numpy()[:, None] - g.y.to_numpy()[None])
        r, c = linear_sum_assignment(np.where(dist <= NEAR_M, dist, 1e6))
        for i, j in zip(r, c):
            if dist[i, j] <= NEAR_M:
                res.loc[d.index[i]] = g.track_id.iloc[j]
    return res


def unit(w):
    info, m = tracks_of(w)
    players = m[m.cls == "player"]
    ball = m[m.cls == "ball"]
    st_ = stitch(players)
    gt = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
    gt = gt[gt.role.isin(["player", "goalkeeper"]) & gt.x.notna()]
    side = gt.drop_duplicates("track_id").set_index("track_id").side
    pid = gt.drop_duplicates("track_id").set_index("track_id").player_id.astype(str)
    ev_frames = np.sort(players.frame.unique())
    ev_frames = ev_frames[ev_frames % EVAL_STEP == 0]
    gte = gt[gt.frame.isin(ev_frames)]
    res = {}
    for name, pos, idcol in (("before", players[players.team.isin(["team_A", "team_B"])], "track_id"),
                             ("after", st_, "slot")):
        q = pos[pos.frame.isin(ev_frames)]
        who = nearest_person(q, gte)
        q = q.assign(person=who)
        # team -> side by majority over this unit
        tm = q.dropna(subset=["person"]).assign(side=lambda d: d.person.map(side))
        team_side = tm.groupby("team").side.agg(lambda s: s.mode().iloc[0]).to_dict()
        q = q.assign(ok_side=q.person.map(side) == q.team.map(team_side))
        good = q[q.person.notna() & q.ok_side]
        res[name] = {"truth": len(gte), "covered": good.person.groupby(good.frame).nunique().sum(),
                     "fills": int((q.get("kind") == "fill").sum()) if "kind" in q else 0,
                     "fills right": int(((q.get("kind") == "fill") & q.person.notna() & q.ok_side).sum()) if "kind" in q else 0,
                     "continuity": good.groupby("person")[idcol].agg(lambda s: s.value_counts().iloc[0] / len(s)),
                     "weights": good.groupby("person").size()}
    # who made the action: rule C on both track sets, actor matched in metres
    fr = json.loads((GPU / w / ".cache" / f"v3_balls_{w}.parquet.meta.json").read_text())["frames"]
    lo, hi = min(fr), max(fr)
    acts = []
    tr_before = pd.concat([players, ball])
    tr_after = pd.concat([st_.assign(track_id=st_.slot)[["frame", "track_id", "cls", "x", "y"]], ball])
    gt_all = gt
    for e in pd.read_csv(CLIPS / f"{w}_events.csv").itertuples():
        if e.label in NOT_ACTOR or pd.isna(e.player_id) or not (lo <= e.frame <= hi):
            continue
        r = {"match": info["match"], "abs": int(info["start_frame"]) + int(e.frame), "label": e.label}
        for name, tr_ in (("before", tr_before), ("after", tr_after)):
            a = SP.actor(tr_, int(e.frame))
            if a is None:
                r[name] = None
                continue
            q = tr_[(tr_.track_id == a) & (tr_.cls == "player")]
            q = q.iloc[[int(np.argmin(abs(q.frame.to_numpy() - e.frame)))]]
            g = gt_all[gt_all.frame == int(q.frame.iloc[0])]
            d = np.hypot(g.x - q.x.iloc[0], g.y - q.y.iloc[0])
            r[name] = bool(len(d) and d.min() <= NEAR_M and pid[g.track_id[d.idxmin()]] == str(int(e.player_id)))
        acts.append(r)
    return info["match"], res, acts


def main():
    for m in MATCHES:
        if not unsealed(m, "stitch: offline player stitching test"):
            raise SystemExit("VEO_UNSEAL needed for M7 / M9")
    units = sorted(p.name for p in GPU.iterdir() if p.name.split("_")[1] in MATCHES
                   and (GPU / p.name / f"output_{p.name}" / "metric_v3s_link.parquet").exists())
    agg, acts = {}, []
    for w in units:
        m, res, a = unit(w)
        acts += a
        for k, v in res.items():
            s = agg.setdefault(m, {}).setdefault(k, {"truth": 0, "covered": 0, "fills": 0, "fills right": 0, "cont": [], "wt": []})
            for f in ("truth", "covered", "fills", "fills right"):
                s[f] += int(v[f])
            s["cont"] += list(v["continuity"]); s["wt"] += list(v["weights"])
        print(w, flush=True)
    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0], "units": len(units)}
    for m, d in agg.items():
        out[m] = {k: {"coverage": round(v["covered"] / v["truth"], 3), "fills": v["fills"],
                      "fill precision": round(v["fills right"] / v["fills"], 3) if v["fills"] else None,
                      "identity continuity (weighted)": round(float(np.average(v["cont"], weights=v["wt"])), 3)}
                  for k, v in d.items()}
    df = pd.DataFrame(acts).drop_duplicates(["match", "abs", "label"])
    for m, g in [("all", df), *df.groupby("match")]:
        out.setdefault("who made the action (rule C, metres)", {})[str(m)] = {
            k: {"n": len(g), "right": int((g[k] == True).sum()), "share": round(float((g[k] == True).mean()), 3),  # noqa: E712
                "wilson": wilson(int((g[k] == True).sum()), len(g)), "given": round(float(g[k].notna().mean()), 3)}  # noqa: E712
            for k in ("before", "after")}
    print(json.dumps(out, indent=1, default=str))
    (HERE / "stitch.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
