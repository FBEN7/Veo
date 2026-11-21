"""Stitching v2 (plan_v2.json): tracklet-majority teams, every tracklet kept,
gaps inside a track filled as well as between linked tracklets.

    VEO_UNSEAL=... python stitch_v2.py
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import stitch as S

MAX_GAP = S.MAX_GAP_S * S.FPS


def with_track_team(players: pd.DataFrame) -> pd.DataFrame:
    team = players.groupby("track_id").team.agg(lambda s: s.mode().iloc[0])
    return players.assign(team=players.track_id.map(team))


def fill_rows(rows: pd.DataFrame, frames: np.ndarray, slot: int, team: str) -> pd.DataFrame:
    """Rows of one slot (sorted, one per frame) plus linear fills of gaps
    up to MAX_GAP frames at the players' frames."""
    f = rows.frame.to_numpy()
    fills = []
    for i in range(len(f) - 1):
        a, b = f[i], f[i + 1]
        if b - a <= 2 or b - a > MAX_GAP:
            continue
        ff = frames[(frames > a) & (frames < b)]
        if not len(ff):
            continue
        s = (ff - a) / (b - a)
        xa, ya, xb, yb = rows.x.iloc[i], rows.y.iloc[i], rows.x.iloc[i + 1], rows.y.iloc[i + 1]
        fills.append(pd.DataFrame({"frame": ff, "track_id": -2, "cls": "player", "team": team,
                                   "x": xa + s * (xb - xa), "y": ya + s * (yb - ya)}))
    out = pd.concat([rows.assign(kind="det"), *[x.assign(kind="fill") for x in fills]], ignore_index=True)
    return out.assign(slot=slot)


def stitch(players: pd.DataFrame) -> pd.DataFrame:
    frames = np.sort(players.frame.unique())
    out, slot = [], 0
    for team, p in players.groupby("team"):
        g = p.sort_values("frame").groupby("track_id")
        tl = pd.DataFrame({"start": g.frame.min(), "end": g.frame.max(),
                           "x0": g.x.first(), "y0": g.y.first(), "x1": g.x.last(), "y1": g.y.last()})
        ids = tl.index.to_numpy()
        nxt = {}
        if team in ("team_A", "team_B") and len(ids) > 1:
            gap = (tl.start.to_numpy()[None, :] - tl.end.to_numpy()[:, None]) / S.FPS
            dist = np.hypot(tl.x0.to_numpy()[None, :] - tl.x1.to_numpy()[:, None],
                            tl.y0.to_numpy()[None, :] - tl.y1.to_numpy()[:, None])
            ok = (gap > 0) & (gap <= S.MAX_GAP_S) & (dist <= S.SLACK_M + S.VMAX * gap)
            r, c = linear_sum_assignment(np.where(ok, dist + 0.5 * gap, 1e6))
            nxt = {int(ids[i]): int(ids[j]) for i, j in zip(r, c) if ok[i, j]}
        has_prev = set(nxt.values())
        for t in ids:
            if int(t) in has_prev:
                continue
            chain, cur = [int(t)], int(t)
            while cur in nxt:
                cur = nxt[cur]
                chain.append(cur)
            rows = p[p.track_id.isin(chain)].sort_values("frame").drop_duplicates("frame").reset_index(drop=True)
            out.append(fill_rows(rows, frames, slot, team))
            slot += 1
    return pd.concat(out, ignore_index=True)


def unit(w):
    info, m = S.tracks_of(w)
    players = with_track_team(m[m.cls == "player"])
    ball = m[m.cls == "ball"]
    st_ = stitch(players)
    gt = pd.read_parquet(S.CLIPS / f"{w}_gt.parquet")
    gt = gt[gt.role.isin(["player", "goalkeeper"]) & gt.x.notna()]
    side = gt.drop_duplicates("track_id").set_index("track_id").side
    pid = gt.drop_duplicates("track_id").set_index("track_id").player_id.astype(str)
    evf = np.sort(players.frame.unique())
    evf = evf[evf % S.EVAL_STEP == 0]
    gte = gt[gt.frame.isin(evf)]
    res = {}
    for name, pos, idcol in (("before", players.assign(kind="det"), "track_id"), ("after", st_, "slot")):
        q = pos[pos.frame.isin(evf)]
        q = q.assign(person=S.nearest_person(q, gte))
        m_ = q.dropna(subset=["person"]).assign(side=lambda d: d.person.map(side))
        team_side = m_[m_.team != "other"].groupby("team").side.agg(lambda s: s.mode().iloc[0]).to_dict()
        q = q.assign(ok_side=(q.team == "other") | (q.person.map(side) == q.team.map(team_side)))
        good = q[q.person.notna() & q.ok_side]
        anyp = q[q.person.notna()]
        fills = q[q.kind == "fill"]
        res[name] = {"truth": len(gte),
                     "covered": int(good.groupby("frame").person.nunique().sum()),
                     "covered any team": int(anyp.groupby("frame").person.nunique().sum()),
                     "fills": len(fills), "fills right": int((fills.person.notna() & fills.ok_side).sum()),
                     "continuity": good.groupby("person")[idcol].agg(lambda s: s.value_counts().iloc[0] / len(s)),
                     "weights": good.groupby("person").size()}
    fr = json.loads((S.GPU / w / ".cache" / f"v3_balls_{w}.parquet.meta.json").read_text())["frames"]
    lo, hi = min(fr), max(fr)
    trs = {"before": pd.concat([players, ball]),
           "after": pd.concat([st_.assign(track_id=st_.slot)[["frame", "track_id", "cls", "x", "y"]], ball])}
    acts = []
    for e in pd.read_csv(S.CLIPS / f"{w}_events.csv").itertuples():
        if e.label in S.NOT_ACTOR or pd.isna(e.player_id) or not (lo <= e.frame <= hi):
            continue
        r = {"match": info["match"], "abs": int(info["start_frame"]) + int(e.frame), "label": e.label}
        for name, tr_ in trs.items():
            a = S.SP.actor(tr_, int(e.frame))
            if a is None:
                r[name] = None
                continue
            q = tr_[(tr_.track_id == a) & (tr_.cls == "player")]
            q = q.iloc[[int(np.argmin(abs(q.frame.to_numpy() - e.frame)))]]
            g = gt[gt.frame == int(q.frame.iloc[0])]
            d = np.hypot(g.x - q.x.iloc[0], g.y - q.y.iloc[0])
            r[name] = bool(len(d) and d.min() <= S.NEAR_M and pid[g.track_id[d.idxmin()]] == str(int(e.player_id)))
        acts.append(r)
    return info["match"], res, acts


def main():
    for m in S.MATCHES:
        if not S.unsealed(m, "stitch v2: offline player stitching test"):
            raise SystemExit("VEO_UNSEAL needed for M7 / M9")
    units = sorted(p.name for p in S.GPU.iterdir() if p.name.split("_")[1] in S.MATCHES
                   and (S.GPU / p.name / f"output_{p.name}" / "metric_v3s_link.parquet").exists())
    agg, acts = {}, []
    for w in units:
        m, res, a = unit(w)
        acts += a
        for k, v in res.items():
            s = agg.setdefault(m, {}).setdefault(k, {"truth": 0, "covered": 0, "covered any team": 0, "fills": 0,
                                                    "fills right": 0, "cont": [], "wt": []})
            for f in ("truth", "covered", "covered any team", "fills", "fills right"):
                s[f] += int(v[f])
            s["cont"] += list(v["continuity"])
            s["wt"] += list(v["weights"])
        print(w, flush=True)
    out = {"plan_v2 sha256": (S.HERE / "plan_v2.json.sha256").read_text().split()[0], "units": len(units)}
    for m, d in agg.items():
        out[m] = {k: {"coverage": round(v["covered"] / v["truth"], 3),
                      "coverage any team": round(v["covered any team"] / v["truth"], 3), "fills": v["fills"],
                      "fill precision": round(v["fills right"] / v["fills"], 3) if v["fills"] else None,
                      "identity continuity (weighted)": round(float(np.average(v["cont"], weights=v["wt"])), 3)}
                  for k, v in d.items()}
    df = pd.DataFrame(acts).drop_duplicates(["match", "abs", "label"])
    for m, g in [("all", df), *df.groupby("match")]:
        out.setdefault("who made the action (rule C, metres)", {})[str(m)] = {
            k: {"n": len(g), "right": int((g[k] == True).sum()), "share": round(float((g[k] == True).mean()), 3),  # noqa: E712
                "wilson": S.wilson(int((g[k] == True).sum()), len(g)), "given": round(float(g[k].notna().mean()), 3)}  # noqa: E712
            for k in ("before", "after")}
    print(json.dumps(out, indent=1, default=str))
    (S.HERE / "stitch_v2.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
