"""The team of an event (plan_v2.json): three ways to give our events a
team, scored on events matched to the truth's; the best one re-run through
plan.json's share and success tests; a ceiling from the truth's actor.

    VEO_UNSEAL=... python team_v2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import team_passes as T  # noqa: E402

L, GROUPS, FPS = T.L, T.GROUPS, T.FPS
sys.path.insert(0, str(T.ROOT))
import score_soccernet as sc  # noqa: E402
from src import passer  # noqa: E402

EVENTS = HERE / "events_v2.parquet"
RADIUS = 3.0


def events(units):
    """Out-of-fold events (unit, match, frame, cls, team A), kept on disk."""
    if EVENTS.exists():
        return pd.read_parquet(EVENTS)
    L.sp.labels_of = T.clip_labels
    d = {}
    for w in units:
        f = T.features(w)
        d[w] = (f, T.targets(w, f.index.to_numpy()))
    probs, thr_by = L.nested(units, d)
    rows = []
    for w in units:
        for e in L.events_of(d[w][0], probs[w], thr_by[w.split("_")[1]]):
            rows.append((w, w.split("_")[1], int(round(e["timestamp_s"] * FPS)), e["event_type"], e["team"]))
    ev = pd.DataFrame(rows, columns=["unit", "match", "frame", "cls", "team_A_method"])
    ev.to_parquet(EVENTS)
    return ev


def team_vote(metric: pd.DataFrame, frame: int):
    """Method C: over -0.4 / +0.2 s, each players' frame with a ball row
    within 2 frames gives the team of its nearest player within 3 m."""
    lo, hi = frame - round(passer.BEFORE_S * FPS), frame + round(passer.AFTER_S * FPS)
    t = metric[(metric.frame >= lo - passer.BALL_TOL) & (metric.frame <= hi + passer.BALL_TOL)]
    ball = t[t.cls == "ball"].drop_duplicates("frame").set_index("frame")
    players = t[(t.cls == "player") & (t.frame >= lo) & (t.frame <= hi)]
    if ball.empty or players.empty:
        return None
    bf = ball.index.to_numpy()
    votes = []
    for f, q in players.groupby("frame"):
        j = np.argmin(abs(bf - f))
        if abs(bf[j] - f) > passer.BALL_TOL:
            continue
        dd = np.hypot(q.x - ball.iloc[j].x, q.y - ball.iloc[j].y).to_numpy()
        if not np.isfinite(dd).any():
            continue
        k = int(np.nanargmin(dd))
        if dd[k] <= RADIUS and q.team.iloc[k] in ("team_A", "team_B"):
            votes.append((q.team.iloc[k], abs(int(f) - frame)))
    if not votes:
        return None
    v = pd.DataFrame(votes, columns=["team", "lag"]).groupby("team").agg(n=("lag", "size"), lag=("lag", "min"))
    return v.sort_values(["n", "lag"], ascending=[False, True]).index[0]


def ceiling(w: str, metric: pd.DataFrame, tm: dict) -> list:
    """Per truth pass / carry: (has a team label, label is the truth's side)
    for our player nearest the truth actor (3 m, nearest players' frame)."""
    lab = T.labels(w)
    lab = lab[lab.label.isin(GROUPS["pass"][0] | GROUPS["carry"][0])]
    g = pd.read_parquet(T.CLIPS / f"{w}_gt.parquet")
    g = g[g.x.notna()].assign(player_id=lambda x: x.player_id.astype(str))
    p = metric[metric.cls == "player"]
    pf = np.sort(p.frame.unique())
    out = []
    for r in lab.itertuples():
        a = g[(g.frame == r.frame) & (g.player_id == str(r.player_id))]
        if a.empty or not len(pf):
            out.append((False, False))
            continue
        f = pf[np.argmin(abs(pf - r.frame))]
        q = p[p.frame == f]
        dd = np.hypot(q.x.to_numpy() - 52.5 - a.x.iloc[0], q.y.to_numpy() - 34.0 - a.y.iloc[0])
        if not len(dd) or dd.min() > RADIUS:
            out.append((False, False))
            continue
        side = tm.get(q.team.iloc[int(dd.argmin())])
        out.append((side is not None, side == r.side))
    return out


def main():
    from src.unseal import unsealed
    for m in T.SEALED:
        if not unsealed(m, "team of events (team_passes/plan_v2): M7 / M9 clips as spotter training and scored data"):
            raise SystemExit("VEO_UNSEAL needed")
    units = sorted(p.name for p in T.TREE.iterdir() if p.name.startswith("st2_"))
    ev = events(units)
    print("events", len(ev), flush=True)
    teams, ceil = [], []
    for w, e in ev.groupby("unit"):
        metric = pd.read_parquet(T.TREE / w / f"output_{w}" / "metric_v3s_link.parquet")
        tm = T.team_map(w)
        pl = metric[(metric.cls == "player") & metric.team.isin(["team_A", "team_B"])]
        track_team = pl.groupby("track_id").team.agg(lambda s: s.mode().iloc[0]).to_dict()
        for r in e.itertuples():
            act = passer.actor(metric, r.frame, FPS)
            teams.append((r.Index, tm.get(r.team_A_method) if r.team_A_method else None,
                          tm.get(track_team.get(act)) if act is not None else None,
                          tm.get(team_vote(metric, r.frame))))
        ceil += ceiling(w, metric, tm)
    tt = pd.DataFrame(teams, columns=["i", "A", "B", "C"]).set_index("i")
    ev = ev.join(tt)
    ev.to_parquet(HERE / "events_v2_teams.parquet")

    # team accuracy on events matched to the truth's (same class, 1 s, one to one)
    hit = {k: 0 for k in "ABC"}
    per = {}
    n_match = 0
    for w, e in ev.groupby("unit"):
        lab = T.labels(w)
        for c in ("pass", "carry"):
            pe = e[e.cls == c].reset_index(drop=True)
            tl = lab[lab.label.isin(GROUPS[c][0])].reset_index(drop=True)
            matched = sc.match_one_to_one(list(pe.frame / FPS), list(tl.frame / FPS), 1.0)[0]
            for pi, ti, _ in matched:
                n_match += 1
                m = per.setdefault(w.split("_")[1], {k: 0 for k in "ABC"} | {"n": 0})
                m["n"] += 1
                for k in "ABC":
                    ok = pe[k].iloc[pi] == tl.side.iloc[ti]
                    hit[k] += ok
                    m[k] += ok
    acc = {k: round(hit[k] / n_match, 4) for k in "ABC"}
    best = max("ABC", key=lambda k: (hit[k], k == "A"))
    print("accuracy", acc, "best", best, flush=True)

    # plan.json's test with the selected method's teams
    rows = []
    for w, e in ev.groupby("unit"):
        e = e.sort_values("frame")
        rows += T.unit_rows(w, list(e.frame / FPS), list(e[best]), list(e.cls))
    df = pd.DataFrame(rows, columns=["unit", "match", "who", "t", "side", "success"])
    df.to_parquet(HERE / "passes_v2.parquet")
    c = np.array(ceil)
    out = {"plan_v2 sha256": (HERE / "plan_v2.json.sha256").read_text().split()[0],
           "events": len(ev), "matched to the truth": n_match,
           "team accuracy on matched events": acc, "selected": best,
           "per match (A, B, C right of n)": {m: [v["A"], v["B"], v["C"], v["n"]] for m, v in sorted(per.items())},
           "ceiling: truth actor's nearest detection": {"truth events": len(c), "with a team label": round(float(c[:, 0].mean()), 4),
                                                       "label is the truth's side": round(float(c[:, 1].mean()), 4),
                                                       "right among labelled": round(float(c[:, 1].sum() / max(c[:, 0].sum(), 1)), 4)},
           "test with the selected method": T.score(df)}
    (HERE / "team_v2.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "test with the selected method"}, indent=1, default=str))
    print(json.dumps({k: v for k, v in out["test with the selected method"].items() if k != "per match"}, indent=1))


if __name__ == "__main__":
    main()
