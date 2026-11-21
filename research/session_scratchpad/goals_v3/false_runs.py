"""goals v3, exploratory reading (after the registered score): what the
held-out false runs and misses of goals_v3.json are.

Per held-out half, with the cell its fold chose (goals_v3.json):
- each false run: when it falls, the labelled actions around it (dead-ball
  gap, the action before), and the ground-truth share over its samples
  (would the rule fire on the truth there too?);
- a split of the detection share at the run's samples: detections matched to
  the truth (players and goalkeepers, one to one within 3 m) given their
  true team (team labels), the truth share of only the people we see
  (missing people), against the full truth share;
- each missed restartable goal: the best detection and truth shares 15-60 s
  after it, and the nearest run start;
- each hit with the wrong scoring team: whether the person nearest the
  centre spot is matched and which team the truth gives.

    VEO_UNSEAL=... python false_runs.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/goals_v2"))
sys.path.insert(0, str(HERE))
import explore as E  # noqa: E402
from score import load  # noqa: E402

F, st = E.F, E.st
FPS = F.FPS
MATCH_M = 3.0


def truth_people(match, half):
    g = st.gsr_table(match, half)
    g = g[(g.frame % F.SAMPLE == 0) & g.role.isin(["player", "goalkeeper"]) & g.x.notna()]
    return g.rename(columns={"side": "team"})[["frame", "x", "y", "team"]]


def pair(d, t):
    """one-to-one det -> truth row index within MATCH_M (Hungarian)."""
    if not len(d) or not len(t):
        return {}
    D = np.hypot(d.x.to_numpy()[:, None] - t.x.to_numpy()[None], d.y.to_numpy()[:, None] - t.y.to_numpy()[None])
    r, c = linear_sum_assignment(np.where(D <= MATCH_M, D, 1e6))
    return {int(i): int(j) for i, j in zip(r, c) if D[i, j] <= MATCH_M}


def share(x, team, ends):
    if not len(x):
        return np.nan
    e = np.array([ends.get(k, 0) for k in team])
    return float(np.mean(np.asarray(x) * e >= 0))


def main():
    from src.unseal import unsealed
    for m in sorted(E.SEALED):
        if not unsealed(m, "goals v3: false-run diagnosis (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    res = json.loads((HERE / "goals_v3.json").read_text())["held out"]["per match"]
    fr_rows, miss_rows, team_rows = [], [], []
    for m in E.MATCHES:
        c = res[m]["cell"]
        cell = (c["T"], c["R"], c["N"])
        for half in (1, 2):
            h = load(m, half)
            tp = truth_people(m, half)
            tends = F.defended_ends(tp)
            ts = E.half_gt(m, half)["s"]
            tr_runs = E.runs(ts, *cell)
            tp_f = {f: g for f, g in tp.groupby("frame")}
            dp = h["people"]
            dp_f = {f: g for f, g in dp.groupby("frame")}
            # det team -> truth side, by majority over matched people on every 10th sample
            votes = []
            for f in sorted(dp_f)[::10]:
                d, t = dp_f[f], tp_f.get(f)
                if t is None:
                    continue
                for i, j in pair(d, t).items():
                    votes.append((d.team.iloc[i], t.team.iloc[j]))
            vt = pd.DataFrame(votes, columns=["dteam", "gteam"])
            side_of = vt.groupby("dteam").gteam.agg(lambda s: s.mode().iloc[0]).to_dict()
            team_acc = float((vt.dteam.map(side_of) == vt.gteam).mean())
            ev = st.events(m)
            ev = ev[ev.half == half].sort_values("frame")
            efr, elab = ev.frame.to_numpy(), ev.label.to_numpy()
            last = int(h["s"].index.max())
            rs = E.runs(h["s"], *cell)
            gf = h["goals"].frame.to_numpy()
            rest = F.restartable(gf, h["actions"])
            mt = F.match_runs([r[0] for r in rs], gf)
            matched = set(mt.values())

            def split(frames):
                out = {"det": [], "det, true teams": [], "truth, seen only": [], "truth": [], "n det": [], "n truth": [], "n matched": [], "team errors": []}
                for f in frames:
                    d, t = dp_f.get(f), tp_f.get(f)
                    if d is None or t is None:
                        continue
                    pr = pair(d, t)
                    out["det"].append(share(d.x, d.team, h["ends"]))
                    ii = list(pr)
                    out["det, true teams"].append(share(d.x.to_numpy()[ii], t.team.to_numpy()[[pr[i] for i in ii]], tends))
                    jj = [pr[i] for i in ii]
                    out["truth, seen only"].append(share(t.x.to_numpy()[jj], t.team.to_numpy()[jj], tends))
                    out["truth"].append(share(t.x, t.team, tends))
                    out["n det"].append(len(d)); out["n truth"].append(len(t)); out["n matched"].append(len(pr))
                    out["team errors"].append(int(sum(side_of.get(d.team.iloc[i]) != t.team.iloc[pr[i]] for i in ii)))
                return {k: round(float(np.nanmean(v)), 3) if len(v) else None for k, v in out.items()}

            for ri, (a, b) in enumerate(rs):
                if ri in matched:
                    continue
                fr = list(range(a, b + 1, F.SAMPLE))
                k = np.searchsorted(efr, a)
                prev_f = efr[k - 1] if k > 0 else None
                next_f = efr[k] if k < len(efr) else None
                g_before = gf[gf <= a]
                fr_rows.append({
                    "match": m, "half": half, "start min": round(a / FPS / 60, 1), "length s": (b - a) // FPS + 1,
                    "to half end s": (last - b) // FPS,
                    "truth fires too": any(x <= b and y >= a for x, y in tr_runs),
                    "action before": elab[k - 1] if k > 0 else None,
                    "s since action before": round((a - prev_f) / FPS, 1) if prev_f is not None else None,
                    "s to action after": round((next_f - a) / FPS, 1) if next_f is not None else None,
                    "OUT/THROW IN/FREE KICK in the 30 s before": bool(np.isin(elab[(efr >= a - 30 * FPS) & (efr <= a)], ["OUT", "THROW IN", "FREE KICK"]).any()),
                    "s since a goal": round((a - g_before.max()) / FPS) if len(g_before) else None,
                    **split(fr)})
            for gi, g in enumerate(gf):
                if not rest[gi] or gi in mt:
                    continue
                win = h["s"].loc[g + 15 * FPS: g + 60 * FPS]
                twin = ts.loc[g + 15 * FPS: g + 60 * FPS]
                starts = np.array([r[0] for r in rs])
                near = int(starts[np.argmin(np.abs(starts - g))]) if len(starts) else None
                miss_rows.append({"match": m, "half": half, "goal min": round(g / FPS / 60, 1),
                                  "best det share 15-60 s": round(float(win.share.max()), 3),
                                  "best truth share 15-60 s": round(float(twin.share.max()), 3),
                                  "T": cell[0], "nearest run start, s after goal": round((near - g) / FPS) if near is not None else None,
                                  **{f"{k} (best det sample)": v for k, v in split([int(win.share.idxmax())]).items()}})
            for gi, ri in mt.items():
                if not rest[gi]:
                    continue
                a, b = rs[ri]
                if F.scorer_kick_spot(h["people"], h["ends"], rs[ri]) == h["side_team"].get(h["goals"].side.iloc[gi]):
                    continue
                d, t = dp_f.get(b), tp_f.get(b)
                i0 = int(np.argmin(np.hypot(d.x, d.y)))
                pr = pair(d, t) if t is not None else {}
                team_rows.append({"match": m, "half": half, "goal min": round(gf[gi] / FPS / 60, 1),
                                  "kicker det dist to spot m": round(float(np.hypot(d.x.iloc[i0], d.y.iloc[i0])), 1),
                                  "kicker matched": i0 in pr,
                                  "kicker det team right": (side_of.get(d.team.iloc[i0]) == t.team.iloc[pr[i0]]) if i0 in pr else None,
                                  "truth nearest to spot m": round(float(np.hypot(t.x, t.y).min()), 1) if t is not None else None})
            print(m, half, "team acc", round(team_acc, 3), "false", sum(1 for r in fr_rows if r["match"] == m and r["half"] == half), flush=True)
    fr = pd.DataFrame(fr_rows)
    miss = pd.DataFrame(miss_rows)
    tm = pd.DataFrame(team_rows)
    det_only = fr[~fr["truth fires too"]]
    summ = {
        "false runs": len(fr),
        "truth fires too (a real formation-like moment)": int(fr["truth fires too"].sum()),
        "detections only": len(det_only),
        "median shares over false runs": fr[["det", "det, true teams", "truth, seen only", "truth"]].median().round(3).to_dict(),
        "detections-only runs where true teams bring the share below T": None,
        "per match": fr.groupby("match").size().to_dict(),
        "action before (counts)": fr["action before"].value_counts().to_dict(),
        "OUT/THROW IN/FREE KICK in the 30 s before": int(fr["OUT/THROW IN/FREE KICK in the 30 s before"].sum()),
        "within 120 s after a goal": int((fr["s since a goal"].fillna(1e9) <= 120).sum()),
        "ending in the last 60 s of the half": int((fr["to half end s"] <= 60).sum()),
        "median s since action before / to action after": [float(fr["s since action before"].median()), float(fr["s to action after"].median())],
        "median people per sample (det / truth / matched)": [float(fr["n det"].median()), float(fr["n truth"].median()), float(fr["n matched"].median())],
        "median team errors per sample": float(fr["team errors"].median()),
    }
    T_of = {m: res[m]["cell"]["T"] for m in res}
    if len(det_only):
        Tm = det_only.match.map(T_of)
        summ["detections-only runs where true teams bring the share below T"] = int((det_only["det, true teams"] < Tm).sum())
        summ["detections-only runs where the seen people's truth share is >= T (missing people)"] = int((det_only["truth, seen only"] >= Tm).sum())
    out = {"reading": "exploratory, after goals_v3.json (registered)", "summary": summ,
           "false runs": fr.to_dict(orient="records"), "missed goals": miss.to_dict(orient="records"),
           "wrong scoring team": tm.to_dict(orient="records")}
    (HERE / "reading_false_runs.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))
    print(miss.to_string())
    print(tm.to_string())


if __name__ == "__main__":
    main()
