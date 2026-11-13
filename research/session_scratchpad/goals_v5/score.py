"""goals v5 (plan.json, hashed first): formation runs kept by a whistle,
the cell chosen leaving one match out over the 20 halves.

    VEO_UNSEAL=... python score.py
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/goals_v2"))
import explore as E  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("goals_v3_score", ROOT / "research/session_scratchpad/goals_v3/score.py")
S3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S3)

F = E.F
F.MATCH_S = (10, 60)          # the plan's window (match_runs and random_baseline read it at call time)
FPS = F.FPS
WH = ROOT / ".cache/whistle"
G = {"T": [0.70, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9], "R": [1.0, 2.0, 3.0, 5.0, None],
     "N": [1, 2, 3], "q": [0.99, 0.995, 0.998, 0.999], "L": [0.2, 0.3, 0.5], "B": [0, 5, 10]}
FORM = list(itertools.product(G["T"], G["R"], G["N"]))
CELLS = list(itertools.product(G["T"], G["R"], G["N"], G["q"], G["L"], G["B"]))


def check_plan():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit("plan.json does not match its sha256")
    return got


def load(m, half):
    h = S3.load(m, half)                       # scan people, goals, actions (complete scans only)
    p = h["people"]
    k = p[p.frame <= 250].groupby("team").x.mean()
    ends = {t: (-1 if v < k.drop(t).iloc[0] else 1) for t, v in k.items()}
    share = F.own_half_share(p, ends, frames=sorted(h["s"].index))
    d0 = np.hypot(p.x, p.y).groupby(p.frame).min().reindex(share.index)
    h.update(s=pd.DataFrame({"share": share, "d0": d0}), ends=ends,
             side_team={"left": next(t for t, e in ends.items() if e < 0),
                        "right": next(t for t, e in ends.items() if e > 0)})
    w = pd.read_parquet(WH / f"{m}_{half}.parquet")
    h["events"] = {}
    for q in G["q"]:
        thr = w.score.quantile(q)
        on = (w.score >= thr).to_numpy()
        t = w.t.to_numpy()
        starts, i = [], 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                starts.append((t[i], (j - i + 1) * 0.1))
                i = j + 1
            else:
                i += 1
        for L in G["L"]:
            h["events"][(q, L)] = np.array([s for s, dur in starts if dur >= L - 1e-9])
    h["runs"] = {c: E.runs(h["s"], *c) for c in FORM}
    q = p[p.team.isin(list(ends)) & p.x.notna() & p.y.notna()]
    h["pf"] = {f: g for f, g in q.groupby("frame")}
    h["pf_frames"] = np.array(sorted(h["pf"]))
    return h


def kicker_other(h, f):
    """formation.scorer_kick_spot's reading on one sample: the team of the
    team-labelled person nearest the spot kicks off; the other scored."""
    g = h["pf"].get(f)
    if g is None or not len(g):
        return None
    k = int(np.hypot(g.x.to_numpy(), g.y.to_numpy()).argmin())
    return F._other(g.team.iloc[k], h["ends"])


def kick_whistle(h, ev_t):
    fr = h["pf_frames"]
    return kicker_other(h, int(fr[int(np.argmin(np.abs(fr - ev_t * FPS)))]))


def evaluate(h, cell, details=False):
    T, R, N, q, L, B = cell
    ev = h["events"][(q, L)]
    kept, cue = [], []
    for a, b in h["runs"][(T, R, N)]:
        inside = ev[(ev >= a / FPS) & (ev <= b / FPS + B)]
        if len(inside):
            kept.append((a, b))
            cue.append(float(inside.min()))
    return score_runs(h, kept, cue, details)


def score_runs(h, kept, cue, details=False):
    gf = h["goals"].frame.to_numpy()
    rest = F.restartable(gf, h["actions"])
    m = F.match_runs([r[0] for r in kept], gf)
    hits = [gi for gi in m if rest[gi]]
    want = {gi: h["side_team"].get(h["goals"].side.iloc[gi]) for gi in hits}
    tw = sum(cue[m[gi]] is not None and kick_whistle(h, cue[m[gi]]) == want[gi] for gi in hits) if cue else 0
    ts = sum(kicker_other(h, kept[m[gi]][1]) == want[gi] for gi in hits)
    out = {"hits": len(hits), "restartable": int(sum(rest)), "false runs": len(kept) - len(m),
           "team right (kick_whistle)": int(tw), "team right (kick_spot)": int(ts), "kept runs": len(kept)}
    if details:
        out.update({"goal frames": gf.tolist(), "rest flags": list(map(bool, rest)), "last": int(h["s"].index.max())})
    return out


def choose(halves, cells, table):
    best = None
    for c in cells:
        hits = sum(table[(h["name"], c)]["hits"] for h in halves)
        fr = sum(table[(h["name"], c)]["false runs"] for h in halves)
        T, R, N, q, L, B = c
        key = (hits, -fr, T, -(R if R is not None else 99), q, L, -B, -N)
        if best is None or key > best[0]:
            best = (key, c)
    return best[1]


def lomo(halves, cells, table, matches):
    per, tot = {}, {}
    for mo in matches:
        c = choose([h for h in halves if h["match"] != mo], cells, table)
        rows = [table[(h["name"], c)] for h in halves if h["match"] == mo]
        r = {k: sum(x[k] for x in rows) for k in rows[0] if isinstance(rows[0][k], int)}
        per[mo] = dict(r, cell=dict(zip(["T", "R", "N", "q", "L", "B"][:len(c)], c)))
        for k, v in r.items():
            tot[k] = tot.get(k, 0) + v
    return tot, per


def main():
    sha = check_plan()
    from src.unseal import unsealed
    for m in sorted(E.SEALED):
        if not unsealed(m, "goals v5: whistle rule scoring (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    halves = []
    for m in E.MATCHES:
        for half in (1, 2):
            h = load(m, half)
            h["name"] = f"{m}_{half}"
            halves.append(h)
            print("loaded", h["name"], flush=True)
    matches = sorted({h["match"] for h in halves})
    table = {(h["name"], c): evaluate(h, c) for h in halves for c in CELLS}
    tot, per = lomo(halves, CELLS, table, matches)
    # without the whistle: formation only, kick-off ends, 10-60 s window
    t0 = {(h["name"], c): score_runs(h, h["runs"][c], [None] * len(h["runs"][c])) for h in halves for c in FORM}
    def choose0(hs):
        best = None
        for c in FORM:
            hits = sum(t0[(h["name"], c)]["hits"] for h in hs); fr = sum(t0[(h["name"], c)]["false runs"] for h in hs)
            key = (hits, -fr, c[0], -(c[1] if c[1] is not None else 99), -c[2])
            if best is None or key > best[0]:
                best = (key, c)
        return best[1]
    tot0 = {"hits": 0, "restartable": 0, "false runs": 0}
    for mo in matches:
        c = choose0([h for h in halves if h["match"] != mo])
        for h in halves:
            if h["match"] == mo:
                for k in tot0:
                    tot0[k] += t0[(h["name"], c)][k]
    # random baseline with the held-out kept-run counts
    base = []
    for mo in matches:
        c = tuple(per[mo]["cell"].values())
        for h in halves:
            if h["match"] == mo:
                d = evaluate(h, c, details=True)
                base.append({"n_runs": d["kept runs"], "goals": d["goal frames"], "restartable": d["rest flags"], "last": d["last"]})
    rb = F.random_baseline(base, 1000, 0)
    allc = choose(halves, CELLS, table)
    found = tot["hits"] / max(tot["restartable"], 1)
    frpm = tot["false runs"] / len(matches)
    team = tot["team right (kick_whistle)"] / max(tot["hits"], 1)
    # diagnostics at the cell chosen on all matches (exploratory)
    q, L = allc[3], allc[4]
    dur = sum(h["last"] if "last" in h else int(h["s"].index.max()) for h in halves) / FPS / 60
    n_ev = sum(len(h["events"][(q, L)]) for h in halves)
    near_ko, n_ko, live_hit, live_n = 0, 0, 0, 0
    for h in halves:
        ev = h["events"][(q, L)]
        acts = np.sort(np.asarray(h["actions"], float))
        gf = h["goals"].frame.to_numpy()
        rest = F.restartable(gf, h["actions"])
        for g, r in zip(gf, rest):
            if not r:
                continue
            ko = acts[acts >= g + 10 * FPS]
            if len(ko):
                n_ko += 1
                near_ko += bool((np.abs(ev - ko[0] / FPS) <= 3).any())
        last = int(h["s"].index.max())
        for a in range(90 * FPS, last - 30 * FPS, 30 * FPS):
            inwin = acts[(acts >= a) & (acts < a + 30 * FPS)]
            gaps = np.diff(np.r_[a, inwin, a + 30 * FPS])
            if len(inwin) and gaps.max() <= 5 * FPS and not ((gf >= a - 90 * FPS) & (gf < a + 30 * FPS)).any():
                live_n += 1
                live_hit += bool(((ev >= a / FPS) & (ev < a / FPS + 30)).any())
    out = {"plan sha256": sha, "halves": len(halves),
           "held out": {"restartable goals found": f"{tot['hits']} / {tot['restartable']}", "share": round(found, 4),
                        "false runs": tot["false runs"], "false runs per match": round(frpm, 2),
                        "scoring team right (kick_whistle)": round(team, 4),
                        "scoring team right (kick_spot)": round(tot["team right (kick_spot)"] / max(tot["hits"], 1), 4),
                        "per match": per},
           "without the whistle (kick-off ends, 10-60 s), held out": dict(tot0, **{"false runs per match": round(tot0["false runs"] / len(matches), 2)}),
           "random-time baseline (held-out kept-run counts), hits": {"mean": round(float(rb.mean()), 2), "95th percentile": float(np.percentile(rb, 95))},
           "cell chosen on all 10 matches": dict(zip(["T", "R", "N", "q", "L", "B"], allc)),
           "diagnostics at that cell (exploratory)": {
               "whistle events per minute": round(n_ev / dur, 2),
               "restartable goals with a whistle event within 3 s of the kick-off action": f"{near_ko} / {n_ko}",
               "30-s live-play stretches holding a whistle event": f"{live_hit} / {live_n}"},
           "bar met (found >= 0.95, false runs <= 1 per match)": bool(found >= 0.95 and frpm <= 1),
           "team bar met (kick_whistle right on >= 0.9 of hits)": bool(team >= 0.9)}
    (HERE / "goals_v5.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "held out"}, indent=1, default=str))
    print(json.dumps({k: v for k, v in out["held out"].items() if k != "per match"}, indent=1))
    for mo, r in per.items():
        print(mo, r)


if __name__ == "__main__":
    main()
