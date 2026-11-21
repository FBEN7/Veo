"""per-player counts from shirt numbers on a follow-cam (plan.json, hashed
first): a ceiling with perfect detection and perfect tracking in view.

    python attribute.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "followcam_sim"))
import simulate as F  # noqa: E402  (camera, ball proxy, projection: unchanged)
from src import soccertrack_v2 as st  # noqa: E402

LOOK_F = 50                     # one look every 2 s
WINDOWS = (2, 6)                # +/- W s around the action
RATES = [(0, 40, 0.0, 0.0), (40, 60, 0.033, 0.067), (60, 80, 0.267, 0.067), (80, 1e9, 0.433, 0.033)]
DRAWS = 200
PASSES = ("PASS", "HIGH PASS", "CROSS")
TYPES = ("shots", "successful passes", "tackles")


def check():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit("plan.json does not match its sha256")
    return got


def rates(px):
    return next((c, w) for lo, hi, c, w in RATES if lo <= px < hi)


def actions(e):
    """the three action types, from one half's actions with a side (sorted)."""
    out = []
    lab, fr, pid, side = e.label.to_numpy(), e.frame.to_numpy(), e.player_id.to_numpy(), e.side.to_numpy()
    for i in range(len(e)):
        t = None
        if lab[i] == "SHOT":
            t = "shots"
        elif lab[i] == "GOAL":
            prev = (lab[:i] == "SHOT") & (pid[:i] == pid[i]) & (fr[:i] >= fr[i] - 50)
            t = None if prev.any() else "shots"
        elif lab[i] == "PLAYER SUCCESSFUL TACKLE":
            t = "tackles"
        elif lab[i] in PASSES and i + 1 < len(e):
            if (fr[i + 1] - fr[i]) / F.FPS <= 6.0 and side[i + 1] == side[i] and lab[i + 1] != "OUT":
                t = "successful passes"
        if t and pid[i] is not None:
            out.append({"type": t, "frame": int(fr[i]), "player_id": pid[i], "side": side[i]})
    return out


def per_half(match, half):
    gsr = st.gsr_table(match, half)
    gsr = gsr[gsr.role.isin(["player", "goalkeeper"]) & gsr.x.notna() & gsr.y.notna()]
    n = int(gsr.frame.max()) + 1
    ev = st.events(match)
    ev = st.with_actors(ev[ev.half == half], gsr)
    cam = np.array([0.0, (34.0 + F.BACK_M) * (1 if st.y_towards_camera(gsr) else -1), F.HEIGHT_M])
    tg = F.aim(F.proxy(ev, n))
    e = ev[ev.side.isin(["left", "right"])].sort_values("frame", kind="stable").reset_index(drop=True)
    acts = actions(e)
    # lineups: per side, number -> player id; per player, his number and side
    who = gsr.dropna(subset=["jersey"]).drop_duplicates("player_id").set_index("player_id")[["jersey", "side"]]
    lineup = {s: {int(j): p for p, j in g.jersey.items()} for s, g in who.groupby("side")}
    pos = {p: d.drop_duplicates("frame").set_index("frame")[["x", "y"]] for p, d in gsr.groupby("player_id")}
    for a in acts:
        a["jersey"] = int(who.jersey.get(a["player_id"], -1)) if a["player_id"] in who.index else -1
        a["looks"] = {}
        d = pos.get(a["player_id"])
        for fov in F.FOVS:
            for w in WINDOWS:
                looks = []
                for k in range(-w // 2, w // 2 + 1):
                    f = a["frame"] + k * LOOK_F
                    if f < 0 or f >= n or np.isnan(tg[f, 0]) or d is None or f not in d.index:
                        continue
                    x, y = d.loc[f].to_numpy(float)
                    ins, y0 = F.project(cam, tg[f], np.array([[x, y, 0.0]]), fov)
                    if not ins[0]:
                        continue
                    _, y1 = F.project(cam, tg[f], np.array([[x, y, F.TALL_M]]), fov)
                    looks.append(rates(abs(y0[0] - y1[0])))
                a["looks"][(fov, w)] = looks
    players = set(gsr.player_id.unique())
    return acts, lineup, players


def attribute(a, lineup, key, rng):
    nums = list(lineup.get(a["side"], {}))
    others = [j for j in nums if j != a["jersey"]]
    votes = {}
    for pc, pw in a["looks"][key]:
        u = rng.random()
        if u < pc:
            r = a["jersey"]
        elif u < pc + pw and others:
            r = others[rng.integers(len(others))]
        else:
            continue
        votes[r] = votes.get(r, 0) + 1
    if not votes:
        return None
    best = max(votes.values())
    top = [j for j, v in votes.items() if v == best]
    return lineup[a["side"]].get(top[0]) if len(top) == 1 else None


def main():
    sha = check()
    per_match = {}
    for match, half in F.HALVES:
        acts, lineup, players = per_half(match, half)
        m = per_match.setdefault(match, {"acts": [], "players": set()})
        m["acts"] += [dict(a, lineup=lineup) for a in acts]
        m["players"] |= players
        print(match, half, len(acts), "actions", flush=True)
    rng = np.random.default_rng(0)
    res = {}
    for fov in F.FOVS:
        for w in WINDOWS:
            key = (fov, w)
            acc = {t: [] for t in TYPES}
            within = {t: [] for t in TYPES}
            within_act = {t: [] for t in TYPES}
            looks = {t: [] for t in TYPES}
            for _ in range(DRAWS):
                for match, m in per_match.items():
                    for t in TYPES:
                        aa = [a for a in m["acts"] if a["type"] == t]
                        true = pd.Series([a["player_id"] for a in aa], dtype=object).value_counts()
                        got = [attribute(a, a["lineup"], key, rng) for a in aa]
                        pred = pd.Series([g for g in got if g is not None], dtype=object).value_counts()
                        if aa:
                            acc[t].append(np.mean([g == a["player_id"] for g, a in zip(got, aa)]))
                        tr = true.reindex(sorted(m["players"])).fillna(0)
                        pr = pred.reindex(sorted(m["players"])).fillna(0)
                        diff = (pr - tr).abs()
                        within[t].append(float((diff <= 1).mean()))
                        if (tr > 0).any():
                            within_act[t].append(float((diff[tr > 0] <= 1).mean()))
            for t in TYPES:
                for a in (a for m in per_match.values() for a in m["acts"] if a["type"] == t):
                    looks[t].append(len(a["looks"][key]))
            res[f"{fov} deg, +/- {w} s"] = {t: {
                "actions attributed to the right player": round(float(np.mean(acc[t])), 3),
                "players within +/- 1 (all who played)": round(float(np.mean(within[t])), 3),
                "players within +/- 1 (with >= 1 true action)": round(float(np.mean(within_act[t])), 3),
                "looks in view per action (mean)": round(float(np.mean(looks[t])), 2)} for t in TYPES}
            print(key, json.dumps(res[f"{fov} deg, +/- {w} s"]), flush=True)
    counts = {t: sum(a["type"] == t for m in per_match.values() for a in m["acts"]) for t in TYPES}
    per_player = {}
    for t in TYPES:
        c = pd.Series([a["player_id"] for m in per_match.values() for a in m["acts"] if a["type"] == t]).value_counts()
        per_player[t] = {"max per player-match (pooled ids)": int(c.max()) if len(c) else 0}
    met = {t: any(v[t]["players within +/- 1 (all who played)"] >= 0.90 for v in res.values()) for t in TYPES}
    out = {"plan sha256": sha, "matches": len(per_match), "actions": counts, "action counts": per_player,
           "met (>= 90% of players within +/- 1 at one view)": met, "results": res}
    (HERE / "actor_id.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1, default=str))


if __name__ == "__main__":
    main()
