"""Identity by role, stage 1 (plan.json): from the truth's positions, can a
player's role in the team's shape stand in for who they are?

    python roles.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src import soccertrack_v2 as st  # noqa: E402

MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578", "128058", "132877"]
FPS, ROUNDS = 25, 20
WINDOWS = {"1 frame": 0, "10 s": 10, "60 s": 60}


def shapes(g: pd.DataFrame, frames) -> dict:
    """frame -> (player ids, positions relative to the team's centroid)."""
    out = {}
    for f, d in g[g.frame.isin(frames)].groupby("frame"):
        xy = d[["x", "y"]].to_numpy(float)
        out[f] = (d.player_id.to_numpy(), xy - xy.mean(axis=0))
    return out


def assign(rel: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Role of each row of rel, one role per player (Hungarian)."""
    cost = ((rel[:, None] - means[None]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    out = np.empty(len(rel), int)
    out[r] = c
    return out


def fit(sh: dict) -> np.ndarray:
    allxy = np.concatenate([v[1] for v in sh.values()])
    means = KMeans(10, n_init=10, random_state=0).fit(allxy).cluster_centers_
    for _ in range(ROUNDS):
        acc, n = np.zeros((10, 2)), np.zeros(10)
        for _, rel in sh.values():
            a = assign(rel, means)
            np.add.at(acc, a, rel)
            np.add.at(n, a, 1)
        means = acc / np.maximum(n, 1)[:, None]
    return means


def roles_windowed(sh: dict, means: np.ndarray, w: int) -> pd.DataFrame:
    """(frame, player, role): per frame (w = 0), or one role per player per
    window of w seconds by summed costs (needs the player followed that long)."""
    rows = []
    if w == 0:
        for f, (ids, rel) in sh.items():
            rows += zip([f] * len(ids), ids, assign(rel, means))
        return pd.DataFrame(rows, columns=["frame", "player", "role"])
    by_win = {}
    for f, (ids, rel) in sh.items():
        by_win.setdefault(f // (w * FPS), []).append((f, ids, rel))
    for _, items in by_win.items():
        players = sorted(set().union(*[set(i) for _, i, _ in items]))
        idx = {p: k for k, p in enumerate(players)}
        cost = np.zeros((len(players), 10))
        for _, ids, rel in items:
            c = ((rel[:, None] - means[None]) ** 2).sum(-1)
            for k, p in enumerate(ids):
                cost[idx[p]] += c[k]
        r, c = linear_sum_assignment(cost)
        role = dict(zip([players[i] for i in r], c))
        for f, ids, _ in items:
            rows += [(f, p, role[p]) for p in ids if p in role]
    return pd.DataFrame(rows, columns=["frame", "player", "role"])


def half_team(g: pd.DataFrame, ev: pd.DataFrame) -> dict:
    sec = np.arange(0, g.frame.max() + 1, FPS)
    sh = shapes(g, sec)
    sh = {f: v for f, v in sh.items() if len(v[0]) == 10}
    means = fit(sh)
    team = set(g.player_id)
    ev = ev[ev.player_id.isin(team)]
    sh_ev = shapes(g, ev.frame.unique())
    out = {"means": means, "seconds": len(sh)}
    for name, w in WINDOWS.items():
        r = roles_windowed(sh, means, w)
        holder = r.groupby("role").player.agg(lambda s: s.value_counts().index[0]).to_dict()
        purity = float((r.role.map(holder) == r.player).mean())
        right = 0
        if w == 0:
            re = roles_windowed(sh_ev, means, 0)
        else:   # the role the actor holds in the window around the action
            win = r.assign(win=r.frame // (w * FPS)).drop_duplicates(["win", "player"]).set_index(["win", "player"]).role
            re = ev.assign(win=ev.frame // (w * FPS)).join(win, on=["win", "player_id"]).rename(columns={"player_id": "player"})
        re = re.dropna(subset=["role"]).drop_duplicates(["frame", "player"])
        hit = ev.merge(re[["frame", "player", "role"]], left_on=["frame", "player_id"], right_on=["frame", "player"])
        right = int((hit.role.astype(int).map(holder) == hit.player_id).sum())
        out[name] = {"purity": round(purity, 4), "actions": len(ev), "actions placed": len(hit), "actions right": right,
                     "player-seconds": len(r)}
        if w == 0:
            out["holder"] = holder
    return out


def main():
    res, pooled = {}, {n: {"right": 0, "actions": 0, "ps": 0, "pure": 0.0} for n in WINDOWS}
    gk_actions, all_actions, cross = 0, 0, []
    for m in MATCHES:
        evm = st.events(m)
        res[m] = {}
        halves = {}
        for h in (1, 2):
            g = st.gsr_table(m, h)
            ev = evm[(evm.half == h) & evm.player_id.notna()].copy()
            ev["player_id"] = ev.player_id.astype(str)
            g = g[g.player_id.notna()].assign(player_id=lambda d: d.player_id.astype(str))
            gk = set(g[g.role == "goalkeeper"].player_id)
            all_actions += len(ev)
            gk_actions += int(ev.player_id.isin(gk).sum())
            out = g[g.role == "player"]
            for side, gs in out.groupby("side"):
                r = half_team(gs, ev[~ev.player_id.isin(gk)])
                halves[(h, frozenset(gs.player_id))] = r
                res[m][f"half {h} {side}"] = {k: v for k, v in r.items() if k not in ("means", "holder")}
                for n in WINDOWS:
                    pooled[n]["right"] += r[n]["actions right"]
                    pooled[n]["actions"] += r[n]["actions"]
                    pooled[n]["ps"] += r[n]["player-seconds"]
                    pooled[n]["pure"] += r[n]["purity"] * r[n]["player-seconds"]
                print(m, h, side, {n: (r[n]["purity"], r[n]["actions right"], r[n]["actions"]) for n in WINDOWS}, flush=True)
        # across halves: the same team (most shared players), roles paired on mirrored means
        one = [k for k in halves if k[0] == 1]
        two = [k for k in halves if k[0] == 2]
        for k1 in one:
            k2 = max(two, key=lambda k: len(k1[1] & k[1]))
            a, b = halves[k1], halves[k2]
            cost = ((a["means"][:, None] + b["means"][None]) ** 2).sum(-1)     # 2nd half mirrored: -means
            r, c = linear_sum_assignment(cost)
            same = sum(a["holder"].get(i) == b["holder"].get(j) for i, j in zip(r, c))
            cross.append(same / 10)
            res[m].setdefault("across halves (same holder)", []).append(same / 10)
    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0], "matches": MATCHES,
           "goalkeepers' share of actions (left out)": round(gk_actions / all_actions, 4),
           "pooled": {n: {"role purity": round(v["pure"] / v["ps"], 4),
                          "actions right": f"{v['right']} / {v['actions']}",
                          "share": round(v["right"] / v["actions"], 4)} for n, v in pooled.items()},
           "across halves, share of roles with the same holder": round(float(np.mean(cross)), 4),
           "per match": res}
    (HERE / "roles.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in out if k != "per match"}, indent=1))


if __name__ == "__main__":
    main()
