import sys
sys.path.insert(0, "/home/user/Veo")
import numpy as np, pandas as pd
from src.team_assignment_v2 import cluster_teams, FIXED_VIEW_CORE_DEPTH

def make(seed):
    rng = np.random.default_rng(seed)
    blue = 225 * np.pi / 180
    def kit(hue, sat, val):
        return np.array([np.cos(hue), np.sin(hue), sat, val])
    feats, depth, truth = {}, {}, {}
    t = 0
    for name, n, mk, (d0, d1) in (
            ("A", 30, lambda: kit(blue + rng.normal(0, 0.1), 0.95 + rng.normal(0, 0.02), 0.42 + rng.normal(0, 0.03)), (0.4, 2.5)),
            ("B", 30, lambda: kit(blue + rng.normal(0, 0.3) if rng.random() > 0.2 else rng.uniform(10, 70) * np.pi / 180,
                                  0.28 + rng.normal(0, 0.05), 0.36 + rng.normal(0, 0.03)), (0.4, 2.5)),
            ("keeper", 3, lambda: kit(30 * np.pi / 180 + rng.normal(0, 0.1), 0.85, 0.6), (0.2, 1.0)),
            ("keeper", 3, lambda: kit(60 * np.pi / 180 + rng.normal(0, 0.1), 0.8, 0.7), (0.2, 1.0)),
            ("referee", 3, lambda: kit(blue + rng.normal(0, 0.1), 0.9, 0.12 + rng.normal(0, 0.02)), (0.5, 2.0)),
            ("bench", 45, lambda: kit(blue + rng.normal(0, 0.1), 0.95 + rng.normal(0, 0.02), 0.13 + rng.normal(0, 0.02)), (-0.5, 0.1))):
        for _ in range(n):
            feats[t], depth[t], truth[t] = mk(), rng.uniform(d0, d1), name
            t += 1
    return feats, depth, truth

def purity(team_map, truth):
    got = pd.DataFrame({"team": pd.Series(team_map), "truth": pd.Series(truth)})
    teams = got[got.team.isin(("team_A", "team_B"))]
    return teams, pd.crosstab(teams.team, teams.truth)

def passes(fm, truth):
    teams, table = purity(fm, truth)
    ok1 = not {"bench", "referee"} & set(table.columns)
    ok2 = sorted(table.idxmax(axis=1)) == ["A", "B"]
    ok3 = table.max(axis=1).sum() / max(len(teams),1) >= 0.95
    ok4 = len(teams) >= 0.8 * 60
    return ok1 and ok2 and ok3 and ok4, (ok1, ok2, ok3, ok4, len(teams))

for seed in range(0, int(sys.argv[1]) if len(sys.argv) > 1 else 1):
    feats, depth, truth = make(seed)
    core = {k for k, d in depth.items() if d >= FIXED_VIEW_CORE_DEPTH}
    row = [seed]
    for kf in ("hsv4", "chroma"):
        for c in (core, None):
            fm = cluster_teams(feats, kit_feature=kf, fit_tracks=c, verbose=False)
            row.append((kf, c is not None, passes(fm, truth)))
    old = cluster_teams(feats, verbose=False)
    _, table = purity(old, truth)
    row.append(("old bench team", "bench" in table.idxmax(axis=1).values))
    print(row)
