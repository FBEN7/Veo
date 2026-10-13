"""E5 scoring: pass + carry F1 at 1 s pooled over windows, COCO vs heatmap
runs, paired bootstrap over 10 s blocks. Usage: python e5_score.py W1 W2"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/Veo")
from score_soccernet import match_one_to_one  # noqa: E402
from score_st2_actions import GROUPS, FPS  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402

TOL, BLOCK = 1.0, 10.0
RUNS = ("events_coco", "events_heatmap")
rows = []  # (window, block, run, tp, n_pred, n_true)
for w in sys.argv[1:]:
    labels = pd.read_csv(DATA_DIR / "soccertrack_v2" / f"{w}_events.csv")
    for run in RUNS:
        events = json.loads((Path(f"output_{w}") / run / "events.json").read_text())
        for g in ("pass", "carry"):
            bas, ours = GROUPS[g]
            truths = list(labels[labels.label.isin(bas)].frame / FPS)
            preds = [e["timestamp_s"] for e in events if e["event_type"] in ours]
            m, _, _ = match_one_to_one(preds, truths, TOL)
            for _, ti, _ in m:
                rows.append((w, int(truths[ti] // BLOCK), run, 1, 0, 0))
            for p in preds:
                rows.append((w, int(p // BLOCK), run, 0, 1, 0))
            for t in truths:
                rows.append((w, int(t // BLOCK), run, 0, 0, 1))
df = pd.DataFrame(rows, columns=["w", "b", "run", "tp", "np", "nt"])
tab = df.groupby(["w", "b", "run"])[["tp", "np", "nt"]].sum().unstack("run", fill_value=0)
blocks = tab.index.tolist()


def f1(t):
    return {r: 2 * t[("tp", r)].sum() / max(t[("np", r)].sum() + t[("nt", r)].sum(), 1)
            for r in RUNS}


full = f1(tab)
for r in RUNS:
    print(r, "TP", int(tab[("tp", r)].sum()), "pred", int(tab[("np", r)].sum()),
          "labels", int(tab[("nt", r)].sum()), "F1", round(full[r], 3))
rng = np.random.default_rng(0)
diffs = []
for _ in range(2000):
    pick = rng.integers(0, len(blocks), len(blocks))
    s = f1(tab.iloc[pick])
    diffs.append(s[RUNS[1]] - s[RUNS[0]])
print("F1 heatmap - coco", round(full[RUNS[1]] - full[RUNS[0]], 3), "95%",
      np.percentile(diffs, [2.5, 97.5]).round(3), "blocks", len(blocks))
