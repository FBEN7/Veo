"""Rule A of passer.py with the ball's gaps filled (ball_interp/plan.json
feasibility: linear, up to 0.5 / 1 s), on the matches v3 was trained on
(M1 is not measurable, M7 / M9 have no linked ball).

    python fill_check.py
"""
import json

import pandas as pd

import passer as P

out = {}
units = sorted(p.name for p in P.GPU.iterdir() if (P.CLIPS / f"{p.name}_events.csv").exists()
               and (P.GPU / p.name / f"output_{p.name}" / "metric_v3s_link.parquet").exists()
               and p.name.split("_")[1] not in P.UNSEEN)
for fill in (0, 12, 25):
    P.FILL = fill
    df = pd.DataFrame([r for w in units for r in P.unit(w)]).drop_duplicates(["match", "half", "abs", "label"])
    out[f"fill {fill / 25:.1f} s"] = {"passes": P.summary(df[df.label.isin(P.PASSES)]), "all actions": P.summary(df)}
    print(fill, json.dumps(out[f"fill {fill / 25:.1f} s"]["passes"]), flush=True)
(P.HERE / "fill_check.json").write_text(json.dumps(out, indent=1, default=str))
