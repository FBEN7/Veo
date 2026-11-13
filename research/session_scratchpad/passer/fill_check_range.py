"""fill_check.py restricted to the labelled actions inside each clip's ball
range (the frames v3 ran on, its sidecar; ball_range.json).

    python fill_check_range.py
"""
import json

import pandas as pd

import passer as P

units = sorted(p.name for p in P.GPU.iterdir() if (P.CLIPS / f"{p.name}_events.csv").exists()
               and (P.GPU / p.name / f"output_{p.name}" / "metric_v3s_link.parquet").exists()
               and p.name.split("_")[1] not in P.UNSEEN)
rng, start = {}, {}
for w in units:
    f = json.loads((P.GPU / w / ".cache" / f"v3_balls_{w}.parquet.meta.json").read_text())["frames"]
    rng[w] = (min(f), max(f))
    start[w] = json.loads((P.CLIPS / f"{w}.json").read_text())["start_frame"]
out = {}
for fill in (0, 12, 25):
    P.FILL = fill
    df = pd.DataFrame([r for w in units for r in P.unit(w)])
    rel = df["abs"] - df.unit.map(start)
    df = df[(rel >= df.unit.map(lambda w: rng[w][0])) & (rel <= df.unit.map(lambda w: rng[w][1]))]
    df = df.drop_duplicates(["match", "half", "abs", "label"])
    key = f"fill {fill / 25:.1f} s"
    out[key] = {"passes": P.summary(df[df.label.isin(P.PASSES)]), "all actions": P.summary(df)}
    print(fill, json.dumps(out[key]), flush=True)
(P.HERE / "fill_check_range.json").write_text(json.dumps(out, indent=1, default=str))
