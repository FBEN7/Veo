"""Rule C with the relinked pitch tracks (plan.json) against the old ones:
the matches v3 was trained on (inside the ball range) and M7 / M9, plus the
verified balls on M9's overlapping clip.

    VEO_UNSEAL=... python score.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "passer"))
import passer as P  # noqa: E402
import possession as C  # noqa: E402
from src.unseal import unsealed  # noqa: E402

RULE = "C [-0.4, +0.2] s"
NEW_PLAYERS = {"128057", "132831"}


def units_of(matches, root):
    out = []
    for p in sorted((P.ROOT / root).iterdir()):
        w = p.name
        if (w.split("_")[1] in matches and (p / f"output_{w}" / "metric_v3s_link_nograss.parquet").exists()
                and (P.CLIPS / f"{w}_events.csv").exists() and (p / f"output_{w}" / "metric_v3s_link.parquet").exists()):
            out.append(w)
    return out


def run(units, root, metric):
    P.GPU, P.METRIC = P.ROOT / root, metric
    rows = []
    for w in units:
        fr = json.loads((P.ROOT / ".cache/gpu" / w / ".cache" / f"v3_balls_{w}.parquet.meta.json").read_text())["frames"]
        rows += C.unit(w, min(fr), max(fr))
    df = pd.DataFrame(rows).drop_duplicates(["match", "abs", "label"])
    return {"all": C.summary(df, RULE), "passes": C.summary(df[df.label.isin(P.PASSES)], RULE),
            "per match": {str(m): C.summary(g, RULE)["share right"] for m, g in df.groupby("match")}}


def verified_m9():
    w = "st2_132831_2nd_f031521"
    s = json.loads((P.CLIPS / f"{w}.json").read_text())["start_frame"]
    f = pd.read_parquet(P.ROOT / ".cache/soccertrack_v2/ball_verified/v5p2/labels/frames.parquet")
    f = f[(f.window == "st2_132831_2nd_f030000") & (f.status == "ball") & (f.abs_frame >= s) & (f.abs_frame < s + 1426)]
    out = {}
    for name in ("metric_v3s_link.parquet", "metric_v3s_link_nograss.parquet"):
        m = pd.read_parquet(P.ROOT / ".cache/gpu_nograss" / w / f"output_{w}" / name)
        b = m[m.cls == "ball"].drop_duplicates("frame").set_index("frame")
        hit = n = 0
        for r in f.itertuples():
            lf = (r.abs_frame - s) - (r.abs_frame - s) % 2
            n += 1
            hit += bool(lf in b.index and np.hypot(b.loc[lf].px_raw - r.u, b.loc[lf].py_raw - r.v) <= 12)
        out[name] = {"verified balls": n, "hit": hit}
    return out


def main():
    for m in NEW_PLAYERS:
        if not unsealed(m, "ball_grass: relinked ball, rule C and verified balls"):
            raise SystemExit("VEO_UNSEAL needed")
    seen = {"117093", "118575", "118576", "118577", "118578", "128058", "132877"}
    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0]}
    for label, matches, root in (("seen matches", seen, ".cache/gpu"), ("M7 / M9", NEW_PLAYERS, ".cache/gpu_nograss")):
        us = units_of(matches, root)
        out[label] = {"units": len(us), "old": run(us, root, "metric_v3s_link.parquet"),
                      "relinked": run(us, root, "metric_v3s_link_nograss.parquet")}
        print(label, json.dumps({k: out[label][k]["all"] if k != "units" else out[label][k] for k in out[label]}), flush=True)
    out["M9 verified clip"] = verified_m9()
    o, n = out["seen matches"]["old"]["all"]["share right"], out["seen matches"]["relinked"]["all"]["share right"]
    a, b = out["M7 / M9"]["old"]["all"]["share right"], out["M7 / M9"]["relinked"]["all"]["share right"]
    out["adopt"] = bool(b > a and n >= o - 0.01)
    print(json.dumps(out, indent=1, default=str))
    (HERE / "score.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
