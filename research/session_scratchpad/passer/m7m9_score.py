"""The registered M7 / M9 test of the actor rule (m7m9_protocol.json): A, B
and C (src/passer.py) on the labelled actions inside each clip's ball
range, scored as possession.py.

    VEO_UNSEAL=... python m7m9_score.py
"""
import json

import pandas as pd

import passer as P
import possession as C
from src.unseal import unsealed

MATCHES = ("128057", "132831")
RULE_C = "C [-0.4, +0.2] s"


def main():
    import sys
    from pathlib import Path
    out_name = "m7m9_test.json"
    if "--gpu" in sys.argv:            # another set of runs (nograss_protocol.json)
        P.GPU = P.ROOT / sys.argv[sys.argv.index("--gpu") + 1]
        out_name = sys.argv[sys.argv.index("--out") + 1]
    for m in MATCHES:
        if not unsealed(m, "passer: registered M7 / M9 actor test"):
            raise SystemExit("VEO_UNSEAL needed for M7 / M9")
    units = sorted(p.name for p in P.GPU.iterdir() if p.name.split("_")[1] in MATCHES
                   and (P.GPU / p.name / f"output_{p.name}" / "metric_v3s_link.parquet").exists())
    rows = []
    for w in units:
        fr = json.loads((P.GPU / w / ".cache" / f"v3_balls_{w}.parquet.meta.json").read_text())["frames"]
        rows += C.unit(w, min(fr), max(fr))
    df = pd.DataFrame(rows).drop_duplicates(["match", "abs", "label"])
    cols = ["A", "B", RULE_C]
    out = {"protocol sha256": (P.HERE / "m7m9_protocol.json.sha256").read_text().split()[0], "units": len(units),
           "all actions": {c: C.summary(df, c) for c in cols},
           "passes": {c: C.summary(df[df.label.isin(P.PASSES)], c) for c in cols},
           "per match": {str(m): {c: C.summary(g, c) for c in cols} for m, g in df.groupby("match")}}
    both = df[df.B.notna() & df[RULE_C].notna()]
    out["paired C vs B"] = {"C right B wrong": int(((both[RULE_C] == True) & (both.B == False)).sum()),  # noqa: E712
                            "B right C wrong": int(((both.B == True) & (both[RULE_C] == False)).sum())}  # noqa: E712
    c, b = out["all actions"][RULE_C], out["all actions"]["B"]
    out["holds"] = bool(c["wilson"][0] >= 0.45 and c["share right"] >= b["share right"])
    print(json.dumps(out, indent=1, default=str))
    (P.HERE / out_name).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
