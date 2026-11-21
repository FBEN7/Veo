"""plan_v7 (hashed first): actor detected at S4, A (conf 0.20, plan_v4's A
files) against A' (each clip's conf_player, kernel veo-nearball-track-v7),
on plan_v6's 221 units.

    VEO_UNSEAL=... python score_v7.py TRK7_DIR
"""
import json
import os
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import score_v4  # noqa: E402

SEALED = {"128057", "132831"}
TRK = ROOT / ".cache/near_ball_trk/trk"
TMP = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
TRK7 = None


def unit(args):
    w, a = args
    d = TMP / w
    d.mkdir()
    for v in ("A", "A2", "B2"):                  # score_v4.unit loads all three
        (d / f"{w}_{v}.parquet").symlink_to(TRK7 / f"{w}_Ap.parquet")
    pick = lambda rows, name: [dict(x, variant=name) for x in rows if x["variant"] == "A" and x["stage"] == "S4"]
    out = pick(score_v4.unit(w, TRK, a), "A") + pick(score_v4.unit(w, d, a), "A'")
    for v in ("A", "A2", "B2"):
        (d / f"{w}_{v}.parquet").unlink()
    d.rmdir()
    return out


def main():
    global TRK7
    from src.unseal import unsealed
    for m in sorted(SEALED):
        if not unsealed(m, "near_ball plan_v7: fixed vs auto-tuned confidence threshold at S4 on the opened M7 / M9 clips"):
            raise SystemExit("VEO_UNSEAL needed")
    TRK7 = Path(sys.argv[1])
    st = pd.read_parquet(HERE / "stages.parquet")
    acts = st[st.stage == "S0 raw"][["unit", "frame", "player_id"]].drop_duplicates()
    acts["f"] = acts.frame
    have = {p.name[:-len("_Ap.parquet")] for p in TRK7.glob("*_Ap.parquet")}
    jobs = [(w, a) for w, a in acts.groupby("unit") if w in have]
    rows = []
    with Pool(os.cpu_count()) as pool:
        for i, r in enumerate(pool.imap_unordered(unit, jobs)):
            rows += r
            if i % 25 == 0:
                print(i, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(HERE / "v7.parquet")
    res = {}
    for v, g in df.groupby("variant"):
        res[v] = {"actions": len(g), "detected": round(float((g.kind == "detected").mean()), 4),
                  "control": round(float(g.control.mean()), 4),
                  "unmatched per action frame": round(float(g.drop_duplicates(["unit", "frame"]).unmatched.mean()), 3),
                  "same track +-1s (detected actors)": round(float(g["same track +-1s"].mean()), 4)}
    per_match = {m: {v: {"detected": round(float((h.kind == "detected").mean()), 3),
                         "unmatched per action frame": round(float(h.drop_duplicates(["unit", "frame"]).unmatched.mean()), 2)}
                     for v, h in g.groupby("variant")} for m, g in df.groupby("match")}
    primary = score_v4.boot(df, "A'", "A", stage="S4")
    a, b = res["A"], res["A'"]
    guards = {"unmatched per action frame, A <= 1.2 x A'": bool(a["unmatched per action frame"] <= 1.2 * b["unmatched per action frame"]),
              "control, A >= A' - 0.01": bool(a["control"] >= b["control"] - 0.01)}
    out = {"plan_v7 sha256": (HERE / "plan_v7.json.sha256").read_text().split()[0],
           "units": len(jobs), "units missing": sorted(set(acts.unit) - have),
           "per arm (S4)": res, "primary: actor detected at S4, A - A'": primary, "guards": guards,
           "bar met (gain >= 0.03, lower bound > 0, guards)": bool(primary["gain"] >= 0.03 and primary["95% bootstrap over matches"][0] > 0
                                                                   and all(guards.values())),
           "per match": per_match}
    (HERE / "v7.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per match"}, indent=1))


if __name__ == "__main__":
    main()
