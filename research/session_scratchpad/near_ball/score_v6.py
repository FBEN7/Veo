"""plan_v6 (hashed first): actor detected at S1, A (conf 0.20) against A'
(each clip's auto-tuned conf_player), on the 9 matches. Local CPU.

    VEO_UNSEAL=... python score_v6.py
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
from src import player_filter  # noqa: E402
import score_v4  # noqa: E402
from diag_v5 import COLS, DET, HI, MARGIN, TRK, track  # noqa: E402

SEALED = {"128057", "132831"}
TMP = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))


def run_dir(w):
    return Path(os.path.realpath(ROOT / f".cache/gpu_shots3/{w}/output_{w}/metric_v3s_link.parquet")).parent / "shots_v3s_link"


def conf_of(w):
    return float(json.loads((run_dir(w) / "profile.json").read_text())["conf_player"])


def unit(args):
    w, a = args
    pitch = json.loads((score_v4.CLIPS / f"{w}.json").read_text())["pitch"]
    det = pd.read_parquet(DET / f"{w}.parquet")
    a0 = pd.read_parquet(TRK / f"{w}_A.parquet")
    arms = {"A": a0[a0.frame <= HI],
            "A'": player_filter.inside_pitch(track(det[det.conf >= conf_of(w)]), pitch, MARGIN)}
    out = []
    for name, t in arms.items():
        d = TMP / name
        d.mkdir(exist_ok=True)
        t = t.assign(team=None, in_s4=False)[COLS]
        for v in ("A", "A2", "B2"):
            t.to_parquet(d / f"{w}_{v}.parquet")
        out += [dict(x, variant=name) for x in score_v4.unit(w, d, a) if x["variant"] == "A" and x["stage"] == "S1"]
    return out


def main():
    from src.unseal import unsealed
    for m in sorted(SEALED):
        if not unsealed(m, "near_ball plan_v6: fixed vs auto-tuned confidence threshold on the opened M7 / M9 clips"):
            raise SystemExit("VEO_UNSEAL needed")
    st = pd.read_parquet(HERE / "stages.parquet")
    acts = st[st.stage == "S0 raw"][["unit", "frame", "player_id"]].drop_duplicates()
    acts["f"] = acts.frame
    confs = {w: conf_of(w) for w in acts.unit.unique()}
    excluded = sorted(w for w, c in confs.items() if c < 0.10)
    jobs = [(w, a) for w, a in acts.groupby("unit") if w not in excluded]
    rows = []
    with Pool(os.cpu_count()) as pool:
        for i, r in enumerate(pool.imap_unordered(unit, jobs)):
            rows += r
            if i % 25 == 0:
                print(i, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(HERE / "v6.parquet")
    res = {}
    for v, g in df.groupby("variant"):
        res[v] = {"actions": len(g), "detected": round(float((g.kind == "detected").mean()), 4),
                  "control": round(float(g.control.mean()), 4),
                  "unmatched per action frame": round(float(g.drop_duplicates(["unit", "frame"]).unmatched.mean()), 3)}
    per_match = {}
    for m, g in df.groupby("match"):
        per_match[m] = {v: {"detected": round(float((h.kind == "detected").mean()), 3),
                            "unmatched per action frame": round(float(h.drop_duplicates(["unit", "frame"]).unmatched.mean()), 2)}
                        for v, h in g.groupby("variant")}
        per_match[m]["median conf_player"] = float(pd.Series([confs[w] for w in g.unit.unique()]).median())
    primary = score_v4.boot(df, "A'", "A", stage="S1")
    a, b = res["A"], res["A'"]
    guards = {"unmatched per action frame, A <= 1.2 x A'": bool(a["unmatched per action frame"] <= 1.2 * b["unmatched per action frame"]),
              "control, A >= A' - 0.01": bool(a["control"] >= b["control"] - 0.01)}
    kept = [w for w, _ in jobs]
    out = {"plan_v6 sha256": (HERE / "plan_v6.json.sha256").read_text().split()[0],
           "units": len(kept), "excluded (conf_player < 0.10)": excluded,
           "share of kept units at conf_player 0.20": round(sum(confs[w] == 0.20 for w in kept) / len(kept), 3),
           "per arm (S1)": res, "primary: actor detected at S1, A - A'": primary, "guards": guards,
           "bar met (gain >= 0.03, lower bound > 0)": bool(primary["gain"] >= 0.03 and primary["95% bootstrap over matches"][0] > 0),
           "per match": per_match}
    (HERE / "v6.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per match"}, indent=1))


if __name__ == "__main__":
    main()
