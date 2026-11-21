"""chain plan v2 (plan_v2.json, hashed first): selection or detection, when
our ball is not within 3 m of the action.

    VEO_UNSEAL=... python cand.py
"""
import json
import os
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src import soccertrack_v2 as st  # noqa: E402
from src.pixel_scale import to_pitch_metres  # noqa: E402
from chain import BALL_DF, BALL_M, CLIPS, LO, HI, SEALED, TREE, boot  # noqa: E402


@lru_cache(None)
def cal(match):
    return st.calibration(match)


def unit(w):
    run = Path(os.path.realpath(TREE / w / f"output_{w}" / "metric_v3s_link.parquet")).parent
    v = pd.read_parquet(run / "v3s_rows.parquet")
    c = cal(w.split("_")[1])
    v = to_pitch_metres(v, lambda uv: st.image_to_pitch(uv, c))
    v = v.assign(x=v.x - 52.5, y=v.y - 34.0)
    v["top"] = v.confidence == v.groupby("frame").confidence.transform("max")
    ev = pd.read_csv(CLIPS / f"{w}_events.csv")
    ev = ev[(ev.frame >= LO) & (ev.frame <= HI) & ev.player_id.notna() & ev.x.notna()]
    out = []
    for r in ev.itertuples():
        f = int(r.frame) - int(r.frame) % 2
        win = v[(v.frame - f).abs() <= BALL_DF]
        d = np.linalg.norm(win[["x", "y"]].to_numpy(float) - [r.x, r.y], axis=1) if len(win) else np.zeros(0)
        near = d <= BALL_M
        out.append({"unit": w, "frame": int(r.frame), "kind": r.kind, "side": r.side, "candidate near": bool(near.any()),
                    "top candidate near": bool((near & win.top.to_numpy()).any()) if len(win) else False})
    return out


def main():
    from src.unseal import unsealed
    for m in sorted(SEALED):
        if not unsealed(m, "chain plan v2: ball candidates (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    ch = pd.read_parquet(HERE / "chain.parquet")
    units = sorted(ch.unit.unique())
    with Pool(os.cpu_count()) as pool:
        c = pd.DataFrame([r for rs in pool.imap_unordered(unit, units) for r in rs])
    # one row per (unit, frame) in c may cover several actions at the same frame: join on position in order
    ch = ch.reset_index(drop=True)
    key = ["unit", "frame", "kind", "side"]
    ch["k"] = ch.groupby(key).cumcount()
    c["k"] = c.groupby(key).cumcount()
    df = ch.merge(c, on=key + ["k"], how="left", validate="one_to_one")
    assert df["candidate near"].notna().all(), "unmatched actions"
    s1 = df["S1 ball found"]
    df["linked ok"] = s1
    df["selection, top-1"] = ~s1 & df["top candidate near"]
    df["selection, lower rank"] = ~s1 & df["candidate near"] & ~df["top candidate near"]
    df["detection"] = ~s1 & ~df["candidate near"]
    df["selection"] = df["selection, top-1"] | df["selection, lower rank"]
    df.to_parquet(HERE / "cand.parquet")
    cats = ["linked ok", "selection, top-1", "selection, lower rank", "detection"]
    share = lambda g: {k: round(float(g[k].mean()), 4) for k in cats}
    kinds = df.kind.where(df.kind.isin(["pass", "carry"]), "other")
    out = {"plan sha256": (HERE / "plan_v2.json.sha256").read_text().split()[0], "actions": len(df),
           "categories (share of all actions)": share(df),
           "selection minus detection": boot(df, "selection", "detection"),
           "per kind": {k: dict(share(g), actions=len(g)) for k, g in df.groupby(kinds)},
           "per match": {m: dict(share(g), actions=len(g)) for m, g in df.groupby("match")}}
    (HERE / "cand.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per match"}, indent=1))


if __name__ == "__main__":
    main()
