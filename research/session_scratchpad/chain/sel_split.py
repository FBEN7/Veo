"""Exploratory (not registered), after cand.json: for the 'selection'
actions, the confidence of the near candidates (the linker sees only those
>= tau 0.10, metric_step.py), and what the linked ball does in the window.

    VEO_UNSEAL=... python sel_split.py
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from src import soccertrack_v2 as st  # noqa: E402
from src.pixel_scale import to_pitch_metres  # noqa: E402
from chain import BALL_DF, BALL_M, CLIPS, LO, HI, SEALED, TREE  # noqa: E402
from cand import cal  # noqa: E402

TAU = 0.10


def unit(w):
    run = Path(os.path.realpath(TREE / w / f"output_{w}" / "metric_v3s_link.parquet")).parent
    c = cal(w.split("_")[1])
    v = to_pitch_metres(pd.read_parquet(run / "v3s_rows.parquet"), lambda uv: st.image_to_pitch(uv, c))
    v = v.assign(x=v.x - 52.5, y=v.y - 34.0)
    m = pd.read_parquet(run / "metric_v3s_link.parquet")
    b = m[m.cls == "ball"].assign(x=lambda d: d.x - 52.5, y=lambda d: d.y - 34.0)
    ev = pd.read_csv(CLIPS / f"{w}_events.csv")
    ev = ev[(ev.frame >= LO) & (ev.frame <= HI) & ev.player_id.notna() & ev.x.notna()]
    out = []
    for r in ev.itertuples():
        f = int(r.frame) - int(r.frame) % 2
        win = v[(v.frame - f).abs() <= BALL_DF]
        d = np.linalg.norm(win[["x", "y"]].to_numpy(float) - [r.x, r.y], axis=1)
        near = win[d <= BALL_M]
        bw = b[(b.frame - f).abs() <= BALL_DF]
        out.append({"unit": w, "frame": int(r.frame), "kind": r.kind, "side": r.side,
                    "near max conf": float(near.confidence.max()) if len(near) else np.nan,
                    "linked rows in window": int(len(bw)),
                    "linked detection_method": ",".join(sorted(set(bw.detection_method.astype(str)))) if len(bw) else ""})
    return out


def main():
    from src.unseal import unsealed
    for m in sorted(SEALED):
        if not unsealed(m, "chain exploratory: selection split (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    cd = pd.read_parquet(HERE / "cand.parquet")
    with Pool(os.cpu_count()) as pool:
        s = pd.DataFrame([r for rs in pool.imap_unordered(unit, sorted(cd.unit.unique())) for r in rs])
    key = ["unit", "frame", "kind", "side"]
    cd["k"] = cd.groupby(key).cumcount(); s["k"] = s.groupby(key).cumcount()
    df = cd.merge(s, on=key + ["k"], how="left", validate="one_to_one")
    sel = df[df.selection]
    n = len(df)
    out = {"status": "exploratory, not registered; after cand.json", "actions": n,
           "selection (share of all actions)": round(len(sel) / n, 4),
           "selection, near candidate below tau 0.10 (never seen by the linker)": round(float((sel["near max conf"] < TAU).sum() / n), 4),
           "selection, near candidate >= 0.10 (seen, not chosen)": round(float((sel["near max conf"] >= TAU).sum() / n), 4),
           "near max conf quartiles (selection)": [round(float(q), 3) for q in sel["near max conf"].quantile([.25, .5, .75])],
           "selection, no linked row in the window": round(float((sel["linked rows in window"] == 0).sum() / n), 4),
           "linked ok, near max conf quartiles": [round(float(q), 3) for q in df[df["linked ok"]]["near max conf"].quantile([.25, .5, .75])]}
    (HERE / "reading_sel_split.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
