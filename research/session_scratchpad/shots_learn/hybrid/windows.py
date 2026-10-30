"""The hybrid ball's windows (protocol.json): per shots unit, the even
frames 350..1074 (the clips' ball range) within 2 s of a frame where the
v4 spotter's out-of-fold probability (oof_v4.parquet) is at least 0.10;
each unit's v4r checkpoint is the fold that did not train on its match
(fold A matches -> v4rB, fold B -> v4rA, M6 / M7 / M9 -> v4rA).
Writes frames.parquet (unit, frame, model).

    python windows.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P_MIN, HALF_S, FPS = 0.10, 2.0, 25.0
FIRST, LAST = 350, 1074
FOLD_A = {"117092", "117093", "118575", "118577"}
FOLD_B = {"118576", "128058", "132877"}


def model_of(match: str) -> str:
    return "v4rB.600" if match in FOLD_A else "v4rA.600"


def main():
    p = pd.read_parquet(HERE / "oof_v4.parquet")
    rows = []
    for (w, m), g in p.groupby(["unit", "match"]):
        fr, pr = g.frame.to_numpy(), g.p.to_numpy()
        hot = fr[pr >= P_MIN]
        inw = np.zeros(len(fr), bool)
        for h in hot:
            inw |= np.abs(fr - h) <= HALF_S * FPS
        keep = fr[inw & (fr >= FIRST) & (fr <= LAST) & (fr % 2 == 0)]
        rows += [(w, int(f), model_of(str(m))) for f in keep]
    out = pd.DataFrame(rows, columns=["unit", "frame", "model"])
    out.to_parquet(HERE / "frames.parquet")
    print(len(out), "frames,", out.unit.nunique(), "units,", out.model.value_counts().to_dict())


if __name__ == "__main__":
    main()
