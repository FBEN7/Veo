"""tdeed_probe (plan.json, hashed first): the virtual camera's centre per
frame for every unit, from our player detections; writes the stage files
the Kaggle kernel reads (units.json, centres.parquet).

    VEO_UNSEAL=... python centres.py OUT_DIR
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "negatives"))
import score as S  # noqa: E402

FPS = 25


def check():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    if hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest() != want:
        raise SystemExit("plan.json does not match its sha256")


def unit_list():
    """(name, match, span or None, source tree) for the 249 clips and 47 windows."""
    S.patch_info()
    wins = json.loads((S.HERE / "windows.json").read_text())["windows"]
    ok = set()
    for p in (S.HERE / "state").glob("*.json"):
        ok |= set(json.loads(p.read_text()).get("ok", []))
    out = [(u["name"], u["match"], u.get("span"), S.V3.NEW) for u in S.EA.units()]
    out += [(w["name"], w["match"], None, S.TREE_NEW) for w in wins if w["name"] in ok]
    out += [(w, w.split("_")[1], None, S.TREE_DEV) for w in S.DEV_WINDOWS]
    return out


def centres(name, tree, n):
    m = pd.read_parquet(tree / name / f"output_{name}" / "metric_v3s_link.parquet",
                        columns=["frame", "cls", "px_raw", "py_raw"])
    p = m[(m.cls == "player") & m.px_raw.notna()].groupby("frame")[["px_raw", "py_raw"]].median()
    c = p.reindex(np.arange(n)).interpolate(limit_direction="both")
    c = c.rolling(2 * FPS, center=True, min_periods=1).median().rolling(FPS, center=True, min_periods=1).mean()
    return pd.DataFrame({"frame": np.arange(n), "cx": c.px_raw.to_numpy(), "cy": c.py_raw.to_numpy()})


def main():
    check()
    from src.unseal import unsealed
    for m in S.V3.NEW_PLAYERS:
        if not unsealed(m, "tdeed probe: crop centres from M7 / M9 player tracks (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    units, rows = [], []
    for name, match, span, tree in unit_list():
        info = S.LS.sd.info_of(name)
        n = int(info["frames"])
        c = centres(name, tree, n)
        if c.cx.isna().any():
            raise SystemExit(f"{name}: no player to centre on")
        rows.append(c.assign(unit=name))
        units.append({"name": name, "match": match, "span": span, "frames": n,
                      "width": int(info["width"]), "height": int(info["height"])})
    pd.concat(rows, ignore_index=True).to_parquet(out / "centres.parquet")
    (out / "units.json").write_text(json.dumps(units, indent=1))
    print(len(units), "units", flush=True)


if __name__ == "__main__":
    main()
