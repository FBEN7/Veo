"""The hybrid ball (protocol.json here): per shots unit, v3b's kept rows
outside the windows and v4r's rows inside them (marks and static points
dropped from v4r's rows as metric_step.py does, over the frames v4r ran
on), relinked without the grass tests as ball_grass/relink.py; then the
shots v4 spotter (learn_v4.py: features, nested leave-one-match-out) on
these tracks against the same spotter on the v3b tracks (cv_v4.json).

    python assemble.py ROWS_DIR          # Kaggle's rows/<unit>.parquet
    VEO_UNSEAL=... python assemble.py --cv
    python assemble.py --control         # the same steps on v3b's own rows (plumbing check)
    VEO_UNSEAL=... python assemble.py --cv-control
"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball/events_tune"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "v2"))
sys.path.insert(0, str(HERE.parent / "v3"))
sys.path.insert(0, str(HERE.parent / "v4"))
import learn_shots as LS  # noqa: E402
import learn_v2 as V  # noqa: E402
import learn_v3 as V3  # noqa: E402
import learn_v4 as V4  # noqa: E402
import score_soccernet as sc  # noqa: E402
import tune_rules as tr  # noqa: E402
from src import ball_spots as bs  # noqa: E402
from src import pixel_scale  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402

HYB = ROOT / ".cache/gpu_shots_hybrid"
CONTROL = ROOT / ".cache/gpu_shots_hybrid_control"   # v3b's own rows put through the same steps


def base_of(w: str) -> Path:
    m = w.split("_")[1]
    return ROOT / (".cache/gpu_nograss" if m in V3.NEW_PLAYERS else ".cache/gpu") / w / f"output_{w}"


def hybrid_rows(w: str, frames: np.ndarray, r4: pd.DataFrame) -> pd.DataFrame:
    cal = st.calibration(w.split("_")[1])
    n = len(frames)
    r4 = bs.drop_marks(r4, bs.spot_marks(r4, cal, n))
    keep = np.ones(len(r4), bool)
    for x, y in tr.static_points(r4, n):
        keep &= np.hypot(r4.px.to_numpy() - x, r4.py.to_numpy() - y) > 1.5 * tr.STATIC_PX
    v3 = pd.read_parquet(base_of(w) / "v3s_rows.parquet")
    lo, hi = window_spans(frames)
    inside = np.zeros(len(v3), bool)
    f = v3.frame.to_numpy()
    for a, b in zip(lo, hi):
        inside |= (f >= a) & (f <= b + 1)            # the odd frame after an even one goes too
    return pd.concat([v3[~inside], r4[keep].reindex(columns=v3.columns)], ignore_index=True) \
        .sort_values(["frame", "confidence"], ascending=[True, False]).reset_index(drop=True)


def window_spans(frames: np.ndarray):
    """Runs of consecutive even frames as (first, last) pairs."""
    f = np.sort(np.asarray(frames, int))
    if not len(f):
        return np.array([], int), np.array([], int)
    cut = np.flatnonzero(np.diff(f) > 2) + 1
    return f[np.r_[0, cut]], f[np.r_[cut - 1, len(f) - 1]]


def build(rows_dir: Path | None, tree: Path):
    """The hybrid tracks into `tree`; `rows_dir` None: the control, v3b's
    kept rows at the window frames in place of v4r's."""
    fr = pd.read_parquet(HERE / "frames.parquet")
    done, missing = 0, []
    for w, g in fr.groupby("unit"):
        base = base_of(w)
        if rows_dir is None:
            v3 = pd.read_parquet(base / "v3s_rows.parquet")
            r4 = v3[v3.frame.isin(g.frame)]
        elif (rows_dir / f"{w}.parquet").exists():
            r4 = pd.read_parquet(rows_dir / f"{w}.parquet")
        else:                                        # no clip on Kaggle: v3b kept
            missing.append(w)
            continue
        out = tree / w / f"output_{w}"
        out.mkdir(parents=True, exist_ok=True)
        rows = hybrid_rows(w, g.frame.to_numpy(), r4)
        rows.to_parquet(out / "v3s_rows.parquet")
        cal = st.calibration(w.split("_")[1])
        dst = sc.swap_ball_rows(base / "events_v1", out / "shots_link", rows)
        # without the ball's grass test no grass fraction is needed (left empty), as relink.py
        pd.read_parquet(dst / "tracks_teams.parquet").assign(grass_frac=float("nan")) \
            .to_parquet(dst / "tracks_grass.parquet")
        with contextlib.redirect_stdout(io.StringIO()):
            merged = sc.merged_from_cached(dst, cal, select="link", top_k=4, tau=0.10,
                                           link_params=tr.LINK, ball_grass_filter=False)
        metric = pixel_scale.to_pitch_metres(merged, lambda uv: st.image_to_pitch(uv, cal))
        metric.to_parquet(out / "metric_v3s_link.parquet")
        shutil.rmtree(dst)                            # the run's copies: 1.6 MB a unit
        done += 1
        if done % 25 == 0:
            print(done, w, flush=True)
    print("built", done, "missing", missing)


def cv(tree: Path, name: str):
    from src.unseal import unsealed
    for m in V3.NEW_PLAYERS:
        if not unsealed(m, "shots hybrid: spotter on the hybrid tracks (M7 / M9 clips as training data)"):
            raise SystemExit("VEO_UNSEAL needed")
    us = V3.new_tree([u for u in V.units() if (LS.GPU / u["name"]).exists()])
    fr = pd.read_parquet(HERE / "frames.parquet")
    for u in us:                                     # units without windows keep their v3b tracks
        d = tree / u["name"] / f"output_{u['name']}"
        if not (d / "metric_v3s_link.parquet").exists():
            d.mkdir(parents=True, exist_ok=True)
            for name in ("metric_v3s_link.parquet", "v3s_rows.parquet"):
                (d / name).symlink_to((V3.NEW / u["name"] / f"output_{u['name']}" / name).resolve())
    LS.GPU = V.GPU = tree
    d = {u["name"]: (V4.features_v4(u), LS.labels(u), u["match"]) for u in us}
    names = list(d)
    probs, thr = LS.nested(d, names, LS.cols_of(d))
    res = LS.score(d, names, probs, thr)
    ref = json.loads((HERE.parent / "v4/cv_v4.json").read_text())["v4"]
    ms = sorted(ref["per match [tp, fp, fn]"])
    dh = np.array([res["per match [tp, fp, fn]"][m][0] - ref["per match [tp, fp, fn]"][m][0] for m in ms], float)
    df = np.array([res["per match [tp, fp, fn]"][m][1] - ref["per match [tp, fp, fn]"][m][1] for m in ms], float)
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(ms), len(ms)) for _ in range(5000)]
    out = {"protocol sha256": (HERE / "protocol.json.sha256").read_text().split()[0], "units": len(us),
           "units with windows": int(fr.unit.nunique()), "hybrid": res, "v3b (cv_v4.json)": ref,
           "hits hybrid - v3b": {"total": float(dh.sum()),
                                 "bootstrap 95% over matches": [float(np.percentile([dh[i].sum() for i in idx], 2.5)),
                                                                float(np.percentile([dh[i].sum() for i in idx], 97.5))]},
           "false alarms hybrid - v3b": {"total": float(df.sum()),
                                         "bootstrap 95% over matches": [float(np.percentile([df[i].sum() for i in idx], 2.5)),
                                                                        float(np.percentile([df[i].sum() for i in idx], 97.5))]}}
    (HERE / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("hits hybrid - v3b", "false alarms hybrid - v3b")}),
          {k: res[k] for k in ("tp", "fp", "fn", "recall", "precision")})


if __name__ == "__main__":
    if sys.argv[1] == "--cv":
        cv(HYB, "cv_hybrid")
    elif sys.argv[1] == "--control":
        build(None, CONTROL)
    elif sys.argv[1] == "--cv-control":
        cv(CONTROL, "cv_control")
    else:
        build(Path(sys.argv[1]), HYB)
