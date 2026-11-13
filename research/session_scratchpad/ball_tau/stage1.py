"""Stage 1 of plan.json for one variant: relink every shots clip with that
confidence cut (metric_v3s_link_<variant>.parquet beside the others), then
score it on the verified balls and at the labelled shots.

    VEO_UNSEAL=... python stage1.py VARIANT      # -> stage1_<VARIANT>.json
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball/events_tune"))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball_interp"))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball_goal"))
import feasibility as F  # noqa: E402
import measure as M  # noqa: E402
import score_soccernet as sc  # noqa: E402
import tune_rules as tr  # noqa: E402
from src import auto_tune, ball_link as bl, pixel_scale, player_filter  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.unseal import unsealed  # noqa: E402

VARIANTS = {"t10": (0.10, 0.10), "t05": (0.05, 0.05), "t02": (0.02, 0.02), "zone03": (0.10, 0.03)}
NEW_PLAYERS = {"128057", "132831"}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def adjust(tracks: pd.DataFrame, cal: dict, tau: float, tau_near: float) -> pd.DataFrame:
    """Near-goal ball candidates' confidence rescaled so that tau_near there
    acts as tau (same cut, same link cost)."""
    if tau_near == tau:
        return tracks
    t = tracks.copy()
    b = t.cls == "ball"
    xy = bl.ground_points(t[b], cal)
    near = np.abs(xy[:, 0]) >= M.NEAR_X
    p = t.loc[b, "confidence"].to_numpy(float)
    q = 1 / (1 + np.exp(-(logit(p) - logit(tau_near) + logit(tau))))
    t.loc[b, "confidence"] = np.where(near, q, p)
    return t


def relink(run: Path, cal: dict, tau: float, tau_near: float) -> pd.DataFrame:
    """merged_from_cached without the grass test, with the variant's cut."""
    profile = auto_tune.VideoProfile.load(run / "profile.json")
    setting = json.loads((run / "pitch.json").read_text())
    tracks = pd.read_parquet(run / "tracks_grass.parquet")
    scale = pixel_scale.estimate_px_per_m(tracks)
    tracks = tracks.assign(px_raw=tracks.px, py_raw=tracks.py)
    capped = player_filter.filter_players(tracks, verbose=False)
    capped = player_filter.inside_pitch(capped, setting["pitch"], setting["margin_px"], classes=("ball",))
    capped = adjust(capped, cal, tau, tau_near)
    with contextlib.redirect_stdout(io.StringIO()):
        merged = sc.select_ball(capped, scale, profile.fps, select="link", cal=cal,
                                link_params=tr.LINK, top_k=4, tau=tau)
    return pixel_scale.to_pitch_metres(merged, lambda uv: st.image_to_pitch(uv, cal))


def verified(tau, tau_near):
    res = {}
    for w, lab in M.WINDOWS.items():
        match = w.split("_")[1]
        cal = st.calibration(match)
        f = pd.read_parquet(lab)
        f = f[(f.window == w) & (f.status == "ball")]
        kept = adjust(F.candidates(w, cal), cal, tau, tau_near)
        players = F.players_of(w)
        out_dir = ROOT / f"output_{w}"
        r = kept[(kept.frame % 2 == 0) & (kept.confidence >= tau)]
        r = r.sort_values("confidence", ascending=False).groupby("frame").head(4)
        if (out_dir / "pitch.json").exists():
            poly = np.asarray(json.loads((out_dir / "pitch.json").read_text())["pitch"], np.float32).reshape(-1, 1, 2)
            r = r[np.asarray([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
                              for x, y in zip(r.px, r.py)], bool)]
        link = bl.link(r, cal, players, c_null=1.0, c_static=2.0, tau=tau).drop_duplicates("frame").set_index("frame")
        hit = wrong = 0
        for q in f.itertuples():
            lf = q.frame - q.frame % 2
            if lf in link.index:
                d = np.hypot(link.loc[lf].px - q.u, link.loc[lf].py - q.v)
                hit += d <= M.HIT
                wrong += d > M.HIT
        res[w] = {"ball frames": len(f), "hit": int(hit), "wrong": int(wrong)}
    n = sum(v["ball frames"] for v in res.values())
    res["all"] = {"ball frames": n, "hit": round(sum(v["hit"] for v in res.values()) / n, 3),
                  "wrong": round(sum(v["wrong"] for v in res.values()) / n, 3)}
    return res


def shots(name, tau, tau_near):
    fname = "metric_v3s_link_nograss.parquet" if name == "t10" else f"metric_v3s_link_{name}.parquet"
    for p in sorted((ROOT / ".cache/gpu").iterdir()):
        w = p.name
        m = w.split("_")[1]
        base = ROOT / (".cache/gpu_nograss" if m in NEW_PLAYERS else ".cache/gpu") / w / f"output_{w}"
        run = base / "shots_v3s_link"
        if name == "t10" or not (run / "tracks_grass.parquet").exists() or (base / fname).exists():
            continue
        relink(run, st.calibration(m), tau, tau_near).to_parquet(base / fname)
    # measure 2 of ball_goal on this variant's tracks
    src = Path(M.__file__).read_text().replace('"metric_v3s_link_nograss.parquet"', repr(fname))
    ns = {"__file__": M.__file__, "__name__": "measure_variant"}
    exec(compile(src, M.__file__, "exec"), ns)
    return ns["measure2"]()


def main():
    name = sys.argv[1]
    tau, tau_near = VARIANTS[name]
    for m in NEW_PLAYERS:
        if not unsealed(m, f"ball_tau: confidence cut {name}"):
            raise SystemExit("VEO_UNSEAL needed")
    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0], "variant": name,
           "tau": tau, "tau near a goal": tau_near, "verified balls": verified(tau, tau_near)}
    print(json.dumps(out["verified balls"]["all"]), flush=True)
    out["shots"] = shots(name, tau, tau_near)
    (HERE / f"stage1_{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out["shots"][k] for k in out["shots"] if k.startswith("shot")}, indent=1))


if __name__ == "__main__":
    main()
