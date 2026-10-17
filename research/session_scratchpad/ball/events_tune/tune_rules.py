"""Step 2a of the events plan: the event rules (src/events.py's EVENT_PARAMS) and
the ball input retuned on the dev windows -- M6's second half (v3 never trained
on it), M7 and M9 (descriptive windows already) -- for pass + carry F1 at 1 s
minus chance, pooled (score_events_paired's objective). Staged like the events
v3 sweep: A ball input (frozen rules), B possession radius x hold, C carry time
x pass distance, A' ball input again. Leave-one-window-out: the stages rerun
on two windows, scored on the third. The test is M1 (events_m1), run once
after this, with the configuration chosen here.

    VEO_UNSEAL=... python tune_rules.py [--out tune.json]
"""
import argparse
import contextlib
import io
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/Veo")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import score_events_paired as sp  # noqa: E402
import score_soccernet as sc  # noqa: E402
import sweep_events_v2 as se  # noqa: E402
from detect_ball_heatmap import read_sidecar  # noqa: E402
from src import ball_track as bt  # noqa: E402
from src import events as ev  # noqa: E402
from src import pixel_scale  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.unseal import unsealed  # noqa: E402

ROOT = Path("/home/user/Veo")
DEV = ["st2_118578_2nd_f015000", "st2_128057_1st_f030000", "st2_132831_2nd_f030000"]
REC = json.loads((ROOT / "research/session_scratchpad/ball/events_v3.json").read_text())
FROZEN = {k: v for k, v in REC["frozen"].items() if k in ev.EVENT_PARAMS}
V4R1 = {"tau": 0.10, "top_k": 4, "c_null": 1.0, "c_static": 2.0, "gap": 6,
        "max_fill_speed": 35.0, "fallback": False}
LINK = {"c_null": 1.0, "c_static": 2.0}
# ball input: rows file, select, top_k, tau, extra pitch gap fill (frames)
BALLS = {
    "v2_link": ("v2_balls_{w}", "link", 4, 0.25, 0),
    "v3_link": ("v3m_balls_{w}", "link", 4, 0.10, 0),
    "v3_track": ("v3p_nofb_balls_{w}", "given", None, None, 0),
    "v3_track_fill12": ("v3p_nofb_balls_{w}", "given", None, None, 12),
    "v3_track_fill25": ("v3p_nofb_balls_{w}", "given", None, None, 25),
    # static points (resting spare balls, marks) dropped anywhere, then as above
    "v3s_link": ("v3s_balls_{w}", "link", 4, 0.10, 0),
    "v3s_track": ("v3ps_nofb_balls_{w}", "given", None, None, 0),
    "v3s_track_fill12": ("v3ps_nofb_balls_{w}", "given", None, None, 12),
}
STATIC_SHARE, STATIC_PX = 0.05, 4.0   # as src/ball_spots' marks, over the whole picture
GRID_B = {"POSSESSION_RADIUS_M": [1.5, 2.0, 3.0, 4.0, 6.0], "MIN_POSSESSION_HOLD_S": [0.1, 0.2, 0.4]}
GRID_C = {"CARRY_MIN_TIME_S": [0.4, 0.8, 1.2], "MIN_PASS_DISTANCE_M": [1.0, 3.0, 5.0]}


def static_points(rows: pd.DataFrame, n_frames: int) -> list:
    """Image points holding a candidate (any rank) on at least STATIC_SHARE of
    the frames within STATIC_PX: a ball in play never stays that long on one
    point; a spare ball at rest, a mark or a post does."""
    cell = np.c_[np.round(rows.px / STATIC_PX), np.round(rows.py / STATIC_PX)].astype(int)
    df = pd.DataFrame({"cx": cell[:, 0], "cy": cell[:, 1], "frame": rows.frame.to_numpy()})
    frames_of = df.groupby(["cx", "cy"]).frame.agg(set)
    pts = []
    for (cx, cy), _ in frames_of.items():
        fr = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                fr |= frames_of.get((cx + dx, cy + dy), set())
        if len(fr) >= STATIC_SHARE * n_frames:
            pts.append((cx * STATIC_PX, cy * STATIC_PX))
    merged = []
    for q in sorted(pts):
        if all(np.hypot(q[0] - m[0], q[1] - m[1]) > 2 * STATIC_PX for m in merged):
            merged.append(q)
    return merged


def static_dropped(w: str) -> Path:
    out = ROOT / f".cache/v3s_balls_{w}.parquet"
    if out.exists() and Path(f"{out}.json").exists():
        return out
    from detect_ball_heatmap import write_sidecar
    src = ROOT / f".cache/v3m_balls_{w}.parquet"
    meta = read_sidecar(src)
    rows = pd.read_parquet(src)
    pts = static_points(rows, len(meta["frames"]))
    keep = np.ones(len(rows), bool)
    for u, v in pts:
        keep &= np.hypot(rows.px.to_numpy() - u, rows.py.to_numpy() - v) > 1.5 * STATIC_PX
    rows[keep].to_parquet(out)
    write_sidecar(out, meta["frames"], **{k: v for k, v in meta.items() if k not in ("frames", "parquet sha256")})
    Path(f"{out}.json").write_text(json.dumps({"from": str(src), "static points": pts,
                                               "rows dropped": int((~keep).sum()), "rows": int(len(rows))}))
    print(f"{w}: {len(pts)} static points, {int((~keep).sum())} of {len(rows)} rows dropped", flush=True)
    return out


def tracked(w: str, cal: dict, static: bool = False) -> Path:
    out = ROOT / f".cache/v3p{'s' if static else ''}_nofb_balls_{w}.parquet"
    if out.exists() and Path(f"{out}.json").exists():
        return out
    src = static_dropped(w) if static else ROOT / f".cache/v3m_balls_{w}.parquet"
    meta = read_sidecar(src)
    info = {}
    tr = bt.track(pd.read_parquet(src), cal, frames=meta["frames"], info=info, **V4R1)
    rows_of(tr).to_parquet(out)
    Path(f"{out}.json").write_text(json.dumps({"from": str(src), "params": V4R1, "counts": info}))
    return out


def rows_of(tr: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"frame": tr.frame.astype(int), "time_s": tr.frame / 25.0, "track_id": -1,
                         "cls": "ball", "px": tr.u, "py": tr.v, "crop_h": tr["size"],
                         "crop_w": tr["size"], "detection_method": "v3_tracked",
                         "confidence": tr.p})


def filled(rows: pd.DataFrame, gap: int, cal: dict) -> pd.DataFrame:
    """The tracked ball with longer pitch gaps filled (ball_track.fill_gaps, 35 m/s)."""
    tr = pd.DataFrame({"frame": rows.frame, "u": rows.px, "v": rows.py, "size": rows.crop_h,
                       "p": rows.confidence, "p_raw": rows.confidence, "logit": 0.0, "source": "detected"})
    return rows_of(bt.fill_gaps(tr, gap, cal, max_speed=35.0))


def metric_of(w: str, ball: str) -> pd.DataFrame:
    cache = ROOT / f"output_{w}/tune_{ball}/metric.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    m = se.match_of(w)
    cal = st.calibration(m)
    name, select, top_k, tau, gap = BALLS[ball]
    if name.startswith("v3s_"):
        static_dropped(w)
    if select == "given":
        tracked(w, cal, static=name.startswith("v3ps"))
    rows = pd.read_parquet(ROOT / f".cache/{name.format(w=w)}.parquet")
    if gap:
        rows = filled(rows, gap, cal)
    src = ROOT / f"output_{w}" / se.SRC
    dst = ROOT / f"output_{w}/tune_{ball}"
    sc.swap_ball_rows(src, dst, rows)
    with contextlib.redirect_stdout(io.StringIO()):
        merged = sc.merged_from_cached(dst, cal, select=select, top_k=top_k, tau=tau,
                                       link_params=LINK if select == "link" else None)
    metric = pixel_scale.to_pitch_metres(merged, lambda uv: st.image_to_pitch(uv, cal))
    metric.to_parquet(cache)
    return metric


def events_of(metric, rules):
    with contextlib.redirect_stdout(io.StringIO()):
        return ev.detect_events(metric, absolute_pitch=True, ball_time_aware=True, params=rules)


def objective(ball, rules, windows, cache):
    key = (ball, json.dumps(rules, sort_keys=True), tuple(windows))
    if key not in cache:
        evs = {w: events_of(metric_of(w, ball), rules) for w in windows}
        s = sp.score(evs, {w: sp.labels_of(w) for w in windows},
                     {w: sp.duration_of(w) for w in windows}, tols=(1.0,), n_chance=100)
        cache[key] = {"objective": s["objective"]["pooled"], "F1": s["1s"]["pooled"]["F1"],
                      "events": s["1s"]["pooled"]["events"], "labels": s["1s"]["pooled"]["labels"],
                      "per window": {w: s["objective"][w] for w in windows}}
    return cache[key]


def stages(windows, cache, log):
    ball, rules = "v3_link", dict(FROZEN)

    def best(cands, stage):
        scored = [(objective(b, r, windows, cache), b, r) for b, r in cands]
        for s, b, r in scored:
            log.append({"stage": stage, "windows": windows, "ball": b, "rules": r, **s})
        return max(scored, key=lambda x: x[0]["objective"])
    _, ball, _ = best([(b, rules) for b in BALLS], "A")
    for stage, grid in (("B", GRID_B), ("C", GRID_C)):
        keys = list(grid)
        cands = [(ball, {**rules, **dict(zip(keys, vals))}) for vals in itertools.product(*grid.values())]
        _, ball, rules = best(cands, stage)
    s, ball, rules = best([(b, rules) for b in BALLS], "A'")
    return ball, rules, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("tune.json")))
    args = ap.parse_args()
    for w in DEV:
        m = se.match_of(w)
        if m in {"128057", "132831"} and not unsealed(m, "events tuning"):
            raise SystemExit("VEO_UNSEAL needed for M7 / M9")
    cache, log = {}, []
    ball, rules, s = stages(DEV, cache, log)
    print("chosen on all dev windows:", ball, rules, s, flush=True)
    baseline = {b: objective(b, FROZEN, DEV, cache) for b in BALLS}
    lowo = {}
    for held in DEV:
        rest = [w for w in DEV if w != held]
        b, r, _ = stages(rest, cache, log)
        lowo[held] = {"ball": b, "rules": r, "on held": objective(b, r, [held], cache),
                      "frozen v2_link on held": objective("v2_link", FROZEN, [held], cache)}
        print("held out", held, lowo[held], flush=True)
    Path(args.out).write_text(json.dumps({"dev": DEV, "chosen": {"ball": ball, "rules": rules, **s},
                                          "frozen rules by ball": baseline, "leave one window out": lowo,
                                          "log": log}, indent=1))
    print("frozen rules by ball:", json.dumps(baseline, indent=1))


if __name__ == "__main__":
    main()
