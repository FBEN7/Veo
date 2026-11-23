"""moving_cam (plan.json): run the product chain on the simulated follow-cam
windows and score distance, top speed and sprints against the truth.

    python evaluate.py run <match>_<half>     # chain on one window (cached)
    python evaluate.py score [--dev]          # test windows (or dev) -> results.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

WIN = Path("/tmp/claude-0/-home-user-veo/b69e5b60-e0fe-5920-8f15-285768d7ea57/scratchpad/mcam/windows")
RUNS = WIN.parent / "runs"
FPS = 25
DEV = ["118575_1", "118576_1"]
TEST = ["117093_1", "117093_2", "132877_1", "132877_2", "118577_1", "118578_1"]
MIN_TRACK_S, MATCH_FRAC, GATE = 10.0, 0.60, 0.75


def check():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    if hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest() != want:
        raise SystemExit("plan.json does not match its sha256")


def run(name):
    """main.py's defaults: detection, teams, re-identification; then both paths."""
    from src import detect_track, stats, moving_camera
    from src.team_assignment import assign_teams
    from src.auto_calibrate import auto_calibrate
    clip = str(WIN / name / "clip.mp4")
    wd = RUNS / name
    wd.mkdir(parents=True, exist_ok=True)
    raw = wd / "tracks.parquet"
    if raw.exists():
        tracks = pd.read_parquet(raw)
    else:
        tracks = detect_track.run(clip, stride=1, model_name=str(ROOT / "yolov8m.pt"), conf_player=0.25,
                                  conf_ball=0.08, max_seconds=0, imgsz=640)
        tracks = assign_teams(clip, tracks)
        tracks = detect_track.reidentify_tracks(clip, tracks)
        tracks.to_parquet(raw)
    moving, info = moving_camera.to_metric(clip, tracks, wd)
    moving.to_parquet(wd / "moving.parquet")
    h_path = wd / "homography.npy"
    if not h_path.exists():
        auto_calibrate(clip, str(h_path), debug=False)
    fixed = stats.to_pitch_coords(tracks, np.load(h_path))
    fixed.to_parquet(wd / "fixed.parquet")
    info["coverage (production)"] = round(moving_camera.coverage(moving, len(np.load(wd / "camera_motion.npy")) / FPS), 3)
    (wd / "info.json").write_text(json.dumps(info, indent=1))
    print(name, info)


def match_tracks(tracks, truth):
    """chain track id -> truth person (track_id), by per-frame nearest box centre."""
    p = tracks[(tracks.cls == "player") & tracks.team.isin(["team_A", "team_B"])]
    tv = truth[truth.in_view & truth.role.isin(["player", "goalkeeper"])]
    by_frame = {f: g for f, g in tv.groupby("out_frame")}
    votes = []
    for r in p.itertuples():
        g = by_frame.get(int(r.frame))
        if g is None:
            continue
        d = np.hypot(g.u.to_numpy() - r.px, g.v.to_numpy() - r.py)
        i = int(np.argmin(d))
        if d[i] <= GATE * g.h.to_numpy()[i]:
            votes.append((r.track_id, int(g.track_id.to_numpy()[i])))
    v = pd.DataFrame(votes, columns=["tid", "person"])
    n = p.groupby("track_id").size()
    out = {}
    for tid, g in v.groupby("tid"):
        top = g.person.value_counts()
        if top.iloc[0] >= MATCH_FRAC * n[tid]:
            out[tid] = int(top.index[0])
    return out


def truth_stats(truth, person, f0, f1, team, tid):
    from src import stats
    t = truth[(truth.track_id == person) & (truth.out_frame >= f0) & (truth.out_frame <= f1)].sort_values("out_frame")
    t = t.dropna(subset=["x", "y"])
    if len(t) < 20:
        return None
    sm = t[["x", "y"]].rolling(FPS, center=True, min_periods=1).mean()
    df = pd.DataFrame({"frame": t.out_frame.to_numpy(), "time_s": t.out_frame.to_numpy() / FPS, "track_id": tid,
                       "cls": "player", "team": team, "x": sm.x.to_numpy(), "y": sm.y.to_numpy()})
    s = stats.physical_stats(df)
    return None if s.empty else s.iloc[0]


def score_window(name):
    from src import stats
    truth = pd.read_parquet(WIN / name / "truth.parquet")
    rows = []
    for path in ("moving", "fixed"):
        tr = pd.read_parquet(RUNS / name / f"{path}.parquet")
        ph = stats.physical_stats(tr).set_index("track_id")
        m = match_tracks(tr, truth)
        p = tr[tr.cls == "player"]
        span = p.groupby("track_id").frame.agg(["min", "max"])
        for tid, person in m.items():
            if tid not in ph.index or ph.loc[tid, "minutes_tracked"] * 60 < MIN_TRACK_S:
                continue
            ts = truth_stats(truth, person, span.loc[tid, "min"], span.loc[tid, "max"], ph.loc[tid, "team"], tid)
            if ts is None:
                continue
            rows.append({"window": name, "path": path, "track": int(tid), "person": person,
                         "seconds": ph.loc[tid, "minutes_tracked"] * 60,
                         "dist": ph.loc[tid, "distance_m"], "dist_truth": ts.distance_m,
                         "top": ph.loc[tid, "top_speed_kmh"], "top_truth": ts.top_speed_kmh,
                         "sprints": ph.loc[tid, "n_sprints"], "sprints_truth": ts.n_sprints})
    tv = truth[truth.role.isin(["player", "goalkeeper"])]
    persons_s = tv.groupby("track_id").size().sum() / FPS
    in_view_s = tv.in_view.sum() / FPS
    return pd.DataFrame(rows), {"truth player-s": round(persons_s), "in crop": round(in_view_s / persons_s, 3)}


def summarise(df, windows):
    """the plan's decision on one path's scored tracks (leave-one-window-out constants)."""
    df = df.copy()
    df["speed_truth"] = df.dist_truth / df.seconds
    const_d, const_t, const_r = [], [], []
    for r in df.itertuples():
        o = df[df.window != r.window]
        const_d.append(r.seconds * o.speed_truth.mean())
        const_t.append(o.top_truth.median())
    df["dist_const"], df["top_const"] = const_d, const_t
    df["e_dist"] = (df.dist - df.dist_truth).abs() / df.dist_truth
    df["e_dist_const"] = (df.dist_const - df.dist_truth).abs() / df.dist_truth
    df["e_top"] = (df.top - df.top_truth).abs()
    df["e_top_const"] = (df.top_const - df.top_truth).abs()
    per = df.groupby("window").agg(n=("track", "size"), e_dist=("e_dist", "median"), e_dist_const=("e_dist_const", "median"),
                                   e_top=("e_top", "median"), e_top_const=("e_top_const", "median"),
                                   bias_dist=("dist", "sum"), truth_dist=("dist_truth", "sum"))
    per["dist ratio"] = per.bias_dist / per.truth_dist
    minutes = df.seconds.sum() / 60
    rate, rate_truth = df.sprints.sum() / minutes, df.sprints_truth.sum() / minutes
    const_rates = []
    for w in windows:
        o = df[df.window != w]
        const_rates.append(o.sprints_truth.sum() / (o.seconds.sum() / 60))
    const_rate = float(np.mean(const_rates)) if const_rates else float("nan")
    med = lambda c: float(df[c].median())
    wins_d = int((per.e_dist < per.e_dist_const).sum())
    wins_t = int((per.e_top < per.e_top_const).sum())
    ratio = rate / rate_truth if rate_truth else float("nan")
    lr = abs(np.log(ratio)) if ratio > 0 else float("inf")
    lr_c = abs(np.log(const_rate / rate_truth)) if rate_truth else float("inf")
    dec = {
        "distance shown": bool(med("e_dist") <= 0.15 and med("e_dist") < med("e_dist_const") and wins_d >= 4),
        "top speed shown": bool(med("e_top") <= 3.0 and med("e_top") < med("e_top_const") and wins_t >= 4),
        "sprints shown": bool(0.67 <= ratio <= 1.5 and lr < lr_c),
    }
    return {
        "scored tracks": int(len(df)), "tracked minutes": round(minutes, 1),
        "distance": {"median rel error": round(med("e_dist"), 3), "constant": round(med("e_dist_const"), 3),
                     "pooled chain / truth": round(float(df.dist.sum() / df.dist_truth.sum()), 3),
                     "windows beating the constant": f"{wins_d} / {len(per)}"},
        "top speed": {"median abs error km/h": round(med("e_top"), 2), "constant": round(med("e_top_const"), 2),
                      "median chain km/h": round(med("top"), 1), "median truth km/h": round(med("top_truth"), 1),
                      "windows beating the constant": f"{wins_t} / {len(per)}"},
        "sprints per tracked minute": {"chain": round(rate, 3), "truth": round(rate_truth, 3),
                                       "chain / truth": round(ratio, 3), "constant": round(const_rate, 3)},
        "per window": per.round(3).reset_index().to_dict("records"),
        "decision": dec,
    }


def score(windows, out):
    frames, cov = [], {}
    for w in windows:
        d, c = score_window(w)
        info = json.loads((RUNS / w / "info.json").read_text())
        c.update({"in scored tracks (moving)": round(float(d[d.path == "moving"].seconds.sum()) / c["truth player-s"], 3),
                  **info})
        cov[w] = c
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(HERE / f"{out}_tracks.csv", index=False)
    res = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0], "windows": windows,
           "moving (deciding)": summarise(df[df.path == "moving"], windows),
           "fixed, one homography (reported)": summarise(df[df.path == "fixed"], windows),
           "coverage and camera per window": cov}
    (HERE / f"{out}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res["moving (deciding)"].items() if k != "per window"}, indent=1, default=float))


if __name__ == "__main__":
    check()
    if sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "score":
        dev = "--dev" in sys.argv
        score(DEV if dev else TEST, "dev_results" if dev else "results")
