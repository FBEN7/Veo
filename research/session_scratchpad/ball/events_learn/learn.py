"""Learned event spotting (passes and carries) on a fixed, calibrated view.

Per detection frame t of a window's cached pipeline (players, teams, pitch
metres; the ball input as in events_tune: v3 with static points dropped,
linked), features from the ball and the players around t, then one
gradient-boosted classifier per class ("a pass / a carry starts within 0.24 s
of t"); events are the peaks of its smoothed probability above a threshold,
at least 0.6 s apart. Validation: leave one match out, nested. The threshold
per class is set on the training matches' own out-of-fold predictions, so
that the events they would give number as many as their labels ("rate";
F1 rises with the event count, and so does chance). The scored fold never
sets its threshold. The older choice, the pooled F1 at 1 s on the scored
out-of-fold predictions themselves ("f1"), is kept for comparison; it is
optimistic.

Feature caches carry a key (calibration, ball input, feature version) and
are recomputed when it changes. For M1, import calib_fix/calib_override.py
first, or the release calibration is used.

    python learn.py cv [--select rate|f1] WINDOW ...
    python learn.py fit OUT.pkl WINDOW ...   # final model on all of them (+ rate thresholds)
    python learn.py apply MODEL.pkl NAME WINDOW ...   # events -> output_<w>/<NAME>/events.json

`apply` adds each event's actor (`actor_track`, the pipeline's track id, or
None) by the possession rule of src/passer.py; cv and fit are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/user/Veo")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball/events_tune"))
import score_events_paired as sp  # noqa: E402
import score_soccernet as sc  # noqa: E402
import tune_rules as tr  # noqa: E402
from src import passer  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402

FPS = 25.0
BALL = "v3s_link"
CLASSES = ("pass", "carry")
POS_S = 0.24          # a frame within this of a label is positive
NMS_S = 0.6           # events of one class at least this far apart
OFFSETS = (-12, -6, -2, 2, 6, 12)   # frames (the pipeline detects every 2nd)
TEAM = {"team_A": 1.0, "team_B": -1.0}
FEATURE_VERSION = 1


def _ball_series(metric: pd.DataFrame, frames: np.ndarray) -> pd.DataFrame:
    b = metric[metric.cls == "ball"].drop_duplicates("frame").set_index("frame")[["x", "y", "confidence"]]
    b = b.reindex(frames)
    dt = 2 / FPS
    vx = (b.x.shift(-1) - b.x.shift(1)) / (2 * dt)
    vy = (b.y.shift(-1) - b.y.shift(1)) / (2 * dt)
    b["speed"] = np.hypot(vx, vy)
    b["has"] = b.x.notna().astype(float)
    return b


def _candidates(w: str, frames: np.ndarray, cal: dict) -> pd.DataFrame:
    """The static-filtered v3 candidates at the detection frames: the best
    confidence, and the count above tau, per frame; pitch metres of the best."""
    rows = pd.read_parquet(ROOT / f".cache/v3s_balls_{w}.parquet")
    rows = rows[rows.frame.isin(frames)]
    best = rows.sort_values("confidence", ascending=False).drop_duplicates("frame").set_index("frame")
    xy = st.image_to_pitch(np.c_[best.px, best.py + best.crop_h / 2], cal)
    out = pd.DataFrame({"c_max": best.confidence, "c_x": xy[:, 0], "c_y": xy[:, 1]}, index=best.index)
    out["c_n"] = rows[rows.confidence >= 0.10].groupby("frame").size()
    return out.reindex(frames)


def _players(metric: pd.DataFrame, frames: np.ndarray, bx: np.ndarray, by: np.ndarray) -> pd.DataFrame:
    """Per frame, from the players to a point (the ball, else the best candidate):
    the two nearest distances, the nearest's team and track, players within 3 m,
    the nearest of each team."""
    p = metric[(metric.cls == "player") & metric.team.isin(TEAM)]
    by_frame = {f: g for f, g in p.groupby("frame")}
    rec = []
    for f, x, y in zip(frames, bx, by):
        g = by_frame.get(f)
        if g is None or not np.isfinite(x):
            rec.append((np.nan,) * 7)
            continue
        d = np.hypot(g.x.to_numpy() - x, g.y.to_numpy() - y)
        o = np.argsort(d)
        team = g.team.to_numpy()
        dA = d[team == "team_A"].min() if (team == "team_A").any() else np.nan
        dB = d[team == "team_B"].min() if (team == "team_B").any() else np.nan
        rec.append((d[o[0]], d[o[1]] if len(o) > 1 else np.nan, TEAM[team[o[0]]],
                    float(g.track_id.to_numpy()[o[0]]), float((d <= 3).sum()), dA, dB))
    return pd.DataFrame(rec, index=frames, columns=["d1", "d2", "near_team", "near_id", "n3", "dA", "dB"])


def features(w: str) -> pd.DataFrame:
    match = json.loads((ROOT / f"data/soccertrack_v2/{w}.json").read_text())["match"]
    cal = st.calibration(str(match))
    tr.static_dropped(w)
    metric = tr.metric_of(w, BALL)
    frames = np.sort(metric[metric.cls == "player"].frame.unique())
    b = _ball_series(metric, frames)
    c = _candidates(w, frames, cal)
    # the point the players are measured to: the linked ball, else the best candidate
    px = np.where(b.x.notna(), b.x, c.c_x)
    py = np.where(b.y.notna(), b.y, c.c_y)
    pl = _players(metric, frames, px, py)
    base = pd.concat([b[["has", "speed", "confidence"]].rename(columns={"confidence": "b_conf"}),
                      c[["c_max", "c_n"]], pl.drop(columns="near_id")], axis=1)
    base["x"], base["y"] = px, py
    base["dAB"] = base.dA - base.dB
    feats = [base.add_suffix("_0")]
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 2
    for o in OFFSETS:
        sh = base.shift(-o // step)
        feats.append(sh[["has", "speed", "c_max", "d1", "near_team", "n3", "dAB"]].add_suffix(f"_{o:+d}"))
    f = pd.concat(feats, axis=1)
    # changes across t: the nearest player, its team, the point's displacement
    k = 6 // step
    nid, nt = pl.near_id, pl.near_team
    f["id_change"] = (nid.shift(k) != nid.shift(-k)).astype(float).where(nid.shift(k).notna() & nid.shift(-k).notna())
    f["team_change"] = (nt.shift(k) != nt.shift(-k)).astype(float).where(nt.shift(k).notna() & nt.shift(-k).notna())
    f["disp"] = np.hypot(pd.Series(px, index=frames).shift(-k) - pd.Series(px, index=frames).shift(k),
                         pd.Series(py, index=frames).shift(-k) - pd.Series(py, index=frames).shift(k))
    f["speed_max_next"] = b.speed[::-1].rolling(6, min_periods=1).max()[::-1].to_numpy()
    f["has_frac_1s"] = b.has.rolling(25 // step, center=True, min_periods=1).mean().to_numpy()
    f["near_team_0"] = pl.near_team.to_numpy()   # kept for the event's team
    f.index.name = "frame"
    return f


def targets(w: str, frames: np.ndarray) -> pd.DataFrame:
    lab = sp.labels_of(w)
    t = frames / FPS
    return pd.DataFrame({c: np.array([np.any(np.abs(np.asarray(lab[c]) - x) <= POS_S) for x in t], float)
                         for c in CLASSES}, index=frames)


def cache_key(w: str) -> str:
    """Calibration, ball input and feature version the features rest on."""
    match = json.loads((ROOT / f"data/soccertrack_v2/{w}.json").read_text())["match"]
    cal = st.calibration(str(match))
    h = hashlib.sha1()
    for k in ("K", "D", "Knew", "Hinv"):
        h.update(np.ascontiguousarray(cal[k], dtype=float).tobytes())
    metric = ROOT / f"output_{w}/tune_{BALL}/metric.parquet"
    h.update(f"{BALL}|{FEATURE_VERSION}|{metric.stat().st_mtime_ns if metric.exists() else 0}".encode())
    return h.hexdigest()


def data(windows):
    out = {}
    for w in windows:
        cache = ROOT / f"output_{w}/learn_features.parquet"
        key_file = cache.with_suffix(".key")
        key = cache_key(w)
        if cache.exists() and key_file.exists() and key_file.read_text() == key:
            f = pd.read_parquet(cache)
        else:
            f = features(w)
            f.to_parquet(cache)
            key_file.write_text(cache_key(w))   # metric.parquet may just have been written
        out[w] = (f, targets(w, f.index.to_numpy()))
    return out


def model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          class_weight="balanced", random_state=0)


def peaks(frames: np.ndarray, p: np.ndarray, thr: float) -> list[int]:
    """Frames of the local maxima of the smoothed probability above thr, NMS_S apart."""
    s = pd.Series(p).rolling(3, center=True, min_periods=1).mean().to_numpy()
    order = np.argsort(-s)
    kept = []
    for i in order:
        if s[i] < thr:
            break
        if all(abs(frames[i] - frames[j]) / FPS >= NMS_S for j in kept):
            kept.append(i)
    return sorted(kept)


def events_of(f: pd.DataFrame, probs: dict, thr: dict) -> list[dict]:
    frames = f.index.to_numpy()
    ev = []
    for c in CLASSES:
        for i in peaks(frames, probs[c], thr[c]):
            team = f.near_team_0.iloc[i]
            ev.append({"event_type": c, "timestamp_s": round(float(frames[i] / FPS), 3),
                       "team": "team_A" if team == 1 else "team_B" if team == -1 else None,
                       "confidence": float(probs[c][i])})
    return sorted(ev, key=lambda e: e["timestamp_s"])


def match_of(w):
    return w.split("_")[1]


def oof(windows, d):
    """Out-of-fold probabilities, leaving one match out."""
    cols = [c for c in d[windows[0]][0].columns if c != "near_team_0"]
    probs = {}
    for m in sorted({match_of(w) for w in windows}):
        train = [w for w in windows if match_of(w) != m]
        if not train:
            raise ValueError("leave-one-match-out needs two matches or more")
        X = pd.concat([d[w][0][cols] for w in train])
        for c in CLASSES:
            y = pd.concat([d[w][1][c] for w in train])
            clf = model().fit(X, y)
            for w in windows:
                if match_of(w) == m:
                    probs.setdefault(w, {})[c] = clf.predict_proba(d[w][0][cols])[:, 1]
    return probs, cols


def best_thresholds(windows, d, probs, grid=np.arange(0.2, 0.96, 0.025)):
    truths = {w: sp.labels_of(w) for w in windows}
    durs = {w: sp.duration_of(w) for w in windows}
    thr = {}
    for c in CLASSES:
        best = None
        for t in grid:
            tp = n_ev = n_lab = 0
            for w in windows:
                f = d[w][0]
                fr = f.index.to_numpy()
                pred = [fr[i] / FPS for i in peaks(fr, probs[w][c], t)]
                m, _, _ = sc.match_one_to_one(pred, truths[w][c], 1.0)
                tp, n_ev, n_lab = tp + len(m), n_ev + len(pred), n_lab + len(truths[w][c])
            f1 = 2 * tp / max(n_ev + n_lab, 1)
            if best is None or f1 > best[0]:
                best = (f1, float(t))
        thr[c] = best[1]
    return thr, truths, durs


def rate_thresholds(windows, d, probs, grid=np.arange(0.2, 0.96, 0.025)):
    """Per class, the threshold whose events number closest to the labels."""
    truths = {w: sp.labels_of(w) for w in windows}
    thr = {}
    for c in CLASSES:
        n_lab = sum(len(truths[w][c]) for w in windows)
        n_ev = {float(t): sum(len(peaks(d[w][0].index.to_numpy(), probs[w][c], t)) for w in windows) for t in grid}
        thr[c] = min(n_ev, key=lambda t: (abs(n_ev[t] - n_lab), -t))
    return thr


def fit_predict(train, test, d, cols):
    X = pd.concat([d[w][0][cols] for w in train])
    out = {w: {} for w in test}
    for c in CLASSES:
        clf = model().fit(X, pd.concat([d[w][1][c] for w in train]))
        for w in test:
            out[w][c] = clf.predict_proba(d[w][0][cols])[:, 1]
    return out


def nested(windows, d):
    """Outer leave-one-match-out; each outer fold's thresholds by rate on its
    training matches' inner out-of-fold predictions."""
    cols = [c for c in d[windows[0]][0].columns if c != "near_team_0"]
    probs, thr_by = {}, {}
    for m in sorted({match_of(w) for w in windows}):
        train = [w for w in windows if match_of(w) != m]
        test = [w for w in windows if match_of(w) == m]
        inner, _ = oof(train, d)
        thr_by[m] = rate_thresholds(train, d, inner)
        probs |= fit_predict(train, test, d, cols)
    return probs, thr_by


def cv(windows, select="rate"):
    d = data(windows)
    truths = {w: sp.labels_of(w) for w in windows}
    durs = {w: sp.duration_of(w) for w in windows}
    if select == "rate":
        probs, thr_by = nested(windows, d)
        evs = {w: events_of(d[w][0], probs[w], thr_by[match_of(w)]) for w in windows}
        thr = thr_by
    else:
        probs, _ = oof(windows, d)
        thr, truths, durs = best_thresholds(windows, d, probs)
        evs = {w: events_of(d[w][0], probs[w], thr) for w in windows}
    s = sp.score(evs, truths, durs, tols=(1.0,), n_chance=200)
    return {"select": select, "thresholds": thr, "1s": s["1s"], "objective": s["objective"]}, evs


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "cv":
        select = "rate"
        if args[0] == "--select":
            select, args = args[1], args[2:]
        res, _ = cv(args, select)
        print(json.dumps(res, indent=1))
    elif cmd == "fit":
        out, windows = args[0], args[1:]
        res, _ = cv(windows, "rate")
        d = data(windows)
        probs, cols = oof(windows, d)
        thr = rate_thresholds(windows, d, probs)
        X = pd.concat([d[w][0][cols] for w in windows])
        models = {c: model().fit(X, pd.concat([d[w][1][c] for w in windows])) for c in CLASSES}
        with open(out, "wb") as fh:
            pickle.dump({"models": models, "cols": cols, "thresholds": thr,
                         "windows": windows, "cv": res}, fh)
        print(json.dumps({"thresholds": thr, "cv": res}, indent=1))
    elif cmd == "apply":
        mpath, name, windows = args[0], args[1], args[2:]
        mdl = pickle.loads(Path(mpath).read_bytes())
        for w in windows:
            f = data([w])[w][0]
            probs = {c: mdl["models"][c].predict_proba(f[mdl["cols"]])[:, 1] for c in CLASSES}
            ev = events_of(f, probs, mdl["thresholds"])
            # who made it: the possession rule (src/passer.py), on the same pitch tracks
            ev = passer.attach(ev, tr.metric_of(w, BALL), FPS)
            out = ROOT / f"output_{w}/{name}/events.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(ev, indent=1))
            print(f"{w} {name}: {len(ev)} events", flush=True)


if __name__ == "__main__":
    main()
