# Kaggle CPU kernel (pushed by push_track.py, plan_v4.json step 2 +
# addendum 1): per unit, the GPU pass's detections (kernel veo-nearball-detall,
# det/<unit>.parquet) through the pipeline's tracker and its fixed-view player
# steps (addendum 2), in three variants:
#   A:  the tracker's output as the pipeline writes it (matched detections only)
#   A2: A, plus teams for tracks shorter than MIN_TRACK_FRAMES from their crops
#   B2: every detection (unmatched ones under new ids), plus that team rule
# each through the outline (margin 10 px), kit features, teams, the roster cap.
# Output: /kaggle/working/trk/<unit>_<A|A2|B2>.parquet (frame, track_id, px, py,
# crop_h, crop_w, team, in_s4). The code under test (src/: constants,
# player_filter, team_assignment_v2) comes from veo-nearball-track-stage.
import glob, os, subprocess, sys, time

T0 = time.time()


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None


import json, hashlib, shutil
stage_ds = os.path.dirname(find("track_stage.json"))
stage = "/kaggle/working/code"                       # the package src/ rebuilt here
os.makedirs(f"{stage}/src", exist_ok=True)
open(f"{stage}/src/__init__.py", "w").close()
for f, want in json.load(open(f"{stage_ds}/track_stage.json"))["sha256"].items():
    n = os.path.basename(f)
    got = hashlib.sha256(open(f"{stage_ds}/{n}", "rb").read()).hexdigest()
    assert got == want, (f, got, want)
    shutil.copy(f"{stage_ds}/{n}", f"{stage}/src/{n}")
code = os.path.dirname(find("supervision-0.30.9-py3-none-any.whl"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                *[os.path.join(code, w) for w in ("supervision-0.30.9-py3-none-any.whl",
                                                  "pydeprecate-0.13.0-py3-none-any.whl")]], check=True)
subprocess.run([sys.executable, "-c", "import supervision, cv2, sklearn, pyarrow; print('supervision', supervision.__version__)"],
               check=True)                            # fail here, not in the workers
det_dir = os.path.dirname(find("units.txt"))          # the GPU kernel's output
os.makedirs("/kaggle/working/trk", exist_ok=True)
print(f"[{time.time() - T0:5.0f}s] setup done", flush=True)

worker = r'''
import glob, json, sys, time
import numpy as np
import pandas as pd
import supervision as sv
stage, det_dir, part, parts = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, stage)
from src import constants as C
from src import player_filter
from src.team_assignment_v2 import assign_teams_v2, track_kit_features, FIXED_VIEW_KIT_FEATURE, FIXED_VIEW_CORE_DEPTH
FPS, STRIDE, CONF, MARGIN, LO, HI = 25.0, 2, 0.20, 10.0, 300, 1125
def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None
def track(d, every):
    tracker = sv.ByteTrack(frame_rate=max(1.0, FPS / STRIDE),
                           track_activation_threshold=C.TRACK_ACTIVATION_THRESHOLD,
                           lost_track_buffer=C.LOST_TRACK_BUFFER,
                           minimum_matching_threshold=C.MINIMUM_MATCHING_THRESHOLD)
    by_f = {f: g for f, g in d.groupby("frame")}
    rows, nid = [], 1_000_000
    for f in range(LO, HI + 1, STRIDE):
        g = by_f.get(f)
        xyxy = g[["x0", "y0", "x1", "y1"]].to_numpy(np.float32) if g is not None else np.zeros((0, 4), np.float32)
        conf = g.conf.to_numpy(np.float32) if g is not None else np.zeros(0, np.float32)
        det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=np.zeros(len(conf), int))
        out = tracker.update_with_detections(det)
        got = [(b, int(t)) for b, t in zip(out.xyxy, out.tracker_id)]
        if every:
            seen = {tuple(b.tolist()) for b, _ in got}
            for b in xyxy:
                if tuple(b.tolist()) not in seen:
                    got.append((b, nid))
                    nid += 1
        for (x1, y1, x2, y2), tid in got:
            h = float(y2 - y1)
            if h < C.MIN_DETECTION_HEIGHT_PX:
                continue
            rows.append(dict(frame=f, time_s=f / FPS, track_id=tid, cls="player", px=float((x1 + x2) / 2),
                             py=float(y2), crop_h=h, crop_w=float(x2 - x1), detection_method="yolo"))
    return pd.DataFrame(rows)
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src import team_assignment_v2 as T
def fit_clusters(feats, fit_tracks):
    """cluster_teams' fit, kept (plan_v4 addendum 2): same inputs, order, seed."""
    tids = list(feats)
    fit = [t for t in tids if t in fit_tracks]
    fallback = len(fit) < 3 * T.N_CLUSTERS
    if fallback:
        fit = tids
    raw = np.array([T.kit_vector(feats[t], FIXED_VIEW_KIT_FEATURE) for t in fit])
    scaler = StandardScaler().fit(raw)
    X = scaler.transform(raw)
    km = KMeans(n_clusters=T.N_CLUSTERS, n_init=10, random_state=0).fit(X)
    if T.N_CLUSTERS >= 3:
        counts = np.bincount(km.labels_, minlength=T.N_CLUSTERS)
        big = np.argsort(counts)[::-1][:2]
        mapping = {int(big[0]): "team_A", int(big[1]): "team_B"}
        mapping.update({c: "other" for c in range(T.N_CLUSTERS) if c not in mapping})
    else:
        mapping = {0: "team_A", 1: "team_B"}
    centres = np.array([km.cluster_centers_[c] for c in sorted(mapping) if mapping[c] != "other"])
    dist = np.linalg.norm(X[:, None, :] - centres[None, :, :], axis=2).min(axis=1)
    inside = np.array([mapping[l] != "other" for l in km.labels_])
    scale = np.median(dist[inside]) if inside.any() else np.median(dist)
    return dict(scaler=scaler, km=km, mapping=mapping, centres=centres, scale=scale, fallback=fallback)
def label(fc, feat, in_core):
    """A track's team from the kept fit, as cluster_teams labels its tracks."""
    if not fc["fallback"] and not in_core:
        return "other"
    x = fc["scaler"].transform(T.kit_vector(feat, FIXED_VIEW_KIT_FEATURE)[None])
    t = fc["mapping"][int(fc["km"].predict(x)[0])]
    if t != "other" and fc["scale"] > 1e-9:
        if np.linalg.norm(x - fc["centres"], axis=1).min() / fc["scale"] > T.RESIDUAL_CUT:
            t = "other"
    return t
def short_teams(video, teamed, feats, core, depth):
    """Teams for tracks shorter than MIN_TRACK_FRAMES from their own crops."""
    p = teamed[teamed.cls == "player"]
    size = p.groupby("track_id").size()
    short = set(size.index[size < T.MIN_TRACK_FRAMES])
    if not short or len(feats) < T.N_CLUSTERS:
        return teamed, 0, 0
    wanted = {}
    for r in p[p.track_id.isin(short)].itertuples():
        wanted.setdefault(int(r.frame), []).append((r.track_id, float(r.px), float(r.py), float(r.crop_h)))
    got = {}
    cap = cv2.VideoCapture(video)
    idx, last = 0, max(wanted)
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, px, py, h in wanted.get(idx, ()):
            f = T._torso_feature(frame, px, py, h)
            if f is not None:
                got.setdefault(tid, []).append(f)
        idx += 1
    cap.release()
    fc = fit_clusters(feats, core)
    team = {tid: label(fc, np.median(fs, axis=0), tid in core) for tid, fs in got.items()}
    out = teamed.copy()
    m = out.track_id.isin(set(team)) & (out.cls == "player")
    out.loc[m, "team"] = out.loc[m, "track_id"].map(team)
    return out, len(short), len(team)
def cap_and_save(teamed, w, v):
    teamed = teamed.reset_index(drop=True)
    teamed["rid"] = np.arange(len(teamed))
    capped = player_filter.filter_players(teamed)
    teamed["in_s4"] = teamed.rid.isin(set(capped.rid))
    teamed[["frame", "track_id", "px", "py", "crop_h", "crop_w", "team", "in_s4"]].to_parquet(
        f"/kaggle/working/trk/{w}_{v}.parquet")
    return f"{v} {len(teamed)} rows {int(teamed.in_s4.sum())} kept"
units = open(f"{det_dir}/units.txt").read().split()[part::parts]
t0 = time.time()
for i, w in enumerate(units):
    video, info = find(f"{w}.mp4"), find(f"{w}.json")
    if video is None or info is None:
        print(f"{i + 1}/{len(units)} {w}: MISSING", flush=True)
        continue
    pitch = json.load(open(info))["pitch"]
    det = pd.read_parquet(f"{det_dir}/det/{w}.parquet")
    d = det[det.conf >= CONF]
    msg = []
    for v, every in (("A", False), ("B", True)):
        tracks = player_filter.inside_pitch(track(d, every), pitch, MARGIN)
        players = tracks[tracks.cls == "player"]
        feats = track_kit_features(video, players)
        depth = player_filter.track_depth(tracks, pitch)
        core = set(depth.index[depth >= FIXED_VIEW_CORE_DEPTH])
        teamed = assign_teams_v2(video, tracks, kit_feature=FIXED_VIEW_KIT_FEATURE, fit_tracks=core,
                                 feats=feats, verbose=False)
        if v == "A":
            msg.append(cap_and_save(teamed, w, "A"))
        # check: the kept fit labels the long tracks as the pipeline did
        if len(feats) >= T.N_CLUSTERS:
            fc = fit_clusters(feats, core)
            tm = teamed[teamed.cls == "player"].drop_duplicates("track_id").set_index("track_id").team
            bad = sum(1 for t, f in feats.items() if label(fc, f, t in core) != tm.get(t))
            msg.append(f"{v} fit check {bad}/{len(feats)} differ")
        teamed2, n_short, n_feat = short_teams(video, teamed, feats, core, depth)
        msg.append(f"{v} short tracks {n_short}, {n_feat} with a kit")
        msg.append(cap_and_save(teamed2, w, "A2" if v == "A" else "B2"))
    print(f"[{time.time() - t0:6.0f}s] {i + 1}/{len(units)} {w}: " + "; ".join(msg), flush=True)
'''
with open("/kaggle/working/worker.py", "w") as f:
    f.write(worker)
PARTS = os.cpu_count() or 4
print("workers", PARTS, flush=True)
procs = [subprocess.Popen([sys.executable, "/kaggle/working/worker.py", stage, det_dir, str(k), str(PARTS)],
                          stdout=open(f"/kaggle/working/tlog_{k}.txt", "w"), stderr=subprocess.STDOUT) for k in range(PARTS)]
for p in procs:
    p.wait()
for k in range(PARTS):
    print(open(f"/kaggle/working/tlog_{k}.txt").read()[-2000:], flush=True)
print(f"[{time.time() - T0:5.0f}s] done", flush=True)
assert all(p.returncode == 0 for p in procs), [p.returncode for p in procs]
