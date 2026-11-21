# Kaggle CPU kernel (pushed by push_kit.py): the kit colour (hsv4 torso
# feature, team_assignment_v2._torso_feature) of every player row listed in
# veo-kit-stage/players.parquet, one sequential pass through each clip's video
# (the shot-clip datasets; M7 / M9 clips only those cut through src/unseal.py).
# Output: /kaggle/working/kit/<unit>.parquet (frame, track_id, h0..h3), logs.
import glob, os, sys, time
from multiprocessing import Pool

T0 = time.time()
stage = os.path.dirname(glob.glob("/kaggle/input/**/players.parquet", recursive=True)[0])
sys.path.insert(0, stage)
os.makedirs("/kaggle/working/kit", exist_ok=True)


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None


def unit(w):
    import cv2
    import numpy as np
    import pandas as pd
    import team_assignment_v2 as ta
    p = pd.read_parquet(f"{stage}/players.parquet", filters=[("unit", "==", w)])
    video = find(f"{w}.mp4")
    if video is None:
        return w, "MISSING", 0
    want = {f: g for f, g in p.groupby("frame")}
    cap = cv2.VideoCapture(video)
    rows, idx, last = [], 0, max(want)
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for r in want.get(idx, p.iloc[:0]).itertuples():
            f = ta._torso_feature(frame, r.px_raw, r.py_raw, r.crop_h)
            if f is not None:
                rows.append((idx, int(r.track_id), *map(float, f)))
        idx += 1
    cap.release()
    out = pd.DataFrame(rows, columns=["frame", "track_id", "h0", "h1", "h2", "h3"])
    out.to_parquet(f"/kaggle/working/kit/{w}.parquet")
    return w, f"frames read {idx}", len(out)


if __name__ == "__main__":
    import pandas as pd
    units = sorted(pd.read_parquet(f"{stage}/players.parquet", columns=["unit"]).unit.unique())
    print(f"{len(units)} units, {os.cpu_count()} cpus", flush=True)
    with Pool(os.cpu_count() or 4) as pool:
        for i, (w, msg, n) in enumerate(pool.imap_unordered(unit, units)):
            print(f"[{time.time() - T0:6.0f}s] {i + 1}/{len(units)} {w}: {msg}, {n} rows", flush=True)
    print("done", flush=True)
