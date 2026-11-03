"""plan_v5 (hashed first): plan_v4's replay at each M7 clip's original
conf_player (A') against the original tracker output (O) and the replay at
0.20 (A). Local CPU, M7's 10 units.

    VEO_UNSEAL=... python diag_v5.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from src import constants as C  # noqa: E402
from src import player_filter  # noqa: E402
import score_v4  # noqa: E402

MATCH, FPS, STRIDE, LO, HI, MARGIN = "128057", 25.0, 2, 300, 1125, 10.0
DET = ROOT / ".cache/near_ball_det/det"
TRK = ROOT / ".cache/near_ball_trk/trk"
COLS = ["frame", "track_id", "px", "py", "crop_h", "crop_w", "team", "in_s4"]


def track(d):
    """track_kernel.track, A mode (matched detections only)."""
    tracker = sv.ByteTrack(frame_rate=max(1.0, FPS / STRIDE),
                           track_activation_threshold=C.TRACK_ACTIVATION_THRESHOLD,
                           lost_track_buffer=C.LOST_TRACK_BUFFER,
                           minimum_matching_threshold=C.MINIMUM_MATCHING_THRESHOLD)
    by_f = {f: g for f, g in d.groupby("frame")}
    rows = []
    for f in range(LO, HI + 1, STRIDE):
        g = by_f.get(f)
        xyxy = g[["x0", "y0", "x1", "y1"]].to_numpy(np.float32) if g is not None else np.zeros((0, 4), np.float32)
        conf = g.conf.to_numpy(np.float32) if g is not None else np.zeros(0, np.float32)
        out = tracker.update_with_detections(sv.Detections(xyxy=xyxy, confidence=conf, class_id=np.zeros(len(conf), int)))
        for (x1, y1, x2, y2), tid in zip(out.xyxy, out.tracker_id):
            h = float(y2 - y1)
            if h < C.MIN_DETECTION_HEIGHT_PX:
                continue
            rows.append(dict(frame=f, track_id=int(tid), cls="player", px=float((x1 + x2) / 2), py=float(y2),
                             crop_h=h, crop_w=float(x2 - x1)))
    return pd.DataFrame(rows)


def xyxy(t):
    return np.stack([t.px - t.crop_w / 2, t.py - t.crop_h, t.px + t.crop_w / 2, t.py], axis=1).astype(float)


def agree(a, b, f0):
    """Share of a's boxes with a box of b on the same frame at IoU >= 0.9."""
    a, b = a[a.frame >= f0], b[b.frame >= f0]
    bf = {f: xyxy(g) for f, g in b.groupby("frame")}
    hit = n = 0
    for f, g in a.groupby("frame"):
        A, B = xyxy(g), bf.get(f)
        n += len(A)
        if B is None or not len(B):
            continue
        lt = np.maximum(A[:, None, :2], B[None, :, :2])
        rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
        inter = np.clip(rb - lt, 0, None).prod(axis=2)
        area = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
        iou = inter / (area(A)[:, None] + area(B)[None, :] - inter)
        hit += int((iou.max(axis=1) >= 0.9).sum())
    return round(hit / max(n, 1), 4)


def main():
    from src.unseal import unsealed
    if not unsealed(MATCH, "near_ball plan_v5: replay at the original conf_player on the opened M7 clips"):
        raise SystemExit("VEO_UNSEAL needed")
    st = pd.read_parquet(HERE / "stages.parquet")
    acts = st[(st.stage == "S0 raw") & st.unit.str.contains(f"_{MATCH}_")][["unit", "frame", "player_id"]].drop_duplicates()
    acts["f"] = acts.frame
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
    rows, units = [], {}
    for w, a in acts.groupby("unit"):
        rd = Path(os.path.realpath(ROOT / f".cache/gpu_shots3/{w}/output_{w}/metric_v3s_link.parquet")).parent / "shots_v3s_link"
        prof = json.loads((rd / "profile.json").read_text())
        setting = json.loads((rd / "pitch.json").read_text())
        pitch = json.loads((score_v4.CLIPS / f"{w}.json").read_text())["pitch"]
        o = player_filter.inside_pitch(pd.read_parquet(rd / "tracks.parquet"), setting["pitch"], setting["margin_px"])
        o = o[(o.cls == "player") & (o.frame >= LO) & (o.frame <= HI)]
        det = pd.read_parquet(DET / f"{w}.parquet")
        a2 = player_filter.inside_pitch(track(det[det.conf >= prof["conf_player"]]), pitch, MARGIN)
        a0 = pd.read_parquet(TRK / f"{w}_A.parquet")
        a0 = a0[a0.frame <= HI]
        info = {"conf_player": prof["conf_player"], "same outline": bool(np.allclose(np.asarray(pitch, float), np.asarray(setting["pitch"], float))),
                "margin": setting["margin_px"], "boxes O / A / A'": [len(o), len(a0), len(a2)]}
        for name, t in (("O", o), ("A", a0), ("A'", a2)):
            d = tmp / name
            d.mkdir(exist_ok=True)
            t = t.assign(team=None, in_s4=False)[COLS]
            for v in ("A", "A2", "B2"):
                t.to_parquet(d / f"{w}_{v}.parquet")
            r = [x for x in score_v4.unit(w, d, a) if x["variant"] == "A" and x["stage"] == "S1"]
            rows += [dict(x, input=name) for x in r]
            if name != "O":
                for f0 in (LO, 350):
                    info[f"{name} frames>={f0}: O found / {name} found in O"] = [agree(o, t, f0), agree(t, o, f0)]
        units[w] = info
        print(w, info, flush=True)
    df = pd.DataFrame(rows)
    det_rate = lambda g: round(float((g.kind == "detected").mean()), 4)
    per_unit = {w: {k: det_rate(g[g.input == k]) for k in ("O", "A", "A'")} for w, g in df.groupby("unit")}
    res = {"plan_v5 sha256": (HERE / "plan_v5.json.sha256").read_text().split()[0],
           "actions": int(len(df) / 3),
           "actor detected at S1": {k: det_rate(df[df.input == k]) for k in ("O", "A", "A'")},
           "per unit": {w: dict(units[w], **{"actor detected O / A / A'": [per_unit[w][k] for k in ("O", "A", "A'")]}) for w in units}}
    s = res["actor detected at S1"]
    res["check O = 0.663"] = abs(s["O"] - 0.663) < 0.001
    allu = list(units.values())
    agree350 = min(min(u["A' frames>=350: O found / A' found in O"]) for u in allu)
    res["reproduced (A' within 0.02 of O, box agreement >= 0.95 both ways from frame 350)"] = bool(abs(s["A'"] - s["O"]) <= 0.02 and agree350 >= 0.95)
    res["lowest unit box agreement from frame 350 (A')"] = agree350
    (HERE / "v5.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per unit"}, indent=1))


if __name__ == "__main__":
    main()
