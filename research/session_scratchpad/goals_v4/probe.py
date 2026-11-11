"""goals v4, stage 1 (plan.json, hashed first): the probe that picks the
ball detector arm at the centre spot and decides go / no go.

Per half: one sequential decode of frames 0 .. (last needed sample + 2 +
the background span); on the kick-off frames (0, 5, ... up to the first
labelled action f0) and on the live-play windows (5 one-second samples
from 40, 70, ..., 310 s, kept if they hold >= 1 labelled action and no
GOAL in the 90 s before), the region around the centre spot (+-12 m by
+-8 m projected through the match's camera, +32 px) goes to
- Y1 / Y2 / Y3: yolov8s, COCO sports ball, crop upscaled x1 / x2 / x3,
  conf floor 0.02; position = the box's bottom centre;
- H: ball_v3b.1500.pt through ball_heatmap.detect_v2 (motion from t-2 /
  t+2, motion_bg background), the region as the outline, tau 0.02, p_cal.
Rows -> .cache/ball_spot/probe_<match>_<half>.parquet (one per half,
skipped when present). `--score` then writes goals_v4/probe.json.

    VEO_UNSEAL=... python probe.py --halves 117092:1 117092:2 ... [--threads 2]
    VEO_UNSEAL=... python probe.py --score
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/shots_goals"))
from src import ball_heatmap as bh  # noqa: E402
from src import motion_bg as mb  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.constants import BALL_CLASS_ID  # noqa: E402

OUT = ROOT / ".cache/ball_spot"
MATCHES = sorted(st.MATCHES)
SEALED = {"128057", "132831"}
REGION_M = (12.0, 8.0)
PAD = 32
OUTER = 96
SCALES = {"Y1": 1, "Y2": 2, "Y3": 3}
CONFS = [0.05, 0.1, 0.2, 0.3, 0.5]
NEAR_M = 2.0
LIVE_MAX = 0.03
GO_RECALL = 0.75
CKPT = ROOT / ".cache/ball_v3b.1500.pt"


def check_plan():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
    if want != got:
        raise SystemExit("plan.json does not match its sha256")
    return got


def todo(match, half):
    ev = st.events(match)
    ev = ev[ev.half == half]
    f0 = int(ev.frame.min())
    ko = sorted(set(range(0, f0 + 1, 5)) | {0})
    goals = ev[ev.label == "GOAL"].frame.to_numpy()
    wins = []
    for s in range(40, 311, 30):
        a = s * 25
        fr = [a + 25 * k for k in range(5)]
        has_act = ((ev.frame >= a) & (ev.frame < a + 125)).any()
        after_goal = ((goals >= a - 90 * 25) & (goals < a)).any()
        if has_act and not after_goal:
            wins.append((s, fr))
    return f0, ko, wins


def boxes(cal, shape):
    H, W = shape[:2]
    sx, sy = REGION_M
    uv = bh.pitch_to_image(np.array([(-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)]), cal)
    x0, y0 = np.floor(uv.min(0)).astype(int) - PAD
    x1, y1 = np.ceil(uv.max(0)).astype(int) + PAD
    R = (max(x0, 0), max(y0, 0), min(x1, W), min(y1, H))
    O = (max(R[0] - OUTER, 0), max(R[1] - OUTER, 0), min(R[2] + OUTER, W), min(R[3] + OUTER, H))
    return R, O


def run_half(match, half, model_y, model_h, info_h, calib_h):
    import formation_scan as fs
    cal = fs.camera(match)
    f0, ko, wins = todo(match, half)
    samples = sorted(set(ko) | {f for _, fr in wins for f in fr})
    need = set()
    for t in samples:
        need |= {t - 2, t, t + 2}
    last = max(samples) + 2 + mb.BG_HALF_SPAN + mb.BG_STEP
    cont = st.open_video(match, half)
    stream = cont.streams.video[0]
    stream.thread_type = "AUTO"
    tb = float(stream.time_base)
    t0 = float(stream.start_time or 0) * tb
    crops, R, O, bgc, outline_local = {}, None, None, None, None
    t_dec = time.time()
    for frame in cont.decode(stream):
        k = int(round((float(frame.pts) * tb - t0) * st.FPS))
        if k > last:
            break
        if k % mb.BG_STEP and k not in need:
            continue
        img = frame.to_ndarray(format="bgr24")
        if R is None:
            R, O = boxes(cal, img.shape)
            ox, oy = O[0], O[1]
            outline_local = np.array([(R[0] - ox, R[1] - oy), (R[2] - ox, R[1] - oy),
                                      (R[2] - ox, R[3] - oy), (R[0] - ox, R[3] - oy)], np.float32)
            oshape = (O[3] - O[1], O[2] - O[0])
            bgc = mb.BackgroundCache(0, last, crop=bh.pitch_box(oshape, outline_local))
        oc = np.ascontiguousarray(img[O[1]:O[3], O[0]:O[2]])
        bgc.add(k, oc)
        if k in need:
            crops[k] = oc
    cont.close()
    bgc.finish()
    t_dec = time.time() - t_dec
    rows = []
    ox, oy = O[0], O[1]
    rx0, ry0, rx1, ry1 = R[0] - ox, R[1] - oy, R[2] - ox, R[3] - oy
    t_det = {a: 0.0 for a in [*SCALES, "H"]}
    kind = {f: "ko" for f in ko}
    win_of = {}
    for s, fr in wins:
        for f in fr:
            kind.setdefault(f, "live")
            win_of[f] = s
    for t in samples:
        cur = crops.get(t)
        if cur is None:
            continue
        reg = cur[ry0:ry1, rx0:rx1]
        for arm, s in SCALES.items():
            tt = time.time()
            im = cv2.resize(reg, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR) if s != 1 else reg
            imgsz = -(-im.shape[1] // 32) * 32
            r = model_y(im, verbose=False, conf=0.02, imgsz=imgsz, classes=[BALL_CLASS_ID])[0]
            xyxy = r.boxes.xyxy.cpu().numpy().reshape(-1, 4) / s
            cf = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, cf):
                rows.append((t, arm, float(c), (x1 + x2) / 2 + R[0], y2 + R[1]))
            t_det[arm] += time.time() - tt
        tt = time.time()
        bg, _ = bgc.get(t)
        det = bh.detect_v2(model_h, crops.get(t - 2), cur, crops.get(t + 2), bg, outline_local,
                           tau=0.02, top_k=8, cal=None, sigma_d=info_h["sigma_d"], sigma_f=info_h["sigma_f"])
        for cu, cv_, p, _, z in det:
            rows.append((t, "H", float(bh.p_cal(z, calib_h)), cu + ox, cv_ + oy))
        t_det["H"] += time.time() - tt
    d = pd.DataFrame(rows, columns=["frame", "arm", "conf", "u", "v"])
    xy = st.image_to_pitch(d[["u", "v"]].to_numpy(float), cal) if len(d) else np.zeros((0, 2))
    d["x"], d["y"] = xy[:, 0], xy[:, 1]
    d["match"], d["half"] = match, half
    d["set"] = d.frame.map(kind)
    d["window"] = d.frame.map(win_of)
    meta = {"match": match, "half": half, "f0": f0, "kick-off frames": ko, "windows": [s for s, _ in wins],
            "samples": len(samples), "region px": list(map(int, R)), "decoded up to": last,
            "decode s": round(t_dec, 1), "detect s per sample": {a: round(v / max(len(samples), 1), 3) for a, v in t_det.items()}}
    return d, meta


def score():
    from src.unseal import unsealed  # noqa: F401  (openings were logged by the runs)
    metas, rows = [], []
    for m in MATCHES:
        for h in (1, 2):
            p = OUT / f"probe_{m}_{h}.parquet"
            if not p.exists():
                raise SystemExit(f"{p.name} missing")
            rows.append(pd.read_parquet(p))
            metas.append(json.loads(p.with_suffix(".json").read_text()))
    d = pd.concat(rows, ignore_index=True)
    d["near"] = np.hypot(d.x, d.y) <= NEAR_M
    live = []
    for mt in metas:
        for s in mt["windows"]:
            for k in range(5):
                live.append((mt["match"], mt["half"], s * 25 + 25 * k))
    live = pd.DataFrame(live, columns=["match", "half", "frame"])
    out = {"plan sha256": check_plan(), "halves": len(metas), "live-play samples": len(live), "arms": {}}
    for arm in [*SCALES, "H"]:
        a = d[(d.arm == arm) & d.near]
        res = {}
        for c in CONFS:
            ac = a[a.conf >= c]
            ko_hit = sum(bool(((ac.match == mt["match"]) & (ac.half == mt["half"]) & (ac.set == "ko")).any()) for mt in metas)
            lv = ac[ac.set == "live"][["match", "half", "frame"]].drop_duplicates()
            res[c] = {"kick-off recall": round(ko_hit / len(metas), 3), "kick-off halves": ko_hit,
                      "live rate": round(len(lv) / len(live), 4)}
        ok = [c for c in CONFS if res[c]["live rate"] <= LIVE_MAX]
        cstar = min(ok) if ok else None
        out["arms"][arm] = {"per conf": {str(c): v for c, v in res.items()}, "c*": cstar,
                            "recall at c*": res[cstar]["kick-off recall"] if cstar is not None else None,
                            "live rate at c*": res[cstar]["live rate"] if cstar is not None else None,
                            "detect s per sample (median over halves)": float(np.median([mt["detect s per sample"][arm] for mt in metas]))}
    order = [*SCALES, "H"]
    cands = [a for a in order if out["arms"][a]["c*"] is not None]
    if cands:
        best = max(cands, key=lambda a: (out["arms"][a]["recall at c*"], -out["arms"][a]["live rate at c*"], -order.index(a)))
        out["chosen arm"], out["chosen c*"] = best, out["arms"][best]["c*"]
        out["go (recall at c* >= 0.75)"] = bool(out["arms"][best]["recall at c*"] >= GO_RECALL)
    else:
        out["chosen arm"], out["go (recall at c* >= 0.75)"] = None, False
    out["per half (chosen arm at c*, kick-off found)"] = {
        f"{mt['match']}_{mt['half']}": bool(((d.arm == out["chosen arm"]) & d.near & (d.conf >= (out.get("chosen c*") or 9))
                                             & (d.match == mt["match"]) & (d.half == mt["half"]) & (d.set == "ko")).any())
        for mt in metas} if out["chosen arm"] else {}
    out["decode s per half (median)"] = float(np.median([mt["decode s"] for mt in metas]))
    (HERE / "probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--halves", nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    check_plan()
    from src.unseal import unsealed
    if a.score:
        return score()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(a.threads)
    import crop_check as cc
    model_y = YOLO(str(cc.MODEL))
    model_h, info_h = bh.load_v2(CKPT)
    calib_h = bh.load_calibration(CKPT)
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in a.halves:
        m, h = spec.split(":")
        h = int(h)
        p = OUT / f"probe_{m}_{h}.parquet"
        if p.exists():
            print(spec, "done already", flush=True)
            continue
        if m in SEALED and not unsealed(m, "goals v4 stage 1: ball probe (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
        t = time.time()
        d, meta = run_half(m, h, model_y, model_h, info_h, calib_h)
        tmp = p.with_suffix(".tmp.parquet")
        d.to_parquet(tmp)
        p.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        tmp.rename(p)
        print(spec, len(d), "rows", round(time.time() - t), "s", meta["detect s per sample"], "decode", meta["decode s"], flush=True)


if __name__ == "__main__":
    main()
