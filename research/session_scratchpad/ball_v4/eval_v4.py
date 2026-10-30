"""Ball detector v4 against v3b on the clicked frames of matches a fold did
not train its clicks on, and on M6 (both folds' guard).

Per clip, one background pass and one decode; every checkpoint runs
`detect_v2` on the same frames (`detect_ball_heatmap.dense_v2`'s loop, all
models at once; the cuda wrapper of kaggle/ball_gpu.py). Then per clicked
frame with a ball: hit@1 (the top candidate within 12 px), hit@4 (one of
the top 4), and per frame without a ball in its crop: a candidate in the
crop. The ranks need no calibration; v4 carries v3b's Platt fit, so a cut
on its confidence is only indicative (`conf10`).

    python eval_v4.py --stage STAGE --clips DIR [DIR ...] --ckpt NAME=PATH ... --out OUT
    (or --part I --parts N per process; then concatenate the parts and)
    python eval_v4.py --stage STAGE --score OUT
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

R_PX = 12.0
CW, CH = 800, 450          # the click page's crop
TAU, TOP_K = 0.05, 8


def find(clips, name):
    for d in clips:
        hit = list(Path(d).rglob(name))
        if hit:
            return hit[0]
    raise SystemExit(f"{name} not found under {clips}")


def load(ckpt, device):
    from src import ball_heatmap as bh

    model, info = bh.load_v2(ckpt)
    if device == "cuda":
        model.to("cuda")
        fwd = model.forward
        model.forward = lambda x: fwd(x.to("cuda", non_blocking=True)).float().cpu()
    return model, info, bh.load_calibration(ckpt)


def dense_many(path, info, models, frames, cal):
    """{name: rows (t, u, v, conf, logit)} at the local `frames`."""
    import cv2
    from detect_ball_heatmap import background_pass
    from src.ball_heatmap import detect_v2, p_cal, pitch_box
    from src.motion_bg import BG_HALF_SPAN, BG_STEP, OFFSET, RingBuffer, bg_sample_frames

    first = int(info["start_frame"])
    last = first + int(info["frames"]) - 1
    wanted = set(int(f) for f in frames)
    need = set()
    for t in wanted:
        need |= set(bg_sample_frames(first + t, first, last))
    box = pitch_box((info["height"], info["width"]), info["pitch"])
    cache, _ = background_pass(path, first, last, need, box)
    out = {k: [] for k in models}

    def run(t, ring):
        bg, n = cache.get(first + t)
        prev, cur, nxt = ring.triple(t)
        for k, (model, mi, calib) in models.items():
            for u, v, p, size, z in detect_v2(model, prev, cur, nxt, bg, info["pitch"], tau=TAU,
                                              top_k=TOP_K, cal=cal, motion_zero=mi["rgb_only"],
                                              sigma_d=mi["sigma_d"], sigma_f=mi["sigma_f"]):
                out[k].append((t, u, v, float(p_cal(z, calib)), z))
        cache.drop_before(first + t - BG_HALF_SPAN - BG_STEP)

    cap = cv2.VideoCapture(str(path))
    ring = RingBuffer(2 * OFFSET + 1)
    idx, end = 0, max(wanted) + OFFSET
    while idx <= end:
        ok, image = cap.read()
        if not ok:
            break
        ring.push(idx, image)
        if idx - OFFSET in wanted:
            run(idx - OFFSET, ring)
        idx += 1
    cap.release()
    for t in range(max(idx - OFFSET, 0), idx):
        if t in wanted:
            run(t, ring)
    return out


def detect(args):
    import torch
    from src import soccertrack_v2 as st

    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for s in args.ckpt:
        k, p = s.split("=", 1)
        models[k] = load(p, device)
    tf = pd.read_parquet(Path(args.stage) / "test_frames.parquet")
    units = sorted(set(tf.unit))[args.part::args.parts]
    rows, t0 = [], time.time()
    for unit, g in tf[tf.unit.isin(units)].groupby("unit"):
        info = json.loads(find(args.clips, f"{unit}.json").read_text())
        res = dense_many(find(args.clips, f"{unit}.mp4"), info, models, sorted(set(g.frame)),
                         st.calibration(str(info["match"])))
        for k, rs in res.items():
            rows += [(k, unit, *r) for r in rs]
        print(f"  {unit}: {g.frame.nunique()} frames [{time.time() - t0:.0f} s]", flush=True)
    d = pd.DataFrame(rows, columns=["model", "unit", "frame", "u", "v", "conf", "logit"])
    d.to_parquet(args.out)
    if args.parts == 1:
        score(args.out, tf)


def score(out, tf=None, stage=None):
    d = pd.read_parquet(out)
    if tf is None:
        tf = pd.read_parquet(Path(stage) / "test_frames.parquet")
    man = {s["id"]: s for s in json.loads(
        (ROOT / "research/session_scratchpad/ball_clicks/manifest.json").read_text())}
    res = []
    for r in tf.itertuples():
        s = man.get(r.shot, {"x0": 0, "y0": 0})          # guard_w: ball frames only
        x0, y0 = s["x0"], s["y0"]
        for k, g in d[(d.unit == r.unit) & (d.frame == r.frame)].groupby("model"):
            g = g.sort_values("conf", ascending=False)
            row = {"model": k, "fold": r.fold, "match": r.match, "status": r.status, "k": r.k,
                   "spotter_hit": r.spotter_hit, "unit": r.unit, "frame": r.frame, "shot": r.shot}
            if r.status == "ball":
                dist = np.hypot(g.u - r.u, g.v - r.v).to_numpy()
                row.update(hit1=bool(len(dist) and dist[0] <= R_PX),
                           hit4=bool((dist[:4] <= R_PX).any()),
                           conf10=bool(((dist <= R_PX) & (g.conf.to_numpy() >= 0.10)).any()))
            elif r.status == "none":
                inside = (g.u.between(x0, x0 + CW) & g.v.between(y0, y0 + CH)).to_numpy()
                row.update(fp_any=bool(inside.any()),
                           fp10=bool((inside & (g.conf.to_numpy() >= 0.10)).any()))
            res.append(row)
        if r.status in ("ball", "none"):
            for k in set(d.model) - set(d[(d.unit == r.unit) & (d.frame == r.frame)].model):
                res.append({"model": k, "fold": r.fold, "match": r.match, "status": r.status,
                            "k": r.k, "spotter_hit": r.spotter_hit, "unit": r.unit,
                            "frame": r.frame, "shot": r.shot,
                            **({"hit1": False, "hit4": False, "conf10": False}
                               if r.status == "ball" else {"fp_any": False, "fp10": False})})
    res = pd.DataFrame(res)
    res.to_parquet(Path(out).with_suffix(".frames.parquet"))
    summary = {}
    for (fold, k), g in res.groupby(["fold", "model"]):
        b, n = g[g.status == "ball"], g[g.status == "none"]
        summary[f"{fold}/{k}"] = {
            "ball frames": len(b), "hit1": round(b.hit1.mean(), 3), "hit4": round(b.hit4.mean(), 3),
            "conf10": round(b.conf10.mean(), 3),
            "after kick (k>=1) hit4": round(b[b.k >= 1].hit4.mean(), 3),
            "missed shots hit4": round(b[~b.spotter_hit.astype(bool)].hit4.mean(), 3),
            "none frames": len(n), "fp_any": round(n.fp_any.mean(), 3) if len(n) else None,
            "fp10": round(n.fp10.mean(), 3) if len(n) else None}
    summary["primary"] = primary(res)
    Path(out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def primary(res: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """protocol.json's endpoint: hit@4 on the clicked ball frames after the
    kick (k >= 1) of folds A and B, each frame by the model of the other fold
    ("<recipe>.<step>", from "<recipe>A.<step>" and "<recipe>B.<step>") against v3b, paired per frame; 95% interval of the
    difference by a bootstrap over shots (a shot's frames move together)."""
    b = res[(res.status == "ball") & res.fold.isin(["A", "B"]) & (res.k >= 1)]
    key = ["unit", "frame", "shot"]
    v3 = b[b.model == "v3b"].set_index(key).hit4
    pairs = sorted({(m.split(".")[0][:-1], m.split(".")[1]) for m in set(b.model)
                    if m != "v3b" and m.split(".")[0].endswith("A")})         # ("v4r", "600")
    out = {}
    for p, s in pairs:
        v4 = b[((b.fold == "A") & (b.model == f"{p}B.{s}"))
               | ((b.fold == "B") & (b.model == f"{p}A.{s}"))].set_index(key).hit4
        m = pd.concat({"v3": v3, "v4": v4}, axis=1, join="inner").astype(float).reset_index()
        d = (m.v4 - m.v3).to_numpy()
        codes, ids = pd.factorize(m.shot)
        groups = [np.flatnonzero(codes == i) for i in range(len(ids))]
        rng = np.random.default_rng(seed)
        boot = [d[np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])].mean()
                for _ in range(n_boot)] if len(groups) else [np.nan]
        out[f"{p}.{s}"] = {"frames": len(m), "shots": len(ids),
                          "v3b hit4": round(float(m.v3.mean()), 3), "v4 hit4": round(float(m.v4.mean()), 3),
                          "difference": round(float(d.mean()), 3),
                          "95% interval": [round(float(np.percentile(boot, 2.5)), 3),
                                           round(float(np.percentile(boot, 97.5)), 3)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage")
    ap.add_argument("--clips", nargs="+")
    ap.add_argument("--ckpt", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--score", help="score these detections (the parts concatenated)")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    args = ap.parse_args()
    if args.score:
        score(args.score, stage=args.stage)
    else:
        detect(args)


if __name__ == "__main__":
    main()
