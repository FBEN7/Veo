"""follow-cam pilot (plan.json, hashed first): score the trained spotter on
M2's two halves (whole videos) from infer.py's per-frame scores.

    python evaluate.py RESULT_DIR      # holds scores/<video>.npz
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCR = Path("/tmp/claude-0/-home-user-veo/b69e5b60-e0fe-5920-8f15-285768d7ea57/scratchpad")
CLASSES = ["PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN", "SHOT",
           "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"]   # score columns 1..12
FPS, STRIDE, NMS_S, TOL_S, DRAWS = 25, 2, 1.0, 2.0, 20
VAL = ["117093_1", "117093_2"]
V4 = {"0.6": 0.212, "0.8": 0.095}   # shots_learn: v4 trajectory spotter, whole-match precision (a different unit)


def check():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit("plan.json does not match its sha256")
    return got


def peaks(s):
    """local maxima, greedy 1-s non-maximum suppression; times in s."""
    t = np.arange(len(s)) * STRIDE / FPS
    order = np.argsort(-s, kind="stable")
    taken = np.zeros(len(s), bool)
    out = []
    half = int(round(NMS_S / 2 * FPS / STRIDE))
    for i in order:
        if taken[max(0, i - half):i + half + 1].any():
            continue
        taken[i] = True
        out.append((t[i], s[i]))
    return out


def flags(dets, labs):
    """per detection (score order), right if it takes the nearest unmatched label within 2 s."""
    dets = sorted(dets, key=lambda d: -d[1])
    lab = np.asarray(labs, float)
    used = np.zeros(len(lab), bool)
    out = []
    for t, sc in dets:
        ok = False
        if len(lab):
            d = np.abs(lab - t)
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] <= TOL_S:
                used[j] = ok = True
        out.append((sc, ok))
    return out


def ap_and_pr(fl, n_lab):
    fl = sorted(fl, key=lambda x: -x[0])
    tp = np.cumsum([ok for _, ok in fl])
    prec = tp / np.arange(1, len(fl) + 1)
    rec = tp / max(n_lab, 1)
    ap = float(np.sum(prec[[ok for _, ok in fl]]) / max(n_lab, 1)) if n_lab else float("nan")

    def p_at(r):
        k = np.where(rec >= r)[0]
        return round(float(prec[k[0]]), 3) if len(k) else None
    return ap, p_at(0.6), p_at(0.8)


def score_class(scores, labels, cols, names):
    fl, n = [], 0
    for v in VAL:
        s = scores[v][:, cols].sum(axis=1)
        lab = [e["t"] for e in labels[v] if e["label"] in names]
        fl += flags(peaks(s), lab)
        n += len(lab)
    return ap_and_pr(fl, n), n


def main():
    sha = check()
    res = Path(sys.argv[1])
    scores = {v: np.load(res / "scores" / f"{v}.npz")["scores"].astype(np.float32) for v in VAL}
    labels = {}
    for v in VAL:
        ann = json.loads((SCR / "followcam_crops" / v / "Labels-ball.json").read_text())["annotations"]
        labels[v] = [{"t": int(a["position"]) / 1000, "label": a["label"]} for a in ann]
    rng = np.random.default_rng(0)
    out = {"plan sha256": sha, "frames": {v: len(scores[v]) for v in VAL}, "per class": {}}
    for i, c in enumerate(CLASSES, start=1):
        (ap, _, _), n = score_class(scores, labels, [i], {c})
        rand = []
        for _ in range(DRAWS):
            rs = {v: rng.random((len(scores[v]), 13)).astype(np.float32) for v in VAL}
            rand.append(score_class(rs, labels, [i], {c})[0][0])
        out["per class"][c] = {"labels": n, "AP": round(ap, 3), "random AP": round(float(np.mean(rand)), 4),
                               "x random": round(ap / max(np.mean(rand), 1e-9), 1)}
    (ap, p6, p8), n = score_class(scores, labels, [8, 12], {"SHOT", "GOAL"})
    out["shots (SHOT + GOAL)"] = {"labels": n, "AP": round(ap, 3), "precision at recall 0.60": p6, "precision at recall 0.80": p8,
                                  "v4 trajectory spotter, whole match (other unit)": V4}
    pc = out["per class"]
    out["gate (SHOT and TACKLE AP >= 5 x random)"] = bool(pc["SHOT"]["x random"] >= 5 and pc["PLAYER SUCCESSFUL TACKLE"]["x random"] >= 5)
    out["mean AP over the 12 classes"] = round(float(np.mean([pc[c]["AP"] for c in CLASSES])), 3)
    (HERE / "pilot_eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
