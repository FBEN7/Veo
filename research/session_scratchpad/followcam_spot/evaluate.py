"""follow-cam pilot (plan.json, hashed first): score the trained spotter on
M2's two halves (whole videos) from infer.py's per-frame scores.

    python evaluate.py RESULT_DIR             # pilot (plan.json): M2 -> pilot_eval.json
    python evaluate.py RESULT_DIR --plan2     # plan2.json: M2 + M10 pooled, and M2 alone -> plan2_eval.json
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
EVAL5 = VAL + ["132877_1", "132877_2"]
PILOT_M2 = HERE / "pilot_eval.json"
V4 = {"0.6": 0.212, "0.8": 0.095}   # shots_learn: v4 trajectory spotter, whole-match precision (a different unit)


def check(plan="plan.json"):
    want = (HERE / f"{plan}.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / plan).read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"{plan} does not match its sha256")
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


def score_class(scores, labels, cols, names, videos=VAL):
    fl, n = [], 0
    for v in videos:
        s = scores[v][:, cols].sum(axis=1)
        lab = [e["t"] for e in labels[v] if e["label"] in names]
        fl += flags(peaks(s), lab)
        n += len(lab)
    return ap_and_pr(fl, n), n


def evaluate(scores, labels, videos, sha, seed=0):
    rng = np.random.default_rng(seed)
    out = {"plan sha256": sha, "frames": {v: len(scores[v]) for v in videos}, "per class": {}}
    for i, c in enumerate(CLASSES, start=1):
        (ap, _, _), n = score_class(scores, labels, [i], {c}, videos)
        rand = []
        for _ in range(DRAWS):
            rs = {v: rng.random((len(scores[v]), 13)).astype(np.float32) for v in videos}
            rand.append(score_class(rs, labels, [i], {c}, videos)[0][0])
        out["per class"][c] = {"labels": n, "AP": round(ap, 3), "random AP": round(float(np.mean(rand)), 4),
                               "random AP p95": round(float(np.percentile(rand, 95)), 4),
                               "x random": round(ap / max(np.mean(rand), 1e-9), 1)}
    (ap, p6, p8), n = score_class(scores, labels, [8, 12], {"SHOT", "GOAL"}, videos)
    out["shots (SHOT + GOAL)"] = {"labels": n, "AP": round(ap, 3), "precision at recall 0.60": p6, "precision at recall 0.80": p8,
                                  "v4 trajectory spotter, whole match (other unit)": V4}
    pc = out["per class"]
    out["gate (SHOT and TACKLE AP >= 5 x random)"] = bool(pc["SHOT"]["x random"] >= 5 and pc["PLAYER SUCCESSFUL TACKLE"]["x random"] >= 5)
    out["mean AP over the 12 classes"] = round(float(np.mean([pc[c]["AP"] for c in CLASSES])), 3)
    return out


def load(res, videos):
    scores = {v: np.load(res / "scores" / f"{v}.npz")["scores"].astype(np.float32) for v in videos}
    labels = {}
    for v in videos:
        ann = json.loads((SCR / "followcam_crops" / v / "Labels-ball.json").read_text())["annotations"]
        labels[v] = [{"t": int(a["position"]) / 1000, "label": a["label"]} for a in ann]
    return scores, labels


def main():
    res = Path(sys.argv[1])
    if "--plan2" not in sys.argv:
        out = evaluate(*load(res, VAL), VAL, check())
        (HERE / "pilot_eval.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        return
    sha = check("plan2.json")
    scores, labels = load(res, EVAL5)
    pooled = evaluate(scores, labels, EVAL5, sha)
    m2 = evaluate(scores, labels, VAL, sha)
    pilot = json.loads(PILOT_M2.read_text())
    pc = pooled["per class"]
    xs, xt = pc["SHOT"]["x random"], pc["PLAYER SUCCESSFUL TACKLE"]["x random"]
    outcome = "A, learning signal" if xs >= 5 and xt >= 5 else "B, trend" if xs >= 2 and xt >= 2 else "C, no signal"
    cmp_ = {c: {"pilot AP": pilot["per class"][c]["AP"], "5 matches AP": m2["per class"][c]["AP"],
                "pilot x random": pilot["per class"][c]["x random"], "5 matches x random": m2["per class"][c]["x random"]}
            for c in ("SHOT", "PLAYER SUCCESSFUL TACKLE", "PASS")}
    cmp_["mean AP over the 12 classes"] = {"pilot": pilot["mean AP over the 12 classes"], "5 matches": m2["mean AP over the 12 classes"]}
    out = {"plan sha256": sha, "decision (M2 + M10)": outcome, "M2 + M10 pooled": pooled, "M2 alone": m2,
           "M2 alone, against the pilot (2 training matches, same schedule)": cmp_}
    (HERE / "plan2_eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"decision": outcome, "pooled SHOT": pc["SHOT"], "pooled TACKLE": pc["PLAYER SUCCESSFUL TACKLE"],
                      "against the pilot": cmp_}, indent=1))


if __name__ == "__main__":
    main()
