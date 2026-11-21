"""The registered M5 test of protocol.json, batch by batch with the early stop.

Clips in the registered order (protocol "test.order"), batches of 8, 7, 7, 7:
cut here, uploaded, run on a Kaggle T4 kernel (players with --calibrated,
the ball over the span +-2 s, metric_step on Kaggle), collected. After each
batch is collected, the frozen model (frozen.json / model.pkl, checked
against their sha256) scores it; the counts accumulate in test.json. The test
stops when success is out of reach (protocol "early stop"). v3 was trained
on M5 (its provenance), so the ball runs with --allow-seen; disclosed.

    python test_m5.py step       # one pass: cut / upload / push / collect / score what is due
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import batches as B  # noqa: E402
import learn_shots as LS  # noqa: E402
import score_soccernet as sc  # noqa: E402

PROTOCOL = json.loads((HERE / "protocol.json").read_text())
ORDER = [0, 7, 12, 15, 6, 21, 27, 19, 3, 24, 22, 9, 23, 11, 17, 20, 16, 1, 14, 13, 10, 8, 5, 26, 18, 2, 28, 25, 4]
SIZES = [8, 7, 7, 7]
NEED = 18
OUT = HERE / "test.json"


def check_frozen():
    if hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest() != \
            (HERE / "protocol.json.sha256").read_text().strip():
        raise SystemExit("protocol.json does not match its sha256")
    if hashlib.sha256((HERE / "frozen.json").read_bytes()).hexdigest() != \
            (HERE / "frozen.json.sha256").read_text().strip():
        raise SystemExit("frozen.json does not match its sha256")
    fr = json.loads((HERE / "frozen.json").read_text())
    if hashlib.sha256((HERE / "model.pkl").read_bytes()).hexdigest() != fr["model sha256"]:
        raise SystemExit("model.pkl does not match frozen.json")
    return fr, pickle.loads((HERE / "model.pkl").read_bytes())


def test_batches() -> list[list[dict]]:
    clips = json.loads((HERE / "clips_test.json").read_text())["clips"]
    ordered = [clips[i] for i in ORDER]
    out, i = [], 0
    for n in SIZES:
        out.append(ordered[i:i + n])
        i += n
    return out


def score_clip(c: dict, mdl: dict) -> dict:
    u = {"name": c["name"], "match": c["match"], "span": c["span_local"]}
    f = LS.features(u)
    lab = LS.labels(u)
    p = mdl["model"].predict_proba(f[mdl["cols"]])[:, 1]
    fr = f.index.to_numpy()
    pred = [fr[i] / LS.FPS for i in LS.peaks(fr, p, mdl["threshold"])]
    m, fp, fn = sc.match_one_to_one(pred, lab, LS.TOL_S)
    return {"labels": len(lab), "events": len(pred), "tp": len(m), "fp": len(fp), "fn": len(fn)}


def step():
    fr, mdl = check_frozen()
    res = json.loads(OUT.read_text()) if OUT.exists() else {
        "protocol sha256": (HERE / "protocol.json.sha256").read_text().strip(),
        "frozen sha256": (HERE / "frozen.json.sha256").read_text().strip(),
        "ball": "v3 with --allow-seen: ball_v3b.1500.pt was trained on M5 (118577, both halves)",
        "batches": {}, "stopped": None}
    if res.get("stopped") or res.get("final"):
        print("test finished:", res.get("stopped") or res.get("final"))
        return
    bs = test_batches()
    B.PLAN = HERE / "clips_test.json"
    B.STATE = HERE / "test_batches"
    B.SEEN = ["118577"]
    B.DS_PREFIX = "veo-st2-m5-b"
    B.batches = lambda: bs
    attempts = sum(len(c["attempts_half"]) for b in bs for c in b)
    for k, b in enumerate(bs):
        if str(k) in res["batches"]:
            continue
        s = B.state(k)
        if not s.get("pushed"):
            B.run(only=k)                  # cut, upload, push when a kernel is free
            return
        if not s.get("collected"):
            B.collect()
            if not B.state(k).get("collected"):
                return
        per = {c["name"]: score_clip(c, mdl) for c in b}
        res["batches"][str(k)] = per
        rows = [v for bb in res["batches"].values() for v in bb.values()]
        hits, fp = sum(r["tp"] for r in rows), sum(r["fp"] for r in rows)
        seen = sum(r["labels"] for r in rows)
        left = attempts - seen
        res["so far"] = {"clips": len(rows), "attempts": seen, "hits": hits, "false": fp,
                         "attempts left": left}
        if hits + left < NEED:
            res["stopped"] = f"fails: recall out of reach after {len(rows)} of 29 clips ({hits} hits + {left} left < {NEED})"
        elif fp > hits + left:
            res["stopped"] = f"fails: precision out of reach after {len(rows)} of 29 clips ({fp} false > {hits} hits + {left} left)"
        OUT.write_text(json.dumps(res, indent=1))
        print(f"batch {k}: {res['so far']}", res.get("stopped") or "", flush=True)
        if res.get("stopped"):
            return
    if len(res["batches"]) == len(bs) and not res.get("stopped"):
        rows = [v for bb in res["batches"].values() for v in bb.values()]
        hits, fp = sum(r["tp"] for r in rows), sum(r["fp"] for r in rows)
        z = 1.96
        p = hits / attempts
        den = 1 + z * z / attempts
        mid = (p + z * z / (2 * attempts)) / den
        half = z * np.sqrt(p * (1 - p) / attempts + z * z / (4 * attempts ** 2)) / den
        prec = hits / max(hits + fp, 1)
        res["final"] = {"recall": f"{hits}/{attempts} = {p:.3f}",
                        "wilson 95%": [round(mid - half, 3), round(mid + half, 3)],
                        "precision": round(prec, 3),
                        "success": bool(hits >= NEED and prec >= 0.5)}
        OUT.write_text(json.dumps(res, indent=1))
        print("final", res["final"])


if __name__ == "__main__":
    step()
