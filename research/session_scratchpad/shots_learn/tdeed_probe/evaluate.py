"""tdeed_probe (plan.json, hashed first): T-DEED's zero-shot shots on our
clips and windows, scored as negatives/score.py; class APs as a sanity check.

    VEO_UNSEAL=... python evaluate.py PROBS_DIR STAGE_DIR
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "negatives"))
import score as S  # noqa: E402

EA, LS = S.EA, S.LS
CLASSES = ["PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN", "SHOT",
           "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"]   # columns 1..12, 0 = background
ARM_O = {"0.6": 0.212, "0.8": 0.095}          # shots_learn/diagnose/diagnose.json, arm O


def check():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit("plan.json does not match its sha256")
    return got


def load(probs, units):
    out = {}
    for u in units:
        f = probs / f"{u['name']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        fr, p = z["frame"], z["p"].astype(np.float32)
        if u["span"]:
            k = (fr >= u["span"][0]) & (fr <= u["span"][1])
            fr, p = fr[k], p[k]
        out[u["name"]] = (fr, p)
    return out


def events(u):
    ev = pd.read_csv(LS.sd.info_of(u["name"])["dir"] / f"{u['name']}_events.csv")
    if u["span"]:
        ev = ev[(ev.frame >= u["span"][0]) & (ev.frame <= u["span"][1])]
    return ev


def dets(units, data, col_score):
    rows = [pd.DataFrame({"unit": u["name"], "match": u["match"], "frame": data[u["name"]][0],
                          "p": col_score(data[u["name"]][1])}) for u in units if u["name"] in data]
    return EA.detections(pd.concat(rows, ignore_index=True))


def minutes(fr):
    return (fr.max() - fr.min()) / LS.FPS / 60


def main():
    sha = check()
    from src.unseal import unsealed
    for m in S.V3.NEW_PLAYERS:
        if not unsealed(m, "tdeed probe: scoring M7 / M9 clips and windows (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    S.patch_info()
    probs, stage = Path(sys.argv[1]), Path(sys.argv[2])
    units = json.loads((stage / "units.json").read_text())
    data = load(probs, units)
    missing = [u["name"] for u in units if u["name"] not in data]
    units = [u for u in units if u["name"] in data]
    shot = lambda p: p[:, 8] + p[:, 12]
    labs = {u["name"]: (sorted((events(u).query("label in @LS.sd.LABELS").frame / LS.FPS).tolist()), u["match"])
            for u in units}
    det = EA.flags(dets(units, data, shot), labs)
    clips = {u["name"] for u in units if u["span"]}
    matches = sorted({u["match"] for u in units})
    per = {}
    for m in matches:
        mu = [u for u in units if u["match"] == m]
        d = det[det.match == m]
        per[m] = {"clip_det": d[d.unit.isin(clips)], "win_det": d[~d.unit.isin(clips)],
                  "n_lab": sum(len(labs[u["name"]][0]) for u in mu if u["name"] in clips),
                  "clip_min": sum(minutes(data[u["name"]][0]) for u in mu if u["name"] in clips),
                  "win_min": sum(minutes(data[u["name"]][0]) for u in mu if u["name"] not in clips)}
    res = {str(r): S.stats(per, matches, r) for r in (0.60, 0.80)}
    max_recall = round(float(det[det.unit.isin(clips)].right.sum()) / max(sum(p["n_lab"] for p in per.values()), 1), 3)
    # sanity: event AP per class (clips + windows), against random scores through the same pipeline
    rng = np.random.default_rng(0)
    rand = {n: (fr, rng.random((len(fr), 13)).astype(np.float32)) for n, (fr, p) in data.items()}
    sanity = {}
    for j, c in enumerate(CLASSES, start=1):
        lc = {u["name"]: (sorted((events(u).query("label == @c").frame / LS.FPS).tolist()), u["match"]) for u in units}
        n_lab = sum(len(v[0]) for v in lc.values())
        if not n_lab:
            continue
        aps = []
        for dd in (data, rand):
            f = EA.flags(dets(units, dd, lambda p: p[:, j]), lc)
            aps.append(round(EA.ap_of(f.score.to_numpy(), f.right.to_numpy(), n_lab)[0], 3))
        sanity[c] = {"labels": n_lab, "AP": aps[0], "AP of random scores": aps[1]}
    p60 = res["0.6"]["whole match: precision"] if res["0.6"] else None
    reading = ("promising" if p60 is not None and p60 >= 0.21
               else "transfer fails" if all(sanity[c]["AP"] <= 2 * sanity[c]["AP of random scores"] for c in ("PASS", "DRIVE"))
               else "in between")
    out = {"plan sha256": sha, "units scored": {"clips": len(clips), "windows": len(units) - len(clips)},
           "units missing": missing,
           "primary: whole-match precision at clip recall 0.60 (arm O 0.212)": p60,
           "at recall 0.80 (arm O 0.095)": res["0.8"]["whole match: precision"] if res["0.8"] else None,
           "highest clip recall reached": max_recall, "reading": reading,
           "operating points": res, "sanity: event AP per class": sanity}
    (HERE / "probe.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
