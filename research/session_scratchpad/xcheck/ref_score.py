"""Independent reference for score_ball_track.py's core numbers (cross-check only).

Reads key.json + key_passes.json (ball_tiles --pairs layout: one record per pass
with "pair frames") and a judged.txt; optional scores JSON for placebos.
Prints a JSON dict of the core numbers.
"""
import json
import math
import sys
from pathlib import Path


def tail(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def labels(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            a = line.split()
            out[int(a[0])] = a[1]
    return out


def main(key_dir, judged, scores=None, coco=None):
    key_dir = Path(key_dir)
    lab = labels(judged)
    key = json.loads((key_dir / "key.json").read_text())
    kp = json.loads((key_dir / "key_passes.json").read_text())
    B = {k["tile"]: lab.get(k["tile"]) == "B" for k in key}
    T, R = kp[0]["tracked"], kp[0]["raw"]
    pos = neg = 0
    tf = rf = 0
    q1b = q1c = 0
    hb = hc = 0
    for p in kp:
        d = 0
        for fr in p["pair frames"]:
            t = fr["tiles"].get(T)
            r = fr["tiles"].get(R)
            tb = t is not None and B[t]
            rb = r is not None and B[r]
            if fr["pair"]:
                d += int(tb and not rb) - int(rb and not tb)
                tf += int(tb and not rb)
                rf += int(rb and not tb)
                if fr["q"] == 1:
                    q1b += int(tb and not rb)
                    q1c += int(rb and not tb)
            if coco and fr["q"] == 1 and coco in fr["tiles"]:
                c = fr["tiles"][coco]
                cb = c is not None and B[c]
                hb += int(tb and not cb)
                hc += int(cb and not tb)
        pos += d > 0
        neg += d < 0
    dec = {}
    for k in key:
        if k["source"] in ("decoy", "corridor-decoy", "offset-decoy"):
            a = dec.setdefault(k["source"], [0, 0])
            a[0] += B[k["tile"]]
            a[1] += 1
    rej = [0, 0]
    for k in key:
        if k["source"] == "rejudge":
            old = labels(Path(k["old_key"]).parent / "judged.txt").get(k["old_tile"]) == "B"
            rej[0] += old == B[k["tile"]]
            rej[1] += 1
    out = {"passes": len(kp), "d>0": pos, "d<0": neg, "sign p": tail(pos, neg),
           "frames tracked only": tf, "frames raw only": rf,
           "q1": [q1b, q1c, tail(q1b, q1c)], "decoys": dec, "rejudge": rej}
    if coco:
        out["H6"] = [hb, hc, tail(hb, hc)]
    if scores:
        res = {}
        for path in scores:
            for r in json.loads(Path(path).read_text())["results"]:
                res[r["window"]] = r
        pl = {}
        for kind in ("pre", "post"):
            th = rh = 0
            for r in res.values():
                if T in r and R in r:
                    for a, b in zip(r[T][f"placebo {kind} hits"], r[R][f"placebo {kind} hits"]):
                        if a is not None and b is not None:
                            th += a
                            rh += b
            pl[kind] = [th, rh]
        out["placebo"] = pl
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    args = sys.argv[1:]
    coco = None
    if "--coco" in args:
        i = args.index("--coco")
        coco = args[i + 1]
        del args[i:i + 2]
    main(args[0], args[1], args[2:] or None, coco)
