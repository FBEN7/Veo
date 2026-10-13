"""Independent recomputation of a pair sitting (no reuse of scorer code)."""
import json, sys, collections
from math import comb

T, RAW, COCO = "T", "ball_v2.4000.pt", "coco"


def sign_p(k, n):
    """One-sided exact binomial P(X >= k | n, 0.5)."""
    if n == 0:
        return 1.0
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def load_judged(path):
    labels, dup, bad = {}, [], []
    for ln, line in enumerate(open(path), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) != 2 or not parts[0].isdigit():
            bad.append((ln, s))
            continue
        t = int(parts[0])
        if t in labels:
            dup.append((t, labels[t], parts[1]))
        labels[t] = parts[1]
    return labels, dup, bad


def main(judged, keyp, passp):
    labels, dup, bad = load_judged(judged)
    key = json.load(open(keyp))
    passes = json.load(open(passp))
    defects = []
    print(f"judged entries {len(labels)}, duplicates {dup}, malformed {bad}")
    print("label counts", dict(collections.Counter(labels.values())))
    kt = [r["tile"] for r in key]
    kdup = [t for t, c in collections.Counter(kt).items() if c > 1]
    if kdup:
        defects.append(f"key tile numbers duplicated {kdup}")
    byt = {r["tile"]: r for r in key}
    missing = sorted(set(byt) - set(labels))
    extra = sorted(set(labels) - set(byt))
    print(f"key tiles {len(key)} (min {min(kt)}, max {max(kt)}, contiguous {sorted(kt) == list(range(1, len(kt) + 1))}); missing judgements {missing}; judged but not in key {extra}")
    if missing or extra or dup or bad:
        defects.append("judgement coverage problem")
    unknown = {t: l for t, l in labels.items() if l not in ("B", "B*", "N", "S")}
    if unknown:
        defects.append(f"unknown labels {unknown}")

    isB = lambda t: t is not None and labels.get(t) == "B"

    # --- consistency key <-> passes
    credited = collections.defaultdict(set)  # tile -> detectors that point at it in passes
    tile_ctx = collections.defaultdict(set)
    q1count = collections.Counter()
    for p in passes:
        if p["tracked"] != T or p["raw"] != RAW:
            defects.append(f"pass {p['window']}/{p['pass']} tracked/raw {p['tracked']}/{p['raw']}")
        frames_seen = collections.Counter(f["frame"] for f in p["pair frames"])
        if any(c > 1 for c in frames_seen.values()):
            defects.append(f"pass {p['window']}/{p['pass']} repeats a pair frame")
        for f in p["pair frames"]:
            if f["q"] == 1:
                q1count[(p["window"], p["pass"])] += 1
            if f["frame"] not in p["frames"]:
                defects.append(f"pass {p['window']}/{p['pass']} pair frame {f['frame']} not in pass frames {p['frames']}")
            for det, t in f["tiles"].items():
                if t is None:
                    continue
                credited[t].add(det)
                tile_ctx[t].add((p["window"], p["pass"], f["frame"], f["q"]))
    for p in passes:
        c = q1count[(p["window"], p["pass"])]
        if c != 1:
            defects.append(f"pass {p['window']}/{p['pass']} has {c} q1 pair frames")
    no_q1 = sum(1 for p in passes if q1count[(p["window"], p["pass"])] == 0)
    multi_q1 = sum(1 for p in passes if q1count[(p["window"], p["pass"])] > 1)
    print(f"passes {len(passes)}; without a q1 pair frame {no_q1}; with >1 q1 pair frame {multi_q1}")

    pair_tiles = {r["tile"] for r in key if r["kind"] == "pair"}
    nshared = collections.Counter()
    for t, dets in credited.items():
        r = byt.get(t)
        if r is None:
            defects.append(f"pass references tile {t} absent from key")
            continue
        if r["kind"] != "pair":
            defects.append(f"pass references non-pair tile {t} ({r['kind']})")
        if set(r["sources"]) != dets:
            defects.append(f"tile {t}: key sources {r['sources']} vs passes credit {sorted(dets)}")
        if "+".join(r["sources"]) != r["source"]:
            defects.append(f"tile {t}: source string {r['source']} vs list {r['sources']}")
        if len(tile_ctx[t]) != 1:
            defects.append(f"tile {t} used in several pass contexts {tile_ctx[t]}")
        (w, ps, fr, q), = list(tile_ctx[t])[:1] or [(None,) * 4]
        if (r["window"], r["pass"], r["frame"]) != (w, ps, fr):
            defects.append(f"tile {t}: key ({r['window']},{r['pass']},{r['frame']}) vs pass ({w},{ps},{fr})")
        if r.get("q") != q:
            defects.append(f"tile {t}: key q {r.get('q')} vs pass q {q}")
        nshared[len(dets)] += 1
    orphan = sorted(pair_tiles - set(credited))
    if orphan:
        defects.append(f"pair tiles never referenced by a pass {orphan}")
    print(f"pair tiles {len(pair_tiles)}, referenced {len(credited)}, by #detectors credited {dict(sorted(nshared.items()))}, orphans {orphan}")
    # dedup sanity: within one pair frame, two detectors with the same tile must be the key's shared tile; with
    # different tiles, check they are not at identical (u,v)
    near_dupe = []
    for p in passes:
        for f in p["pair frames"]:
            ts = {d: t for d, t in f["tiles"].items() if t is not None}
            dist = sorted(set(ts.values()))
            for i in range(len(dist)):
                for j in range(i + 1, len(dist)):
                    a, b = byt[dist[i]], byt[dist[j]]
                    dd = ((a["u"] - b["u"]) ** 2 + (a["v"] - b["v"]) ** 2) ** 0.5
                    if dd < 3:
                        near_dupe.append((dist[i], dist[j], round(dd, 2)))
    print(f"distinct tiles at the same pair frame closer than 3 px (possible missed dedup): {near_dupe}")

    # --- statistics
    npos = nneg = nzero = 0
    fc = collections.Counter()
    by = collections.defaultdict(collections.Counter)
    q1 = {RAW: [0, 0], COCO: [0, 0]}
    q1n = {RAW: 0, COCO: 0}
    for p in passes:
        d = 0
        for f in p["pair frames"]:
            tB = isB(f["tiles"].get(T))
            rB = isB(f["tiles"].get(RAW))
            cell = "only tracked" if tB and not rB else "only raw" if rB and not tB else "both" if tB else "neither"
            fc[cell] += 1
            by["source " + f["track source"]][cell] += 1
            by["parity " + f["parity"]][cell] += 1
            by["source x parity " + f["track source"] + " " + f["parity"]][cell] += 1
            by["match " + p["match"]][cell] += 1
            d += (cell == "only tracked") - (cell == "only raw")
            if f["q"] == 1:
                for other in (RAW, COCO):
                    if other not in f["tiles"]:
                        continue
                    q1n[other] += 1
                    oB = isB(f["tiles"].get(other))
                    if tB and not oB:
                        q1[other][0] += 1
                    elif oB and not tB:
                        q1[other][1] += 1
        if d > 0:
            npos += 1
        elif d < 0:
            nneg += 1
        else:
            nzero += 1
    print(f"H5 passes d>0 {npos}, d<0 {nneg}, d=0 {nzero}; one-sided sign p = {sign_p(npos, npos + nneg):.4f}")
    print(f"pair frames {sum(fc.values())}: only tracked {fc['only tracked']}, only raw {fc['only raw']}, both {fc['both']}, neither {fc['neither']}")
    for o in (RAW, COCO):
        b, c = q1[o]
        if q1n[o]:
            print(f"q1 McNemar T vs {o}: {q1n[o]} q1 frames, T only {b}, {o} only {c}, one-sided p = {sign_p(b, b + c):.4f}")
    for k in sorted(by):
        v = by[k]
        print(f"  {k}: only tracked {v['only tracked']}, only raw {v['only raw']}, both {v['both']}, neither {v['neither']}")
    dk = collections.defaultdict(lambda: [0, 0])
    for r in key:
        if r["kind"] in ("pair", "rejudge"):
            continue
        dk[r["kind"]][1] += 1
        dk[r["kind"]][0] += labels.get(r["tile"]) == "B"
    print("decoys judged B:", {k: f"{a} of {n}" for k, (a, n) in sorted(dk.items())})
    dkany = {k: sum(1 for r in key if r["kind"] == k and labels.get(r["tile"]) in ("B", "B*")) for k in dk}
    print("decoys judged B or B*:", dkany)
    print("DEFECTS:" if defects else "DEFECTS: none")
    for d in defects:
        print("  -", d)


if __name__ == "__main__":
    main(*sys.argv[1:4])
