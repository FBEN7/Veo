"""Tiles for the second judge per chunk: every tile the first judge did not
label N, every tile it was unsure of, and a deterministic ~10% of the rest
(the random recheck that estimates the first judge's misses)."""
import json, sys
from pathlib import Path
V = Path(__file__).parent
unsure = {l.split()[0]: {int(t) for t in l.split()[1:]} for l in (V / "train_B_unsure.txt").read_text().splitlines() if l.strip()}
out = {}
for ci, c in enumerate(["chunk_01", "chunk_02", "chunk_03", "chunk_04"]):
    f = V / "train_B_judgements" / f"{c}_j1.txt"
    j1 = {int(a): b for a, b in (l.split()[:2] for l in f.read_text().splitlines() if l.strip())}
    if len(j1) < 100:
        print(c, "incomplete", len(j1)); continue
    u = unsure.get(c, set())
    rand = [t for t in range(1, 101) if j1[t] == "N" and t not in u and (t + ci * 3) % 10 == 0]
    re = [t for t in range(1, 101) if j1[t] != "N" or t in u or t in rand]
    out[c] = {"recheck": re, "random": rand}
    print(c, "recheck", len(re), "random", len(rand), "non-N", sum(v != "N" for v in j1.values()))
(V / "train_B_recheck.json").write_text(json.dumps(out))
