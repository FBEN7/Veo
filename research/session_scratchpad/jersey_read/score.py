"""identity step 2a: score the committed readings against the key (plan.json + addendum1.json)."""
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    sha = {}
    for name in ("plan.json", "addendum1.json", "readings.json"):
        want = (HERE / f"{name}.sha256").read_text().split()[0]
        got = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"{name} does not match its sha256")
        sha[name] = got
    rd, key = json.loads((HERE / "readings.json").read_text()), json.loads((HERE / "key.json").read_text())
    p2 = pd.DataFrame(key["part2"]).assign(read=lambda d: d.id.map(rd["part2"]))
    p2["truth"] = p2.jersey.astype(int).astype(str)
    p2["given"], p2["correct"] = p2.read != "?", p2.read == p2.truth
    bands = {}
    for b in ("<40", "40-60", "60-80", ">=80"):
        x = p2[p2.band == b]
        bands[b] = {"crops": len(x), "read": round(float(x.given.mean()), 3), "correct": round(float(x.correct.mean()), 3),
                    "wrong among read": round(float((x.given & ~x.correct).sum() / max(x.given.sum(), 1)), 3),
                    "box height median (px)": round(float(x.bh.median()), 1)}
    p3 = pd.DataFrame(key["part3"]).assign(read=lambda d: d.id.map(rd["part3"]))
    p3["truth"] = p3.jersey.astype(int).astype(str)
    p3["given"], p3["correct"] = p3.read != "?", p3.read == p3.truth
    seg = p3.groupby("seg").agg(match=("match", "first"), bh=("bh", "median"), crops_read=("given", "sum"),
                                correct=("correct", "any"), wrong=("given", lambda s: bool((s & ~p3.loc[s.index, "correct"]).any())))
    n_ok, n_bad = int(seg.correct.sum()), int(seg.wrong.sum())
    out = {"sha256": sha, "part 2 per band (our box height)": bands,
           "part 2 all": {"crops": len(p2), "read": int(p2.given.sum()), "correct": int(p2.correct.sum()),
                          "wrong": int((p2.given & ~p2.correct).sum())},
           "part 3 segments": seg.reset_index().round(1).to_dict("records"),
           "part 3: segments with a correct read / with a wrong read (of 12)": [n_ok, n_bad],
           "part 3: crops read correctly / wrongly (of 120)": [int(p3.correct.sum()), int((p3.given & ~p3.correct).sum())],
           "decision (>= 6 correct and <= 1 wrong)": "numbers worth a GPU step" if n_ok >= 6 and n_bad <= 1 else "numbers not readable enough on this footage",
           "wrong reads": p2[p2.given & ~p2.correct][["id", "read", "truth", "bh"]].to_dict("records") + p3[p3.given & ~p3.correct][["id", "read", "truth", "bh"]].round(1).to_dict("records")}
    (HERE / "score.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("part 3 segments", "sha256")}, indent=1, default=str))
    print(seg)


if __name__ == "__main__":
    main()
