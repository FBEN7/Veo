"""Run eval_ball_detector.reuse from the cwd's code on the given scores; dump results."""
import json, sys
sys.path.insert(0, ".")
import eval_ball_detector as ev
out, paths = sys.argv[1], sys.argv[2:]
results, meta = ev.reuse(paths, log=lambda *a: None)
names = list(dict.fromkeys(k for r in results for k, v in r.items()
                           if isinstance(v, dict) and "candidates" in v))
kinds = {n: ev.kind_of(n, results, meta) for n in names}
summary = ev.summarise(results, names, kinds)
json.dump({"results": results, "summary": summary}, open(out, "w"))
