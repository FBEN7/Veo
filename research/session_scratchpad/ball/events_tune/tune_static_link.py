"""Step 2a, second candidate (decided after tune_rules.py, from its per-window
scores: the tracked ball fails on M9, the static-point filter + linker is the
only input that holds there): stages B and C of tune_rules.py with the ball
fixed to v3s_link, pooled objective over the dev windows, and leave one
window out. M1 then scores both candidates and the current pipeline."""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tune_rules as t  # noqa: E402

cache, log = {}, []


def stages(windows, ball="v3s_link"):
    rules = dict(t.FROZEN)
    for stage, grid in (("B", t.GRID_B), ("C", t.GRID_C)):
        keys = list(grid)
        cands = [{**rules, **dict(zip(keys, v))} for v in itertools.product(*grid.values())]
        scored = [(t.objective(ball, r, windows, cache), r) for r in cands]
        log.extend({"stage": stage, "windows": windows, "rules": r, **s} for s, r in scored)
        rules = max(scored, key=lambda x: x[0]["objective"])[1]
    return rules, t.objective(ball, rules, windows, cache)


rules, s = stages(t.DEV)
print("v3s_link chosen:", rules, s, flush=True)
lowo = {}
for held in t.DEV:
    r, _ = stages([w for w in t.DEV if w != held])
    lowo[held] = {"rules": r, "on held": t.objective("v3s_link", r, [held], cache),
                  "frozen v2_link on held": t.objective("v2_link", t.FROZEN, [held], cache)}
    print("held", held, lowo[held]["on held"]["objective"], "frozen v2", lowo[held]["frozen v2_link on held"]["objective"], flush=True)
Path(__file__).with_name("tune_static_link.json").write_text(json.dumps(
    {"ball": "v3s_link", "chosen": {"rules": rules, **s}, "leave one window out": lowo, "log": log}, indent=1))
