"""The registered M1 events test (m1_registration.json): each configuration on
each M1 window through tune_rules.py's code path (the ball input, then the
event rules on the pitch tracks), written as output_<w>/<config>/events.json
for score_events_paired.py.

    python apply_m1.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import score_soccernet as sc  # noqa: E402
import tune_rules as t  # noqa: E402

REG = json.loads(Path(__file__).with_name("m1_registration.json").read_text())
for w in REG["windows"]:
    for name, cfg in REG["configurations"].items():
        out = t.ROOT / f"output_{w}/{name}/events.json"
        if out.exists():
            continue
        evs = t.events_of(t.metric_of(w, cfg["ball"]), cfg["rules"])
        out.parent.mkdir(parents=True, exist_ok=True)
        sc.write_events(out, evs)
        print(f"{w} {name}: {len(evs)} events", flush=True)
