"""The learned spotter's M1 test (registered in m1_learned_registration.json;
a second use of M1, so weaker evidence than a fresh test).

Checks the registered hashes, loads M1's refitted camera (calib_fix/
calib_override.py, as the E0 rerun it is compared with used), recomputes M1's
features (keyed caches) and writes output_<w>/test_L1/events.json; then the
scoring is score_events_paired.py on test_E0 and test_L1, once.

    python apply_m1.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/user/Veo")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball/calib_fix"))
sys.path.insert(0, str(HERE))
import calib_override  # noqa: E402,F401  (before anything reads st.calibration)
import learn  # noqa: E402

reg = json.loads((HERE / "m1_learned_registration.json").read_text())
for rel, sha in reg["sha256"].items():
    got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    if got != sha:
        sys.exit(f"{rel}: sha256 {got} differs from the registered {sha}")
sys.argv = ["learn.py", "apply", str(ROOT / reg["model"]), reg["config"], *reg["windows"]]
learn.main()
