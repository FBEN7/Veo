"""moving_cam on TeamTrack (plan_teamtrack.json): evaluate.py's functions,
unchanged, pointed at the TeamTrack windows.

    python evaluate_tt.py run tt_0360       # chain on one window (cached)
    python evaluate_tt.py score [--dev]     # -> tt_results.json / tt_dev_results.json
"""
import hashlib
import sys

import evaluate as ev
from render_tt import HERE, OUT

DEV = ["tt_0180"]
TEST = ["tt_0000", "tt_0360", "tt_0720", "tt_1080", "tt_1440", "tt_1800"]

if __name__ == "__main__":
    ev.check()
    want = (HERE / "plan_teamtrack.json.sha256").read_text().split()[0]
    if hashlib.sha256((HERE / "plan_teamtrack.json").read_bytes()).hexdigest() != want:
        raise SystemExit("plan_teamtrack.json does not match its sha256")
    ev.WIN, ev.RUNS = OUT, OUT.parent / "tt_runs"
    if sys.argv[1] == "run":
        ev.run(sys.argv[2])
    elif sys.argv[1] == "score":
        dev = "--dev" in sys.argv
        ev.score(DEV if dev else TEST, "tt_dev_results" if dev else "tt_results")
