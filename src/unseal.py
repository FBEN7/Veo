"""The one way to open the test matches M7 (128057) and M9 (132831):
protocol v5 part 2 (`.cache/soccertrack_v2/ball_verified/v5p2_registration.json`).

`unsealed(match, what)` is True only when
- the environment variable VEO_UNSEAL holds the registration file's sha256,
- the match is one of its test matches, and
- every file the registration froze still has its registered sha256;
each opening is appended to `v5p2_unseal.log` (time, match, what). Without
VEO_UNSEAL it is False, so the scripts' "never opened" guards stay as they
were; with a wrong value, or a frozen file changed, it stops the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / ".cache/soccertrack_v2/ball_verified/v5p2_registration.json"
LOG = ROOT / ".cache/soccertrack_v2/ball_verified/v5p2_unseal.log"
ENV = "VEO_UNSEAL"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unsealed(match: str, what: str = "") -> bool:
    want = os.environ.get(ENV)
    if not want:
        return False
    if not REGISTRATION.exists():
        raise SystemExit(f"{ENV} is set but {REGISTRATION.name} does not exist")
    if _sha(REGISTRATION) != want:
        raise SystemExit(f"{ENV} is not the sha256 of {REGISTRATION.name}")
    reg = json.loads(REGISTRATION.read_text())
    if str(match) not in reg["test matches"]:
        return False
    changed = [f for f, h in reg["frozen sha256"].items() if _sha(ROOT / f) != h]
    if changed:
        raise SystemExit(f"files changed since the registration: {changed}")
    with open(LOG, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{match}\t{what}\n")
    return True
