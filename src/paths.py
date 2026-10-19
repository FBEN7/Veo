"""Where footage, labels and caches live, outside the repository.

Videos, SoccerNet label files and anything derived from them are under a
non-commercial agreement and must not be committed. Scripts find them
through these two settings instead of a path written into the code:

    VEO_DATA_DIR    footage and label files        (default: ./data)
    VEO_CACHE_DIR   intermediate results to reuse  (default: ./.cache)

Both defaults are gitignored.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("VEO_DATA_DIR", REPO / "data"))
CACHE_DIR = Path(os.environ.get("VEO_CACHE_DIR", REPO / ".cache"))
