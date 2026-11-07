"""Use a refitted camera model where one exists (.cache/calib_fix/<match>.npz,
fit_players.py), without touching the frozen src/soccertrack_v2.py: importing
this module rebinds `soccertrack_v2.calibration`, which every caller looks up
at call time. Matches without a refit keep the release's model."""
from pathlib import Path

import numpy as np

from src import soccertrack_v2 as st

FIX = Path("/home/user/Veo/.cache/calib_fix")
_release = st.calibration


def calibration(match: str) -> dict:
    p = FIX / f"{match}.npz"
    if p.exists():
        z = np.load(p)
        return {"K": z["K"], "D": z["D"], "Knew": z["Knew"], "Hinv": z["Hinv"]}
    return _release(match)


st.calibration = calibration
