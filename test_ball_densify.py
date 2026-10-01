"""Which frames get full-resolution detection, and how they join the tracks.

The detection itself needs the model and footage; this checks the parts
around it that decide what is scanned and what the path sees. Runs
standalone and exits non-zero on failure.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.ball_densify import FULLRES_FILE, goal_view_frames, with_fullres


def main() -> int:
    ok = True

    got = goal_view_frames([8, 12, 40], grid=4, n_frames=42)
    want = [6, 7, 8, 9, 10, 11, 12, 13, 14, 38, 39, 40, 41]
    good = got == want
    ok &= good
    print(f"   goal-view frames around poses 8, 12, 40: {got}  "
          f"{'ok' if good else 'WRONG, want ' + str(want)}")

    stored = pd.DataFrame({"frame": [1, 2], "time_s": [0.04, 0.08],
                           "track_id": [-1, -1], "cls": ["ball", "ball"],
                           "px": [10.0, 11.0], "py": [5.0, 5.0],
                           "crop_h": [8.0, 8.0],
                           "detection_method": ["yolo", "yolo"],
                           "confidence": [0.3, 0.3]})
    with tempfile.TemporaryDirectory() as tmp:
        same = with_fullres(stored, Path(tmp))
        good = same is stored
        ok &= good
        print(f"   no cache: tracks unchanged  {'ok' if good else 'WRONG'}")
        extra = stored.iloc[:1].assign(frame=3, px=50.0,
                                       detection_method="yolo_fullres")
        extra.to_parquet(Path(tmp) / FULLRES_FILE)
        merged = with_fullres(stored, Path(tmp))
        good = (len(merged) == 3 and
                list(merged.detection_method) == ["yolo", "yolo",
                                                  "yolo_fullres"])
        ok &= good
        print(f"   cache: {len(merged)} rows, full-res candidate appended  "
              f"{'ok' if good else 'WRONG'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
