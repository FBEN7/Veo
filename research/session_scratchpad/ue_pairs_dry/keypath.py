"""ball_tiles --pairs main on the U-E sealed file with a fake video (constant
noise frames): checks the written key / key_passes structurally with
score_ball_track's checks. Prints only counts and ok."""
import json, runpy, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/user/Veo")
import cv2

class FakeCap:
    img = np.random.default_rng(0).integers(0, 255, (1080, 4096, 3), dtype=np.uint8)
    def __init__(self, path): pass
    def read(self): return True, self.img
    def release(self): pass

cv2.VideoCapture = FakeCap
X = Path(sys.argv[1])
sys.argv = ["ball_tiles.py", "--scores", str(X / "S_UE.json"), "--pairs", "T", "ball_v2.4000.pt",
            "--extra", "coco@q1", "--decoys", "10", "--corridor-decoys", "10",
            "--offset-decoys", "6", "--seed", "4", "--out", str(X / "tiles")]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path("/home/user/Veo/ball_tiles.py", run_name="__main__")
import score_ball_track as sb
key = json.loads((X / "tiles_key" / "key.json").read_text())
passes = json.loads((X / "tiles_key" / "key_passes.json").read_text())
assert not (X / "tiles" / "key.json").exists()
by_tile = {k["tile"]: k for k in key}
T, R, extra = sb.sitting_of(passes)
sb.check_tiles(passes, by_tile)
fake = {t: "N" for t in by_tile}
rows = sb.frame_rows(passes, {t: False for t in by_tile}, T, R)
for q in [1] + list(extra.values()):
    sb.mcnemar_at(rows, T, R if q == 1 else list(extra)[0], q, len(passes))
pair = [k for k in key if k["kind"] == "pair"]
print("tiles", len(key), "sheets", len(list((X / "tiles").glob("sheet_*.jpg"))),
      "kinds", dict(Counter(k["kind"] for k in key)))
print("pair tiles: px/m None", sum(k.get("px/m") is None for k in pair),
      "track source None", sum(k.get("track source") is None for k in pair),
      "parity", dict(Counter(k["parity"] for k in pair)),
      "keys", sorted({kk for k in pair for kk in k}))
fr = [f for p in passes for f in p["pair frames"]]
print("pair frames: px/m None", sum(f["px/m"] is None for f in fr),
      "why", dict(Counter(f["why"] for f in fr)))
# Offset decoys: 18-20 px from base, >= 13 px from every candidate at that frame.
S = json.loads((X / "S_UE.json").read_text())
res = {r["window"]: r for r in S["results"]}
ok = 0
for k in key:
    if k["kind"] != "offset-decoy":
        continue
    d = np.hypot(k["u"] - k["base"][0], k["v"] - k["base"][1])
    rows_ = [row for v in res[k["window"]].values() if isinstance(v, dict) and "candidates" in v
             for row in v["candidates"].get(str(k["frame"])) or []]
    clear = min(np.hypot(r[0] - k["u"], r[1] - k["v"]) for r in rows_)
    ok += 18 <= d <= 20 and clear >= 13
print("offset decoys geometry ok", ok, "of", sum(k["kind"] == "offset-decoy" for k in key))
print("structure ok")
