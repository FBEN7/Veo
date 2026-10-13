"""Legacy-path regression of ball_tiles.py: HEAD vs working tree, with a fake
video (constant frames), same args; compares key.json, key_passes.json and
the sheets byte for byte. Prints only file names and equal / differ."""
import hashlib, runpy, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/user/Veo")
import cv2

class FakeCap:
    imgs = [np.random.default_rng(k).integers(0, 255, (1080, 4096, 3), dtype=np.uint8)
            for k in range(3)]
    def __init__(self, path):
        self.i = 0
    def read(self):
        self.i += 1
        return True, self.imgs[self.i % 3]
    def release(self):
        pass

cv2.VideoCapture = FakeCap
S = "/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball"
R = Path("/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/tiles_reg")
runs = {
    "pf": ["--scores", f"{S}/heatmap_T.json", f"{S}/coco_T2.json", "--pass-frame", "1",
           "--take", "ball_heatmap.3000.pt", "coco", "--dedupe-px", "6", "--decoys", "5",
           "--corridor-decoys", "4", "--passes-per-match", "3", "--seed", "3"],
    "rand": ["--scores", f"{S}/heatmap_T.json", f"{S}/coco_T2.json", "--take",
             "ball_heatmap.3000.pt=6", "coco=5", "--tau", "0.2", "--decoys", "4",
             "--seed", "1"],
}
which = sys.argv[1]
src = R / "old" / "ball_tiles.py" if which == "old" else Path("/home/user/Veo/ball_tiles.py")
for name, args in runs.items():
    out = R / which / name
    sys.argv = ["ball_tiles.py", *args, "--out", str(out)]
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(src), run_name="__main__")
    (R / f"{which}_{name}.stdout").write_text(buf.getvalue().replace(str(R / which), "OUT"))
