"""detect_ball_heatmap.py --v2 with ball_v3b.1500.pt at stride 1 over a whole
window, the model on cuda (the GPU picked by CUDA_VISIBLE_DEVICES) through a
wrapped `ball_heatmap.load_v2`; repo code unchanged, the same rows as the CPU
(compare_st2_118578_2nd_f015000.json). Run from the repo root.

    python ball_gpu.py W [--allow-seen] [--threads N]

A clip of a sealed match (M7 / M9) runs only when opened.json beside this
script lists it: the clip was cut on the repo machine through src/unseal.py
(that opening logged there; VEO_UNSEAL never comes here, owner decision
2026-10-20), so detect_ball_heatmap's test-match guard is lifted for that
clip alone. Any other window of a sealed match is still refused.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
import detect_ball_heatmap as dbh  # noqa: E402
from src import ball_heatmap as bh  # noqa: E402

w = sys.argv[1]
threads = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv else "4"
allow_seen = "--allow-seen" in sys.argv
opened = Path(__file__).with_name("opened.json")
if opened.exists() and w in json.loads(opened.read_text())["clips"]:
    dbh.TEST_MATCHES = set()
orig = bh.load_v2


def load_on(ckpt):
    model, info = orig(ckpt)
    model.to("cuda")
    fwd = model.forward
    model.forward = lambda x: fwd(x.to("cuda", non_blocking=True)).float().cpu()
    return model, info


bh.load_v2 = load_on
torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
sys.argv = ["detect_ball_heatmap.py", "--v2", "--window", w,
            "--weights", ".cache/ball_v3b.1500.pt", "--stride", "1",
            "--threads", threads, "--out", f".cache/v3_balls_{w}.parquet"]
if allow_seen:
    sys.argv.append("--allow-seen")
dbh.main()
