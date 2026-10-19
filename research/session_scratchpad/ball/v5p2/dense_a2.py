"""Protocol v5 part 2, amendment 2 (dense_v5p2.py with 250-frame chunks, two at a
time: worker restarts came before a 750-frame chunk's 19 minutes). Restart-proof dense v3 run (stride 1, no --allow-seen): detect_ball_heatmap.py --v2 over
--range chunks (each its own parquet + sidecar, skipped when already valid),
then the chunks concatenated into .cache/v3_balls_<w>.parquet with a merged
sidecar (range None; a split run gives the same rows, see _check)."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/user/Veo")
from detect_ball_heatmap import read_sidecar, write_sidecar  # noqa: E402

W = sys.argv[1]
STEP = 250
WORKERS = 2
root = Path("/home/user/Veo")
info = json.loads((root / "data/soccertrack_v2" / f"{W}.json").read_text())
n = int(info["frames"])
cdir = root / ".cache" / f"v3_chunks_{W}"
cdir.mkdir(exist_ok=True)
# One process per window: a live holder of the lock means skip (exit 0,
# nothing merged); a dead one's lock (worker restart) is taken over.
import os  # noqa: E402
lock = root / ".cache" / f"v3_chunks_{W}.lock"
try:
    lock.mkdir()
except FileExistsError:
    try:
        pid = int((lock / "pid").read_text())
        if b"dense_a2" not in Path(f"/proc/{pid}/cmdline").read_bytes():
            raise OSError("pid reused")
        print(f"{W}: locked by live pid {pid}; skipped", flush=True)
        sys.exit(0)
    except (OSError, ValueError):
        pass
(lock / "pid").write_text(str(os.getpid()))
from concurrent.futures import ThreadPoolExecutor  # noqa: E402


def chunk(lo):
    hi = min(lo + STEP, n)
    out = cdir / f"{lo:05d}_{hi:05d}.parquet"
    try:
        meta = read_sidecar(out)
    except ValueError:
        print(f"chunk {lo}-{hi}: running", flush=True)
        subprocess.run([sys.executable, "detect_ball_heatmap.py", "--v2", "--window", W,
                        "--weights", ".cache/ball_v3b.1500.pt", "--stride", "1",
                        "--threads", "2", "--range", str(lo), str(hi), "--out", str(out)],
                       cwd=root, check=True)
        meta = read_sidecar(out)
        print(f"chunk {lo}-{hi}: done", flush=True)
    else:
        print(f"chunk {lo}-{hi}: done already", flush=True)
    return out, meta


with ThreadPoolExecutor(WORKERS) as pool:
    done = list(pool.map(chunk, range(0, n, STEP)))
parts = [pd.read_parquet(out) for out, _ in done]
metas = [meta for _, meta in done]
keys = ("window", "match", "half", "start_frame", "stride", "extra_frames", "ckpt",
        "ckpt sha256", "rgb_only", "sigma", "calib", "tau", "top_k", "detection_method")
for m in metas[1:]:
    assert all(m[k] == metas[0][k] for k in keys), "chunks disagree"
df = pd.concat(parts, ignore_index=True)
final = root / ".cache" / f"v3_balls_{W}.parquet"
df.to_parquet(final)
frames = sorted(f for m in metas for f in m["frames"])
write_sidecar(final, frames, **{k: metas[0][k] for k in keys}, range=None,
              requested=sum(m["requested"] for m in metas),
              chunks=[{"range": m["range"], "parquet sha256": m["parquet sha256"]}
                      for m in metas])
print(f"merged {len(frames)} frames, {len(df)} rows -> {final}", flush=True)
import shutil  # noqa: E402
shutil.rmtree(lock, ignore_errors=True)
