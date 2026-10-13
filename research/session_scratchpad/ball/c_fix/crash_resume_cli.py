"""Fix 2 on real data, through the CLI: a clean --window run on the training
window st2_118575_1st_f015000, and the same run killed (SIGKILL) once its
partial parquet holds rows, then run again: same strips.parquet, same JPEG
bytes, no leftovers. Also: a stream-source run (simulated by a strips.parquet
of stream rows) refuses the window run into the same directory."""
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path("/home/user/Veo")
D = Path(__file__).resolve().parent
W = "st2_118575_1st_f015000"
ARGS = ["--window", W, "--frames", "15000", "15800", "--flush-every", "3"]
clean, crash = D / "win_clean", D / "win_crash"
for d in (clean, crash):
    shutil.rmtree(d, ignore_errors=True)


def run(out, **kw):
    return subprocess.run([sys.executable, "build_ball_strips.py", *ARGS, "--out", str(out)],
                          cwd=REPO, capture_output=True, text=True, **kw)


print("== clean")
r = run(clean)
print(r.stdout.strip(), r.stderr.strip()[-500:])
assert r.returncode == 0

print("== crash")
part = crash / f"strips_118575_1_{W}_15000-15800.partial.parquet"
p = subprocess.Popen([sys.executable, "build_ball_strips.py", *ARGS, "--out", str(crash)],
                     cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
n = 0
t0 = time.time()
while p.poll() is None and time.time() - t0 < 300:
    if part.exists():
        try:
            n = len(pd.read_parquet(part))
        except Exception:
            n = 0
        if n >= 6:
            p.send_signal(signal.SIGKILL)
            break
    time.sleep(0.05)
p.wait()
print(f"killed (return code {p.returncode}) with {n} rows saved; output: "
      f"{p.stdout.read().strip()}")
assert p.returncode == -signal.SIGKILL and n >= 6
jpgs = sorted(x.name for x in crash.glob("*.jpg"))
print(f"after the kill: {len(jpgs)} JPEGs, strips.parquet exists: "
      f"{(crash / 'strips.parquet').exists()}, partial rows {len(pd.read_parquet(part))}")
assert not (crash / "strips.parquet").exists()

print("== resume")
r = run(crash)
print(r.stdout.strip(), r.stderr.strip()[-500:])
assert r.returncode == 0 and "resuming" in r.stdout

print("== compare")
ta, tb = pd.read_parquet(clean / "strips.parquet"), pd.read_parquet(crash / "strips.parquet")
pd.testing.assert_frame_equal(ta, tb)
fa = sorted(x.name for x in clean.glob("*.jpg"))
fb = sorted(x.name for x in crash.glob("*.jpg"))
assert fa == fb == sorted(ta.file)
same = all((clean / f).read_bytes() == (crash / f).read_bytes() for f in fa)
left = sorted(x.name for x in crash.iterdir() if "partial" in x.name or x.suffix == ".tmp")
print(f"strips.parquet equal ({len(ta)} rows, kinds {ta.kind.value_counts().to_dict()}), "
      f"{len(fa)} JPEGs byte-identical: {same}; leftovers: {left}")
assert same and not left

print("== one source per directory")
mixed = D / "win_mixed"
shutil.rmtree(mixed, ignore_errors=True)
mixed.mkdir()
ta.head(3).assign(source="stream").to_parquet(mixed / "strips.parquet")
r = run(mixed)
print(f"window run into a dir of stream rows: return code {r.returncode}; "
      f"{(r.stderr.strip().splitlines() or [''])[-1]}")
assert r.returncode != 0 and "another directory" in r.stderr
assert not list(mixed.glob("*.jpg"))
shutil.rmtree(mixed)
print("OK")
