"""Refusal paths of sweep_events_v2 (E2E-5, F1) on training windows only;
nothing measured is printed."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/user/Veo")
from detect_ball_heatmap import write_sidecar  # noqa: E402
from sweep_events_v2 import balls_meta, calibration_of  # noqa: E402

S = Path(__file__).resolve().parent
B = S.parent
W1, W2 = "st2_118575_1st_f015000", "st2_118576_1st_f030000"
CAL = B / "e2e/ckpt/smoke_v2.20.pt.calib.json"
cal = calibration_of(CAL)
side = {"path": str(CAL), "sha256": cal["sha256"], "a": cal["a"], "b": cal["b"],
        "tau": cal["tau"]}

rows = pd.read_parquet(f"/home/user/Veo/output_{W1}/events_v1/tracks.parquet")
rows = rows[rows.cls == "ball"]
pq1 = S / "balls_118575.parquet"
rows.to_parquet(pq1)
write_sidecar(pq1, sorted(rows.frame.unique()), window=W1, calib=side,
              **{"ckpt sha256": cal["ckpt sha256"]})
pq2 = S / "balls_118576_nosidecar.parquet"
rows.head(5).to_parquet(pq2)
Path(f"{pq2}.meta.json").unlink(missing_ok=True)
pq3 = S / "balls_118575_otherfit.parquet"
rows.to_parquet(pq3)
write_sidecar(pq3, sorted(rows.frame.unique()), window=W1,
              calib=side | {"a": side["a"] + 0.1}, **{"ckpt sha256": cal["ckpt sha256"]})

m = balls_meta(W1, pq1, cal, cal["tau"])
print("accepted:", m["ckpt sha256"][:12], m["calib"]["a"], m["calib"]["tau"])


def run(*argv):
    r = subprocess.run([sys.executable, "sweep_events_v2.py", *map(str, argv)],
                       cwd="/home/user/Veo", capture_output=True, text=True, timeout=300)
    msg = (r.stderr.strip().splitlines() or ["<none>"])[-1]
    print(f"exit {r.returncode}: {msg[:300]}")
    return r.returncode


common = ["--windows", W1, W2, "--allow-train", "--out", S / "never.json"]
assert run("--balls", f"{W1}={pq1}", f"{W2}={pq2}", "--calib", CAL, *common) == 1
assert run("--balls", f"{W1}={pq1}", f"{W2}={pq2}", "--tau", 0.2, *common) == 1
assert run("--balls", f"{W1}={pq3}", f"{W2}={pq2}", "--calib", CAL, *common) == 1
assert run("--balls", f"{W1}={pq1}", f"{W2}={pq1}", "--calib", CAL, *common) == 1
assert not (S / "never.json").exists()

# --apply: the smoke record has no tau; a crafted record (a copy with the
# smoke calibration, a altered) refuses the rows before anything is written.
smoke = B / "F/smoke_v3.json"
assert run("--apply", smoke, "--window", W1, "--balls", f"{W1}={pq1}",
           "--name", f"{S}/apply_{{w}}") == 1
rec = json.loads(smoke.read_text())
rec.update(tau=cal["tau"], calibration=cal | {"a": cal["a"] + 0.1})
crafted = S / "crafted_v3.json"
crafted.write_text(json.dumps(rec))
Path(f"{crafted}.sha256").write_text(
    hashlib.sha256(crafted.read_bytes()).hexdigest() + f"  {crafted.name}\n")
assert run("--apply", crafted, "--window", W1, "--balls", f"{W1}={pq1}",
           "--name", f"{S}/apply_{{w}}") == 1
assert not (S / f"apply_{W1}").exists()
print("real-data refusals: ok")
