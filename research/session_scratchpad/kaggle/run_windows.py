"""The window pipeline on the Kaggle GPUs (backend_note.json): for each window,
the same steps as research/session_scratchpad/ball/events_m1/driver.sh, minus
the CPU-only ones that run back here from the returned files (marks.py, v2):

1. players: eval_soccertrack_v2.py --run --calibrated --model yolov8s.pt
   --stride 2 --pitch-margin 50 -> output_<w>/ (ultralytics picks the GPU;
   fp32). --calibrated skips the ground-plane fit (365 of 554 s on a clip),
   whose result nothing downstream reads: step 2 takes only tracks.parquet,
   profile.json and clip.json from this run (events_run.py);
2. teams and events_v1: research/session_scratchpad/events_run.py <w> events_v1;
3. ball: ball_gpu.py, detect_ball_heatmap.py --v2 with ball_v3b.1500.pt at
   stride 1 over the whole window -> .cache/v3_balls_<w>.parquet (+ sidecar).

The ball does not read the players, so the two run side by side, each on its
own GPU of the T4 x2 machine: a players lane (1 then 2, window after window)
on GPU 0 and a ball lane on GPU 1, each with half of the 4 CPU threads. The
steps and their arguments are those of the one-GPU run that matched the CPU
(compare_st2_118578_2nd_f015000.json); only the scheduling changes.

Each window's files go to /kaggle/working/<w>.tar.gz once both lanes are done
with it, with timings in <w>.json. A failed step is logged, what finished is
packed, and the lanes go on. A window written W:seen runs the ball step with
--allow-seen (windows whose CPU v3 run had it).

    python run_windows.py --root /home/user/Veo W1 W2:seen ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
THREADS = "2"


def lane_env(root: Path, gpu: int) -> dict:
    return {**os.environ, "PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": THREADS, "MKL_NUM_THREADS": THREADS,
            "OPENBLAS_NUM_THREADS": THREADS}


def step(rec: dict, name: str, cmd: list, root: Path, env: dict, log: Path):
    t = time.perf_counter()
    with open(log, "a") as f:     # each line stamped with seconds since the step began
        proc = subprocess.Popen([sys.executable, "-u", *cmd], cwd=root, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            f.write(f"[{time.perf_counter() - t:7.1f}] {line}")
        r = proc.wait()
        r = subprocess.CompletedProcess(cmd, r)
    rec["steps"][name] = round(time.perf_counter() - t, 1)
    if r.returncode:
        rec["failed"].append(name)
    print(f"{rec['window']}: {name} {rec['steps'][name]} s"
          f"{' FAILED (see ' + log.name + ')' if r.returncode else ''}", flush=True)
    return r.returncode == 0


def players_lane(root: Path, specs: list, recs: dict, out: Path):
    env = lane_env(root, 0)
    for w, _, _ in specs:
        rec, log = recs[w], out / f"{w}.players.log"
        if wait_ready(w, rec) and step(rec, "players", ["eval_soccertrack_v2.py", f"data/soccertrack_v2/{w}.mp4",
                                 "--out", f"output_{w}", "--run", "--calibrated",
                                 "--model", "yolov8s.pt", "--stride", "2",
                                 "--pitch-margin", "50"], root, env, log):
            step(rec, "events_v1", ["research/session_scratchpad/events_run.py", w, "events_v1"],
                 root, env, log)
        rec["done"].add("players")
        finish(root, w, rec, out)


def ball_lane(root: Path, specs: list, recs: dict, out: Path):
    env = lane_env(root, 1)
    for w, seen, rng in specs:
        if not wait_ready(w, recs[w]):
            recs[w]["done"].add("ball")
            finish(root, w, recs[w], out)
            continue
        cmd = [str(HERE / "ball_gpu.py"), w, "--threads", THREADS] + (["--allow-seen"] if seen else [])
        if rng:
            cmd += ["--range", str(rng[0]), str(rng[1])]
        step(recs[w], "ball_v3", cmd, root, env, out / f"{w}.ball.log")
        recs[w]["done"].add("ball")
        finish(root, w, recs[w], out)


LOCK = threading.Lock()
READY: dict = {}          # w -> threading.Event, set once its mp4 is there (--cut)


def wait_ready(w: str, rec: dict) -> bool:
    ev = READY.get(w)
    if ev is None:
        return True
    ev.wait()
    return "cut" not in rec["failed"]


def cutter(root: Path, specs: list, recs: dict):
    """--cut: the units' mp4 from the gated release by range requests (the
    same st.cut_clip as fetch_soccertrack_v2.py; HF_TOKEN from the kernel's
    secret), one after the other ahead of the lanes; the unit's .json gets
    the frames, width and height actually written."""
    sys.path.insert(0, str(root))
    from src import soccertrack_v2 as st

    for w, _, _ in specs:
        rec = recs[w]
        info_path = root / "data/soccertrack_v2" / f"{w}.json"
        t = time.perf_counter()
        try:
            info = json.loads(info_path.read_text())
            mp4 = root / "data/soccertrack_v2" / f"{w}.mp4"
            got = st.cut_clip(str(info["match"]), int(info["half"]), int(info["cut"]["start"]),
                              int(info["cut"]["count"]), mp4, crf=int(info["cut"]["crf"]))
            info.update(frames=got["frames"], width=got["width"], height=got["height"],
                        path=str(mp4))
            if info_path.is_symlink():
                info_path.unlink()
            info_path.write_text(json.dumps(info, indent=1))
        except Exception as e:                 # the unit is skipped by both lanes
            rec["failed"].append("cut")
            rec["cut_error"] = f"{type(e).__name__}: {e}"[:500]
        rec["steps"]["cut"] = round(time.perf_counter() - t, 1)
        print(f"{w}: cut {rec['steps']['cut']} s{' FAILED' if 'cut' in rec['failed'] else ''}",
              flush=True)
        READY[w].set()


def finish(root: Path, w: str, rec: dict, out: Path):
    """Pack the window once both lanes are done with it."""
    with LOCK:
        if rec["done"] != {"players", "ball"}:
            return
        with tarfile.open(out / f"{w}.tar.gz", "w:gz") as t:
            if (root / f"output_{w}").exists():
                t.add(root / f"output_{w}", arcname=f"output_{w}")
            for f in (root / ".cache").glob(f"v3_balls_{w}.parquet*"):
                t.add(f, arcname=f".cache/{f.name}")
        rec["ok"] = not rec["failed"]
        (out / f"{w}.json").write_text(json.dumps({**rec, "done": sorted(rec["done"])}, indent=1))
        if w in READY:                         # a cut clip: free the disk
            (root / "data/soccertrack_v2" / f"{w}.mp4").unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cut", action="store_true",
                    help="cut each unit's mp4 here first (its .json carries 'cut'); needs HF_TOKEN")
    ap.add_argument("windows", nargs="+",
                    help="W[:seen][@A-B]: :seen = --allow-seen on the ball, @A-B = the ball on local frames [A, B) only")
    args = ap.parse_args()
    import torch

    root, out = Path(args.root), Path.cwd()
    specs = []
    for s in args.windows:
        s, _, rng = s.partition("@")
        specs.append((s.split(":")[0], s.endswith(":seen"),
                      tuple(int(x) for x in rng.split("-")) if rng else None))
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(gpus) < 2:
        raise SystemExit(f"two GPUs expected (T4 x2), found {gpus}")
    recs = {w: {"window": w, "gpus": gpus, "torch": torch.__version__, "allow_seen": seen,
                "ball_range": rng, "steps": {}, "failed": [], "done": set()}
            for w, seen, rng in specs}
    t0 = time.perf_counter()
    funcs = [players_lane, ball_lane]
    if args.cut:
        READY.update({w: threading.Event() for w, _, _ in specs})
        threading.Thread(target=cutter, args=(root, specs, recs), daemon=True).start()
    lanes = [threading.Thread(target=f, args=(root, specs, recs, out)) for f in funcs]
    for t in lanes:
        t.start()
    for t in lanes:
        t.join()
    print(f"all lanes done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
