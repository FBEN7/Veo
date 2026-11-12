"""shots negatives (plan.json, hashed first): draw the ordinary-play windows,
cut them here, upload in batches of 10, run them on Kaggle, collect.
Resumable: each batch's state in state/<k>.json.

    python drive.py draw
    VEO_UNSEAL=... python drive.py run        # one pass: cut / upload / push what is due
    python drive.py collect                    # download finished batches
    python drive.py status
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/goals_v2"))

WINDOWS = HERE / "windows.json"
STATE = HERE / "state"
DATA = ROOT / "data/negatives/soccertrack_v2"
STAGE = Path(os.environ.get("VEO_STAGE", "/tmp/claude-0/veo_stage_neg"))
OUT = ROOT / ".cache/gpu_windows"
SLUGS = ["veo-gpu-run", "veo-gpu-run2"]
SEEN = {"117093", "118575", "118576", "118578", "128058", "132877"}   # v3 trained on them
SEALED = {"128057", "132831"}
SIZE = 10
DS_PREFIX = "veo-st2-neg-b"
LEN = 4500
GAP = 750
DEV_WINDOWS = ["st2_117093_1st_f015000", "st2_118575_1st_f015000", "st2_118576_1st_f030000",
               "st2_118578_1st_f030000", "st2_118578_2nd_f015000", "st2_128058_1st_f030000",
               "st2_132877_1st_f015000"]


def check_plan():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    if hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest() != want:
        raise SystemExit("plan.json does not match its sha256")


def kaggle(*args, check=True) -> str:
    env = {**os.environ, "KAGGLE_API_TOKEN": os.environ.get("KAGGLE_TOKEN", "")}
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, env=env)
    if check and r.returncode:
        raise SystemExit(f"kaggle {args[0]} {args[1]}: {r.stderr[-500:]}")
    return r.stdout


def busy_ranges() -> dict:
    """(match, half) -> [(a, b)] half-frame ranges to keep clear of."""
    out: dict = {}
    for f in ("shots_learn/clips_train.json", "shots_learn/clips_test.json", "kaggle/shot_clips_dev.json"):
        for c in json.loads((ROOT / "research/session_scratchpad" / f).read_text())["clips"]:
            out.setdefault((c["match"], int(c["half"])), []).append((c["cut_start"], c["cut_start"] + c["cut_frames"]))
    for w in DEV_WINDOWS:
        _, m, h, f = w.split("_")
        a = int(f[1:])
        out.setdefault((m, 1 if h == "1st" else 2), []).append((a, a + LEN))
    return out


def _valid(n, taken):
    a = np.arange(3000, n - 6000 + 1)
    ok = np.ones(len(a), bool)
    for lo, hi in taken:
        ok &= (a + LEN + GAP <= lo) | (a >= hi + GAP)
    return a[ok]


def draw():
    """plan.json + addendum1.json: uniform over the valid starts; shortfalls
    drawn after all halves, from the other half of the match, then any."""
    check_plan()
    for name in ("addendum1.json",):
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != (HERE / f"{name}.sha256").read_text().split()[0]:
            raise SystemExit(f"{name} does not match its sha256")
    import explore as E
    rng = np.random.default_rng(20261112)
    busy = busy_ranges()
    halves, taken, n_of = [], {}, {}
    for m in E.MATCHES:
        for half in (1, 2):
            scan = pd.read_parquet(ROOT / f".cache/formation_scan/{m}_{'1st' if half == 1 else '2nd'}.parquet",
                                   columns=["frame"])
            n_of[(m, half)] = int(scan.frame.max()) + 25
            taken[(m, half)] = list(busy.get((m, half), []))
            halves.append((m, half))
    wins, short = [], []

    def one(key):
        v = _valid(n_of[key], taken[key])
        if not len(v):
            return False
        a = int(rng.choice(v))
        taken[key].append((a, a + LEN))
        m, half = key
        wins.append({"name": f"st2_{m}_{'1st' if half == 1 else '2nd'}_f{a:06d}", "match": m, "half": half,
                     "start": a, "frames": LEN, "half_frames_est": n_of[key]})
        return True

    for key in halves:
        for _ in range(2):
            if not one(key):
                short.append(key)
    moved = []
    for key in short:
        other = (key[0], 3 - key[1])
        order = [other] + [h for h in halves if h != other]
        for k2 in order:
            if one(k2):
                moved.append({"from": list(key), "to": list(k2)})
                break
    WINDOWS.write_text(json.dumps({"seed": 20261112, "shortfalls moved": moved, "windows": wins}, indent=1))
    print(len(wins), "windows drawn; moved", moved)


def batches():
    w = json.loads(WINDOWS.read_text())["windows"]
    return [w[i:i + SIZE] for i in range(0, len(w), SIZE)]


def state(k):
    p = STATE / f"{k}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def mark(k, **kw):
    STATE.mkdir(exist_ok=True)
    (STATE / f"{k}.json").write_text(json.dumps({**state(k), **kw}, indent=1))


def cut(w):
    from fetch_soccertrack_v2 import fetch
    if (DATA / f"{w['name']}.mp4").exists() and (DATA / f"{w['name']}.json").exists():
        return
    if w["match"] in SEALED:
        from src.unseal import unsealed
        if not unsealed(w["match"], "shots negatives: cut an ordinary-play window (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    fetch(w["match"], w["half"], w["start"] / 1500, LEN / 1500, DATA)


def slug_free(slug, for_push=False):
    s = kaggle("kernels", "status", f"francoisbernard/{slug}", check=False).lower()
    if "running" in s or "queued" in s:
        return False
    if for_push:
        for p in STATE.glob("*.json"):
            x = json.loads(p.read_text())
            if x.get("slug") == slug and x.get("pushed") and not x.get("collected"):
                return False
    return True


def run(only=None):
    check_plan()
    for k, b in enumerate(batches()):
        if only is not None and k != only:
            continue
        s = state(k)
        if s.get("pushed"):
            continue
        if not s.get("uploaded"):
            t = time.time()
            for w in b:
                cut(w)
            mark(k, cut=True, cut_s=round(time.time() - t), windows=[w["name"] for w in b])
            d = STAGE / f"b{k}"
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
            for w in b:
                for suf in (".mp4", ".json", "_gt.parquet", "_events.csv"):
                    try:
                        os.link(DATA / f"{w['name']}{suf}", d / f"{w['name']}{suf}")
                    except OSError:
                        shutil.copy(DATA / f"{w['name']}{suf}", d)
            for m in {w["match"] for w in b}:
                for f in (ROOT / ".cache/soccertrack_v2/calib").glob(f"{m}_*"):
                    shutil.copy(f, d)
            ds = f"{DS_PREFIX}{k}"
            (d / "dataset-metadata.json").write_text(json.dumps(
                {"title": ds, "id": f"francoisbernard/{ds}", "licenses": [{"name": "CC-BY-4.0"}]}))
            kaggle("datasets", "create", "-p", str(d), check=False)
            for _ in range(240):
                if (kaggle("datasets", "status", f"francoisbernard/{ds}", check=False).strip().splitlines() or [""])[-1] == "ready":
                    break
                time.sleep(30)
            else:
                raise SystemExit(f"dataset {ds} not ready")
            mark(k, uploaded=True, dataset=f"francoisbernard/{ds}")
            shutil.rmtree(d)
            for w in b:
                (DATA / f"{w['name']}.mp4").unlink()
            print(f"batch {k}: {len(b)} windows cut and uploaded", flush=True)
        slug = next((x for x in SLUGS if slug_free(x, for_push=True)), None)
        if slug is None:
            print(f"batch {k}: uploaded, waiting for a free kernel", flush=True)
            return
        ws = [f"{w['name']}{':seen' if w['match'] in SEEN else ''}" for w in b]
        r = subprocess.run([sys.executable, str(ROOT / "research/session_scratchpad/kaggle/push_run.py"),
                            "--slug", slug, *ws, "--data", state(k)["dataset"]],
                           capture_output=True, text=True,
                           env={**os.environ, "KAGGLE_API_TOKEN": os.environ.get("KAGGLE_TOKEN", "")})
        if r.returncode:
            raise SystemExit(f"push batch {k}: {r.stderr[-500:]}")
        mark(k, pushed=True, slug=slug, pushed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        print(f"batch {k}: pushed to {slug}", flush=True)


def collect():
    for p in sorted(STATE.glob("*.json"), key=lambda p: int(p.stem)):
        k, s = int(p.stem), json.loads(p.read_text())
        if not s.get("pushed") or s.get("collected") or not slug_free(s["slug"]):
            continue
        out = STAGE / f"out_b{k}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        kaggle("kernels", "output", f"francoisbernard/{s['slug']}", "-p", str(out), check=False)
        ok = []
        for w in s["windows"]:
            tgz, rec = out / f"{w}.tar.gz", out / f"{w}.json"
            if tgz.exists() and rec.exists() and json.loads(rec.read_text()).get("ok"):
                (OUT / w).mkdir(parents=True, exist_ok=True)
                with tarfile.open(tgz) as t:
                    t.extractall(OUT / w)
                ok.append(w)
        mark(k, collected=True, ok=ok, n=len(s["windows"]))
        shutil.rmtree(out, ignore_errors=True)
        print(f"batch {k}: {len(ok)}/{len(s['windows'])} collected", flush=True)


def status():
    for k, b in enumerate(batches()):
        s = state(k)
        print(k, {x: s.get(x) for x in ("cut", "uploaded", "pushed", "slug", "collected")},
              f"ok {len(s.get('ok', []))}/{len(b)}" if s.get("collected") else "")


def cut_only(k):
    for w in batches()[k]:
        cut(w)
        print("cut", w["name"], flush=True)


if __name__ == "__main__":
    cmd, arg = sys.argv[1], (int(sys.argv[2]) if len(sys.argv) > 2 else None)
    if cmd == "run":
        run(arg)
    elif cmd == "cut":
        cut_only(arg)
    else:
        {"draw": draw, "collect": collect, "status": status}[cmd]()
