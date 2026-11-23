"""possession on a whole match (plan.json, hashed first): tile 128057's two
halves in 3-min windows, cut them here, upload in batches of 8, run them on
Kaggle (run_windows.py unchanged), collect. Resumable: state/<k>.json.

    VEO_UNSEAL=... python drive.py run      # one pass: cut / upload / push what is due
    python drive.py collect                  # download finished batches
    python drive.py status
    VEO_UNSEAL=... python drive.py loop      # run + collect every 5 min until all collected (or 110 min)
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

MATCH = "128057"
HALF_FRAMES = {1: 67625, 2: 70350}
LEN, MIN_REST, SIZE = 4500, 1500, 8
STATE = HERE / "state"
DATA = ROOT / "data/possession_match/soccertrack_v2"
STAGE = Path("/tmp/claude-0/-home-user-veo/b69e5b60-e0fe-5920-8f15-285768d7ea57/scratchpad/stage_pm")
OUT = ROOT / ".cache/gpu_match"
SLUGS = ["veo-gpu-run", "veo-gpu-run2"]
DS_PREFIX = "veo-st2-match-b"
REASON = "possession on a whole match: cut 128057 windows (development use, seal lifted 2026-11-05)"


def check_plan():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    if hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest() != want:
        raise SystemExit("plan.json does not match its sha256")


def windows():
    out = []
    for half, n in HALF_FRAMES.items():
        starts = list(range(0, n, LEN))
        spans = [[a, min(a + LEN, n)] for a in starts]
        if spans[-1][1] - spans[-1][0] < MIN_REST and len(spans) > 1:
            last = spans.pop()
            spans[-1][1] = last[1]
        for a, b in spans:
            out.append({"name": f"st2_{MATCH}_{'1st' if half == 1 else '2nd'}_f{a:06d}", "half": half, "start": a, "frames": b - a})
    return out


def batches():
    w = windows()
    return [w[i:i + SIZE] for i in range(0, len(w), SIZE)]


def kaggle(*args, check=True) -> str:
    env = {**os.environ, "KAGGLE_API_TOKEN": os.environ.get("KAGGLE_TOKEN", "")}
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, env=env)
    if check and r.returncode:
        raise SystemExit(f"kaggle {args[0]} {args[1]}: {r.stderr[-500:]}")
    return r.stdout


def state(k):
    p = STATE / f"{k}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def mark(k, **kw):
    STATE.mkdir(exist_ok=True)
    (STATE / f"{k}.json").write_text(json.dumps({**state(k), **kw}, indent=1))


def cut(w):
    from fetch_soccertrack_v2 import fetch
    from src.unseal import unsealed
    if (DATA / f"{w['name']}.mp4").exists() and (DATA / f"{w['name']}.json").exists():
        return
    if not unsealed(MATCH, REASON):
        raise SystemExit("VEO_UNSEAL needed")
    fetch(MATCH, w["half"], w["start"] / 1500, w["frames"] / 1500, DATA)


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


def push(k, names, datasets, slug):
    r = subprocess.run([sys.executable, str(ROOT / "research/session_scratchpad/kaggle/push_run.py"),
                        "--slug", slug, *names, "--opened", *names, "--data", *datasets],
                       capture_output=True, text=True,
                       env={**os.environ, "KAGGLE_API_TOKEN": os.environ.get("KAGGLE_TOKEN", "")})
    if r.returncode:
        raise SystemExit(f"push batch {k}: {r.stderr[-500:]}")


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
                for attempt in range(5):   # the proxy resets long transfers now and then: retry with backoff
                    try:
                        cut(w)
                        break
                    except Exception as e:  # noqa: BLE001
                        if attempt == 4:
                            raise
                        print(f"cut {w['name']}: {type(e).__name__}, retry {attempt + 1}", flush=True)
                        time.sleep(30 * 2 ** attempt)
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
            for f in (ROOT / ".cache/soccertrack_v2/calib").glob(f"{MATCH}_*"):
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
        push(k, [w["name"] for w in b], [state(k)["dataset"]], slug)
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
        (HERE / "logs").mkdir(exist_ok=True)
        for w in s["windows"]:
            tgz, rec = out / f"{w}.tar.gz", out / f"{w}.json"
            if tgz.exists() and rec.exists() and json.loads(rec.read_text()).get("ok"):
                (OUT / w).mkdir(parents=True, exist_ok=True)
                with tarfile.open(tgz) as t:
                    t.extractall(OUT / w)
                ok.append(w)
            else:
                for f in out.glob(f"{w}.*"):
                    if not f.name.endswith(".tar.gz"):
                        shutil.copy(f, HERE / "logs" / f"b{k}_{f.name}")
        mark(k, collected=True, ok=ok, n=len(s["windows"]))
        shutil.rmtree(out, ignore_errors=True)
        print(f"batch {k}: {len(ok)}/{len(s['windows'])} collected", flush=True)


RERUN = 99   # state/99.json: the one rerun the plan allows


def rerun(slug):
    if state(RERUN).get("pushed"):
        raise SystemExit("the rerun was already pushed")
    ok, ds = set(), []
    for k in range(len(batches())):
        ok |= set(state(k).get("ok", []))
        ds.append(state(k)["dataset"])
    names = [w["name"] for w in windows() if w["name"] not in ok]
    push(RERUN, names, ds, slug)
    mark(RERUN, pushed=True, slug=slug, windows=names, datasets=ds, pushed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(f"rerun: {len(names)} windows pushed to {slug}", flush=True)


def status():
    for k, b in enumerate(batches()):
        s = state(k)
        print(k, {x: s.get(x) for x in ("cut", "uploaded", "pushed", "slug", "collected")},
              f"ok {len(s.get('ok', []))}/{len(b)}" if s.get("collected") else "")
    if state(RERUN):
        print("rerun", state(RERUN).get("slug"), state(RERUN).get("collected"), len(state(RERUN).get("ok", [])))


def loop(limit_s=110 * 60):
    t0 = time.time()
    while time.time() - t0 < limit_s:
        collect()
        run()
        if all(state(k).get("collected") for k in range(len(batches()))):
            print("all batches collected", flush=True)
            return
        time.sleep(300)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "rerun":
        rerun(sys.argv[2])
    elif cmd == "windows":
        print(json.dumps(windows(), indent=0))
    else:
        {"collect": collect, "status": status, "loop": loop}[cmd]()
