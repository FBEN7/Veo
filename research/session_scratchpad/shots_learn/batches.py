"""Training clips of protocol.json, batch by batch (owner decision 2026-10-21:
cut here, not on Kaggle): cut the mp4s here (st.cut_clip, as fetch does),
upload a private dataset, push the batch to a free Kaggle kernel (two
alternate), delete the local mp4s once uploaded. Resumable: every step
writes its mark in batches/<k>.json and is skipped when marked. Results are
fetched later with `collect`.

    python batches.py run [--only K]     # one pass: cut / upload / push what is due
    python batches.py collect            # download finished batches into .cache/gpu
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src import soccertrack_v2 as st  # noqa: E402

HERE = Path(__file__).resolve().parent
PLAN = HERE / "clips_train.json"
STATE = HERE / "batches"
CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
STAGE = Path(os.environ.get("VEO_STAGE", "/tmp/veo_stage"))
SLUGS = ["veo-gpu-run", "veo-gpu-run2"]
SEEN = ["117093", "118575", "118576", "118578", "128058", "132877"]   # v3 trained on them
SIZE = 20


def kaggle(*args, check=True) -> str:
    env = {**os.environ, "KAGGLE_API_TOKEN": os.environ.get("KAGGLE_TOKEN", "")}
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, env=env)
    if check and r.returncode:
        raise SystemExit(f"kaggle {args[0]} {args[1]}: {r.stderr[-500:]}")
    return r.stdout


def batches() -> list[list[dict]]:
    clips = json.loads(PLAN.read_text())["clips"]
    return [clips[i:i + SIZE] for i in range(0, len(clips), SIZE)]   # in plan order


def state(k: int) -> dict:
    p = STATE / f"{k}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def mark(k: int, **kw):
    STATE.mkdir(exist_ok=True)
    s = {**state(k), **kw}
    (STATE / f"{k}.json").write_text(json.dumps(s, indent=1))


def cut(c: dict):
    mp4 = CLIPS / f"{c['name']}.mp4"
    info_path = CLIPS / f"{c['name']}.json"
    info = json.loads(info_path.read_text())
    if mp4.exists() and info.get("cut_done"):
        return
    got = st.cut_clip(c["match"], c["half"], c["cut_start"], c["cut_frames"], mp4, crf=18)
    info.update(frames=got["frames"], width=got["width"], height=got["height"],
                path=str(mp4), cut_done=True)
    info_path.write_text(json.dumps(info, indent=1))


def slug_free(slug: str, for_push: bool = False) -> bool:
    """Not running; for a push, also no batch on it still to collect (the
    API serves only a kernel's latest output)."""
    s = kaggle("kernels", "status", f"francoisbernard/{slug}", check=False)
    if any(x in s.lower() for x in ("running", "queued")):
        return False
    if for_push:
        for p in STATE.glob("*.json"):
            st_ = json.loads(p.read_text())
            if st_.get("slug") == slug and st_.get("pushed") and not st_.get("collected"):
                return False
    return True


def run(only: int | None):
    bs = batches()
    for k, b in enumerate(bs):
        if only is not None and k != only:
            continue
        s = state(k)
        if s.get("pushed"):
            continue
        if not s.get("uploaded"):
            t = time.time()
            for c in b:
                cut(c)
            mark(k, cut=True, cut_s=round(time.time() - t), clips=[c["name"] for c in b])
            d = STAGE / f"b{k}"
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
            for c in b:
                for suf in (".mp4", ".json", "_gt.parquet", "_events.csv"):
                    try:
                        os.link(CLIPS / f"{c['name']}{suf}", d / f"{c['name']}{suf}")
                    except OSError:
                        shutil.copy(CLIPS / f"{c['name']}{suf}", d)
            for m in {c["match"] for c in b}:
                for f in (ROOT / ".cache/soccertrack_v2/calib").glob(f"{m}_*"):
                    shutil.copy(f, d)
            ds = f"veo-st2-train-b{k}"
            (d / "dataset-metadata.json").write_text(json.dumps(
                {"title": ds, "id": f"francoisbernard/{ds}", "licenses": [{"name": "CC-BY-4.0"}]}))
            kaggle("datasets", "create", "-p", str(d), check=False)
            while kaggle("datasets", "status", f"francoisbernard/{ds}", check=False).strip() != "ready":
                time.sleep(30)
            mark(k, uploaded=True, dataset=f"francoisbernard/{ds}")
            shutil.rmtree(d)
            for c in b:
                (CLIPS / f"{c['name']}.mp4").unlink()
            print(f"batch {k}: {len(b)} clips cut and uploaded", flush=True)
        slug = next((x for x in SLUGS if slug_free(x, for_push=True)), None)
        if slug is None:
            print(f"batch {k}: uploaded, waiting for a free kernel", flush=True)
            return
        plan = STAGE / f"plan_b{k}.json"
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(json.dumps({"clips": b}))
        r = subprocess.run([sys.executable, str(ROOT / "research/session_scratchpad/kaggle/push_run.py"),
                            "--plan", str(plan), "--slug", slug, "--seen", *SEEN,
                            "--data", state(k)["dataset"]],
                           capture_output=True, text=True,
                           env={**os.environ, "KAGGLE_API_TOKEN": os.environ.get("KAGGLE_TOKEN", "")})
        if r.returncode:
            raise SystemExit(f"push batch {k}: {r.stderr[-500:]}")
        mark(k, pushed=True, slug=slug, pushed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        print(f"batch {k}: pushed to {slug}", flush=True)


def collect():
    gpu = ROOT / ".cache/gpu"
    for p in sorted(STATE.glob("*.json"), key=lambda p: int(p.stem)):
        k, s = int(p.stem), json.loads(p.read_text())
        if not s.get("pushed") or s.get("collected") or not slug_free(s["slug"]):
            continue
        out = STAGE / f"out_b{k}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        kaggle("kernels", "output", f"francoisbernard/{s['slug']}", "-p", str(out), check=False)
        ok = 0
        for w in s["clips"]:
            tgz, rec = out / f"{w}.tar.gz", out / f"{w}.json"
            if tgz.exists() and rec.exists() and json.loads(rec.read_text()).get("ok"):
                (gpu / w).mkdir(parents=True, exist_ok=True)
                with tarfile.open(tgz) as t:
                    t.extractall(gpu / w)
                ok += 1
        mark(k, collected=True, ok=ok, n=len(s["clips"]))
        print(f"batch {k}: {ok}/{len(s['clips'])} collected", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "collect"))
    ap.add_argument("--only", type=int)
    a = ap.parse_args()
    run(a.only) if a.cmd == "run" else collect()
