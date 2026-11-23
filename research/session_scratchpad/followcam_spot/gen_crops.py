"""follow-cam pilot (plan.json, hashed first): one half's follow-cam crops.

Downloads the half's panorama (one at a time, deleted after), decodes every
2nd frame, crops 0.35 x the width around the truth players' smoothed median
position, scales to 796 x 448 and encodes H.264 at 12.5 fps. Writes, under
OUT/<match>_<half>/: crops.mp4, centres.npz (crop box per kept frame) and
Labels-ball.json (T-DEED's SoccerNetBall format). Resumable: a half whose
done.json exists is skipped; a partial download is resumed.

    python gen_crops.py 118575 1 [--seconds 60]
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src import soccertrack_v2 as st  # noqa: E402

FPS, STRIDE, CROP_FRAC, OW, OH = 25, 2, 0.35, 796, 448
SCRATCH = Path("/tmp/claude-0/-home-user-veo/b69e5b60-e0fe-5920-8f15-285768d7ea57/scratchpad")
OUT = SCRATCH / "followcam_crops"
DL = SCRATCH / "panorama_dl"


def check():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    if hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest() != want:
        raise SystemExit("plan.json does not match its sha256")


def centres(match, half):
    """per frame (25 fps), the camera centre in panorama pixels."""
    g = st.gsr_table(match, half)
    g = g[g.role.isin(["player", "goalkeeper"]) & g.bx.notna()]
    c = g.assign(cx=g.bx + g.bw / 2, cy=g.by + g.bh / 2).groupby("frame")[["cx", "cy"]].median()
    n = int(g.frame.max()) + 1
    c = c.reindex(np.arange(n)).interpolate(limit_direction="both")
    c = c.rolling(2 * FPS, center=True, min_periods=1).median().rolling(FPS, center=True, min_periods=1).mean()
    return c.to_numpy(), n


def download(match, half):
    """the half's panorama, resumed with range requests if a partial file is there."""
    import requests
    DL.mkdir(parents=True, exist_ok=True)
    path = DL / f"{match}_{half}.mp4"
    done = DL / f"{match}_{half}.ok"
    if done.exists():
        return path
    for attempt in range(8):
        try:
            have = path.stat().st_size if path.exists() else 0
            u = st.signed_url(st.video_path(match, half))
            r = requests.get(u, headers={"Range": f"bytes={have}-"} if have else {}, stream=True, timeout=120)
            if r.status_code == 416:
                break
            r.raise_for_status()
            if have and r.status_code != 206:
                have = 0
            with open(path, "ab" if have else "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
            break
        except Exception as e:  # noqa: BLE001  (the proxy resets long transfers now and then)
            print(f"download {match} {half}: {type(e).__name__}, retry {attempt + 1}", flush=True)
            time.sleep(min(30 * 2 ** attempt, 600))
    else:
        raise SystemExit(f"download {match} {half} failed")
    done.write_text("ok")
    return path


def labels(match, half):
    ev = st.events(match)
    ev = ev[ev.half == half].sort_values("frame")
    ann = []
    for r in ev.itertuples():
        ms = int(round(r.frame / FPS * 1000))
        ann.append({"gameTime": f"1 - {ms // 60000:02d}:{ms // 1000 % 60:02d}", "label": r.label,
                    "position": str(ms), "team": r.side or "", "visibility": "visible"})
    return {"annotations": ann}


def main():
    check()
    ap = argparse.ArgumentParser()
    ap.add_argument("match")
    ap.add_argument("half", type=int)
    ap.add_argument("--seconds", type=float, default=0, help="test: only the first N seconds")
    a = ap.parse_args()
    name = f"{a.match}_{a.half}" + (f"_test{int(a.seconds)}" if a.seconds else "")
    od = OUT / name
    if (od / "done.json").exists():
        print(name, "already done", flush=True)
        return
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cen, n = centres(a.match, a.half)
    video = download(a.match, a.half)
    t_dl = time.time() - t0
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                       "-of", "json", str(video)], capture_output=True, text=True).stdout)["streams"][0]
    w, h = probe["width"], probe["height"]
    s = OW / (CROP_FRAC * w)
    sw, sh = int(round(w * s / 2) * 2), int(round(h * s / 2) * 2)
    dec = ["ffmpeg", "-v", "error", "-threads", "0", "-i", str(video)]
    if a.seconds:
        dec += ["-t", str(a.seconds)]
    dec += ["-vf", f"select=not(mod(n\\,{STRIDE})),scale={sw}:{sh}", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    enc = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS / STRIDE),
           "-i", "pipe:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(od / "crops.mp4")]
    pd_ = subprocess.Popen(dec, stdout=subprocess.PIPE)
    pe = subprocess.Popen(enc, stdin=subprocess.PIPE)
    boxes, k, size = [], 0, sw * sh * 3
    while True:
        buf = pd_.stdout.read(size)
        if len(buf) < size:
            break
        img = np.frombuffer(buf, np.uint8).reshape(sh, sw, 3)
        f = min(k * STRIDE, n - 1)
        cx, cy = cen[f] * s
        x0 = int(np.clip(round(cx - OW / 2), 0, sw - OW))
        y0 = int(np.clip(round(cy - OH / 2), 0, max(sh - OH, 0)))
        crop = img[y0:y0 + OH, x0:x0 + OW]
        if crop.shape[0] < OH:
            crop = np.pad(crop, ((0, OH - crop.shape[0]), (0, 0), (0, 0)))
        pe.stdin.write(np.ascontiguousarray(crop).tobytes())
        boxes.append((k * STRIDE, x0 / s, y0 / s, OW / s, OH / s))
        k += 1
    pd_.wait()
    pe.stdin.close()
    pe.wait()
    np.savez_compressed(od / "centres.npz", boxes=np.array(boxes, np.float32))
    (od / "Labels-ball.json").write_text(json.dumps(labels(a.match, a.half)))
    rec = {"match": a.match, "half": a.half, "panorama": [w, h], "scale": s, "kept frames": k, "gsr frames": n,
           "download s": round(t_dl), "crop s": round(time.time() - t0 - t_dl), "mp4 MB": round((od / "crops.mp4").stat().st_size / 1e6, 1)}
    (od / "done.json").write_text(json.dumps(rec, indent=1))
    print(name, rec, flush=True)
    if not a.seconds:
        video.unlink()
        (DL / f"{a.match}_{a.half}.ok").unlink()


if __name__ == "__main__":
    main()
