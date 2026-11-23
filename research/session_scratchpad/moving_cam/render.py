"""moving_cam (plan.json, hashed first): one 3-min simulated follow-cam window.

Crops SoccerTrack v2's panorama to 0.35 of its width at 16:9, the crop
centre following the truth players' median (2-s rolling median, then 1-s
mean), scales it to 1920 x 1080 and encodes H.264 at 4.4 Mbit/s, 25 fps.
Frames are fetched by range requests (no full download). Writes, under
OUT/<match>_<half>/: clip.mp4, truth.parquet (per frame, every person's
pitch metres and box centre / height in the output frame) and done.json.

    python render.py 117093 1
"""
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

FPS, START, COUNT = 25, 15000, 4500
CROP_FRAC, OW, OH, BITRATE = 0.35, 1920, 1080, "4400k"
OUT = Path("/tmp/claude-0/-home-user-veo/b69e5b60-e0fe-5920-8f15-285768d7ea57/scratchpad/mcam/windows")


def check():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    if hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest() != want:
        raise SystemExit("plan.json does not match its sha256")


def centres(g):
    p = g[g.role.isin(["player", "goalkeeper"]) & g.bx.notna()]
    c = p.assign(cx=p.bx + p.bw / 2, cy=p.by + p.bh / 2).groupby("frame")[["cx", "cy"]].median()
    n = int(g.frame.max()) + 1
    c = c.reindex(np.arange(n)).interpolate(limit_direction="both")
    c = c.rolling(2 * FPS, center=True, min_periods=1).median().rolling(FPS, center=True, min_periods=1).mean()
    return c.to_numpy()


def main():
    check()
    match, half = sys.argv[1], int(sys.argv[2])
    od = OUT / f"{match}_{half}"
    if (od / "done.json").exists():
        print(od.name, "already done")
        return
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    g = st.gsr_table(match, half)
    cen = centres(g)
    src = st.open_video(match, half)
    enc = None
    boxes = np.zeros((COUNT, 3))   # x0, y0, scale per output frame
    n = 0
    for k, img in st.read_frames(src, START, COUNT):
        h, w = img.shape[:2]
        cw = int(round(CROP_FRAC * w))
        ch = int(round(cw * OH / OW))
        cx, cy = cen[min(k, len(cen) - 1)]
        x0 = int(np.clip(round(cx - cw / 2), 0, w - cw))
        y0 = int(np.clip(round(cy - ch / 2), 0, h - ch))
        import cv2
        out = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (OW, OH), interpolation=cv2.INTER_LINEAR)
        if enc is None:
            enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
                                    "-r", str(FPS), "-i", "pipe:0", "-c:v", "libx264", "-preset", "veryfast",
                                    "-b:v", BITRATE, "-maxrate", BITRATE, "-bufsize", "8800k", "-pix_fmt", "yuv420p",
                                    str(od / "clip.mp4")], stdin=subprocess.PIPE)
        enc.stdin.write(out.tobytes())
        boxes[k - START] = (x0, y0, OW / cw)
        n += 1
    src.close()
    enc.stdin.close()
    enc.wait()
    if n != COUNT:
        raise SystemExit(f"{od.name}: {n} frames read, {COUNT} wanted")
    win = g[(g.frame >= START) & (g.frame < START + COUNT)].copy()
    b = boxes[(win.frame - START).to_numpy()]
    win["out_frame"] = win.frame - START
    win["u"] = (win.bx + win.bw / 2 - b[:, 0]) * b[:, 2]
    win["v"] = (win.by + win.bh / 2 - b[:, 1]) * b[:, 2]
    win["h"] = win.bh * b[:, 2]
    win["in_view"] = (win.u >= 0) & (win.u < OW) & (win.v >= 0) & (win.v < OH)
    win.to_parquet(od / "truth.parquet")
    np.save(od / "crop_boxes.npy", boxes)
    (od / "done.json").write_text(json.dumps({"match": match, "half": half, "start": START, "frames": n,
                                              "panorama": [w, h], "crop": [cw, ch], "scale": OW / cw,
                                              "seconds": round(time.time() - t0)}, indent=1))
    print(od.name, "done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
