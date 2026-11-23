"""moving_cam on TeamTrack (plan_teamtrack.json, hashed first): one 3-min
simulated follow-cam window from six consecutive soccer_side clips.

Full-height 16:9 crop of the 6500 x 1000 panorama (1778 x 1000), centre x
following the annotated players' median box centre (2-s rolling median then
1-s mean), scaled to 1920 x 1080, H.264 4.4 Mbit/s, 25 fps. Writes, under
OUT/tt_<start>/: clip.mp4, truth.parquet (same columns evaluate.py reads),
crop_boxes.npy, done.json (with the per-clip identity scores).

    python render_tt.py 360        # window starting at 360 s
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
SCR = Path("/tmp/claude-0/-home-user-veo/b69e5b60-e0fe-5920-8f15-285768d7ea57/scratchpad")
TT = SCR / "tt_dl/teamtrack/teamtrack/soccer_side"
TRAJ = SCR / "tt_traj/teamtrack-trajectory/teamtrack-trajectory/Soccer"
OUT = SCR / "mcam/tt_windows"
FPS, CLIP_F, N_CLIPS = 25, 750, 6
PW, PH = 6500, 1000
CW, CH = round(PH * 16 / 9), PH
OW, OH, BITRATE = 1920, 1080, "4400k"
MIN_ID_SCORE = 1.0
TRAJ_FILES = ((0, 0), (20, 240), (40, 480), (42, 504))   # file suffix, first frame


def check():
    for f in ("plan.json", "addendum1.json", "plan_teamtrack.json"):
        want = (HERE / f"{f}.sha256").read_text().split()[0]
        if hashlib.sha256((HERE / f).read_bytes()).hexdigest() != want:
            raise SystemExit(f"{f} does not match its sha256")


def clip_name(s):
    return f"F_{'20200220' if s < 900 else '20220220'}_1_{s:04d}_{s + 30:04d}"


def find(sub, name):
    hits = sorted(TT.glob(f"*/{sub}/{name}"))
    if len(hits) != 1:
        raise SystemExit(f"{sub}/{name}: {len(hits)} files")
    return hits[0]


def clip_truth(s):
    """per-frame boxes (frame, person, team, left, top, w, h) and metres, one clip."""
    name = clip_name(s)
    ann = pd.read_csv(find("annotations", name + ".csv"), header=[0, 1, 2], index_col=0)
    players = [c for c in ann.columns.droplevel(2).unique() if c[0] != "BALL"]
    get = lambda t, p, a: ann[(t, p, a)].to_numpy(dtype=float)
    left = np.stack([get(t, p, "bb_left") for t, p in players], 1)
    top = np.stack([get(t, p, "bb_top") for t, p in players], 1)
    w = np.stack([get(t, p, "bb_width") for t, p in players], 1)
    h = np.stack([get(t, p, "bb_height") for t, p in players], 1)
    traj = np.full((CLIP_F, 44), np.nan)
    split = find("videos", name + ".mp4").parent.parent.name
    for k, f0 in TRAJ_FILES:
        traj[f0:f0 + 240] = np.loadtxt(TRAJ / split / f"{name}_{k}.txt", delimiter=",")
    tx, ty = traj[:744, 0::2], traj[:744, 1::2]
    bx, by = (left + w / 2)[:744], (top + h)[:744]
    score = np.zeros((22, 22))
    for i in range(22):
        m = ~np.isnan(bx[:, i])
        for j in range(22):
            score[i, j] = np.corrcoef(bx[m, i], tx[m, j])[0, 1] + np.corrcoef(by[m, i], ty[m, j])[0, 1]
    r, c = linear_sum_assignment(-score)
    ok = float(score[r, c].min()) >= MIN_ID_SCORE
    rows = []
    for i, (t, p) in enumerate(players):
        rows.append(pd.DataFrame({
            "clip_frame": np.arange(CLIP_F), "track_id": int(t) * 100 + int(p), "team": int(t),
            "role": "player", "kept": ok, "bx": left[:, i], "by": top[:, i], "bw": w[:, i], "bh": h[:, i],
            "x": traj[:, 2 * c[i]] if ok else np.nan, "y": traj[:, 2 * c[i] + 1] if ok else np.nan}))
    return pd.concat(rows, ignore_index=True), {"clip": name, "min id score": round(float(score[r, c].min()), 3),
                                                "kept": ok}


def centres(g, n):
    c = g.assign(cx=g.bx + g.bw / 2, cy=g.by + g.bh / 2).groupby("frame")[["cx", "cy"]].median()
    c = c.reindex(np.arange(n)).interpolate(limit_direction="both")
    return c.rolling(2 * FPS, center=True, min_periods=1).median().rolling(FPS, center=True, min_periods=1).mean().to_numpy()


def main():
    check()
    start = int(sys.argv[1])
    od = OUT / f"tt_{start:04d}"
    if (od / "done.json").exists():
        print(od.name, "already done")
        return
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    parts, ids = [], []
    for k in range(N_CLIPS):
        g, info = clip_truth(start + 30 * k)
        g["frame"] = g.clip_frame + k * CLIP_F
        parts.append(g)
        ids.append(info)
    g = pd.concat(parts, ignore_index=True).sort_values(["track_id", "frame"])
    # frames 744-749 of each clip have no trajectory row: linear, within each person's window
    g[["x", "y"]] = g.groupby("track_id")[["x", "y"]].transform(
        lambda s: s.interpolate(limit_area="inside").ffill(limit=6))
    g.loc[~g.kept, ["x", "y"]] = np.nan          # a dropped clip stays without truth
    # are annotation ids the same people across clip boundaries? box-centre jump, px
    jumps = []
    for k in range(1, N_CLIPS):
        a = g[g.frame == k * CLIP_F - 1].set_index("track_id")
        z = g[g.frame == k * CLIP_F].set_index("track_id")
        d = np.hypot((a.bx + a.bw / 2) - (z.bx + z.bw / 2), (a.by + a.bh / 2) - (z.by + z.bh / 2))
        jumps.append(round(float(d.median()), 1))
    n = N_CLIPS * CLIP_F
    cen = centres(g[g.bx.notna()], n)
    boxes = np.zeros((n, 3))
    scale = OW / CW
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "pipe:0", "-c:v", "libx264", "-preset", "veryfast",
                            "-b:v", BITRATE, "-maxrate", BITRATE, "-bufsize", "8800k", "-pix_fmt", "yuv420p",
                            str(od / "clip.mp4")], stdin=subprocess.PIPE)
    f = 0
    for k in range(N_CLIPS):
        cap = cv2.VideoCapture(str(find("videos", clip_name(start + 30 * k) + ".mp4")))
        for _ in range(CLIP_F):
            ok, img = cap.read()
            if not ok:
                raise SystemExit(f"{clip_name(start + 30 * k)}: short at frame {f}")
            if img.shape[:2] != (PH, PW):
                raise SystemExit(f"unexpected panorama size {img.shape}")
            x0 = int(np.clip(round(cen[f, 0] - CW / 2), 0, PW - CW))
            enc.stdin.write(cv2.resize(img[0:CH, x0:x0 + CW], (OW, OH), interpolation=cv2.INTER_LINEAR).tobytes())
            boxes[f] = (x0, 0, scale)
            f += 1
        cap.release()
    enc.stdin.close()
    enc.wait()
    b = boxes[g.frame.to_numpy()]
    g["out_frame"] = g.frame
    g["u"] = (g.bx + g.bw / 2 - b[:, 0]) * b[:, 2]
    g["v"] = (g.by + g.bh / 2 - b[:, 1]) * b[:, 2]
    g["h"] = g.bh * b[:, 2]
    g["in_view"] = (g.u >= 0) & (g.u < OW) & (g.v >= 0) & (g.v < OH)
    g.to_parquet(od / "truth.parquet")
    np.save(od / "crop_boxes.npy", boxes)
    (od / "done.json").write_text(json.dumps({"start s": start, "frames": f, "crop": [CW, CH], "scale": scale,
                                              "identity per clip": ids, "median box jump at clip joins (px)": jumps,
                                              "seconds": round(time.time() - t0)}, indent=1))
    print(od.name, "done", round(time.time() - t0), "s", [i["min id score"] for i in ids], "joins", jumps)


if __name__ == "__main__":
    main()
