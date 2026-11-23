"""tdeed_probe (plan.json): T-DEED SoccerNetBall_challenge1 on a broadcast-like
virtual camera cut from our panoramic videos; per unit, the per-frame class
probabilities (every 2nd frame) as its inference.py computes them.

    python run_tdeed.py --repo T-DEED --ckpt ckpt.pt --stage STAGE --videos DIR [DIR ...]
                        --out OUT [--units N1 N2 ...] [--device cuda]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

CLIP, STEP, STRIDE = 100, 25, 2        # inference.py for soccernetball: clip 100, overlap 75, stride 2
OW, OH = 796, 448
CROP_FRAC = 0.35


def model_of(repo, ckpt, device):
    sys.path.insert(0, repo)
    import timm
    import torch
    create = timm.create_model
    timm.create_model = lambda name, pretrained=False, **k: create(name, pretrained=False, **k)  # weights come from the checkpoint
    if device == "cpu":
        torch.nn.Module.cuda = lambda self, *a, **k: self
    from model.model import TDEEDModel
    cfg = json.load(open(os.path.join(repo, "config/SoccerNetBall/SoccerNetBall_challenge1.json")))
    args = SimpleNamespace(**cfg)
    args.crop_dim = None if cfg["crop_dim"] <= 0 else cfg["crop_dim"]
    m = TDEEDModel(device=device, args=args)
    m._model.update_pred_head([cfg["num_classes"] + 1, cfg["pretrain"]["num_classes"] + 1])
    m._num_classes = cfg["num_classes"] + 1 + cfg["pretrain"]["num_classes"] + 1
    m._model.to(device)
    m.load(torch.load(ckpt, map_location=device, weights_only=True))
    return m


def frames(video, w, h, cen):
    """Every 2nd frame, scaled so the crop is OW wide, cropped around the centre."""
    s = OW / (CROP_FRAC * w)
    sw, sh = int(round(w * s / 2) * 2), int(round(h * s / 2) * 2)
    cmd = ["ffmpeg", "-v", "error", "-threads", "0", "-i", video,
           "-vf", f"select=not(mod(n\\,{STRIDE})),scale={sw}:{sh}", "-vsync", "0",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, k = [], 0
    while True:
        buf = p.stdout.read(sw * sh * 3)
        if len(buf) < sw * sh * 3:
            break
        img = np.frombuffer(buf, np.uint8).reshape(sh, sw, 3)
        f = min(k * STRIDE, len(cen) - 1)
        cx, cy = cen[f] * s
        x0 = int(np.clip(round(cx - OW / 2), 0, sw - OW))
        y0 = int(np.clip(round(cy - OH / 2), 0, max(sh - OH, 0)))
        crop = img[y0:y0 + OH, x0:x0 + OW]
        if crop.shape[0] < OH:
            crop = np.pad(crop, ((0, OH - crop.shape[0]), (0, 0), (0, 0)))
        out.append(crop.transpose(2, 0, 1).copy())
        k += 1
    p.wait()
    return np.stack(out)


def predict(model, fr, device, batch=4):
    n = len(fr)
    if n < CLIP:
        fr = np.concatenate([fr, np.zeros((CLIP - n,) + fr.shape[1:], np.uint8)])
    starts = list(range(0, len(fr) - CLIP + 1, STEP))
    if starts[-1] != len(fr) - CLIP:
        starts.append(len(fr) - CLIP)
    acc = np.zeros((len(fr), 13), np.float32)
    sup = np.zeros(len(fr), np.int32)
    for i in range(0, len(starts), batch):
        b = starts[i:i + batch]
        seq = np.stack([fr[a:a + CLIP] for a in b])
        _, sc = model.predict(seq, use_amp=(device == "cuda"))
        for a, p in zip(b, sc):
            acc[a:a + CLIP] += p
            sup[a:a + CLIP] += 1
    return (acc / np.maximum(sup, 1)[:, None])[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", nargs="*")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    units = json.load(open(os.path.join(a.stage, "units.json")))
    if a.units:
        units = [u for u in units if u["name"] in set(a.units)]
    cen = pd.read_parquet(os.path.join(a.stage, "centres.parquet"))
    model = model_of(a.repo, a.ckpt, a.device)
    os.makedirs(a.out, exist_ok=True)
    for u in units:
        dst = os.path.join(a.out, f"{u['name']}.npz")
        if os.path.exists(dst):
            continue
        hits = [h for d in a.videos for h in glob.glob(os.path.join(d, "**", f"{u['name']}.mp4"), recursive=True)]
        if not hits:
            print(u["name"], "no video", flush=True)
            continue
        t = time.time()
        c = cen[cen.unit == u["name"]].sort_values("frame")[["cx", "cy"]].to_numpy()
        fr = frames(hits[0], u["width"], u["height"], c)
        p = predict(model, fr, a.device, a.batch)
        np.savez_compressed(dst, frame=np.arange(len(p)) * STRIDE, p=p.astype(np.float16))
        print(u["name"], len(fr), "frames", round(time.time() - t, 1), "s", flush=True)


if __name__ == "__main__":
    main()
