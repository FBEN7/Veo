"""follow-cam pilot (plan.json): per-frame class scores of a trained T-DEED
on whole crops.mp4 videos (12.5 fps = every 2nd frame), as its inference
does: clips of 100, overlap 75, softmax averaged over overlapping clips.
Read in chunks so a half never sits in memory whole.

    python infer.py --repo T-DEED --config cfg.json --ckpt last.pt --videos DIR --names 117093_1 ... --out OUT
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

CLIP, STEP, CHUNK = 100, 25, 1000


def model_of(repo, cfg_path, ckpt, device):
    sys.path.insert(0, repo)
    import timm
    import torch
    create = timm.create_model
    timm.create_model = lambda name, pretrained=False, **k: create(name, pretrained=False, **k)  # weights come from the checkpoint
    if device == "cpu":   # local smoke test only
        torch.nn.Module.cuda = lambda self, *a, **k: self
        torch.Tensor.cuda = lambda self, *a, **k: self
    from model.model import TDEEDModel
    cfg = json.load(open(cfg_path))
    args = SimpleNamespace(**cfg)
    args.crop_dim = None if cfg["crop_dim"] <= 0 else cfg["crop_dim"]
    args.pretrain = None
    m = TDEEDModel(device=device, args=args)
    m.load(torch.load(ckpt, map_location=device))
    return m


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
        _, sc = model.predict(np.stack([fr[a:a + CLIP] for a in b]), use_amp=(device == "cuda"))
        for a, p in zip(b, sc):
            acc[a:a + CLIP] += p
            sup[a:a + CLIP] += 1
    return acc[:n], sup[:n]


def frames(path):
    cap = cv2.VideoCapture(path)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    cap.release()


def run(model, path, device, batch=4):
    """chunks of CHUNK frames overlapping by CLIP; scores summed with their support."""
    acc, sup, buf, base = [], [], [], 0
    tot_acc, tot_sup = np.zeros((0, 13), np.float32), np.zeros(0, np.int32)

    def flush(buf, base, last):
        nonlocal tot_acc, tot_sup
        a, s = predict(model, np.stack(buf), device, batch)
        end = base + len(buf)
        if end > len(tot_acc):
            tot_acc = np.concatenate([tot_acc, np.zeros((end - len(tot_acc), 13), np.float32)])
            tot_sup = np.concatenate([tot_sup, np.zeros(end - len(tot_sup), np.int32)])
        tot_acc[base:end] += a
        tot_sup[base:end] += s

    for f in frames(path):
        buf.append(f)
        if len(buf) == CHUNK:
            flush(buf, base, False)
            buf = buf[-CLIP:]
            base += CHUNK - CLIP
    if len(buf) > CLIP or base == 0:
        flush(buf, base, True)
    return tot_acc / np.maximum(tot_sup, 1)[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = model_of(a.repo, a.config, a.ckpt, a.device)
    for name in a.names:
        t = time.time()
        sc = run(m, os.path.join(a.videos, f"{name}.mp4"), a.device, a.batch)
        np.savez_compressed(os.path.join(a.out, f"{name}.npz"), scores=sc.astype(np.float16))
        print(name, sc.shape, f"{time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
