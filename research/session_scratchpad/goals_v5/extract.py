"""goals v5 (plan.json, hashed first): whistle features of a half's audio.

One pass over the half's audio stream (the file is streamed; only audio is
decoded): channels averaged, resampled to 48 kHz if needed, 0.1-s bins with
a Hann window; per bin ratio (2.5-4.5 kHz energy / energy above 200 Hz),
tonality (max / median |X| in 2.5-4.5 kHz), score = ratio * ln(tonality)
(0 when tonality <= 1) and rms. t = audio pts time - video start time.
-> .cache/whistle/<match>_<half>.parquet and .json; skipped when present.

    VEO_UNSEAL=... python extract.py 117092:1 117092:2 ...
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src import soccertrack_v2 as st  # noqa: E402

OUT = ROOT / ".cache/whistle"
SR = 48000
BIN = 4800
SEALED = {"128057", "132831"}


def check_plan():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    if hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest() != want:
        raise SystemExit("plan.json does not match its sha256")


def features(x: np.ndarray, t0: float) -> pd.DataFrame:
    n = len(x) // BIN
    X = np.abs(np.fft.rfft(x[:n * BIN].reshape(n, BIN) * np.hanning(BIN), axis=1))
    f = np.fft.rfftfreq(BIN, 1 / SR)
    band = X[:, (f >= 2500) & (f <= 4500)]
    e_all = (X[:, f >= 200] ** 2).sum(1)
    ratio = (band ** 2).sum(1) / np.maximum(e_all, 1e-12)
    ton = band.max(1) / np.maximum(np.median(band, 1), 1e-12)
    score = np.where(ton > 1, ratio * np.log(np.maximum(ton, 1)), 0.0)
    rms = np.sqrt((x[:n * BIN].reshape(n, BIN) ** 2).mean(1))
    return pd.DataFrame({"t": t0 + np.arange(n) * BIN / SR, "score": score, "ratio": ratio,
                         "tonality": ton, "rms": rms})


def half(match: str, h: int):
    import av
    c = st.open_video(match, h)
    a, v = c.streams.audio[0], c.streams.video[0]
    v0 = float(v.start_time or 0) * float(v.time_base)
    res = av.AudioResampler(format="flt", layout="mono", rate=SR)
    chunks, t_first = [], None
    tic = time.time()
    for pkt in c.demux(a):
        for fr in pkt.decode():
            if t_first is None:
                t_first = float(fr.pts * a.time_base) - v0
            for o in res.resample(fr):
                chunks.append(o.to_ndarray().reshape(-1))
    for o in res.resample(None):
        chunks.append(o.to_ndarray().reshape(-1))
    c.close()
    x = np.concatenate(chunks).astype(np.float32)
    d = features(x, t_first)
    meta = {"match": match, "half": h, "stream rate": a.rate, "channels": a.channels,
            "seconds decoded": round(len(x) / SR, 1), "first t": round(t_first, 3),
            "run s": round(time.time() - tic, 1)}
    return d, meta


def main():
    check_plan()
    from src.unseal import unsealed
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[1:]:
        m, h = spec.split(":")
        p = OUT / f"{m}_{h}.parquet"
        if p.exists():
            print(spec, "done already", flush=True)
            continue
        if m in SEALED and not unsealed(m, "goals v5: whistle features (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
        for attempt in range(6):
            try:
                d, meta = half(m, int(h))
                break
            except Exception as e:  # streamed file: signed URLs expire, the proxy may reset
                print(spec, "attempt", attempt, "failed:", type(e).__name__, str(e)[:120], flush=True)
                time.sleep(10)
        else:
            raise SystemExit(f"{spec}: failed 6 times")
        tmp = p.with_suffix(".tmp.parquet")
        d.to_parquet(tmp)
        p.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        tmp.rename(p)
        print(spec, meta, flush=True)


if __name__ == "__main__":
    main()
