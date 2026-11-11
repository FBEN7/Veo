"""Exploratory (after goals_v5.json): how loud is the kick-off whistle?
Per restartable goal, the max whistle score within +-3 s of the kick-off action
(first labelled action >= 10 s after the goal), as a percentile of the half's scores;
and the same for random times, for contrast."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import score as S
rows, rnd, silent = [], [], {}
rng = np.random.default_rng(0)
for m in S.E.MATCHES:
    for half in (1, 2):
        h = S.S3.load(m, half)
        w = pd.read_parquet(S.WH / f"{m}_{half}.parquet")
        t, sc = w.t.to_numpy(), w.score.to_numpy()
        srt = np.sort(sc)
        pct = lambda v: float(np.searchsorted(srt, v) / len(srt))
        acts = np.sort(np.asarray(h["actions"], float))
        gf = h["goals"].frame.to_numpy()
        for g, r in zip(gf, S.F.restartable(gf, h["actions"])):
            if not r:
                continue
            ko = acts[acts >= g + 10 * S.FPS][0] / S.FPS
            win = (t >= ko - 3) & (t <= ko + 3)
            rows.append({"match": m, "half": half, "ko_s": round(ko, 1), "pct": round(pct(sc[win].max()), 4),
                         "rms_pct": round(float((w.rms.to_numpy() <= w.rms[win].max()).mean()), 3)})
        if (w.rms > 0).any():
            for x in rng.uniform(120, t.max() - 120, 20):
                win = (t >= x - 3) & (t <= x + 3)
                rnd.append(pct(sc[win].max()))
        silent.setdefault(m, []).append(bool((w.rms == 0).all()))
d = pd.DataFrame(rows)
mute = sorted(m for m, v in silent.items() if all(v))
a = d[~d.match.isin(mute)]
out = {"halves with an all-zero audio track": {m: silent[m] for m in silent if any(silent[m])},
       "restartable goals": len(d), "on audible matches": len(a),
       "audible: kick-off max score percentile, quartiles": a.pct.quantile([.25, .5, .75]).round(4).tolist(),
       "audible: share >= 0.99 / 0.98 / 0.95": [round(float((a.pct >= q).mean()), 3) for q in (0.99, 0.98, 0.95)],
       "kick-off max score percentile, quartiles": d.pct.quantile([.25, .5, .75]).round(4).tolist(),
       "share >= 0.99 / 0.98 / 0.95": [round(float((d.pct >= q).mean()), 3) for q in (0.99, 0.98, 0.95)],
       "random 6-s windows (audible halves), share >= 0.99 / 0.98 / 0.95": [round(float((np.array(rnd) >= q).mean()), 3) for q in (0.99, 0.98, 0.95)],
       "per match median percentile": d.groupby("match").pct.median().round(3).to_dict(),
       "per goal": d.to_dict("records")}
(HERE / "reading_ko_whistle.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "per goal"}, indent=1))
