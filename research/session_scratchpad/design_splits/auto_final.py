"""Read-only: the frozen automatic measure (FF-auto: 4 m corridor, t in
[0,1], |t-s| <= 0.3, >= 1 m from every player) per pass on T and V-old, for
v1 heatmap and COCO; SD of the paired difference for power."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
from eval_ball_detector import frames_of, ground
D = "/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/"
DATA = "/home/user/Veo/data/soccertrack_v2"
res = {}
for f in ("ball/heatmap_T.json", "ball/coco_T2.json", "ball/V.json"):
    for r in json.load(open(D + f))["results"]:
        res.setdefault(r["window"], {}).update({k: v for k, v in r.items() if isinstance(v, dict)})
HM = "ball_heatmap.3000.pt"
def ffauto(c, p, fr, people, cal, W=4.0, R=1.0, dt=0.3):
    if not len(c) or fr not in people: return 0.0
    gp = ground(np.array(c)[:1], cal)[0]
    if not np.isfinite(gp).all(): return 0.0
    A, B = np.array(p["from"]), np.array(p["to"]); ab = B - A
    t = float(np.dot(gp - A, ab) / np.dot(ab, ab)); tc = np.clip(t, 0, 1)
    perp = np.hypot(*(gp - (A + tc * ab)))
    q = p["frames"].index(fr); s = 0.25 + q * 0.5 / 3
    near = np.hypot(*(people[fr] - gp).T).min()
    return float(perp <= W and 0 <= t <= 1 and abs(t - s) <= dt and near >= R)
out = []
for w, r in res.items():
    S = "V" if w.startswith(("st2_117093", "st2_132877")) else "T"
    info = json.load(open(f"{DATA}/{w}.json")); cal = st.calibration(info["match"])
    gt = pd.read_parquet(f"{DATA}/{w}_gt.parquet")
    people = {fr: q[["x", "y"]].to_numpy(float) for fr, q in gt.groupby("frame")}
    for i, p in enumerate(frames_of(w)["passes"]):
        row = dict(S=S, w=w, i=i, minute=p["frames"][0] // 1500)
        for name in (HM, "coco"):
            row[name] = np.mean([ffauto(r[name]["candidates"].get(str(fr), []), p, fr, people, cal) for fr in p["frames"]])
        out.append(row)
df = pd.DataFrame(out)
df.to_csv(D + "design_splits/ffauto_per_pass.csv", index=False)
for S, x in df.groupby("S"):
    d = x[HM] - x["coco"]
    rng = np.random.default_rng(0)
    bs = [d.values[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    print(f"{S}: passes {len(x)} v1 {x[HM].mean():.3f} coco {x['coco'].mean():.3f} diff {d.mean():+.3f} "
          f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] SD(diff) {d.std(ddof=1):.3f} corr {np.corrcoef(x[HM], x['coco'])[0,1]:.2f}")
sd = df[df.S == "V"].eval(f"`{HM}` - coco").std(ddof=1)
for n in (38, 120, 170, 208, 250):
    se = sd / np.sqrt(n)
    print(f"n={n}: SE {se:.3f}, MDE (80% power, two-sided 0.05) {2.8 * se:.3f}; with design effect 1.3: {2.8 * se * np.sqrt(1.3):.3f}")
