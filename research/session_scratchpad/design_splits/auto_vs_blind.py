"""Read-only: per-source agreement of the chosen automatic measure, and the
automatic free-flight hit over all four frames per pass (T and V), next to
the blind result."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
from src.ball_heatmap import pitch_to_image
from eval_ball_detector import frames_of, ground
D = "/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/"
DATA = "/home/user/Veo/data/soccertrack_v2"
CFG = [(3.0, 20, 1.0, 0.0, 1.0), (2.5, 20, 1.0, 0.0, 1.0), (4.0, 20, 1.0, 0.0, 1.0), (1.0, 0, 2.0, 0.1, 0.9)]
g = pd.read_csv(D + "design_splits/tiles_ff_geometry.csv")
g["B"] = g.j == "B"
det = g[g.src != "decoy"].sort_values("tdir", key=lambda s: s.isin(["tiles_ff", "tiles_ffV"]), ascending=False).drop_duplicates(["w", "f", "src"])
for W, P, R, lo, hi in CFG[:1]:
    a = ((det.perp <= W) | (det.px <= P)) & (det.near >= R) & (det.t >= lo) & (det.t <= hi)
    for (s, src), x in det.assign(a=a).groupby(["set", "src"]):
        tp = int((x.a & x.B).sum()); fp = int((x.a & ~x.B).sum()); fn = int((~x.a & x.B).sum()); tn = int((~x.a & ~x.B).sum())
        print(f"cfg W{W} P{P} R{R}: {s} {src:22s} tp {tp} fp {fp} fn {fn} tn {tn}  auto-rate {(tp+fp)/len(x):.2f} judged-rate {(tp+fn)/len(x):.2f}")
    print("  false autos:", det.assign(a=a)[a & ~det.B][["set", "src", "j", "perp", "px", "near", "t"]].round(2).values.tolist())
    print("  missed B:", det.assign(a=a)[~a & det.B][["set", "src", "perp", "px", "near", "t"]].round(2).values.tolist())

res = {}
for f in ("ball/heatmap_T.json", "ball/coco_T2.json", "ball/V.json"):
    for r in json.load(open(D + f))["results"]:
        res.setdefault(r["window"], {}).update({k: v for k, v in r.items() if isinstance(v, dict)})
HM = "ball_heatmap.3000.pt"
for W, P, R, lo, hi in CFG:
    per = {"T": {HM: [], "coco": []}, "V": {HM: [], "coco": []}}
    for w, r in res.items():
        S = "V" if w.startswith(("st2_117093", "st2_132877")) else "T"
        info = json.load(open(f"{DATA}/{w}.json")); cal = st.calibration(info["match"])
        sets = frames_of(w)
        gt = pd.read_parquet(f"{DATA}/{w}_gt.parquet")
        people = {fr: q[["x", "y"]].to_numpy(float) for fr, q in gt.groupby("frame")}
        for p in sets["passes"]:
            A, B = np.array(p["from"]), np.array(p["to"]); ab = B - A
            uvs = pitch_to_image(A + np.linspace(0, 1, 201)[:, None] * ab, cal)
            for name in (HM, "coco"):
                if name not in r:
                    continue
                hits = []
                for fr in p["frames"]:
                    c = np.array(r[name]["candidates"].get(str(fr), []))
                    if not len(c) or fr not in people:
                        hits.append(0.0); continue
                    gp = ground(c[:1], cal)[0]
                    if not np.isfinite(gp).all():
                        hits.append(0.0); continue
                    t = float(np.dot(gp - A, ab) / np.dot(ab, ab)); tc = np.clip(t, 0, 1)
                    perp = np.hypot(*(gp - (A + tc * ab)))
                    px = np.hypot(*(uvs - np.array([c[0, 0], c[0, 1] + c[0, 3] / 2])).T).min()
                    near = np.hypot(*(people[fr] - gp).T).min()
                    hits.append(float((perp <= W or px <= P) and near >= R and lo <= t <= hi))
                per[S][name].append(np.mean(hits))
    for S in ("T", "V"):
        a, b = np.array(per[S][HM]), np.array(per[S]["coco"])
        d = a - b; rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"W{W} P{P} R{R} t{lo}-{hi} {S}: passes {len(a)} heatmap {a.mean():.3f} coco {b.mean():.3f} diff {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
