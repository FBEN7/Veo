"""Where the ball is lost near the goals (plan.json).

    VEO_UNSEAL=... python measure.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball_interp"))
import feasibility as F  # noqa: E402
from src import ball_link as bl  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.unseal import unsealed  # noqa: E402

VER = ROOT / ".cache/soccertrack_v2/ball_verified"
WINDOWS = {"st2_118578_2nd_f015000": VER / "A/frames.parquet", "st2_118578_2nd_f037500": VER / "A/frames.parquet",
           "st2_128057_1st_f030000": VER / "v5p2/labels/frames.parquet",
           "st2_132831_2nd_f030000": VER / "v5p2/labels/frames.parquet"}
HIT = 12.0
NEAR_X = 52.5 - 16.5
CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
SHOTS = {"SHOT", "GOAL"}
PASSES = {"PASS", "HIGH PASS", "CROSS"}


def pipeline_link(rows, cal, players, poly):
    """The pipeline's ball choice without the grass test: players' frames
    (even), confidence >= 0.10, top 4 a frame, inside the outline, link."""
    r = rows[(rows.frame % 2 == 0) & (rows.confidence >= 0.10)]
    r = r.sort_values("confidence", ascending=False).groupby("frame").head(4)
    if poly is not None:
        ins = [cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in zip(r.px, r.py)]
        r = r[np.asarray(ins, bool)]
    return bl.link(r, cal, players, c_null=1.0, c_static=2.0, tau=0.10)


def measure1():
    rows_out = []
    for w, lab in WINDOWS.items():
        match = w.split("_")[1]
        cal = st.calibration(match)
        f = pd.read_parquet(lab)
        f = f[(f.window == w) & (f.status == "ball")]
        raw = pd.read_parquet(ROOT / f".cache/v3_balls_{w}.parquet")
        kept = F.candidates(w, cal)
        out_dir = ROOT / f"output_{w}"
        players = F.players_of(w)
        poly = None
        if (out_dir / "pitch.json").exists():
            poly = np.asarray(json.loads((out_dir / "pitch.json").read_text())["pitch"], np.float32).reshape(-1, 1, 2)
        link = pipeline_link(kept, cal, players, poly).drop_duplicates("frame").set_index("frame")
        rby, kby = {k: g for k, g in raw.groupby("frame")}, {k: g for k, g in kept.groupby("frame")}
        g = st.image_to_pitch(np.c_[f.u, f.v + f["size"] / 2], cal)
        for (r, (gx, gy)) in zip(f.itertuples(), g):
            def any_hit(d):
                return bool(d is not None and len(d) and (np.hypot(d.px - r.u, d.py - r.v) <= HIT).any())
            rr = rby.get(r.frame)
            top = rr.sort_values("confidence", ascending=False).iloc[0] if rr is not None and len(rr) else None
            lf = r.frame - r.frame % 2
            rows_out.append({
                "window": w, "match": match, "zone": "near a goal" if abs(gx) >= NEAR_X else "elsewhere",
                "size": r.size, "raw any": any_hit(rr), "kept any": any_hit(kby.get(r.frame)),
                "top1": bool(top is not None and np.hypot(top.px - r.u, top.py - r.v) <= HIT),
                "linked": bool(lf in link.index and np.hypot(link.loc[lf].px - r.u, link.loc[lf].py - r.v) <= HIT)})
    d = pd.DataFrame(rows_out)
    cols = ["raw any", "kept any", "top1", "linked"]
    res = {}
    for key, g in [("all", d), *[(z, x) for z, x in d.groupby("zone")], *[(f"{m} {z}", x) for (m, z), x in d.groupby(["match", "zone"])]]:
        res[key] = {"n": len(g), **{c: round(float(g[c].mean()), 3) for c in cols}, "ball size px": round(float(g["size"].median()), 1)}
    return res


def measure2():
    rows = []
    for p in sorted((ROOT / ".cache/gpu").iterdir()):
        w = p.name
        m = w.split("_")[1]
        if m == "117092" or not (CLIPS / f"{w}_events.csv").exists():
            continue
        base = ROOT / (".cache/gpu_nograss" if m in ("128057", "132831") else ".cache/gpu") / w / f"output_{w}"
        if not ((base / "v3s_rows.parquet").exists() and (base / "metric_v3s_link_nograss.parquet").exists()):
            continue
        fr = json.loads((p / ".cache" / f"v3_balls_{w}.parquet.meta.json").read_text())["frames"]
        lo, hi = min(fr), max(fr)
        kept = pd.read_parquet(base / "v3s_rows.parquet")
        met = pd.read_parquet(base / "metric_v3s_link_nograss.parquet")
        ball = met[met.cls == "ball"].drop_duplicates("frame").set_index("frame")
        gt = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
        gt = gt.assign(pid=gt.player_id.astype(str))
        kby = {k: g for k, g in kept.groupby("frame")}
        gby = {k: g for k, g in gt.groupby("frame")}
        for e in pd.read_csv(CLIPS / f"{w}_events.csv").itertuples():
            kind = "shot" if e.label in SHOTS else "pass" if e.label in PASSES else None
            if kind is None or pd.isna(e.player_id) or not (lo + 10 <= e.frame <= hi - 10):
                continue
            pid = str(int(e.player_id))
            cand = link = False
            gx = np.nan
            for f in range(int(e.frame) - 10, int(e.frame) + 11, 2):
                g = gby.get(f)
                if g is None:
                    continue
                s = g[g.pid == pid]
                if s.empty:
                    continue
                s = s.iloc[0]
                fx, fy, h = s.bx + s.bw / 2, s.by + s.bh, s.bh
                if f == int(e.frame) - int(e.frame) % 2:
                    gx = s.x
                k = kby.get(f)
                if k is not None and (np.hypot(k.px - fx, k.py - fy) <= h).any():
                    cand = True
                if f in ball.index and np.hypot(ball.loc[f].px_raw - fx, ball.loc[f].py_raw - fy) <= h:
                    link = True
            rows.append({"match": m, "kind": kind, "near goal": bool(abs(gx) >= NEAR_X) if gx == gx else None,
                         "candidate at feet": cand, "linked at feet": link})
    d = pd.DataFrame(rows)
    res = {}
    for key, g in [*[(k, x) for k, x in d.groupby("kind")],
                   *[(f"{k}, {'near a goal' if n else 'elsewhere'}", x) for (k, n), x in d.dropna(subset=['near goal']).groupby(["kind", "near goal"])]]:
        res[key] = {"n": len(g), "candidate at feet": round(float(g["candidate at feet"].mean()), 3),
                    "linked at feet": round(float(g["linked at feet"].mean()), 3)}
    res["shots per match"] = {m: {"n": len(g), "candidate": round(float(g["candidate at feet"].mean()), 2),
                                  "linked": round(float(g["linked at feet"].mean()), 2)}
                              for m, g in d[d.kind == "shot"].groupby("match")}
    return res


def main():
    for m in ("128057", "132831"):
        if not unsealed(m, "ball_goal: ball near the goals"):
            raise SystemExit("VEO_UNSEAL needed")
    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0],
           "measure 1 (verified balls)": measure1()}
    print(json.dumps(out, indent=1), flush=True)
    out["measure 2 (labelled shots and passes, ball at the actor's feet)"] = measure2()
    print(json.dumps(out["measure 2 (labelled shots and passes, ball at the actor's feet)"], indent=1))
    (HERE / "measure.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
