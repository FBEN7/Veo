"""Where we lose the player on the ball (plan.json): per labelled action,
the actor detected, taken by a neighbour, missed or merged in the image;
the track around the action; the position error; a control of the other
players on the same frame.

    VEO_UNSEAL=... python near_ball.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import eval_soccertrack_v2 as ev  # noqa: E402

TREE = ROOT / ".cache/gpu_shots3"
CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
SEALED = {"128057", "132831"}
SKIP = {"117092"}              # M1: truth boxes in another pixel frame
LO, HI, WIN = 350, 1075, 24


def clip(w: str) -> list:
    m = pd.read_parquet(TREE / w / f"output_{w}" / "metric_v3s_link.parquet")
    lab = pd.read_csv(CLIPS / f"{w}_events.csv")
    lab = lab[(lab.frame >= LO) & (lab.frame <= HI) & lab.player_id.notna()]
    if lab.empty:
        return []
    gt = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
    gt = gt[gt.bx.notna()]
    truth = ev.truth(CLIPS / f"{w}_gt.parquet")
    centres = {int(f) - int(f) % 2 for f in lab.frame}
    frames = sorted({c + k for c in centres for k in range(-WIN, WIN + 1, 2)})
    p = m[(m.cls == "player") & m.frame.isin(frames)]
    det = ev.boxes(p)
    tr = truth[truth.frame.isin(frames)]
    det["person"] = ev.match_feet(det, tr)
    by_f = {f: g for f, g in det.groupby("frame")}
    tr_f = {f: g for f, g in tr.groupby("frame")}
    out = []
    for r in lab.itertuples():
        f = int(r.frame) - int(r.frame) % 2
        a = gt[(gt.frame == f) & (gt.player_id.astype(str) == str(int(r.player_id)))]
        d, t = by_f.get(f), tr_f.get(f)
        if a.empty or d is None or t is None:
            continue
        actor = int(a.track_id.iloc[0])
        role = a.role.iloc[0]
        au, av = float(a.bx.iloc[0] + a.bw.iloc[0] / 2), float(a.by.iloc[0] + a.bh.iloc[0])
        fu, fv = ((d.x0 + d.x1) / 2).to_numpy(), d.y1.to_numpy()
        dist = np.hypot(fu - au, fv - av)
        hit = d[d.person == actor]
        if len(hit):
            kind, row = "detected", hit.iloc[0]
        elif (dist <= ev.FOOT_PX).any():
            kind, row = "taken", d.iloc[int(dist.argmin())]
        else:
            kind, row = "missed", None
        merged = None
        if row is not None:
            tu, tv = ((t.x0 + t.x1) / 2).to_numpy(), t.y1.to_numpy()
            inside = (tu >= row.x0) & (tu <= row.x1) & (tv >= row.y0) & (tv <= row.y1 + 5)
            merged = bool(inside.sum() >= 2)
        same = switch = pos = None
        if kind == "detected":
            tid = row.track_id
            win = det[(det.frame >= f - WIN) & (det.frame <= f + WIN)]
            act = win[win.person == actor]
            n_fr = len({fr for fr in range(f - WIN, f + WIN + 1, 2) if fr in by_f})
            same = float((act.track_id == tid).sum() / max(n_fr, 1))
            mine = win[(win.track_id == tid) & win.person.notna()]
            switch = bool((mine.person != actor).any())
            pos = float(np.hypot(row.x - 52.5 - a.x.iloc[0], row.y - 34.0 - a.y.iloc[0]))
        others = t[(t.person != actor) & (t.role == "player")]
        ctrl = float(others.person.isin(set(d.person.dropna())).mean()) if len(others) else np.nan
        out.append({"unit": w, "match": w.split("_")[1], "frame": int(r.frame), "label": r.label, "role": role,
                    "kind": kind, "merged": merged, "same track +-1s": same, "switch": switch,
                    "pitch m": pos, "control detected": ctrl})
    return out


def main():
    from src.unseal import unsealed
    for m in sorted(SEALED):
        if not unsealed(m, "near_ball diagnostic: the opened M7 / M9 shot clips (scored data)"):
            raise SystemExit("VEO_UNSEAL needed")
    units = sorted(p.name for p in TREE.iterdir() if p.name.startswith("st2_") and p.name.split("_")[1] not in SKIP)
    rows = []
    for i, w in enumerate(units):
        rows += clip(w)
        if i % 25 == 0:
            print(i, w, len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(HERE / "actions.parquet")
    k = df.kind.value_counts(normalize=True).round(4).to_dict()
    det = df[df.kind == "detected"]
    out = {"plan sha256": (HERE / "plan.json.sha256").read_text().split()[0], "units": len(units), "actions": len(df),
           "actor (share of actions)": k,
           "merged (share of actions with a detection near the actor)": round(float(df.merged.dropna().mean()), 4),
           "control: other outfield players detected on the same frames": round(float(df["control detected"].mean()), 4),
           "detected actors: same track over +-1 s (mean share of frames)": round(float(det["same track +-1s"].mean()), 4),
           "detected actors: their track carries another person within +-1 s": round(float(det.switch.mean()), 4),
           "detected actors: pitch distance (m)": {"median": round(float(det["pitch m"].median()), 2),
                                                   "p75": round(float(det["pitch m"].quantile(0.75)), 2),
                                                   "beyond 3 m": round(float((det["pitch m"] > 3).mean()), 4)},
           "per match: detected / taken / missed": {m: g.kind.value_counts(normalize=True).round(3).to_dict() for m, g in df.groupby("match")},
           "by label: detected": df.groupby("label").kind.apply(lambda s: round(float((s == "detected").mean()), 3)).to_dict()}
    (HERE / "near_ball.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
