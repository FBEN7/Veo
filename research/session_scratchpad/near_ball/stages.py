"""Where the pipeline loses the player on the ball (plan_v3.json): per
action, the actor detected / taken / missed at each cached stage, from the
raw detector to the pipeline's output.

    VEO_UNSEAL=... python stages.py DET_DIR     # DET_DIR: the kernel's det/<unit>.parquet
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import eval_soccertrack_v2 as ev  # noqa: E402
from src import player_filter  # noqa: E402

CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
TREE = ROOT / ".cache/gpu_shots3"
SEALED = {"128057", "132831"}
STAGES = ["S0 raw", "S1 tracker", "S2 teams", "S3 assigned", "S4 roster cap", "S5 output"]
UNASSIGNED = ("", "-1", "unknown", "unassigned", "nan", "None")
CHECK = {}


def run_dir(w: str) -> Path:
    """The stages the output was built from (plan_v3_addendum1.json)."""
    return Path(os.path.realpath(TREE / w / f"output_{w}" / "metric_v3s_link.parquet")).parent / "shots_v3s_link"


def output(w: str) -> pd.DataFrame:
    return pd.read_parquet(TREE / w / f"output_{w}" / "metric_v3s_link.parquet")


def stage_tables(w: str, det: pd.DataFrame) -> dict:
    info = json.loads((CLIPS / f"{w}.json").read_text())
    poly = np.asarray(info["pitch"], np.float32).reshape(-1, 1, 2)
    d = det[(det["pass"] == "s_c10") & (det.conf >= 0.20)]
    keep = [cv2.pointPolygonTest(poly, (float((a + b) / 2), float(y)), False) >= 0
            for a, b, y in zip(d.x0, d.x1, d.y1)]
    out = {"S0 raw": d[keep][["frame", "x0", "y0", "x1", "y1"]].assign(track_id=-1, team=None)}
    rd = run_dir(w)
    setting = json.loads((rd / "pitch.json").read_text())
    t1 = player_filter.inside_pitch(pd.read_parquet(rd / "tracks.parquet"), setting["pitch"], setting["margin_px"])
    out["S1 tracker"] = ev.boxes(t1)
    grass = pd.read_parquet(rd / "tracks_grass.parquet")
    s2 = ev.boxes(grass)
    out["S2 teams"] = s2
    out["S3 assigned"] = s2[s2.team.notna() & ~s2.team.astype(str).isin(UNASSIGNED)]
    out["S4 roster cap"] = ev.boxes(player_filter.filter_players(grass))
    out["S5 output"] = ev.boxes(output(w))
    return out


def s4_in_output(tables: dict) -> float:
    """Share of the output's player rows found in S4 (same frame, pixel)."""
    key = lambda d: set(zip(d.frame, ((d.x0 + d.x1) / 2).round(2), d.y1.round(2)))
    s5 = key(tables["S5 output"])
    return len(s5 & key(tables["S4 roster cap"])) / max(len(s5), 1)


def unit(w: str, det: pd.DataFrame, acts: pd.DataFrame) -> list:
    gt = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
    gt = gt[gt.bx.notna()]
    truth = ev.truth(CLIPS / f"{w}_gt.parquet")
    frames = sorted(set(acts.f))
    tr = truth[truth.frame.isin(frames)]
    tr_f = {f: g for f, g in tr.groupby("frame")}
    tables = stage_tables(w, det)
    CHECK[w] = s4_in_output(tables)
    # S3 track lengths over the whole clip (the roster cap's tie-breaker)
    s3 = tables["S3 assigned"]
    s3_len = s3.groupby("track_id").size()
    out = []
    for s, d in tables.items():
        d = d[d.frame.isin(frames)].reset_index(drop=True)
        d["person"] = ev.match_feet(d, tr) if len(d) else pd.Series(dtype=float)
        by_f = {f: g for f, g in d.groupby("frame")}
        for r in acts.itertuples():
            a = gt[(gt.frame == r.f) & (gt.player_id.astype(str) == str(int(r.player_id)))]
            t = tr_f.get(r.f)
            if a.empty or t is None:
                continue
            actor = int(a.track_id.iloc[0])
            g = by_f.get(r.f, d.iloc[:0])
            au, av = float(a.bx.iloc[0] + a.bw.iloc[0] / 2), float(a.by.iloc[0] + a.bh.iloc[0])
            dist = np.hypot(((g.x0 + g.x1) / 2).to_numpy() - au, g.y1.to_numpy() - av)
            hit = g[g.person == actor]
            kind = "detected" if len(hit) else ("taken" if len(dist) and (dist <= ev.FOOT_PX).any() else "missed")
            others = t[(t.person != actor) & (t.role == "player")]
            ctrl = float(others.person.isin(set(g.person.dropna())).mean()) if len(others) else np.nan
            row = {"unit": w, "match": w.split("_")[1], "frame": r.f, "player_id": int(r.player_id), "stage": s,
                   "kind": kind, "control": ctrl, "n det": len(g), "team": None, "track len": np.nan,
                   "len rank": np.nan, "same label": np.nan}
            if s == "S3 assigned" and len(hit):
                h = hit.iloc[0]
                same = g[g.team == h.team]
                lens = same.track_id.map(s3_len)
                row.update({"team": str(h.team), "track len": int(s3_len[h.track_id]),
                            "len rank": int((lens > s3_len[h.track_id]).sum() + 1), "same label": len(same)})
            out.append(row)
    return out


def main():
    from src.unseal import unsealed
    for m in sorted(SEALED):
        if not unsealed(m, "near_ball plan_v3: pipeline stages on the opened M7 / M9 clips"):
            raise SystemExit("VEO_UNSEAL needed")
    det_dir = Path(sys.argv[1])
    acts = pd.read_parquet(HERE / "actions.parquet")[["unit", "frame"]]
    lab = pd.concat([pd.read_csv(CLIPS / f"{w}_events.csv").assign(unit=w) for w in sorted(set(acts.unit))])
    acts = acts.merge(lab[["unit", "frame", "player_id"]].dropna(), on=["unit", "frame"]).drop_duplicates(["unit", "frame", "player_id"])
    acts["f"] = acts.frame - acts.frame % 2
    rows = []
    for i, (w, a) in enumerate(acts.groupby("unit")):
        rows += unit(w, pd.read_parquet(det_dir / f"{w}.parquet"), a)
        if i % 25 == 0:
            print(i, w, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(HERE / "stages.parquet")
    key = ["unit", "frame", "player_id"]
    wide = df.pivot_table(index=key, columns="stage", values="kind", aggfunc="first")[STAGES]
    n = len(wide)
    det = wide == "detected"
    per_stage = {s: {"detected": round(float(det[s].mean()), 4),
                     "taken": round(float((wide[s] == "taken").mean()), 4),
                     "missed": round(float((wide[s] == "missed").mean()), 4),
                     "control": round(float(df[df.stage == s].control.mean()), 4),
                     "detections per action frame": round(float(df[df.stage == s]["n det"].mean()), 2)} for s in STAGES}
    lost = det["S0 raw"] & ~det["S5 output"]
    first = {}
    for s_prev, s in zip(STAGES[:-1], STAGES[1:]):
        here = lost & det[STAGES[:STAGES.index(s)]].all(axis=1) & ~det[s]
        first[s] = round(float(here.sum() / n), 4)
    first["regained then lost (not monotone)"] = round(float(lost.sum() / n) - sum(first.values()), 4)
    s3 = df[df.stage == "S3 assigned"].set_index(key)
    cap_lost = det["S3 assigned"] & ~det["S4 roster cap"]
    cl = s3.loc[cap_lost[cap_lost].index]
    kept = s3.loc[(det["S3 assigned"] & det["S4 roster cap"])[lambda x: x].index]
    per_match = {}
    for m in sorted(df.match.unique()):
        idx = wide.index.get_level_values("unit").str.split("_").str[1] == m
        per_match[m] = {s: round(float(det[s][idx].mean()), 3) for s in STAGES}
    chk = pd.Series(CHECK)
    out = {"plan_v3 sha256": (HERE / "plan_v3.json.sha256").read_text().split()[0],
           "addendum1 sha256": (HERE / "plan_v3_addendum1.json.sha256").read_text().split()[0], "actions": n,
           "check: output's player rows found in S4": {"min": round(float(chk.min()), 4), "mean": round(float(chk.mean()), 4),
                                                       "units below 0.99": int((chk < 0.99).sum())},
           "per stage": per_stage,
           "raw has the actor, output does not (share of actions)": round(float(lost.sum() / n), 4),
           "first stage where it is lost (share of actions)": first,
           "lost at the roster cap": {
               "share of actions": round(float(cap_lost.sum() / n), 4),
               "actor's label at S3": cl.team.value_counts(normalize=True).round(3).to_dict(),
               "actor's track length (frames at stride 2), median": float(cl["track len"].median()),
               "same, actors kept by the cap": float(kept["track len"].median()),
               "actor's length rank among its label on the frame, median": float(cl["len rank"].median()),
               "same-label detections on the frame, median": float(cl["same label"].median())},
           "per match: detected": per_match}
    (HERE / "stages.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
