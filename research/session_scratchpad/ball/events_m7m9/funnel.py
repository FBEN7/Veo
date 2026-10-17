"""Where pass / carry / shot labels are lost on M7 / M9 (descriptive): per label,
is the actor detected, and how far from the action's pitch position is (a) the
events' selected ball (tracks_merged of each run), (b) v3's top-1 and (c) any of
v3's top 4 (dense rows, spot marks dropped), within 2 frames."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402

NEAR_M, NEAR_F = 3.0, 2
GROUP = {"PASS": "pass", "HIGH PASS": "pass", "CROSS": "pass", "DRIVE": "carry", "SHOT": "shot", "GOAL": "shot"}
out = []
for w in ("st2_128057_1st_f030000", "st2_132831_2nd_f030000"):
    info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
    cal = st.calibration(info["match"])
    lab = pd.read_csv(DATA_DIR / "soccertrack_v2" / f"{w}_events.csv")
    lab = lab[lab.label.isin(GROUP) & lab.x.notna()]

    def on_pitch(df, u="px_raw", v="py_raw"):
        xy = st.image_to_pitch(np.c_[df[u], df[v] + df.crop_h / 2], cal)
        return df.assign(x_m=xy[:, 0], y_m=xy[:, 1])

    sel = {d: on_pitch(pd.read_parquet(f"output_{w}/{d}/tracks_merged.parquet").query("cls == 'ball'"))
           for d in ("events_v1", "ev_v2", "ev_v3m")}
    v3 = pd.read_parquet(f".cache/v3m_balls_{w}.parquet")
    ucol = "px_raw" if "px_raw" in v3 else "px"
    vcol = "py_raw" if "py_raw" in v3 else "py"
    v3 = on_pitch(v3, ucol, vcol)
    score = "confidence"
    v3 = v3.sort_values(["frame", score], ascending=[True, False])
    v3["rank"] = v3.groupby("frame").cumcount()
    for r in lab.itertuples():
        rec = {"window": w[4:10], "group": GROUP[r.label], "frame": int(r.frame)}
        for d, b in sel.items():
            b = b[(b.frame - r.frame).abs() <= NEAR_F]
            rec[d] = float(np.hypot(b.x_m - r.x, b.y_m - r.y).min()) if len(b) else np.inf
        c = v3[(v3.frame - r.frame).abs() <= NEAR_F]
        t1 = c[c["rank"] == 0]
        rec["v3 top-1"] = float(np.hypot(t1.x_m - r.x, t1.y_m - r.y).min()) if len(t1) else np.inf
        t4 = c[c["rank"] < 4]
        rec["v3 top-4"] = float(np.hypot(t4.x_m - r.x, t4.y_m - r.y).min()) if len(t4) else np.inf
        out.append(rec)
d = pd.DataFrame(out)
cols = ["events_v1", "ev_v2", "ev_v3m", "v3 top-1", "v3 top-4"]
near = d[cols].le(NEAR_M)
print(f"share of labels with a ball within {NEAR_M} m of the action (within {NEAR_F} frames):")
print(pd.concat([d[["window", "group"]], near], axis=1).groupby(["window", "group"]).mean().round(2)
      .assign(n=d.groupby(["window", "group"]).size()))
print(near.assign(group=d.group).groupby("group").mean().round(2))
print("v3 score column:", score, "| v3 columns:", list(v3.columns)[:14])

# The funnel for ev_v3m, per label: actor detected (as score_st2_actions), ball near
# (selected, and v3's top 4, within 2 and 12 frames), and an event of its group
# within 1 s.
from eval_soccertrack_v2 import boxes, match_feet, truth  # noqa: E402

rows = []
for w in ("st2_128057_1st_f030000", "st2_132831_2nd_f030000"):
    info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
    cal = st.calibration(info["match"])
    lab = pd.read_csv(DATA_DIR / "soccertrack_v2" / f"{w}_events.csv")
    lab["player_id"] = lab.player_id.astype("Int64").astype(str)
    lab = lab[lab.label.isin(GROUP) & lab.x.notna()]
    raw = pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{w}_gt.parquet")
    pid_of = raw.groupby("track_id").player_id.agg(
        lambda s: str(s.dropna().mode().iloc[0]) if s.notna().any() else None)
    gt = truth(DATA_DIR / "soccertrack_v2" / f"{w}_gt.parquet")
    run = f"output_{w}/ev_v3m"
    merged = pd.read_parquet(f"{run}/tracks_merged.parquet")
    gt_s = gt[gt.frame % 2 == 0]
    det = boxes(merged)
    det = det[det.frame % 2 == 0]
    det = det.assign(person=match_feet(det, gt_s))
    by_frame = {f: g for f, g in det.groupby("frame")}
    ball = merged.query("cls == 'ball'")
    xy = st.image_to_pitch(np.c_[ball.px_raw, ball.py_raw + ball.crop_h / 2], cal)
    ball = ball.assign(x_m=xy[:, 0], y_m=xy[:, 1])
    v3 = pd.read_parquet(f".cache/v3m_balls_{w}.parquet").sort_values(["frame", "confidence"], ascending=[True, False])
    v3 = v3[v3.groupby("frame").cumcount() < 4]
    xy = st.image_to_pitch(np.c_[v3.px, v3.py + v3.crop_h / 2], cal)
    v3 = v3.assign(x_m=xy[:, 0], y_m=xy[:, 1])
    ev = json.loads(open(f"{run}/events.json").read())
    for r in lab.itertuples():
        g = GROUP[r.label]
        near = [by_frame[f] for f in range(r.frame - 2, r.frame + 3) if f in by_frame]
        who = {pid_of.get(p) for x in near for p in x.person.dropna()}

        def close(b, nf):
            b = b[(b.frame - r.frame).abs() <= nf]
            return bool(len(b)) and bool((np.hypot(b.x_m - r.x, b.y_m - r.y) <= NEAR_M).any())
        rows.append({"window": w[4:10], "group": g, "actor detected": r.player_id in who,
                     "selected ball near": close(ball, 2), "selected ball near (0.5 s)": close(ball, 12),
                     "v3 top-4 near": close(v3, 2), "v3 top-4 near (0.5 s)": close(v3, 12),
                     "event of its type within 1 s": any(
                         e["event_type"] in ({"pass"} if g == "pass" else {"carry"} if g == "carry" else {"shot", "goal"})
                         and abs(e["timestamp_s"] - r.frame / 25) <= 1.0 for e in ev)})
f = pd.DataFrame(rows)
f["actor + selected ball"] = f["actor detected"] & f["selected ball near"]
print("\nfunnel (ev_v3m), share of labels:")
print(f.drop(columns="window").groupby("group").mean().round(2).T.assign(all=f.drop(columns=["window", "group"]).mean().round(2)))
both = f[f["actor + selected ball"]]
print("\nevent found within 1 s | actor + ball there:", round(both["event of its type within 1 s"].mean(), 2), f"({len(both)} labels)")
nob = f[~f["actor + selected ball"]]
print("event found within 1 s | not both:", round(nob["event of its type within 1 s"].mean(), 2), f"({len(nob)} labels)")
