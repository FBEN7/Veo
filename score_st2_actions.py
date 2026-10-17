"""Score a SoccerTrack v2 window's detected events (`events.json`, from
`eval_soccertrack_v2.py --run --calibrated`) against its ball actions
(BAS, `<window>_events.csv` from `fetch_soccertrack_v2.py`).

Per group, the labels and our events are matched one to one, nearest
first, within a tolerance (0.5 s by default, also 1 and 2 s: actions are
1.4 s apart at the median, 0.8 s at the 10th percentile):

    pass    PASS, HIGH PASS, CROSS            <- pass
    carry   DRIVE                             <- carry
    won     PLAYER SUCCESSFUL TACKLE, BLOCK   <- tackle, recovery, interception
    shot    SHOT, GOAL                        <- shot, goal
    out     OUT                               <- out_of_play

THROW IN, FREE KICK and HEADER have no detector here and are not scored.
For each group: labels, our events, matches, precision, recall, F1, the F1
the same number of events would get at random times (`chance`, mean of
2000 draws) and how often chance does as well (`p`).

For matched events, the team and the actor:

- team: our team_A / team_B mapped to the sides once per window from the
  detections matched to outfield players (as `eval_soccertrack_v2.
  team_scores` does), so the mapping does not lean on the events scored;
- actor: our event's track, at the event's frame (nearest sampled frame
  within 2), matched by its feet to a ground-truth person among all that
  frame's detections, compared with the action's `player_id`.

And two ceilings per window, on the labels: the actor is detected (a
detection matched to them within 2 frames), and the ball is there (the
selected ball within 3 m of the actor's ground-truth position within 2
frames) -- without a ball no ball event can be found.

    python score_st2_actions.py st2_117093_1st_f015000 ... [--dir NAME]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from eval_soccertrack_v2 import boxes, match_feet, truth
from score_soccernet import match_one_to_one
from src.paths import DATA_DIR

FPS = 25
GROUPS = {
    "pass": ({"PASS", "HIGH PASS", "CROSS"}, {"pass"}),
    "carry": ({"DRIVE"}, {"carry"}),
    "won": ({"PLAYER SUCCESSFUL TACKLE", "BALL PLAYER BLOCK"},
            {"tackle", "recovery", "interception"}),
    "shot": ({"SHOT", "GOAL"}, {"shot", "goal"}),
    "out": ({"OUT"}, {"out_of_play"}),
}
TOLERANCES_S = (0.5, 1.0, 2.0)
NEAR_FRAMES = 2
BALL_NEAR_M = 3.0


def side_of_team(det: pd.DataFrame, person: pd.Series,
                 gt: pd.DataFrame) -> dict:
    """team_A / team_B -> left / right, from detections matched to outfield
    players."""
    side = gt.groupby("person").side.agg(lambda s: s.mode().iloc[0])
    role = gt.groupby("person").role.agg(lambda s: s.mode().iloc[0])
    df = pd.DataFrame({"team": det.team, "person": person}).dropna()
    df = df[df.team.isin(("team_A", "team_B"))
            & (df.person.map(role) == "player")]
    if df.empty:
        return {}
    table = pd.crosstab(df.team, df.person.map(side))
    r, c = linear_sum_assignment(-table.to_numpy())
    return {table.index[i]: table.columns[j] for i, j in zip(r, c)}


def actor_of(event: dict, det: pd.DataFrame, person: pd.Series):
    """The ground-truth person our event's track is matched to at the
    event's frame (nearest sampled frame within NEAR_FRAMES), or None."""
    tid = event.get("player_track_id")
    if tid is None or tid == -1:
        return None
    f = int(round(event["timestamp_s"] * FPS))
    rows = det[det.track_id == tid]
    rows = rows[(rows.frame - f).abs() <= NEAR_FRAMES]
    if rows.empty:
        return None
    k = (rows.frame - f).abs().idxmin()
    p = person.get(k)
    return None if pd.isna(p) else int(p)


def chance(n_pred: int, truths, tol: float, duration: float, rng,
           draws: int = 2000) -> np.ndarray:
    out = np.empty(draws)
    for k in range(draws):
        rand = sorted(rng.uniform(0, duration, n_pred))
        m, _, _ = match_one_to_one(rand, truths, tol)
        p, r = len(m) / max(n_pred, 1), len(m) / max(len(truths), 1)
        out[k] = 2 * p * r / (p + r) if p + r else 0.0
    return out


def score_window(window: str, sub: str | None = None,
                 seed: int = 1) -> dict:
    out_dir = Path(f"output_{window}") / (sub or "")
    events = json.loads((out_dir / "events.json").read_text())
    labels = pd.read_csv(DATA_DIR / "soccertrack_v2" / f"{window}_events.csv")
    labels["player_id"] = labels.player_id.astype("Int64").astype(str)
    raw = pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{window}_gt.parquet")
    pid_of = raw.groupby("track_id").player_id.agg(
        lambda s: str(s.dropna().mode().iloc[0]) if s.notna().any() else None)
    gt = truth(DATA_DIR / "soccertrack_v2" / f"{window}_gt.parquet")
    merged = pd.read_parquet(out_dir / "tracks_merged.parquet")
    stride = int(np.gcd.reduce(merged.frame.unique().astype(int))) or 1
    gt_s = gt[gt.frame % stride == 0]
    det = boxes(merged)
    det = det[det.frame % stride == 0]
    person = match_feet(det, gt_s)
    sides = side_of_team(det, person, gt_s)
    duration = (gt.frame.max() + 1) / FPS
    rng = np.random.default_rng(seed)
    report = {"window": window, "labels": int(len(labels)),
              "events": int(len(events)), "team map": sides}

    # Ceilings on the labels: actor detected, ball near the actor.
    by_frame = {f: g for f, g in det.assign(person=person).groupby("frame")}
    from src import soccertrack_v2 as st

    match = json.loads((DATA_DIR / "soccertrack_v2" / f"{window}.json")
                       .read_text())["match"]
    ball = merged[merged.cls == "ball"].copy()
    xy = st.image_to_pitch(np.c_[ball.px_raw, ball.py_raw + ball.crop_h / 2],
                           st.calibration(match))
    ball["x_m"], ball["y_m"] = xy[:, 0], xy[:, 1]
    actor_seen = ball_there = 0
    for lab in labels.itertuples():
        near = [by_frame[f] for f in range(lab.frame - NEAR_FRAMES,
                                           lab.frame + NEAR_FRAMES + 1)
                if f in by_frame]
        who = {pid_of.get(p) for g in near for p in g.person.dropna()}
        actor_seen += lab.player_id in who
        b = ball[(ball.frame - lab.frame).abs() <= NEAR_FRAMES]
        if len(b) and pd.notna(lab.x):
            d = np.hypot(b.x_m - lab.x, b.y_m - lab.y)
            ball_there += bool((d <= BALL_NEAR_M).any())
    report["actor detected"] = round(actor_seen / max(len(labels), 1), 3)
    report["ball near actor"] = round(ball_there / max(len(labels), 1), 3)

    groups = {}
    for g, (bas, ours) in GROUPS.items():
        lab = labels[labels.label.isin(bas)].reset_index(drop=True)
        ev = [e for e in events if e["event_type"] in ours]
        truths = list(lab.frame / FPS)
        preds = [e["timestamp_s"] for e in ev]
        row = {"labels": len(lab), "ours": len(ev)}
        for tol in TOLERANCES_S:
            m, _, _ = match_one_to_one(preds, truths, tol)
            p = len(m) / len(preds) if preds else 0.0
            r = len(m) / len(truths) if truths else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            key = f"{tol:g}s"
            row[key] = {"matched": len(m), "P": round(p, 3), "R": round(r, 3),
                        "F1": round(f1, 3)}
            if preds and truths:
                c = chance(len(preds), truths, tol, duration, rng)
                row[key]["chance F1"] = round(float(c.mean()), 3)
                row[key]["p"] = round(float((c >= f1).mean()), 3)
            if tol == TOLERANCES_S[0] and m:
                team_ok = team_n = actor_ok = actor_n = 0
                for pi, ti, _ in m:
                    e, l = ev[pi], lab.iloc[ti]
                    if e.get("team") in sides:
                        team_n += 1
                        team_ok += sides[e["team"]] == l.side
                    a = actor_of(e, det, person)
                    if a is not None:
                        actor_n += 1
                        actor_ok += pid_of.get(a) == l.player_id
                row[key].update({
                    "team right": f"{team_ok}/{team_n}",
                    "actor right": f"{actor_ok}/{actor_n}"})
        groups[g] = row
    report["groups"] = groups
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows", nargs="+")
    ap.add_argument("--dir", help="subdirectory of output_<window> holding "
                                  "the run (events.json, tracks_merged)")
    args = ap.parse_args()
    for w in args.windows:
        print(json.dumps(score_window(w, args.dir)), flush=True)


if __name__ == "__main__":
    main()
