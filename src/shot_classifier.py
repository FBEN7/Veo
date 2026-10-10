"""Shots from the learned classifier, given a clip's candidate readings.

`train_shot_classifier.py` fits gradient boosted trees on the readings
`src/shot_features.py` takes around every candidate strike, with the hand-set
rules' verdict (`goal_plane.find_shots`) as two more inputs; `--save` writes
the model. This applies it: the probability of a shot at each candidate,
then the most likely candidates, at least `EVENT_GAP_S` apart, down to a
threshold.

The model is trained on footage under a non-commercial agreement, so like
every weight file here it is kept out of the repository and passed in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .goal_plane import MERGE_S, PITCH_WIDTH_M

# Detected events are at least this far apart: at two seconds the build-up
# before a shot and the clearance after it came out as events of their own
# (EVENT_ACCURACY.md, "A learned shot classifier").
EVENT_GAP_S = 4.0
# Held out by match on 36 windows, 0.5 found 30 of 44 attempts at 77%.
THRESHOLD = 0.5
RULE_KIND = {"attempt": 1.0, "off target": 2.0, "on target": 3.0,
             "goal": 4.0}


def rule_inputs(times, rule_shots):
    """(rule_dt, rule_kind) per candidate time: seconds from the nearest
    shot the rules found (NaN if none in the clip), and what they called it
    if within `MERGE_S`, else 0."""
    dt, kind = [], []
    for t in times:
        if not rule_shots:
            dt.append(np.nan)
            kind.append(0.0)
            continue
        when, outcome = min(rule_shots, key=lambda s: abs(t - s[0]))
        dt.append(float(t - when))
        kind.append(RULE_KIND.get(outcome, 0.0)
                    if abs(t - when) <= MERGE_S else 0.0)
    return dt, kind


def events(times, probs, threshold, gap=EVENT_GAP_S):
    """Indices of the peaks of `probs`, at least `gap` seconds apart,
    in time order."""
    times = np.asarray(times, dtype=float)
    kept = []
    for i in np.argsort(-np.asarray(probs)):
        if probs[i] < threshold:
            break
        if all(abs(times[i] - times[j]) >= gap for j in kept):
            kept.append(int(i))
    return sorted(kept, key=lambda i: times[i])


def predict(blob, table: pd.DataFrame, rule_shots, threshold=THRESHOLD):
    """Shots in one clip: a list of dicts with the candidate's frame, time,
    position and the model's probability, earliest first.

    `blob` is what `train_shot_classifier.py --save` wrote; `rule_shots` is
    [(time, outcome)] from the rules on the same ball and poses.
    """
    if table.empty:
        return []
    table = table.copy()
    if "rule_dt" in blob["features"]:
        table["rule_dt"], table["rule_kind"] = rule_inputs(
            table.time_s.tolist(), rule_shots)
    for c in blob["features"]:
        if c not in table.columns:
            table[c] = np.nan
    probs = blob["model"].predict_proba(table[blob["features"]])[:, 1]
    out = []
    for i in events(table.time_s.to_numpy(), probs, threshold):
        row = table.iloc[i]
        speed = row.get("ground_speed_s", np.nan)
        out.append({"frame": int(row.frame), "time_s": float(row.time_s),
                    "crossing_s": float(row.time_s),
                    "x": float(row.x_m),
                    "y": float(row.off_centre_m) + PITCH_WIDTH_M / 2.0,
                    "distance_m": float(row.distance_m),
                    "speed_ms": float(speed) if pd.notna(speed) else np.nan,
                    "goal": "left", "outcome": "attempt",
                    "across_m": np.nan, "height_m": np.nan,
                    "read": f"classifier, p={probs[i]:.2f}",
                    "probability": float(probs[i])})
    return out


def with_rules(found, rule_shots):
    """The classifier's shots, each replaced by the rules' reading of it
    where the rules found the same shot within `MERGE_S` -- that reading has
    the crossing, the outcome and the goal check -- plus every goal the rules
    confirmed that the classifier left out: a goal is confirmed by the
    kick-off after it, the strongest evidence there is."""
    out, used = [], set()
    for shot in found:
        near = [k for k, r in enumerate(rule_shots)
                if k not in used and abs(r["time_s"] - shot["time_s"])
                <= MERGE_S]
        if near:
            k = min(near, key=lambda k: abs(rule_shots[k]["time_s"]
                                             - shot["time_s"]))
            used.add(k)
            out.append({**rule_shots[k], "probability": shot["probability"],
                        "read": f"{rule_shots[k]['read']}; classifier "
                                f"p={shot['probability']:.2f}"})
        else:
            out.append(shot)
    for k, r in enumerate(rule_shots):
        if k not in used and r.get("outcome") == "goal":
            out.append({**r, "read": f"{r['read']}; kept as a goal, below "
                                     f"the classifier's threshold"})
    return sorted(out, key=lambda s: s["time_s"])
