"""What the wrong outcome calls look like at the moment possession changes.

`analyse_pass_outcome.py` shows the outcome field is independent of the truth
and that fixing team assignment would not repair it. That moves the fault into
possession transfer itself, and this looks at the transfers rather than the
calls: for every pass we emit that matches a labelled pass, it measures what
the pipeline saw when it decided who received the ball.

Split by the truth -- did the pass actually keep possession -- the question is
which of these measurements separates a correct call from a wrong one. A
quantity that looks the same on both sides is not the mechanism.

    python diagnose_transfer.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_pass_outcome import (UPLOADS, WINDOWS, TOLERANCE_S, match,
                                  truth_passes)
from score_soccernet import run_pipeline
from src import events as ev


_BALL_CACHE: dict[int, pd.DataFrame] = {}


def _ball_cache(metric: pd.DataFrame) -> pd.DataFrame:
    """The event detector's own ball timeline, computed once per window."""
    key = id(metric)
    if key not in _BALL_CACHE:
        _BALL_CACHE[key] = ev._ball_kinematics(metric)
    return _BALL_CACHE[key]


def transfer_detail(metric: pd.DataFrame, e: dict) -> dict | None:
    """What was true at the frame the receiver was credited.

    Rebuilt from the same tracks the event came from, using the same
    possession function, so the numbers describe the actual decision rather
    than an approximation of it.
    """
    end_t = float(e["timestamp_s"]) + float(e.get("pass_interval_s") or 0.0)
    receiver = int(e.get("receiver_track_id", -1))
    passer = int(e.get("player_track_id", -1))
    if receiver == -1:
        return None

    ball = _ball_cache(metric)
    if ball is None or ball.empty:
        return None

    players = metric[(metric.cls == "player")
                     & (metric.team.isin(["team_A", "team_B"]))]

    # The frame whose time is closest to when the ball arrived.
    row = ball.iloc[(ball.time_s - end_t).abs().argsort().iloc[0]]
    frame = int(row["frame"])
    bx, by = float(row["bx"]), float(row["by"])

    here = players[players.frame == frame]
    if here.empty:
        return None
    d = np.hypot(here.px - bx, here.py - by)

    rec = here[here.track_id == receiver]
    rec_dist = float(np.hypot(rec.px - bx, rec.py - by).iloc[0]) if len(rec) else np.nan

    passer_team = str(e.get("team"))
    same = here[here.team == passer_team]
    other = here[here.team != passer_team]
    d_same = float(np.hypot(same.px - bx, same.py - by).min()) if len(same) else np.nan
    d_other = float(np.hypot(other.px - bx, other.py - by).min()) if len(other) else np.nan

    return dict(
        receiver_dist_m=rec_dist,
        nearest_m=float(d.min()),
        margin_m=float(np.sort(d)[1] - np.sort(d)[0]) if len(d) > 1 else np.nan,
        nearest_same_m=d_same,
        nearest_other_m=d_other,
        n_within_8m=int((d <= 8.0).sum()),
        n_within_3m=int((d <= 3.0).sum()),
        pass_distance_m=float(e.get("pass_distance_m") or np.nan),
        pass_interval_s=float(e.get("pass_interval_s") or np.nan),
        inferred=bool(e.get("inferred_from_unknown_bridge")),
        passer=passer,
        receiver=receiver,
    )


def main():
    rows = []
    for name, out_dir, labels_name, offset in WINDOWS:
        clip = json.loads((Path(out_dir) / "clip.json").read_text())["path"]
        events, _, metric = run_pipeline(clip, Path(out_dir), return_tracks=True)

        ours = [e for e in events if e.get("event_type") == "pass"]
        truth = truth_passes(UPLOADS / labels_name, offset, True)
        pairs = match([e["timestamp_s"] for e in ours],
                      [t["t"] for t in truth], TOLERANCE_S)

        for pi, ti in pairs:
            det = transfer_detail(metric, ours[pi])
            if det is None:
                continue
            det.update(window=name, ours=ours[pi].get("outcome"),
                       truth=truth[ti]["outcome"])
            det["correct"] = det["ours"] == det["truth"]
            rows.append(det)

    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing matched")
        return
    df.to_csv("transfer_detail.csv", index=False)

    cols = ["receiver_dist_m", "nearest_m", "margin_m", "nearest_same_m",
            "nearest_other_m", "n_within_8m", "n_within_3m",
            "pass_distance_m", "pass_interval_s"]

    print(f"\n{len(df)} matched passes across {df.window.nunique()} windows\n")

    print("A. by what we said, split by what was true")
    print(df.groupby(["truth", "ours"]).size().to_string())

    print("\nB. medians -- passes that truly kept the ball, by our call")
    kept = df[df.truth == "success"]
    print(kept.groupby("ours")[cols].median().round(2).to_string())

    print("\nC. medians -- passes that truly lost the ball, by our call")
    lost = df[df.truth == "intercepted"]
    print(lost.groupby("ours")[cols].median().round(2).to_string())

    print("\nD. the wrong calls that matter most: truly kept, we said "
          f"intercepted (n={len(kept[kept.ours == 'intercepted'])})")
    wrong = kept[kept.ours == "intercepted"]
    right = kept[kept.ours == "success"]
    print(f"{'':22s} {'wrong':>8s} {'right':>8s}")
    for c in cols:
        print(f"  {c:20s} {wrong[c].median():8.2f} {right[c].median():8.2f}")

    print("\nE. how crowded the credit decision is")
    print(f"  median players within 8 m of the ball at the transfer: "
          f"{df.n_within_8m.median():.0f}")
    print(f"  median players within 3 m: {df.n_within_3m.median():.0f}")
    print(f"  receiver was the nearest player: "
          f"{(df.receiver_dist_m <= df.nearest_m + 1e-6).mean():.0%}")
    print(f"  a teammate of the passer was nearer than the receiver: "
          f"{(df.nearest_same_m < df.receiver_dist_m - 0.01).mean():.0%}")
    print(f"  receiver further than 3 m from the ball: "
          f"{(df.receiver_dist_m > 3.0).mean():.0%}")
    print(f"  receiver further than 5 m from the ball: "
          f"{(df.receiver_dist_m > 5.0).mean():.0%}")

    print("\nwritten transfer_detail.csv")


if __name__ == "__main__":
    main()
