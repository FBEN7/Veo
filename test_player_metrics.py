"""Per-player ball metrics credit the right player.

Interceptions are recorded as the failed pass's outcome, with the
interceptor in `intercepted_by_track_id`. Counting them compared that field
against a name that was never defined, so any match with an intercepted
pass raised NameError while building the report. The synthetic end-to-end
run has no interceptions, which is how it went unnoticed.

Runs standalone and exits non-zero on failure.
"""

import sys

import pandas as pd

from src.advanced_metrics import compute_ball_interaction_metrics


def main() -> int:
    tracks = pd.DataFrame({"cls": ["player"] * 3,
                           "team": ["team_A", "team_B", "team_B"],
                           "track_id": [1, 2, 3]})
    events = [
        {"event_type": "pass", "player_track_id": 1,
         "outcome": "intercepted", "intercepted_by_track_id": 2},
        {"event_type": "pass", "player_track_id": 1, "outcome": "completed"},
        {"event_type": "shot", "player_track_id": 3, "outcome": "on_target"},
    ]
    got = compute_ball_interaction_metrics(events, tracks)
    checks = [
        ("the interceptor is credited", got[2]["interceptions"], 1),
        ("the passer is not", got[1]["interceptions"], 0),
        ("nobody else is", got[3]["interceptions"], 0),
        ("touches count passes", got[1]["touches"], 2),
        ("shots on target", got[3]["shots_on_target"], 1),
    ]
    ok = True
    for name, value, want in checks:
        good = value == want
        ok &= good
        print(f"   {name:<28s} {value} (want {want})  "
              f"{'ok' if good else 'WRONG'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
