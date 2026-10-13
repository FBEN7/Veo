"""One row per action: the event format events, labels and reports share.

The fields follow SPADL (Decroos et al., "Actions speak louder than goals",
KDD 2019; the `socceraction` library), the format the analytics literature
converts Opta, StatsBomb and Wyscout events into, extended with what video
needs: the frames an action starts and ends on, and the receiver of a
pass, as StatsBomb's open data records it.

- `type`: pass, carry, shot, tackle, recovery (an interception or a loose
  ball won), clearance, goal ... A carry is its own action between a
  reception and the player's next action (StatsBomb's convention), so a
  pass is never also a carry.
- `start_frame` / `end_frame`: the touch that starts the action and the
  moment it ends -- for a pass, the release and the reception; for a
  carry, the reception and the next touch away; for a shot, the strike and
  where the ball stops or crosses the line.
- `player_from` / `player_to`: identities (`src/player_identity.py`), not
  tracker ids; `player_to` only for passes, the player who received it.
- `start_x`, `start_y`, `end_x`, `end_y`: pitch metres where known.
- `length_m`: from start to end; empty for tackles and recoveries.
- `result`: success or fail (a pass reaching a team-mate, a shot on
  target ...).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

TYPES = ("pass", "carry", "shot", "tackle", "recovery", "clearance",
         "goal", "out")
NO_LENGTH = {"tackle", "recovery", "goal", "out"}


@dataclass
class Action:
    type: str
    start_frame: int
    end_frame: int
    team: str | None = None
    player_from: str | None = None
    player_to: str | None = None
    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    result: str | None = None
    fps: float = 25.0

    @property
    def start_s(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_s(self) -> float:
        return self.end_frame / self.fps

    @property
    def length_m(self) -> float | None:
        if self.type in NO_LENGTH or None in (self.start_x, self.start_y,
                                              self.end_x, self.end_y):
            return None
        return math.hypot(self.end_x - self.start_x,
                          self.end_y - self.start_y)

    def row(self) -> dict:
        out = asdict(self)
        out.update(start_s=round(self.start_s, 2), end_s=round(self.end_s, 2),
                   length_m=(None if self.length_m is None
                             else round(self.length_m, 1)))
        return out


def from_pipeline(event: dict, fps: float, who=None) -> Action:
    """An `events.detect_events` event in this format. `who(track_id)`
    names the identity of a track; without it the track id is kept.

    The pipeline stamps one time per event, so start and end are that frame
    and, for passes and carries, the frame `pass_interval_s` /
    `carry_duration_s` later.
    """
    name = who or (lambda tid: None if tid is None else f"track {tid}")
    start = int(round(float(event["timestamp_s"]) * fps))
    span = float(event.get("pass_interval_s")
                 or event.get("carry_duration_s") or 0.0)
    kind = event["event_type"]
    return Action(
        type=kind, start_frame=start, end_frame=start + int(round(span * fps)),
        team=event.get("team"), player_from=name(event.get("player_track_id")),
        player_to=(name(event.get("receiver_track_id"))
                   if kind == "pass" else None),
        start_x=event.get("location_x"), start_y=event.get("location_y"),
        end_x=event.get("end_location_x"), end_y=event.get("end_location_y"),
        result=event.get("outcome"), fps=fps)
