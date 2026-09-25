"""Chances created and assists: the pass that set up a shot.

These are the cheapest events this pipeline can add, because they are not
perception at all. A chance created is a pass followed by a shot from the
same team; an assist is that pass when the shot goes in. Everything needed
is already detected -- passes with a team and a receiver, shots with a time
and a position -- so this is a join, not a detector.

Being a join is also what makes it honest to build before the shot detector
can be validated on real footage. The rule can be checked on its own, against
labelled matches, without any of this project's perception being involved:
the SoccerNet ball-action labels carry PASS, SHOT, GOAL and the team for
each, so the same rule can run over them and be judged against what football
produces.

## The rule, and the two ways it goes wrong

Look back from a shot for the most recent pass by the same team. Two
mistakes are available and they pull in opposite directions.

Too permissive, and a pass three moves earlier gets the credit for a goal
its taker had nothing to do with. The guard is an opposition touch: if the
other team touched the ball between the pass and the shot, possession
turned over and the pass created nothing.

Too strict, and only the last action before the shot counts, which loses
every shot where the receiver took a touch first -- and a touch before
shooting is the normal case, not the exception. Measured on labelled
matches, requiring the pass to be *immediately* before the shot links only
32% of shots, where football assists roughly half to two thirds.

So: most recent same-team pass, within a time window, with no opposition
touch in between.
"""

from __future__ import annotations

# How far back a shot may reach for the pass that made it. Long enough for a
# touch and a stride, short enough that a pass from the previous phase of
# play cannot claim the credit.
#
# Chosen on one labelled match with the other never looked at, because
# tuning on a number and then quoting it is how a measurement stops meaning
# anything. On that match the assisted share runs 27% at zero touches
# whatever the window, then 55% at 3 s, 64% at 5 s and 68% at 8 s with one.
# Five seconds sits in the middle of the 45-75% football produces.
MAX_LOOKBACK_S = 5.0

# Actions that count as the ball being played to someone.
PASS_LIKE = ("pass", "cross", "high pass", "free kick", "corner", "throw in")

# Actions by the same team that may sit between the pass and the shot
# without breaking the link: the shooter controlling the ball.
OWN_TOUCHES = ("carry", "drive", "header", "take on", "dribble")


def _is_pass(label: str) -> bool:
    return str(label).strip().lower() in PASS_LIKE


def _is_own_touch(label: str) -> bool:
    return str(label).strip().lower() in OWN_TOUCHES


# How many touches of their own the shooter may take between receiving the
# pass and shooting. One is controlling the ball and the pass still made the
# chance; several is a dribble, and football credits the dribbler.
MAX_OWN_TOUCHES = 1


def link_chances(actions, shots, max_lookback_s: float = MAX_LOOKBACK_S,
                 max_own_touches: int = MAX_OWN_TOUCHES):
    """Attach the creating pass to each shot, where there is one.

    `actions` is every on-ball action in time order, each a dict with at
    least `timestamp_s`, `team` and a label under `event_type`. `shots` is
    the same shape, each also carrying `is_goal` where that is known.

    Returns a list, one entry per shot, of

        {"shot": shot, "pass": creating pass or None, "is_assist": bool}

    A shot with no creating pass is not a failure. Shots come from rebounds,
    interceptions, dribbles and set pieces taken directly, and football
    attributes no assist to those either.
    """
    ordered = sorted(actions, key=lambda a: float(a["timestamp_s"]))
    out = []
    for shot in sorted(shots, key=lambda s: float(s["timestamp_s"])):
        when = float(shot["timestamp_s"])
        team = shot.get("team")
        creator, blocked, touches = None, False, 0

        for action in reversed(ordered):
            at = float(action["timestamp_s"])
            if at >= when:
                continue
            if when - at > max_lookback_s:
                break
            label = action.get("event_type", "")
            other = action.get("team")

            # The other team touched it: possession turned over, and
            # anything before this belongs to a different phase of play.
            if team is not None and other is not None and other != team:
                blocked = True
                break
            if _is_pass(label):
                creator = action
                break
            if _is_own_touch(label):
                touches += 1
                if touches > max_own_touches:
                    break           # a dribble, not a pass being finished
                continue            # the shooter controlling it
            # Anything else by this team is not a pass and not a touch on
            # the way to one; keep looking, it may be noise.

        if blocked:
            creator = None
        out.append({"shot": shot, "pass": creator,
                    "is_assist": bool(creator) and bool(shot.get("is_goal"))})
    return out


def summarise(links):
    """Counts a coach would recognise."""
    chances = sum(1 for link in links if link["pass"])
    assists = sum(1 for link in links if link["is_assist"])
    return {"shots": len(links), "chances_created": chances,
            "assists": assists,
            "share_assisted": chances / len(links) if links else 0.0}


def per_team(links):
    """Chances created and assists, by the team that made them."""
    out: dict = {}
    for link in links:
        creator = link["pass"]
        if not creator:
            continue
        team = creator.get("team", "?")
        row = out.setdefault(team, {"chances_created": 0, "assists": 0})
        row["chances_created"] += 1
        if link["is_assist"]:
            row["assists"] += 1
    return out
