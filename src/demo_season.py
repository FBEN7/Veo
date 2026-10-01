"""Deterministic fictional match data for the Riverside Athletic dashboard demo."""

from __future__ import annotations

import random
from datetime import date


CLUB = "Riverside Athletic"

ROSTER = [
    {"track_id": 1, "jersey_number": None, "name": "Pascal Lecoq", "position": "Trainer", "positions": ["Trainer"], "photo_slug": "pascal-lecoq", "pace": 5.1, "speed": 27.2},
    {"track_id": 2, "jersey_number": 1, "name": "Mathieu Minet", "position": "Goalkeeper", "positions": ["Goalkeeper"], "photo_slug": "mathieu-minet", "pace": 5.1, "speed": 27.2},
    {"track_id": 3, "jersey_number": 10, "name": "Charles Linden", "position": "Forward", "positions": ["Forward", "Centre forward"], "photo_slug": "charles-linden", "pace": 9.4, "speed": 29.8},
    {"track_id": 4, "jersey_number": 88, "name": "Benjamin Vermaut", "position": "Defender", "positions": ["Defender", "Centre back"], "photo_slug": "benjamin-vermaut", "pace": 9.7, "speed": 30.2},
    {"track_id": 5, "jersey_number": 30, "name": "Baudry Roisin", "position": "Goalkeeper", "positions": ["Goalkeeper"], "photo_slug": "baudry-roisin", "pace": 5.0, "speed": 26.8},
    {"track_id": 6, "jersey_number": 8, "name": "Dimitri Vagenende", "position": "Midfielder", "positions": ["Midfielder", "Centre midfielder"], "photo_slug": "dimitri-vagenende", "pace": 10.8, "speed": 31.0},
    {"track_id": 7, "jersey_number": 20, "name": "Alex Juprelle", "position": "Defender", "positions": ["Defender", "Right back"], "photo_slug": "alex-juprelle", "pace": 10.1, "speed": 31.5},
    {"track_id": 8, "jersey_number": 17, "name": "François Bernard", "position": "Midfielder", "positions": ["Midfielder", "Attacking midfielder"], "photo_slug": "francois-bernard", "pace": 11.0, "speed": 32.1},
    {"track_id": 9, "jersey_number": 9, "name": "Clément Piette", "position": "Forward", "positions": ["Forward", "Centre forward"], "photo_slug": "clement-piette", "pace": 9.1, "speed": 32.8},
    {"track_id": 10, "jersey_number": 7, "name": "David Durieux", "position": "Midfielder", "positions": ["Midfielder", "Centre midfielder"], "photo_slug": "david-durieux", "pace": 11.2, "speed": 32.4},
    {"track_id": 11, "jersey_number": 12, "name": "Doan Vu-Duc", "position": "Player", "positions": ["Player", "Midfielder"], "photo_slug": "doan-vu-duc", "pace": 10.4, "speed": 33.4},
]

FIXTURES = [
    {"date": "2026-08-02", "opponent": "Northbridge FC", "score": (2, 1), "opponent_score": (1, 0), "scorers": [(10, 7, 22), (9, 8, 71)], "possession": 0.54},
    {"date": "2026-08-09", "opponent": "Westhaven Town", "score": (1, 1), "opponent_score": (1, 1), "scorers": [(11, 9, 64)], "possession": 0.49},
    {"date": "2026-08-16", "opponent": "Ashford United", "score": (0, 1), "opponent_score": (1, 0), "scorers": [], "possession": 0.57},
    {"date": "2026-08-23", "opponent": "Kingsley Rovers", "score": (3, 0), "opponent_score": (0, 0), "scorers": [(10, 7, 18), (11, 8, 52), (7, 6, 78)], "possession": 0.58},
    {"date": "2026-08-30", "opponent": "Millbrook Athletic", "score": (2, 0), "opponent_score": (0, 0), "scorers": [(9, 4, 35), (10, 11, 83)], "possession": 0.52},
    {"date": "2026-09-06", "opponent": "Dalesford FC", "score": (1, 2), "opponent_score": (2, 1), "scorers": [(9, 8, 68)], "possession": 0.46},
    {"date": "2026-09-20", "opponent": "Redcliffe City", "score": (2, 1), "opponent_score": (1, 0), "scorers": [(9, 10, 31), (10, 11, 76)], "possession": 0.51},
    {"date": "2026-09-27", "opponent": "Eastborough FC", "score": (2, 2), "opponent_score": (2, 1), "scorers": [(11, 7, 14), (10, None, 89)], "possession": 0.55},
]


def _lineup(fixture_index: int) -> list[dict]:
    goalkeeper = ROSTER[1] if fixture_index % 3 else ROSTER[4]
    outfield = [player for player in ROSTER if player["position"] not in {"Goalkeeper", "Trainer"}]
    rotated = outfield[fixture_index % len(outfield):] + outfield[:fixture_index % len(outfield)]
    lineup = [goalkeeper, *rotated[:14]]
    required_ids = {
        contributor_id
        for scorer_id, assister_id, _ in FIXTURES[fixture_index]["scorers"]
        for contributor_id in (scorer_id, assister_id)
        if contributor_id is not None
    }
    for required_id in sorted(required_ids):
        if any(player["track_id"] == required_id for player in lineup):
            continue
        replacement = next(
            (index for index in range(len(lineup) - 1, 0, -1)
             if lineup[index]["track_id"] not in required_ids),
            None,
        )
        if replacement is not None:
            lineup[replacement] = next(player for player in outfield if player["track_id"] == required_id)
    return lineup


def _player_line(player: dict, fixture: dict, fixture_index: int, roster_index: int) -> dict:
    rng = random.Random(3100 + fixture_index * 67 + player["track_id"] * 13)
    position = player["position"]
    minutes = rng.randint(66, 90) if roster_index < 11 else rng.randint(18, 34)
    minutes = 90 if position == "Goalkeeper" and roster_index == 0 else minutes
    distance = round(player["pace"] * minutes / 90 * rng.uniform(0.92, 1.08) * 1000, 0)

    if position == "Goalkeeper":
        attempted = rng.randint(20, 34)
        shots = rng.randint(0, 1)
        tackles = rng.randint(0, 1)
    elif "back" in position or position == "Centre back":
        attempted = rng.randint(24, 48)
        shots = rng.choices([0, 1, 2], weights=[82, 16, 2])[0]
        tackles = rng.randint(2, 6)
    elif "midfielder" in position:
        attempted = rng.randint(35, 67)
        shots = rng.choices([0, 1, 2, 3], weights=[36, 42, 18, 4])[0]
        tackles = rng.randint(1, 5)
    else:
        attempted = rng.randint(17, 39)
        shots = rng.choices([0, 1, 2, 3, 4], weights=[20, 38, 26, 13, 3])[0]
        tackles = rng.randint(0, 3)

    passes_completed = round(attempted * rng.uniform(0.77, 0.94))
    scorer_data = [row for row in fixture["scorers"] if row[0] == player["track_id"]]
    assists = sum(1 for row in fixture["scorers"] if row[1] == player["track_id"])
    goals = len(scorer_data)
    shots = max(shots, goals)
    xg = sum(rng.uniform(0.2, 0.52) for _ in scorer_data) + sum(rng.uniform(0.025, 0.21) for _ in range(shots - goals))
    position_options = player["positions"]
    played_position = position_options[fixture_index % len(position_options)]
    team_possession = fixture["possession"]
    individual_possession = round(rng.uniform(0.025, 0.105) * minutes / 90, 3)
    rating = 6.15 + goals * 0.92 + assists * 0.55 + (passes_completed / max(attempted, 1) - 0.8) * 2 + rng.uniform(-0.52, 0.5)

    return {
        "track_id": player["track_id"],
        "name": player["name"],
        "jersey_number": player.get("jersey_number") or player["track_id"],
        "photo_url": f"/static/players/{player['photo_slug']}.jpg",
        "team": CLUB,
        "position": played_position,
        "preferred_position": player["position"],
        "minutes_tracked": minutes,
        "distance_m": distance,
        "top_speed_kmh": round(rng.uniform(player["speed"] - 2.4, player["speed"]), 1),
        "n_sprints": rng.randint(7, 25) if position != "Goalkeeper" else rng.randint(1, 4),
        "n_passes": passes_completed,
        "passes_completed": passes_completed,
        "passes_attempted": attempted,
        "pass_completion_rate": round(passes_completed / attempted, 3),
        "n_shots": shots,
        "n_goals": goals,
        "goals": goals,
        "assists": assists,
        "xg": round(xg, 2),
        "possession_pct": individual_possession,
        "tackles": tackles,
        "interceptions": rng.randint(0, 4) if position != "Goalkeeper" else 0,
        "carries": rng.randint(8, 27) if position != "Goalkeeper" else rng.randint(0, 3),
        "carry_distance_m": round(rng.uniform(180, 620) * minutes / 90) if position != "Goalkeeper" else round(rng.uniform(0, 35)),
        "rating": round(min(9.5, max(5.7, rating)), 1),
        "_team_possession": team_possession,
    }


def _opponent_stats(fixture: dict, rng: random.Random) -> dict:
    return {
        "team": fixture["opponent"],
        "possession_pct": round(1 - fixture["possession"], 3),
        "total_distance_km": round(rng.uniform(105, 121), 1),
        "n_passes": rng.randint(330, 485),
        "passes_completed": rng.randint(275, 415),
        "passes_attempted": rng.randint(350, 505),
        "pass_completion_rate": round(rng.uniform(0.75, 0.87), 3),
        "n_shots": rng.randint(7, 15),
        "n_goals": fixture["score"][1],
        "xg": round(rng.uniform(0.55, 1.95), 2),
        "total_sprints": rng.randint(115, 190),
        "tackles": rng.randint(12, 23),
        "assists": fixture["score"][1],
        "carries": rng.randint(160, 250),
        "carry_distance_m": rng.randint(4600, 7900),
    }


def build_demo_season() -> dict:
    """Build eight fictional fixtures with at most fifteen Riverside players each."""
    matches = []
    reports = {}

    for index, fixture in enumerate(FIXTURES):
        match_id = index + 1
        rng = random.Random(8700 + index)
        lineup = _lineup(index)
        players = [_player_line(player, fixture, index, squad_index) for squad_index, player in enumerate(lineup)]
        goals_for, goals_against = fixture["score"]
        team_distance = round(sum(player["distance_m"] for player in players) / 1000, 1)
        passes_attempted = sum(player["passes_attempted"] for player in players)
        passes_completed = sum(player["passes_completed"] for player in players)
        club_xg = round(sum(player["xg"] for player in players), 2)
        result = "W" if goals_for > goals_against else "L" if goals_for < goals_against else "D"
        opponent = fixture["opponent"]
        label = f"{CLUB} vs {opponent}"

        club_stats = {
            "team": CLUB,
            "possession_pct": fixture["possession"],
            "total_distance_km": team_distance,
            "total_sprints": sum(player["n_sprints"] for player in players),
            "n_passes": passes_completed,
            "passes_completed": passes_completed,
            "passes_attempted": passes_attempted,
            "pass_completion_rate": round(passes_completed / passes_attempted, 3),
            "n_shots": sum(player["n_shots"] for player in players),
            "n_goals": goals_for,
            "xg": club_xg,
            "tackles": sum(player["tackles"] for player in players),
            "assists": sum(player["assists"] for player in players),
            "carries": sum(player["carries"] for player in players),
            "carry_distance_m": sum(player["carry_distance_m"] for player in players),
        }
        away_stats = _opponent_stats(fixture, rng)

        goal_events = []
        for scorer_id, assist_id, minute in fixture["scorers"]:
            scorer = next(player for player in ROSTER if player["track_id"] == scorer_id)
            assister = next((player for player in ROSTER if player["track_id"] == assist_id), None)
            goal_events.append({
                "event_type": "goal",
                "timestamp_s": minute * 60,
                "team": CLUB,
                "player_track_id": scorer_id,
                "player_name": scorer["name"],
                "assist_track_id": assist_id,
                "assist_name": assister["name"] if assister else None,
                "minute": minute,
            })
        for goal_number in range(goals_against):
            minute = 27 + goal_number * 31 + index % 7
            goal_events.append({
                "event_type": "goal",
                "timestamp_s": minute * 60,
                "team": opponent,
                "player_track_id": None,
                "player_name": f"{['Callum Ward', 'Mason Ellis', 'Theo James'][goal_number % 3]}",
                "assist_track_id": None,
                "assist_name": None,
                "minute": minute,
            })
        goal_events.sort(key=lambda event: event["timestamp_s"])
        running_score = {CLUB: 0, opponent: 0}
        for event in goal_events:
            running_score[event["team"]] += 1
            event["score_at_goal"] = f"{running_score[CLUB]}–{running_score[opponent]}"

        events = {
            "goal": goals_for + goals_against,
            "shot": club_stats["n_shots"] + away_stats["n_shots"],
            "pass": passes_attempted + away_stats["passes_attempted"],
            "tackle": club_stats["tackles"] + away_stats["tackles"],
        }
        match = {
            "id": match_id,
            "label": label,
            "date": fixture["date"],
            "duration_s": 90 * 60,
            "home_team": CLUB,
            "away_team": opponent,
            "home_score": goals_for,
            "away_score": goals_against,
            "result": result,
            "demo": True,
        }
        matches.append(match)
        reports[match_id] = {
            "team_stats": [club_stats, away_stats],
            "events": events,
            "players": players,
            "goals": goal_events,
            "scoreline": {"for": goals_for, "against": goals_against},
            "demo": True,
        }

    return {
        "club_team": CLUB,
        "season": "2026/27",
        "demo": True,
        "data_notice": "Fictional demonstration data, created to preview the club dashboard.",
        "matches": matches,
        "reports": reports,
        "roster_size": len(ROSTER),
        "players_per_match_max": max(len(report["players"]) for report in reports.values()),
        "generated_on": date.today().isoformat(),
    }