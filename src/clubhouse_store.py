"""Persistent club identity, fixture, and league-result storage."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator


_SCHEMA = """
CREATE TABLE IF NOT EXISTS club_settings (
    setting_key TEXT PRIMARY KEY,
    setting_value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS club_fixtures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_date TEXT NOT NULL,
    kickoff TEXT NOT NULL DEFAULT '',
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    venue TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (match_date, home_team, away_team)
);

CREATE TABLE IF NOT EXISTS league_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_score INTEGER NOT NULL CHECK (home_score >= 0),
    away_score INTEGER NOT NULL CHECK (away_score >= 0),
    venue TEXT NOT NULL DEFAULT '',
    reported_by TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (match_date, home_team, away_team),
    CHECK (home_team <> away_team)
);
"""


def _connect(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _connection(database_path: str | Path) -> Iterator[sqlite3.Connection]:
    connection = _connect(database_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_clubhouse_store(database_path: str | Path) -> None:
    with _connection(database_path) as connection:
        connection.executescript(_SCHEMA)
        connection.execute(
            "INSERT OR IGNORE INTO club_settings (setting_key, setting_value) VALUES ('club_name', 'Riverside Athletic')"
        )


def get_clubhouse_data(database_path: str | Path) -> dict:
    init_clubhouse_store(database_path)
    today = date.today().isoformat()
    with _connection(database_path) as connection:
        settings = dict(connection.execute("SELECT setting_key, setting_value FROM club_settings"))
        fixtures = connection.execute(
            """SELECT f.id, f.match_date, f.kickoff, f.home_team, f.away_team, f.venue
               FROM club_fixtures AS f
               WHERE f.match_date >= ?
                 AND NOT EXISTS (
                     SELECT 1 FROM league_results AS r
                     WHERE r.match_date = f.match_date
                       AND r.home_team = f.home_team AND r.away_team = f.away_team
                 )
               ORDER BY f.match_date, f.kickoff, f.id""",
            (today,),
        ).fetchall()
        results = connection.execute(
            """SELECT id, match_date, home_team, away_team, home_score, away_score,
                      venue, reported_by, updated_at
             FROM league_results ORDER BY match_date DESC, id DESC"""
        ).fetchall()

    club_name = settings.get("club_name", "Riverside Athletic")
    totals: dict[str, dict[str, int | str]] = {}

    def ensure_team(team: str) -> dict:
        return totals.setdefault(team, {
            "team": team, "played": 0, "won": 0, "drawn": 0, "lost": 0,
            "goals_for": 0, "goals_against": 0, "goal_difference": 0, "points": 0,
        })

    ensure_team(club_name)
    for result in results:
        home = ensure_team(result["home_team"])
        away = ensure_team(result["away_team"])
        home_score, away_score = result["home_score"], result["away_score"]
        for team, scored, conceded in ((home, home_score, away_score), (away, away_score, home_score)):
            team["played"] += 1
            team["goals_for"] += scored
            team["goals_against"] += conceded
            team["goal_difference"] = team["goals_for"] - team["goals_against"]
        if home_score > away_score:
            home["won"] += 1
            away["lost"] += 1
            home["points"] += 3
        elif home_score < away_score:
            away["won"] += 1
            home["lost"] += 1
            away["points"] += 3
        else:
            home["drawn"] += 1
            away["drawn"] += 1
            home["points"] += 1
            away["points"] += 1

    table = sorted(
        totals.values(),
        key=lambda row: (-row["points"], -row["goal_difference"], -row["goals_for"], row["team"].casefold()),
    )
    for position, team in enumerate(table, start=1):
        team["position"] = position
    return {
        "club_name": club_name,
        "fixtures": [dict(fixture) for fixture in fixtures],
        "results": [dict(result) for result in results],
        "table": table,
    }


def save_club_name(database_path: str | Path, club_name: str) -> None:
    init_clubhouse_store(database_path)
    with _connection(database_path) as connection:
        current = connection.execute(
            "SELECT setting_value FROM club_settings WHERE setting_key='club_name'"
        ).fetchone()
        previous_name = current["setting_value"] if current else "Riverside Athletic"
        connection.execute(
            "INSERT INTO club_settings (setting_key, setting_value) VALUES ('club_name', ?) "
            "ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value",
            (club_name,),
        )
        if previous_name != club_name:
            for table in ("club_fixtures", "league_results"):
                connection.execute(
                    f"UPDATE {table} SET home_team=? WHERE home_team=?",
                    (club_name, previous_name),
                )
                connection.execute(
                    f"UPDATE {table} SET away_team=? WHERE away_team=?",
                    (club_name, previous_name),
                )


def add_fixture(database_path: str | Path, fixture: dict[str, str]) -> int:
    init_clubhouse_store(database_path)
    with _connection(database_path) as connection:
        cursor = connection.execute(
            """INSERT INTO club_fixtures
               (match_date, kickoff, home_team, away_team, venue, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                fixture["match_date"], fixture["kickoff"], fixture["home_team"],
                fixture["away_team"], fixture["venue"], datetime.now(timezone.utc).isoformat(),
            ),
        )
        return int(cursor.lastrowid)


def save_league_result(database_path: str | Path, result: dict[str, str | int]) -> None:
    init_clubhouse_store(database_path)
    with _connection(database_path) as connection:
        connection.execute(
            """INSERT INTO league_results
               (match_date, home_team, away_team, home_score, away_score, venue, reported_by, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(match_date, home_team, away_team) DO UPDATE SET
                   home_score=excluded.home_score, away_score=excluded.away_score,
                   venue=excluded.venue, reported_by=excluded.reported_by,
                   updated_at=excluded.updated_at""",
            (
                result["match_date"], result["home_team"], result["away_team"],
                result["home_score"], result["away_score"], result["venue"],
                result["reported_by"], datetime.now(timezone.utc).isoformat(),
            ),
        )