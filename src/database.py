"""SQLite database layer for football match event storage.

Schema inspired by the StatsBomb event model (industry standard):
  - events: one row per detected on-pitch action, with typed columns and a
    JSON 'attributes' blob for extensibility without schema migrations.
  - player_stats / team_stats: pre-aggregated per match for fast report rendering.

The database file lives at output/match.db and is fully portable (single file).

Typical usage:
    db = MatchDatabase("output/match.db")
    db.init()
    match_id = db.insert_match(label="Team A vs B", video_path="data/match.mp4",
                               duration_s=5400.0)
    db.insert_events(match_id, events)          # list[dict]
    db.insert_player_stats(match_id, player_rows)  # list[dict]
    db.insert_team_stats(match_id, team_rows)       # list[dict]
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS matches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT    NOT NULL,
    date        TEXT    NOT NULL,   -- ISO-8601 date of analysis
    video_path  TEXT,
    duration_s  REAL,
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id          INTEGER NOT NULL REFERENCES matches(id),
    event_type        TEXT    NOT NULL,   -- 'pass'|'shot'|'goal'|'out_of_play'
    timestamp_s       REAL    NOT NULL,   -- seconds from video start
    team              TEXT,               -- 'team_A'|'team_B'|'unknown'
    player_track_id   INTEGER,            -- track_id of acting player (-1 = unknown)
    location_x        REAL,               -- pitch metres (origin = top-left corner)
    location_y        REAL,
    end_location_x    REAL,               -- destination (passes / shots)
    end_location_y    REAL,
    outcome           TEXT,               -- 'success'|'fail'|'goal'|'on_target'|'off_target'|'saved'
    attributes        TEXT    DEFAULT '{}' -- JSON blob for future fields
);

CREATE TABLE IF NOT EXISTS player_stats (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id         INTEGER NOT NULL REFERENCES matches(id),
    track_id         INTEGER NOT NULL,
    team             TEXT,
    distance_m       REAL,
    top_speed_kmh    REAL,
    n_sprints        INTEGER,
    minutes_tracked  REAL,
    n_passes         INTEGER DEFAULT 0,
    n_shots          INTEGER DEFAULT 0,
    n_goals          INTEGER DEFAULT 0,
    possession_pct   REAL    DEFAULT 0.0,
    rating           REAL    DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS team_stats (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id         INTEGER NOT NULL REFERENCES matches(id),
    team             TEXT    NOT NULL,
    possession_pct   REAL    DEFAULT 0.0,
    total_distance_km REAL   DEFAULT 0.0,
    total_sprints    INTEGER DEFAULT 0,
    n_passes         INTEGER DEFAULT 0,
    n_shots          INTEGER DEFAULT 0,
    n_goals          INTEGER DEFAULT 0,
    field_tilt_x     REAL    DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_events_match    ON events(match_id);
CREATE INDEX IF NOT EXISTS idx_events_type     ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_team     ON events(team);
CREATE INDEX IF NOT EXISTS idx_player_match    ON player_stats(match_id);
CREATE INDEX IF NOT EXISTS idx_team_match      ON team_stats(match_id);
"""


class MatchDatabase:
    """Thin wrapper around sqlite3 for football match data persistence."""

    def __init__(self, db_path: str = "output/match.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Create tables and indexes if they don't exist yet."""
        with self._connect() as conn:
            conn.executescript(_DDL)
        print(f"[DB] Initialised → {self.db_path}")

    # ------------------------------------------------------------------
    # Inserts
    # ------------------------------------------------------------------

    def insert_match(
        self,
        label: str,
        video_path: str = "",
        duration_s: float = 0.0,
    ) -> int:
        """Insert a match record and return its auto-incremented id."""
        now = datetime.now(timezone.utc).isoformat()
        today = datetime.now(timezone.utc).date().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO matches (label, date, video_path, duration_s, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (label, today, str(video_path), duration_s, now),
            )
            match_id = cur.lastrowid
        print(f"[DB] Match #{match_id} — '{label}'")
        return match_id

    def insert_events(self, match_id: int, events: list[dict[str, Any]]) -> None:
        """Bulk-insert event dicts. Unknown keys go into the JSON attributes blob."""
        _known = {
            "event_type", "timestamp_s", "team", "player_track_id",
            "location_x", "location_y", "end_location_x", "end_location_y",
            "outcome",
        }
        rows = []
        for ev in events:
            extra = {k: v for k, v in ev.items() if k not in _known}
            rows.append((
                match_id,
                ev.get("event_type", "unknown"),
                float(ev.get("timestamp_s", 0.0)),
                ev.get("team"),
                ev.get("player_track_id"),
                ev.get("location_x"),
                ev.get("location_y"),
                ev.get("end_location_x"),
                ev.get("end_location_y"),
                ev.get("outcome"),
                json.dumps(extra),
            ))
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO events "
                "(match_id, event_type, timestamp_s, team, player_track_id,"
                " location_x, location_y, end_location_x, end_location_y,"
                " outcome, attributes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        print(f"[DB] Inserted {len(rows)} events")

    def insert_player_stats(
        self, match_id: int, player_rows: list[dict[str, Any]]
    ) -> None:
        rows = [
            (
                match_id,
                int(r["track_id"]),
                r.get("team"),
                r.get("distance_m"),
                r.get("top_speed_kmh"),
                r.get("n_sprints"),
                r.get("minutes_tracked"),
                int(r.get("n_passes", 0)),
                int(r.get("n_shots", 0)),
                int(r.get("n_goals", 0)),
                float(r.get("possession_pct", 0.0)),
                float(r.get("rating", 0.0)),
            )
            for r in player_rows
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO player_stats "
                "(match_id, track_id, team, distance_m, top_speed_kmh, n_sprints,"
                " minutes_tracked, n_passes, n_shots, n_goals, possession_pct, rating)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        print(f"[DB] Inserted player stats for {len(rows)} tracks")

    def insert_team_stats(
        self, match_id: int, team_rows: list[dict[str, Any]]
    ) -> None:
        rows = [
            (
                match_id,
                r["team"],
                float(r.get("possession_pct", 0.0)),
                float(r.get("total_distance_km", 0.0)),
                int(r.get("total_sprints", 0)),
                int(r.get("n_passes", 0)),
                int(r.get("n_shots", 0)),
                int(r.get("n_goals", 0)),
                float(r.get("field_tilt_x", 0.0)),
            )
            for r in team_rows
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO team_stats "
                "(match_id, team, possession_pct, total_distance_km, total_sprints,"
                " n_passes, n_shots, n_goals, field_tilt_x)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
        print(f"[DB] Inserted team stats for {len(rows)} teams")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def event_summary(self, match_id: int) -> dict[str, int]:
        """Return event counts by type for a given match."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event_type, COUNT(*) FROM events"
                " WHERE match_id=? GROUP BY event_type",
                (match_id,),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_events(self, match_id: int) -> list[dict[str, Any]]:
        """Return all events for a match as a list of dicts."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM events WHERE match_id=? ORDER BY timestamp_s",
                (match_id,),
            ).fetchall()
        return [dict(r) for r in rows]
