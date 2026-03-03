"""SQLModel database schemas for Sightline.

Structs for VODs, Matches, MapGames, Rounds, PlayerPositions,
KillEvents, StatSnapshots, Strategies, and PlaystyleProfiles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel, create_engine, Session


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

class VOD(SQLModel, table=True):
    """A raw VOD file record."""
    id: int | None = Field(default=None, primary_key=True)
    filename: str
    filepath: str
    source_url: str = ""
    duration_sec: float = 0.0
    resolution: str = ""
    processed: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DBMatch(SQLModel, table=True):
    """A full match (best-of-N series)."""
    __tablename__ = "matches"

    id: int | None = Field(default=None, primary_key=True)
    vod_id: int = Field(foreign_key="vod.id")
    opponent: str
    event_name: str = ""
    result: str = ""  # "3-1"
    date_played: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DBMapGame(SQLModel, table=True):
    """A single map/game within a match."""
    __tablename__ = "map_games"

    id: int | None = Field(default=None, primary_key=True)
    match_id: int = Field(foreign_key="matches.id")
    map_name: str
    mode: str  # "snd", "hardpoint", "control"
    game_number: int = 1
    faze_score: int = 0
    opponent_score: int = 0


class DBRound(SQLModel, table=True):
    """A round within a map game (primarily SND)."""
    __tablename__ = "rounds"

    id: int | None = Field(default=None, primary_key=True)
    map_game_id: int = Field(foreign_key="map_games.id")
    round_number: int
    side: str = "unknown"
    outcome: str = "unknown"
    win_condition: str = "unknown"
    start_time: float = 0.0
    end_time: float = 0.0


class DBPlayerPosition(SQLModel, table=True):
    """A tracked player position at a specific timestamp."""
    __tablename__ = "player_positions"

    id: int | None = Field(default=None, primary_key=True)
    round_id: int = Field(foreign_key="rounds.id")
    player_id: int
    team: str
    x: float
    y: float
    grid_cell: str = ""
    timestamp: float


class DBKillEvent(SQLModel, table=True):
    """A kill event extracted from the kill feed."""
    __tablename__ = "kill_events"

    id: int | None = Field(default=None, primary_key=True)
    round_id: int = Field(foreign_key="rounds.id")
    killer: str
    victim: str
    weapon: str = "unknown"
    killer_grid_cell: str = ""
    timestamp: float


class DBStatSnapshot(SQLModel, table=True):
    """A player stat snapshot from the roster table."""
    __tablename__ = "stat_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    round_id: int = Field(foreign_key="rounds.id")
    player_name: str
    team: str
    kills: int = 0
    deaths: int = 0
    streak: int = 0
    time_on_obj: float = 0.0
    timestamp: float


class DBStrategySuggestion(SQLModel, table=True):
    """An AI-generated strategy suggestion."""
    __tablename__ = "strategy_suggestions"

    id: int | None = Field(default=None, primary_key=True)
    match_id: int = Field(foreign_key="matches.id")
    round_id: Optional[int] = Field(default=None, foreign_key="rounds.id")
    suggestion_type: str  # "pre_round", "loss_analysis", "scouting"
    content: str
    confidence: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DBPlaystyleProfile(SQLModel, table=True):
    """A team's playstyle fingerprint."""
    __tablename__ = "playstyle_profiles"

    id: int | None = Field(default=None, primary_key=True)
    team_name: str
    map_name: str = ""
    mode: str = ""
    playstyle_vector: str = ""  # JSON-serialized float list
    pace_score: float = 0.0
    trade_rate: float = 0.0
    first_blood_rate: float = 0.0
    adaptation_score: float = 0.0
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Database engine
# ---------------------------------------------------------------------------

def get_engine(database_url: str = "sqlite:///./data/sightline.db"):
    """Create and return a SQLAlchemy engine."""
    engine = create_engine(database_url, echo=False)
    return engine


def init_db(database_url: str = "sqlite:///./data/sightline.db") -> None:
    """Initialize the database — create all tables."""
    engine = get_engine(database_url)
    SQLModel.metadata.create_all(engine)


def get_session(database_url: str = "sqlite:///./data/sightline.db") -> Session:
    """Create a new database session."""
    engine = get_engine(database_url)
    return Session(engine)
