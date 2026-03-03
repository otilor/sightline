"""Core domain models for Sightline.

Enums for game state, dataclasses for positions, events, rounds,
matches, strategies, and playstyle profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GameMode(str, Enum):
    SND = "snd"
    HARDPOINT = "hardpoint"
    CONTROL = "control"
    MENU = "menu"
    UNKNOWN = "unknown"


class Side(str, Enum):
    ATTACK = "attack"
    DEFENSE = "defense"
    UNKNOWN = "unknown"


class RoundOutcome(str, Enum):
    WIN = "win"
    LOSS = "loss"
    UNKNOWN = "unknown"


class WinCondition(str, Enum):
    ELIMINATION = "elimination"
    BOMB_DETONATION = "bomb_detonation"
    BOMB_DEFUSE = "bomb_defuse"
    TIME_EXPIRED = "time_expired"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PlayerPosition:
    """A single tracked position on the minimap."""
    player_id: int
    team: str
    x: float
    y: float
    grid_cell: str
    timestamp: float


@dataclass
class KillEvent:
    """A single kill extracted from the kill feed."""
    killer: str
    victim: str
    weapon: str
    killer_grid_cell: str
    timestamp: float


@dataclass
class PlayerStatSnapshot:
    """A snapshot of a player's stats from the roster table."""
    player_name: str
    team: str
    kills: int
    deaths: int
    streak: int
    time_on_obj: float
    timestamp: float


@dataclass
class ScoreboardSnapshot:
    """A snapshot of the scoreboard state."""
    faze_score: int
    opponent_score: int
    game_clock: float
    mode: GameMode
    round_number: int | None = None
    hill_timer: float | None = None
    timestamp: float = 0.0


@dataclass
class Round:
    """A single round (SND) or segment with all associated data."""
    round_number: int
    side: Side
    outcome: RoundOutcome
    win_condition: WinCondition
    start_time: float
    end_time: float
    positions: list[PlayerPosition] = field(default_factory=list)
    kill_events: list[KillEvent] = field(default_factory=list)
    stat_snapshots: list[PlayerStatSnapshot] = field(default_factory=list)


@dataclass
class MapGame:
    """A single map within a best-of series."""
    map_name: str
    mode: GameMode
    faze_score: int = 0
    opponent_score: int = 0
    rounds: list[Round] = field(default_factory=list)


@dataclass
class Match:
    """A full best-of-N series between Faze and an opponent."""
    opponent: str
    event_name: str
    result: str = ""  # e.g. "3-1"
    map_games: list[MapGame] = field(default_factory=list)
    vod_filename: str = ""


@dataclass
class StrategySuggestion:
    """An AI-generated strategy recommendation."""
    suggestion_type: str  # "pre_round", "loss_analysis", "scouting"
    content: str
    confidence: float
    source_round_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlaystyleProfile:
    """Aggregated playstyle fingerprint for a team."""
    team_name: str
    playstyle_vector: list[float] = field(default_factory=list)
    route_cluster_distributions: dict[str, dict[str, float]] = field(default_factory=dict)
    formation_cluster_distributions: dict[str, float] = field(default_factory=dict)
    pace_score: float = 0.0
    trade_rate: float = 0.0
    first_blood_rate: float = 0.0
    adaptation_score: float = 0.0


@dataclass
class GameplayWindow:
    """A contiguous window of detected gameplay in a VOD."""
    start_sec: float
    end_sec: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec
