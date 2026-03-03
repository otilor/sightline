"""Kill event feature engineering.

Computes first blood timing, trade timing/success rate, kill heatmaps,
weapon distributions, and multi-kill frequency from raw kill events.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from cod_analyst.game.models import KillEvent


@dataclass
class KillEventFeatures:
    """Aggregated engagement features for a team in a single round."""
    team: str

    # First blood
    first_blood_time: float | None = None
    got_first_blood: bool = False

    # Trading
    avg_trade_time: float = 0.0
    trade_success_rate: float = 0.0
    total_deaths: int = 0
    traded_deaths: int = 0

    # Kill positions
    kill_heatmap: dict[str, int] = field(default_factory=dict)   # grid_cell -> count
    death_heatmap: dict[str, int] = field(default_factory=dict)  # grid_cell -> count

    # Weapons
    weapon_distribution: dict[str, int] = field(default_factory=dict)  # weapon -> count

    # Multi-kills
    double_kills: int = 0
    triple_kills: int = 0
    quad_kills: int = 0


def compute_kill_features(
    kill_events: list[KillEvent],
    team_players: set[str],
    trade_window_sec: float = 2.0,
    multi_kill_window_sec: float = 5.0,
) -> KillEventFeatures:
    """Compute engagement features from kill events for one team.

    Parameters
    ----------
    kill_events : list[KillEvent]
        All kill events in the round (both teams), sorted by timestamp.
    team_players : set[str]
        Names of players on this team.
    trade_window_sec : float
        Maximum time gap for a kill to count as a trade.
    multi_kill_window_sec : float
        Time window to group kills for multi-kill detection.

    Returns
    -------
    KillEventFeatures
        Computed engagement features.
    """
    features = KillEventFeatures(team="faze" if team_players else "unknown")

    if not kill_events:
        return features

    # Sort by timestamp
    sorted_events = sorted(kill_events, key=lambda e: e.timestamp)

    # First blood
    features.first_blood_time = sorted_events[0].timestamp
    features.got_first_blood = sorted_events[0].killer in team_players

    # Kill/death positions
    kill_cells: Counter = Counter()
    death_cells: Counter = Counter()
    weapon_counts: Counter = Counter()

    team_deaths: list[KillEvent] = []
    team_kills: list[KillEvent] = []

    for event in sorted_events:
        if event.killer in team_players:
            team_kills.append(event)
            if event.killer_grid_cell:
                kill_cells[event.killer_grid_cell] += 1
            if event.weapon:
                weapon_counts[event.weapon] += 1

        if event.victim in team_players:
            team_deaths.append(event)
            # Death position is the killer's position (approximation)
            if event.killer_grid_cell:
                death_cells[event.killer_grid_cell] += 1

    features.kill_heatmap = dict(kill_cells)
    features.death_heatmap = dict(death_cells)
    features.weapon_distribution = dict(weapon_counts)

    # Trading analysis
    trade_times: list[float] = []
    traded_count = 0

    for death in team_deaths:
        # Look for a revenge kill within trade_window_sec
        for kill in team_kills:
            if death.timestamp < kill.timestamp <= death.timestamp + trade_window_sec:
                if kill.victim == death.killer or True:  # Any trade counts
                    traded_count += 1
                    trade_times.append(kill.timestamp - death.timestamp)
                    break

    features.total_deaths = len(team_deaths)
    features.traded_deaths = traded_count
    features.trade_success_rate = traded_count / max(len(team_deaths), 1)
    features.avg_trade_time = sum(trade_times) / max(len(trade_times), 1) if trade_times else 0.0

    # Multi-kill detection
    for player in team_players:
        player_kills = [e for e in team_kills if e.killer == player]
        if len(player_kills) < 2:
            continue

        # Group kills within window
        i = 0
        while i < len(player_kills):
            streak = 1
            j = i + 1
            while j < len(player_kills) and \
                  player_kills[j].timestamp - player_kills[i].timestamp < multi_kill_window_sec:
                streak += 1
                j += 1

            if streak == 2:
                features.double_kills += 1
            elif streak == 3:
                features.triple_kills += 1
            elif streak >= 4:
                features.quad_kills += 1

            i = j if j > i + 1 else i + 1

    return features
