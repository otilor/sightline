"""Per-player movement feature engineering.

Computes speed, grid-cell transitions, direction-to-objective,
time-in-zone, idle time, and first-move cell from raw trajectories.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from cod_analyst.game.models import PlayerPosition


@dataclass
class MovementFeatures:
    """Computed movement features for a single player in a single round."""
    player_id: int
    team: str

    # Speed profile
    avg_speed: float = 0.0
    max_speed: float = 0.0

    # Grid-cell transitions: ordered sequence of cells visited
    cell_sequence: list[str] = field(default_factory=list)

    # Direction to objective (cosine similarity to each bombsite)
    direction_to_objectives: dict[str, float] = field(default_factory=dict)

    # 8-directional heading bins (count per direction)
    heading_bins: dict[str, int] = field(default_factory=dict)

    # Time in zone: fraction of round spent in each cell
    time_in_zone: dict[str, float] = field(default_factory=dict)

    # Idle time: fraction of round with speed ≈ 0
    idle_fraction: float = 0.0

    # First cell entered after round start
    first_move_cell: str = ""


_DIRECTION_BINS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _angle_to_bin(dx: float, dy: float) -> str:
    """Convert displacement vector to 8-directional bin."""
    angle = math.atan2(-dy, dx)  # Negative dy because y increases downward
    # Convert to 0–360
    degrees = math.degrees(angle) % 360
    # 8 bins of 45° each, starting from N
    idx = int((degrees + 22.5) // 45) % 8
    # Map: 0=E, 1=NE, 2=N, ... → remap to start from N
    remap = [2, 1, 0, 7, 6, 5, 4, 3]
    return _DIRECTION_BINS[remap[idx]]


def compute_movement_features(
    positions: list[PlayerPosition],
    objective_positions: dict[str, tuple[float, float]] | None = None,
    idle_threshold: float = 0.005,
) -> MovementFeatures:
    """Compute movement features from a player's position trajectory.

    Parameters
    ----------
    positions : list[PlayerPosition]
        Sorted time series of positions for one player in one round.
    objective_positions : dict, optional
        Mapping from objective name to (x, y) center coordinate.
    idle_threshold : float
        Speed below which the player is considered idle.

    Returns
    -------
    MovementFeatures
        Computed feature set.
    """
    if not positions:
        return MovementFeatures(player_id=0, team="unknown")

    features = MovementFeatures(
        player_id=positions[0].player_id,
        team=positions[0].team,
    )

    # ---- Speed and heading ----
    speeds: list[float] = []
    heading_counts = Counter()
    idle_frames = 0

    for i in range(1, len(positions)):
        prev, curr = positions[i - 1], positions[i]
        dt = curr.timestamp - prev.timestamp
        if dt <= 0:
            continue

        dx = curr.x - prev.x
        dy = curr.y - prev.y
        dist = math.sqrt(dx * dx + dy * dy)
        speed = dist / dt
        speeds.append(speed)

        if speed < idle_threshold:
            idle_frames += 1
        else:
            heading_counts[_angle_to_bin(dx, dy)] += 1

    features.avg_speed = float(np.mean(speeds)) if speeds else 0.0
    features.max_speed = float(np.max(speeds)) if speeds else 0.0
    features.heading_bins = dict(heading_counts)
    features.idle_fraction = idle_frames / max(len(positions) - 1, 1)

    # ---- Grid-cell transition sequence ----
    cell_seq: list[str] = []
    for pos in positions:
        if not cell_seq or pos.grid_cell != cell_seq[-1]:
            cell_seq.append(pos.grid_cell)
    features.cell_sequence = cell_seq

    # ---- First move cell ----
    if len(cell_seq) >= 2:
        features.first_move_cell = cell_seq[1]  # Second cell = first cell they moved to
    elif cell_seq:
        features.first_move_cell = cell_seq[0]

    # ---- Time in zone ----
    total_time = positions[-1].timestamp - positions[0].timestamp if len(positions) > 1 else 1.0
    zone_time: dict[str, float] = Counter()

    for i in range(1, len(positions)):
        dt = positions[i].timestamp - positions[i - 1].timestamp
        zone_time[positions[i - 1].grid_cell] += dt

    features.time_in_zone = {cell: t / total_time for cell, t in zone_time.items()}

    # ---- Direction to objectives ----
    if objective_positions and len(positions) >= 2:
        last_pos = positions[-1]
        prev_pos = positions[-2]
        move_vec = np.array([last_pos.x - prev_pos.x, last_pos.y - prev_pos.y])
        move_norm = np.linalg.norm(move_vec)

        for obj_name, (ox, oy) in objective_positions.items():
            obj_vec = np.array([ox - last_pos.x, oy - last_pos.y])
            obj_norm = np.linalg.norm(obj_vec)

            if move_norm > 0 and obj_norm > 0:
                cosine = float(np.dot(move_vec, obj_vec) / (move_norm * obj_norm))
                features.direction_to_objectives[obj_name] = cosine
            else:
                features.direction_to_objectives[obj_name] = 0.0

    return features
