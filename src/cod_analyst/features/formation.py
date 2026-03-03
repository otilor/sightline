"""Team formation feature engineering.

Computes centroid, spatial spread, convex hull area, pairwise distances,
buddy pair ratio, and centroid velocity from all player positions at each timestamp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cod_analyst.game.models import PlayerPosition


@dataclass
class FormationSnapshot:
    """Formation features at a single timestamp."""
    timestamp: float
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    spatial_spread: float = 0.0
    convex_hull_area: float = 0.0
    pairwise_distances: list[float] = field(default_factory=list)
    buddy_pair_ratio: float = 0.0
    num_players: int = 0


@dataclass
class FormationFeatures:
    """Aggregated formation features for a team across an entire round."""
    team: str

    # Averages over the round
    avg_spread: float = 0.0
    avg_hull_area: float = 0.0
    avg_buddy_ratio: float = 0.0
    avg_centroid_velocity: float = 0.0

    # Time series
    snapshots: list[FormationSnapshot] = field(default_factory=list)

    # Centroid trajectory
    centroid_path: list[tuple[float, float, float]] = field(default_factory=list)  # (x, y, t)


def _convex_hull_area(points: np.ndarray) -> float:
    """Compute convex hull area of 2D points using Shoelace formula.

    Returns 0 if fewer than 3 points.
    """
    if len(points) < 3:
        return 0.0

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return float(hull.volume)  # In 2D, 'volume' is the area
    except Exception:
        # Fallback for collinear points
        return 0.0


def compute_formation_snapshot(
    positions: list[PlayerPosition],
    timestamp: float,
) -> FormationSnapshot:
    """Compute formation features for a single timestamp.

    Parameters
    ----------
    positions : list[PlayerPosition]
        All player positions for one team at one timestamp.
    timestamp : float
        Current timestamp.

    Returns
    -------
    FormationSnapshot
        Formation features at this moment.
    """
    if not positions:
        return FormationSnapshot(timestamp=timestamp)

    coords = np.array([[p.x, p.y] for p in positions])
    n = len(coords)

    # Centroid
    centroid = coords.mean(axis=0)

    # Spatial spread (std of distances from centroid)
    dists_from_centroid = np.linalg.norm(coords - centroid, axis=1)
    spread = float(np.std(dists_from_centroid))

    # Convex hull area
    hull_area = _convex_hull_area(coords)

    # Pairwise distances
    pairwise = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            pairwise.append(d)

    # Buddy pair ratio
    buddy_ratio = 0.0
    if pairwise:
        buddy_ratio = min(pairwise) / max(pairwise) if max(pairwise) > 0 else 1.0

    return FormationSnapshot(
        timestamp=timestamp,
        centroid_x=float(centroid[0]),
        centroid_y=float(centroid[1]),
        spatial_spread=spread,
        convex_hull_area=hull_area,
        pairwise_distances=pairwise,
        buddy_pair_ratio=buddy_ratio,
        num_players=n,
    )


def compute_formation_features(
    positions_by_time: dict[float, list[PlayerPosition]],
    team: str,
) -> FormationFeatures:
    """Compute formation features for one team across an entire round.

    Parameters
    ----------
    positions_by_time : dict[float, list[PlayerPosition]]
        Mapping from timestamp to list of player positions for one team.
    team : str
        Team identifier.

    Returns
    -------
    FormationFeatures
        Aggregated formation features.
    """
    features = FormationFeatures(team=team)

    timestamps = sorted(positions_by_time.keys())
    snapshots = []

    for t in timestamps:
        snap = compute_formation_snapshot(positions_by_time[t], t)
        snapshots.append(snap)
        features.centroid_path.append((snap.centroid_x, snap.centroid_y, t))

    features.snapshots = snapshots

    if not snapshots:
        return features

    # Compute averages
    features.avg_spread = float(np.mean([s.spatial_spread for s in snapshots]))
    features.avg_hull_area = float(np.mean([s.convex_hull_area for s in snapshots]))
    features.avg_buddy_ratio = float(np.mean([s.buddy_pair_ratio for s in snapshots]))

    # Centroid velocity
    velocities = []
    for i in range(1, len(snapshots)):
        prev, curr = snapshots[i - 1], snapshots[i]
        dt = curr.timestamp - prev.timestamp
        if dt > 0:
            dx = curr.centroid_x - prev.centroid_x
            dy = curr.centroid_y - prev.centroid_y
            v = math.sqrt(dx * dx + dy * dy) / dt
            velocities.append(v)

    features.avg_centroid_velocity = float(np.mean(velocities)) if velocities else 0.0

    return features
