"""Playstyle embedding — UMAP projection for team fingerprints.

Aggregates route and formation cluster distributions into a playstyle
vector, then projects to 2D with UMAP for visualization and comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from cod_analyst.ml.formation_clustering import FormationProfile
from cod_analyst.ml.route_clustering import RouteCluster

logger = logging.getLogger(__name__)


@dataclass
class PlaystyleVector:
    """High-dimensional playstyle representation for one team."""
    team_name: str
    vector: np.ndarray = field(default_factory=lambda: np.zeros(0))
    feature_names: list[str] = field(default_factory=list)


@dataclass
class PlaystyleMap:
    """2D UMAP projection of multiple teams' playstyles."""
    team_names: list[str] = field(default_factory=list)
    coords_2d: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    vectors: list[PlaystyleVector] = field(default_factory=list)


def build_playstyle_vector(
    route_clusters: list[RouteCluster],
    formation_profile: FormationProfile,
    first_blood_rate: float = 0.0,
    trade_rate: float = 0.0,
    avg_trade_time: float = 0.0,
    avg_round_pace: float = 0.0,
) -> PlaystyleVector:
    """Build a playstyle feature vector from ML outputs.

    Combines:
    - Route cluster frequency distribution
    - Formation cluster frequency distribution
    - Engagement statistics (first blood rate, trade rate, etc.)

    Parameters
    ----------
    route_clusters : list[RouteCluster]
        Discovered route patterns.
    formation_profile : FormationProfile
        Formation clustering results.
    first_blood_rate, trade_rate, avg_trade_time, avg_round_pace : float
        Aggregate engagement statistics.

    Returns
    -------
    PlaystyleVector
        High-dimensional playstyle representation.
    """
    features: list[float] = []
    names: list[str] = []

    # Route cluster frequencies (pad to fixed size)
    max_routes = 6
    for i in range(max_routes):
        if i < len(route_clusters):
            features.append(route_clusters[i].frequency)
            names.append(f"route_{route_clusters[i].label}_freq")
        else:
            features.append(0.0)
            names.append(f"route_{i}_freq")

    # Formation cluster frequencies
    max_formations = 5
    for i in range(max_formations):
        if i < len(formation_profile.clusters):
            fc = formation_profile.clusters[i]
            features.extend([fc.frequency, fc.avg_spread, fc.avg_buddy_ratio])
            names.extend([
                f"form_{fc.label}_freq",
                f"form_{fc.label}_spread",
                f"form_{fc.label}_buddy",
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
            names.extend([f"form_{i}_freq", f"form_{i}_spread", f"form_{i}_buddy"])

    # Engagement stats
    features.extend([first_blood_rate, trade_rate, avg_trade_time, avg_round_pace])
    names.extend(["first_blood_rate", "trade_rate", "avg_trade_time", "avg_round_pace"])

    return PlaystyleVector(
        team_name=formation_profile.team,
        vector=np.array(features, dtype=np.float32),
        feature_names=names,
    )


def project_playstyles(
    vectors: list[PlaystyleVector],
    n_neighbors: int = 5,
    min_dist: float = 0.3,
) -> PlaystyleMap:
    """Project playstyle vectors to 2D using UMAP.

    Parameters
    ----------
    vectors : list[PlaystyleVector]
        Playstyle vectors for each team.
    n_neighbors : int
        UMAP n_neighbors parameter (local vs global structure).
    min_dist : float
        UMAP min_dist parameter (clustering tightness).

    Returns
    -------
    PlaystyleMap
        2D projection with team labels.
    """
    if len(vectors) < 3:
        # UMAP needs at least a few points
        logger.warning("Not enough teams (%d) for UMAP projection", len(vectors))
        coords = np.array([[v.vector.mean(), v.vector.std()] for v in vectors])
        return PlaystyleMap(
            team_names=[v.team_name for v in vectors],
            coords_2d=coords,
            vectors=vectors,
        )

    data = np.array([v.vector for v in vectors])

    # Normalize
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0
    data_norm = (data - mean) / std

    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, len(vectors) - 1),
            min_dist=min_dist,
            random_state=42,
        )
        coords_2d = reducer.fit_transform(data_norm)
    except ImportError:
        logger.warning("umap-learn not installed, using PCA fallback")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(data_norm)

    logger.info("Playstyle projection: %d teams mapped to 2D", len(vectors))

    return PlaystyleMap(
        team_names=[v.team_name for v in vectors],
        coords_2d=coords_2d,
        vectors=vectors,
    )
