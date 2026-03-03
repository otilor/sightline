"""Formation clustering — GMM for team-level formation discovery.

Discovers recurring team formations using Gaussian Mixture Models
with soft cluster assignments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.mixture import GaussianMixture

from cod_analyst.features.formation import FormationFeatures

logger = logging.getLogger(__name__)


@dataclass
class FormationCluster:
    """A discovered team formation pattern."""
    cluster_id: int
    label: str
    frequency: float
    avg_spread: float
    avg_hull_area: float
    avg_buddy_ratio: float
    centroid_position: tuple[float, float]  # Average team centroid


@dataclass
class FormationProfile:
    """Complete formation profile for a team on a map."""
    team: str
    map_name: str
    clusters: list[FormationCluster] = field(default_factory=list)
    soft_assignments: list[np.ndarray] = field(default_factory=list)  # Per-round probabilities


def formation_features_to_vector(features: FormationFeatures) -> np.ndarray:
    """Convert FormationFeatures into a fixed-length feature vector.

    Vector: [avg_spread, avg_hull_area, avg_buddy_ratio, avg_centroid_velocity,
             centroid_x_mean, centroid_y_mean]
    """
    if not features.centroid_path:
        return np.zeros(6, dtype=np.float32)

    centroids = np.array([(x, y) for x, y, _ in features.centroid_path])

    return np.array([
        features.avg_spread,
        features.avg_hull_area,
        features.avg_buddy_ratio,
        features.avg_centroid_velocity,
        centroids[:, 0].mean(),
        centroids[:, 1].mean(),
    ], dtype=np.float32)


def cluster_formations(
    round_features: list[FormationFeatures],
    n_components: int = 4,
    min_rounds: int = 5,
) -> FormationProfile:
    """Cluster team formations using Gaussian Mixture Models.

    Parameters
    ----------
    round_features : list[FormationFeatures]
        Per-round formation features for one team on one map.
    n_components : int
        Number of formation clusters to discover.
    min_rounds : int
        Minimum rounds needed for clustering.

    Returns
    -------
    FormationProfile
        Discovered formations with soft assignments.
    """
    team = round_features[0].team if round_features else "unknown"

    if len(round_features) < min_rounds:
        logger.warning("Not enough rounds (%d) for formation clustering", len(round_features))
        return FormationProfile(team=team, map_name="")

    # Build feature matrix
    vectors = np.array([formation_features_to_vector(f) for f in round_features])

    # Normalize features
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    std[std == 0] = 1.0
    vectors_norm = (vectors - mean) / std

    # Determine optimal n_components (cap at available data)
    n_comp = min(n_components, len(round_features) // 2, len(round_features))
    if n_comp < 2:
        n_comp = 2

    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_comp,
        covariance_type="full",
        n_init=5,
        max_iter=200,
        random_state=42,
    )
    gmm.fit(vectors_norm)

    # Get soft assignments
    probs = gmm.predict_proba(vectors_norm)
    hard_labels = gmm.predict(vectors_norm)

    # Build cluster profiles
    clusters: list[FormationCluster] = []

    for k in range(n_comp):
        member_indices = [i for i, l in enumerate(hard_labels) if l == k]
        member_features = [round_features[i] for i in member_indices]

        if not member_features:
            continue

        avg_spread = float(np.mean([f.avg_spread for f in member_features]))
        avg_hull = float(np.mean([f.avg_hull_area for f in member_features]))
        avg_buddy = float(np.mean([f.avg_buddy_ratio for f in member_features]))

        # Average centroid position
        all_centroids = []
        for f in member_features:
            if f.centroid_path:
                all_centroids.extend([(x, y) for x, y, _ in f.centroid_path])

        centroid_pos = (0.5, 0.5)
        if all_centroids:
            arr = np.array(all_centroids)
            centroid_pos = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))

        # Auto-label based on characteristics
        if avg_buddy < 0.3:
            label = f"Split Setup {chr(65 + k)}"
        elif avg_spread < 0.1:
            label = f"Stack {chr(65 + k)}"
        else:
            label = f"Default {chr(65 + k)}"

        clusters.append(FormationCluster(
            cluster_id=k,
            label=label,
            frequency=len(member_indices) / len(round_features),
            avg_spread=avg_spread,
            avg_hull_area=avg_hull,
            avg_buddy_ratio=avg_buddy,
            centroid_position=centroid_pos,
        ))

    clusters.sort(key=lambda c: c.frequency, reverse=True)

    logger.info(
        "Formation clustering (%s): %d clusters from %d rounds — %s",
        team, len(clusters), len(round_features),
        ", ".join(f"{c.label} ({c.frequency:.0%})" for c in clusters),
    )

    return FormationProfile(
        team=team,
        map_name="",
        clusters=clusters,
        soft_assignments=[probs[i] for i in range(len(probs))],
    )
