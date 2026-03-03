"""Route clustering — DTW + DBSCAN for individual player routes.

Discovers recurring movement patterns per player per map using
Dynamic Time Warping distance and density-based clustering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


@dataclass
class RouteCluster:
    """A discovered route pattern."""
    cluster_id: int
    label: str  # e.g., "Route A", or aliased name
    member_count: int
    frequency: float  # Fraction of total rounds
    representative: list[str]  # Most common grid-cell sequence
    all_sequences: list[list[str]] = field(default_factory=list)


def _dtw_distance(seq_a: list[str], seq_b: list[str], all_cells: list[str] | None = None) -> float:
    """Compute Dynamic Time Warping distance between two grid-cell sequences.

    Uses cell-to-cell distance as the local cost (0 if same cell, 1 if different).
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float(max(n, m))

    # Cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0.0 if seq_a[i - 1] == seq_b[j - 1] else 1.0
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return dtw[n, m] / max(n, m)  # Normalize by length


def _compute_dtw_distance_matrix(sequences: list[list[str]]) -> np.ndarray:
    """Compute pairwise DTW distance matrix."""
    n = len(sequences)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = _dtw_distance(sequences[i], sequences[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def cluster_routes(
    sequences: list[list[str]],
    eps: float = 0.4,
    min_samples: int = 3,
) -> list[RouteCluster]:
    """Cluster player grid-cell transition sequences using DTW + DBSCAN.

    Parameters
    ----------
    sequences : list[list[str]]
        Grid-cell transition sequences, one per round.
    eps : float
        DBSCAN epsilon (max DTW distance for neighbors).
    min_samples : int
        DBSCAN minimum cluster size.

    Returns
    -------
    list[RouteCluster]
        Discovered route clusters, sorted by frequency.
    """
    if len(sequences) < min_samples:
        logger.warning("Not enough sequences (%d) for clustering", len(sequences))
        return []

    # Compute DTW distance matrix
    logger.info("Computing DTW distance matrix for %d sequences...", len(sequences))
    dist_matrix = _compute_dtw_distance_matrix(sequences)

    # DBSCAN with precomputed distances
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = clustering.fit_predict(dist_matrix)

    # Build clusters
    unique_labels = set(labels)
    clusters: list[RouteCluster] = []

    for label in sorted(unique_labels):
        if label == -1:
            continue  # Skip noise points

        member_indices = [i for i, l in enumerate(labels) if l == label]
        member_sequences = [sequences[i] for i in member_indices]

        # Find representative (most common subsequence or median)
        # Use shortest sequence that's still in the cluster
        representative = min(member_sequences, key=len)

        clusters.append(RouteCluster(
            cluster_id=label,
            label=f"Route {chr(65 + label)}",  # Route A, B, C...
            member_count=len(member_indices),
            frequency=len(member_indices) / len(sequences),
            representative=representative,
            all_sequences=member_sequences,
        ))

    # Sort by frequency descending
    clusters.sort(key=lambda c: c.frequency, reverse=True)

    # Count noise
    noise_count = sum(1 for l in labels if l == -1)

    logger.info(
        "Route clustering: %d clusters, %d noise (%.0f%% classified)",
        len(clusters), noise_count,
        (1 - noise_count / len(sequences)) * 100,
    )

    return clusters
