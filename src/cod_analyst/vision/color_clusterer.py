"""CIELAB color clustering — unsupervised team identification.

Replaces hardcoded HSV ranges with K-Means clustering in perceptually
uniform CIELAB space, plus temporal voting for noise smoothing.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class TeamColorProfile:
    """Learned color centroids for each team."""
    faze_centroid_lab: np.ndarray | None = None
    opponent_centroid_lab: np.ndarray | None = None
    is_calibrated: bool = False


@dataclass
class DotColorSample:
    """Color sample from a single detected dot."""
    dot_id: int
    lab_color: np.ndarray  # Mean Lab color of the dot region
    frame_timestamp: float


class ColorClusterer:
    """Assigns team identity to detected player dots using CIELAB clustering.

    Pipeline:
    1. Extract dominant pixel colors from each detected dot's bounding box
    2. Convert BGR to CIELAB
    3. K-Means (k=2) to find two team color clusters
    4. Calibration frame anchors which cluster = Faze
    5. Temporal voting smooths noise
    """

    def __init__(self, temporal_window: int = 10):
        self.temporal_window = temporal_window
        self.profile = TeamColorProfile()
        self._vote_history: dict[int, deque[str]] = defaultdict(
            lambda: deque(maxlen=temporal_window)
        )
        self._kmeans: KMeans | None = None

    def extract_dot_color(
        self,
        minimap: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Extract the dominant CIELAB color from a dot's bounding box.

        Parameters
        ----------
        minimap : np.ndarray
            The minimap image (BGR).
        bbox : tuple
            (x1, y1, x2, y2) bounding box of the detected dot.

        Returns
        -------
        np.ndarray
            Mean CIELAB color vector [L, a, b].
        """
        x1, y1, x2, y2 = bbox
        h, w = minimap.shape[:2]

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros(3, dtype=np.float32)

        crop = minimap[y1:y2, x1:x2]

        # Convert to CIELAB
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).astype(np.float32)

        # Take center pixels (inner 50%) to avoid edge artifacts
        ch, cw = lab.shape[:2]
        margin_h, margin_w = ch // 4, cw // 4
        center = lab[
            max(0, margin_h): max(1, ch - margin_h),
            max(0, margin_w): max(1, cw - margin_w),
        ]

        return center.reshape(-1, 3).mean(axis=0)

    def cluster_teams(self, color_samples: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Run K-Means (k=2) on collected CIELAB color samples.

        Parameters
        ----------
        color_samples : list[np.ndarray]
            List of CIELAB color vectors from detected dots.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two cluster centroids in CIELAB space.
        """
        if len(color_samples) < 2:
            raise ValueError("Need at least 2 color samples to cluster")

        data = np.array(color_samples, dtype=np.float32)
        self._kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        self._kmeans.fit(data)

        return self._kmeans.cluster_centers_[0], self._kmeans.cluster_centers_[1]

    def calibrate(
        self,
        faze_reference_color_bgr: np.ndarray,
        cluster_a: np.ndarray,
        cluster_b: np.ndarray,
    ) -> None:
        """Calibrate which cluster corresponds to Faze.

        Uses a reference BGR color (e.g., from loading screen) to
        determine which CIELAB cluster is closer to Faze's team color.

        Parameters
        ----------
        faze_reference_color_bgr : np.ndarray
            Reference BGR color for Faze (from calibration frame).
        cluster_a, cluster_b : np.ndarray
            The two CIELAB cluster centroids.
        """
        # Convert reference to CIELAB
        ref_pixel = np.array([[faze_reference_color_bgr]], dtype=np.uint8)
        ref_lab = cv2.cvtColor(ref_pixel, cv2.COLOR_BGR2Lab).astype(np.float32)[0, 0]

        # Assign closest cluster to Faze
        dist_a = np.linalg.norm(ref_lab - cluster_a)
        dist_b = np.linalg.norm(ref_lab - cluster_b)

        if dist_a <= dist_b:
            self.profile.faze_centroid_lab = cluster_a
            self.profile.opponent_centroid_lab = cluster_b
        else:
            self.profile.faze_centroid_lab = cluster_b
            self.profile.opponent_centroid_lab = cluster_a

        self.profile.is_calibrated = True
        logger.info(
            "Color calibration complete: Faze=L%.0f,a%.0f,b%.0f  Opponent=L%.0f,a%.0f,b%.0f",
            *self.profile.faze_centroid_lab, *self.profile.opponent_centroid_lab,
        )

    def classify_dot(self, lab_color: np.ndarray) -> str:
        """Classify a single dot as 'faze' or 'opponent' by centroid distance.

        Returns 'unknown' if not calibrated.
        """
        if not self.profile.is_calibrated:
            return "unknown"

        dist_faze = np.linalg.norm(lab_color - self.profile.faze_centroid_lab)
        dist_opp = np.linalg.norm(lab_color - self.profile.opponent_centroid_lab)

        return "faze" if dist_faze <= dist_opp else "opponent"

    def classify_with_voting(self, dot_id: int, lab_color: np.ndarray) -> str:
        """Classify a dot with temporal majority voting.

        Collects classifications across frames for the same tracked dot
        and returns the majority vote.
        """
        raw_class = self.classify_dot(lab_color)
        self._vote_history[dot_id].append(raw_class)

        # Majority vote
        history = self._vote_history[dot_id]
        faze_count = sum(1 for v in history if v == "faze")
        opp_count = sum(1 for v in history if v == "opponent")

        if faze_count > opp_count:
            return "faze"
        elif opp_count > faze_count:
            return "opponent"
        else:
            return raw_class

    def reset_votes(self) -> None:
        """Reset vote history (e.g., at round boundaries)."""
        self._vote_history.clear()
