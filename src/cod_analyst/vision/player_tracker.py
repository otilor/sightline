"""Player tracker — ByteTrack persistent IDs across minimap frames.

Takes per-frame detections and produces continuous trajectories
with stable player IDs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from cod_analyst.game.models import PlayerPosition
from cod_analyst.vision.player_detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackedPlayer:
    """A tracked player dot with persistent ID and position history."""
    track_id: int
    team: str  # "faze" or "opponent"
    positions: list[PlayerPosition] = field(default_factory=list)
    is_active: bool = True

    @property
    def last_position(self) -> PlayerPosition | None:
        return self.positions[-1] if self.positions else None


class PlayerTracker:
    """ByteTrack-based multi-object tracker for minimap player dots.

    Maintains persistent IDs for detected dots across frames and
    builds per-player trajectory histories.
    """

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self._tracker = None
        self._tracks: dict[int, TrackedPlayer] = {}
        self._next_id: int = 0

    def _init_tracker(self) -> None:
        """Initialize the ByteTrack tracker from supervision."""
        try:
            import supervision as sv
            self._tracker = sv.ByteTrack(
                track_activation_threshold=self.track_thresh,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
                frame_rate=5,
            )
            logger.info("ByteTrack tracker initialized")
        except ImportError:
            logger.error("supervision not installed. Install with: pip install supervision")
            raise

    def update(
        self,
        detections: list[Detection],
        team_labels: dict[int, str],
        timestamp: float,
        grid_cell_fn: callable | None = None,
    ) -> list[TrackedPlayer]:
        """Update tracker with new frame detections.

        Parameters
        ----------
        detections : list[Detection]
            Detections from the current frame.
        team_labels : dict[int, str]
            Mapping from detection index to team label ("faze"/"opponent").
        timestamp : float
            Current frame timestamp in seconds.
        grid_cell_fn : callable, optional
            Function mapping (x, y) -> grid cell string.

        Returns
        -------
        list[TrackedPlayer]
            Currently active tracked players.
        """
        if self._tracker is None:
            self._init_tracker()

        import supervision as sv

        if not detections:
            return list(self._tracks.values())

        # Build supervision Detections
        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
        )

        # Run tracker
        tracked = self._tracker.update_with_detections(sv_detections)

        # Process tracked results
        active_ids = set()

        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])
            active_ids.add(track_id)

            x1, y1, x2, y2 = tracked.xyxy[i]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Normalize to [0, 1]
            # Note: these are minimap-relative coordinates
            # The caller should know the minimap dimensions
            team = team_labels.get(i, "unknown")
            grid_cell = grid_cell_fn(cx, cy) if grid_cell_fn else ""

            position = PlayerPosition(
                player_id=track_id,
                team=team,
                x=float(cx),
                y=float(cy),
                grid_cell=grid_cell,
                timestamp=timestamp,
            )

            if track_id not in self._tracks:
                self._tracks[track_id] = TrackedPlayer(
                    track_id=track_id,
                    team=team,
                )

            self._tracks[track_id].positions.append(position)
            self._tracks[track_id].is_active = True

        # Mark inactive tracks
        for tid, track in self._tracks.items():
            if tid not in active_ids:
                track.is_active = False

        return [t for t in self._tracks.values() if t.is_active]

    def get_all_tracks(self) -> dict[int, TrackedPlayer]:
        """Return all tracked players (active and inactive)."""
        return self._tracks

    def get_trajectories(self, team: str | None = None) -> dict[int, list[PlayerPosition]]:
        """Extract trajectories for all (or filtered) players.

        Parameters
        ----------
        team : str, optional
            Filter by team ("faze" or "opponent").

        Returns
        -------
        dict[int, list[PlayerPosition]]
            Mapping from track_id to position history.
        """
        result = {}
        for tid, track in self._tracks.items():
            if team and track.team != team:
                continue
            if track.positions:
                result[tid] = track.positions
        return result

    def reset(self) -> None:
        """Reset all tracking state (e.g., at round boundaries)."""
        self._tracker = None
        self._tracks.clear()
        self._next_id = 0
        self._init_tracker()
