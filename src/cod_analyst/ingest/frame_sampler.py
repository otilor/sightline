"""Adaptive frame sampler — three-tier sampling within gameplay windows.

Tier 1 (Discovery): 0.5 fps — handled by gameplay_detector, not here.
Tier 2 (Tactical): 4-5 fps — primary extraction rate.
Tier 3 (Burst): 10+ fps — around engagement events.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass, field

import cv2
import numpy as np

from cod_analyst.config import AppConfig
from cod_analyst.game.models import GameplayWindow
from cod_analyst.ingest.video_loader import VideoMeta, downscale_frame

logger = logging.getLogger(__name__)


@dataclass
class SampledFrame:
    """A single sampled frame with metadata."""
    image: np.ndarray
    timestamp: float
    frame_number: int
    tier: str  # "tactical" or "burst"
    gameplay_window_idx: int


@dataclass
class BurstTrigger:
    """Marks a timestamp region for burst sampling."""
    center_sec: float
    window_sec: float = 5.0

    @property
    def start(self) -> float:
        return max(0.0, self.center_sec - self.window_sec)

    @property
    def end(self) -> float:
        return self.center_sec + self.window_sec


@dataclass
class _SamplerState:
    """Internal state tracking for burst mode."""
    burst_triggers: list[BurstTrigger] = field(default_factory=list)
    _current_burst_idx: int = 0

    def is_in_burst(self, timestamp: float) -> bool:
        """Check if the given timestamp falls within any burst window."""
        for trigger in self.burst_triggers:
            if trigger.start <= timestamp <= trigger.end:
                return True
        return False

    def add_burst(self, center_sec: float, window_sec: float = 5.0) -> None:
        """Register a new burst trigger (e.g., from engagement detection)."""
        self.burst_triggers.append(BurstTrigger(center_sec, window_sec))
        # Keep sorted by center time
        self.burst_triggers.sort(key=lambda b: b.center_sec)


def sample_gameplay(
    cap: cv2.VideoCapture,
    meta: VideoMeta,
    windows: list[GameplayWindow],
    cfg: AppConfig,
    burst_triggers: list[BurstTrigger] | None = None,
) -> Generator[SampledFrame, None, None]:
    """Sample frames from gameplay windows with adaptive rates.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Opened video capture.
    meta : VideoMeta
        Video metadata.
    windows : list[GameplayWindow]
        Detected gameplay time windows from gameplay_detector.
    cfg : AppConfig
        Application configuration.
    burst_triggers : list[BurstTrigger], optional
        Pre-computed burst trigger points. Additional triggers can be
        registered dynamically via the state object.

    Yields
    ------
    SampledFrame
        Frames sampled at the appropriate tier rate.
    """
    tactical_interval = 1.0 / cfg.video.sample_fps_tactical
    burst_interval = 1.0 / cfg.video.sample_fps_burst
    target_w, target_h = cfg.video.target_resolution

    state = _SamplerState(burst_triggers=burst_triggers or [])

    total_tactical = 0
    total_burst = 0

    for win_idx, window in enumerate(windows):
        current_time = window.start_sec

        logger.debug(
            "Sampling window %d/%d: %.1fs – %.1fs (%.1fs)",
            win_idx + 1, len(windows), window.start_sec, window.end_sec, window.duration,
        )

        while current_time < window.end_sec:
            # Determine tier
            in_burst = state.is_in_burst(current_time)
            tier = "burst" if in_burst else "tactical"
            interval = burst_interval if in_burst else tactical_interval

            # Read frame
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
            ok, frame = cap.read()
            if not ok:
                current_time += interval
                continue

            # Downscale
            frame = downscale_frame(frame, target_w, target_h)

            # Emit
            frame_number = int(current_time * meta.fps)
            yield SampledFrame(
                image=frame,
                timestamp=current_time,
                frame_number=frame_number,
                tier=tier,
                gameplay_window_idx=win_idx,
            )

            if in_burst:
                total_burst += 1
            else:
                total_tactical += 1

            current_time += interval

    logger.info(
        "Sampling complete: %d tactical + %d burst = %d total frames across %d windows",
        total_tactical, total_burst, total_tactical + total_burst, len(windows),
    )
