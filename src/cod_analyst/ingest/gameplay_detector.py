"""Gameplay detector — scans a VOD at low FPS to find gameplay windows.

Primary signal: minimap presence in bottom-left ROI via template matching.
Secondary: kill feed presence, scoreboard format.
Produces a list of GameplayWindow time ranges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from cod_analyst.config import AppConfig
from cod_analyst.game.models import GameplayWindow
from cod_analyst.ingest.video_loader import VideoMeta, crop_roi, downscale_frame

logger = logging.getLogger(__name__)


@dataclass
class _DetectionState:
    """Internal state for tracking gameplay detection across frames."""
    in_gameplay: bool = False
    window_start: float = 0.0
    consecutive_positive: int = 0
    consecutive_negative: int = 0

    # Require N consecutive frames to flip state — debouncing
    enter_threshold: int = 3
    exit_threshold: int = 6


def _has_minimap_content(roi: np.ndarray, threshold: float = 25.0) -> bool:
    """Check if the minimap ROI has gameplay content.

    Gameplay minimaps have rich color variation and structure.
    Non-gameplay (black, loading, desk) has low variance.
    """
    if roi is None or roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

    # Check mean brightness — pure black = no minimap
    mean_val = np.mean(gray)
    if mean_val < 15:
        return False

    # Check edge density — minimaps have map structure = edges
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / edges.size

    # Check color variance — minimaps have colored dots
    if len(roi.shape) == 3:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        saturation_mean = np.mean(hsv[:, :, 1])
    else:
        saturation_mean = 0.0

    # Minimap has both structure (edges) and color (saturation)
    has_structure = edge_ratio > 0.02
    has_color = saturation_mean > 20.0

    return has_structure and has_color


def _has_killfeed_content(roi: np.ndarray) -> bool:
    """Check if the kill feed ROI has text content (secondary signal)."""
    if roi is None or roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

    # Kill feed text is high-contrast white on dark
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    text_ratio = np.count_nonzero(binary) / binary.size

    # Some text present but not too much (would be a graphic overlay)
    return 0.005 < text_ratio < 0.3


def detect_gameplay(
    cap: cv2.VideoCapture,
    meta: VideoMeta,
    cfg: AppConfig,
    min_window_duration: float = 30.0,
) -> list[GameplayWindow]:
    """Scan a VOD at discovery FPS to find gameplay time windows.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Opened video capture.
    meta : VideoMeta
        Video metadata.
    cfg : AppConfig
        Application configuration.
    min_window_duration : float
        Minimum duration in seconds for a valid gameplay window.

    Returns
    -------
    list[GameplayWindow]
        Sorted list of gameplay time windows.
    """
    discovery_fps = cfg.video.sample_fps_discovery
    frame_interval = 1.0 / discovery_fps
    minimap_roi = cfg.minimap.roi_pct
    killfeed_roi = cfg.hud.killfeed_roi_pct

    state = _DetectionState()
    windows: list[GameplayWindow] = []

    current_time = 0.0
    total_frames_checked = 0

    logger.info(
        "Starting gameplay detection on %s at %.1f fps (checking every %.1fs)",
        meta.filepath, discovery_fps, frame_interval,
    )

    while current_time < meta.duration_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ok, frame = cap.read()
        if not ok:
            break

        frame = downscale_frame(frame, *cfg.video.target_resolution)
        total_frames_checked += 1

        # Extract ROIs
        minimap_crop = crop_roi(frame, minimap_roi)
        killfeed_crop = crop_roi(frame, killfeed_roi)

        # Check signals
        minimap_present = _has_minimap_content(minimap_crop)
        killfeed_present = _has_killfeed_content(killfeed_crop)

        # Minimap is primary; kill feed is secondary confirmation
        is_gameplay = minimap_present or (killfeed_present and state.in_gameplay)

        if is_gameplay:
            state.consecutive_positive += 1
            state.consecutive_negative = 0
        else:
            state.consecutive_negative += 1
            state.consecutive_positive = 0

        # State transitions with debouncing
        if not state.in_gameplay and state.consecutive_positive >= state.enter_threshold:
            state.in_gameplay = True
            # Backdate window start by the debounce frames
            state.window_start = max(0.0, current_time - (state.enter_threshold * frame_interval))
            logger.debug("Gameplay started at %.1fs", state.window_start)

        elif state.in_gameplay and state.consecutive_negative >= state.exit_threshold:
            state.in_gameplay = False
            window_end = current_time - (state.exit_threshold * frame_interval)
            duration = window_end - state.window_start

            if duration >= min_window_duration:
                windows.append(GameplayWindow(
                    start_sec=state.window_start,
                    end_sec=window_end,
                ))
                logger.debug(
                    "Gameplay window: %.1fs – %.1fs (%.1fs)",
                    state.window_start, window_end, duration,
                )
            else:
                logger.debug(
                    "Discarded short window: %.1fs – %.1fs (%.1fs < %.1fs)",
                    state.window_start, window_end, duration, min_window_duration,
                )

        current_time += frame_interval

    # Close any open window
    if state.in_gameplay:
        duration = meta.duration_sec - state.window_start
        if duration >= min_window_duration:
            windows.append(GameplayWindow(
                start_sec=state.window_start,
                end_sec=meta.duration_sec,
            ))

    total_gameplay = sum(w.duration for w in windows)
    logger.info(
        "Gameplay detection complete: %d windows, %.1fs gameplay / %.1fs total (%.0f%%), "
        "%d frames checked",
        len(windows),
        total_gameplay,
        meta.duration_sec,
        (total_gameplay / meta.duration_sec * 100) if meta.duration_sec > 0 else 0,
        total_frames_checked,
    )

    return windows
