"""Video loader — opens VODs, extracts metadata, provides frame access.

Handles resolution validation, 720p downscaling, and random-access reads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMeta:
    """Metadata extracted from a video file."""
    filepath: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    codec: str


def load_video(filepath: str | Path) -> tuple[cv2.VideoCapture, VideoMeta]:
    """Open a video file and return the capture object + metadata.

    Raises FileNotFoundError if *filepath* does not exist.
    Raises ValueError if the file cannot be opened by OpenCV.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Video not found: {filepath}")

    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {filepath}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    duration_sec = frame_count / fps if fps > 0 else 0.0

    meta = VideoMeta(
        filepath=str(filepath),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        codec=codec,
    )
    logger.info(
        "Loaded video: %s — %dx%d @ %.1f fps, %.1fs (%d frames)",
        filepath.name, width, height, fps, duration_sec, frame_count,
    )
    return cap, meta


def read_frame_at(cap: cv2.VideoCapture, timestamp_sec: float) -> np.ndarray | None:
    """Seek to *timestamp_sec* and read a single frame. Returns None on failure."""
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ok, frame = cap.read()
    return frame if ok else None


def downscale_frame(frame: np.ndarray, target_width: int = 1280, target_height: int = 720) -> np.ndarray:
    """Downscale a frame to the target resolution if larger."""
    h, w = frame.shape[:2]
    if w <= target_width and h <= target_height:
        return frame
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def crop_roi(frame: np.ndarray, roi_pct: list[float]) -> np.ndarray:
    """Crop a region of interest from a frame using percentage coordinates.

    *roi_pct* = [x%, y%, w%, h%] where each value is in [0.0, 1.0].
    """
    h, w = frame.shape[:2]
    x1 = int(roi_pct[0] * w)
    y1 = int(roi_pct[1] * h)
    x2 = int((roi_pct[0] + roi_pct[2]) * w)
    y2 = int((roi_pct[1] + roi_pct[3]) * h)
    return frame[y1:y2, x1:x2]
