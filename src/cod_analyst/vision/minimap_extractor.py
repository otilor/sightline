"""Minimap extractor — crops and preprocesses the minimap ROI.

Provides cleaned minimap frames for downstream player detection
and color clustering.
"""

from __future__ import annotations

import cv2
import numpy as np

from cod_analyst.config import AppConfig
from cod_analyst.ingest.video_loader import crop_roi


def extract_minimap(frame: np.ndarray, cfg: AppConfig) -> np.ndarray:
    """Crop the minimap region from a full frame and preprocess.

    Parameters
    ----------
    frame : np.ndarray
        Full 720p BGR frame.
    cfg : AppConfig
        Application configuration (minimap ROI).

    Returns
    -------
    np.ndarray
        Cropped and preprocessed minimap image (BGR).
    """
    roi = crop_roi(frame, cfg.minimap.roi_pct)

    # Light preprocessing: denoise to reduce compression artifacts
    roi = cv2.GaussianBlur(roi, (3, 3), 0)

    return roi


def enhance_dots(minimap: np.ndarray) -> np.ndarray:
    """Enhance player dot visibility for better detection.

    Increases saturation and contrast to make colored dots pop
    against the muted map background.
    """
    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Boost saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)

    # Increase value contrast
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)

    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
