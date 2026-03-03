"""Scoreboard OCR — extracts score, clock, mode from top-center panel.

Parses team scores, game clock, mode label, round counter, and
mode-specific info (hill timer, BO5 status).
"""

from __future__ import annotations

import logging
import re

import cv2
import numpy as np

from cod_analyst.config import AppConfig
from cod_analyst.game.models import GameMode, ScoreboardSnapshot
from cod_analyst.ingest.video_loader import crop_roi

logger = logging.getLogger(__name__)

_reader = None


def _get_reader(languages: list[str] | None = None):
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(languages or ["en"], gpu=True, verbose=False)
    return _reader


def _preprocess_scoreboard(roi: np.ndarray) -> np.ndarray:
    """Preprocess scoreboard ROI for OCR — high contrast white text on dark."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 160, 255, cv2.THRESH_BINARY)

    h, w = binary.shape
    if h < 100:
        scale = 100 / h
        binary = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return binary


def _parse_scores(text: str) -> tuple[int, int] | None:
    """Parse score text like '3 - 1' or '3-1' or '3  1'."""
    match = re.search(r"(\d+)\s*[-–—:]\s*(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Fallback: two numbers separated by whitespace
    nums = re.findall(r"\d+", text)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])

    return None


def _parse_clock(text: str) -> float | None:
    """Parse game clock like '1:45' or '0:23' into seconds."""
    match = re.search(r"(\d+):(\d{2})", text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return minutes * 60 + seconds
    return None


def _detect_mode_from_text(text: str) -> GameMode:
    """Detect game mode from scoreboard text content."""
    text_upper = text.upper()
    if "SEARCH" in text_upper or "S&D" in text_upper or "SND" in text_upper:
        return GameMode.SND
    elif "HARDPOINT" in text_upper or "HP" in text_upper:
        return GameMode.HARDPOINT
    elif "CONTROL" in text_upper or "CTRL" in text_upper:
        return GameMode.CONTROL
    return GameMode.UNKNOWN


def extract_scoreboard(
    frame: np.ndarray,
    cfg: AppConfig,
    timestamp: float,
) -> ScoreboardSnapshot:
    """Extract scoreboard data from the top-center ROI.

    Parameters
    ----------
    frame : np.ndarray
        Full 720p BGR frame.
    cfg : AppConfig
        Application configuration.
    timestamp : float
        Current frame timestamp.

    Returns
    -------
    ScoreboardSnapshot
        Parsed scoreboard state.
    """
    reader = _get_reader(cfg.hud.ocr_languages)

    roi = crop_roi(frame, cfg.hud.scoreboard_roi_pct)
    processed = _preprocess_scoreboard(roi)
    results = reader.readtext(processed)

    # Combine all text
    all_text = " ".join(text for _, text, conf in results if conf > 0.3)

    # Parse scores
    scores = _parse_scores(all_text)
    faze_score = scores[0] if scores else 0
    opp_score = scores[1] if scores else 0

    # Parse clock
    clock = _parse_clock(all_text)

    # Detect mode
    mode = _detect_mode_from_text(all_text)

    # Parse round number (SND specific)
    round_number = None
    round_match = re.search(r"R(?:OUND)?\s*(\d+)", all_text, re.IGNORECASE)
    if round_match:
        round_number = int(round_match.group(1))

    # Parse hill timer (Hardpoint specific)
    hill_timer = None
    hill_match = re.search(r"(\d+:\d{2})\s*$", all_text)
    if hill_match and mode == GameMode.HARDPOINT:
        hill_timer = _parse_clock(hill_match.group(1))

    snapshot = ScoreboardSnapshot(
        faze_score=faze_score,
        opponent_score=opp_score,
        game_clock=clock or 0.0,
        mode=mode,
        round_number=round_number,
        hill_timer=hill_timer,
        timestamp=timestamp,
    )

    logger.debug(
        "Scoreboard at %.1fs: %d-%d, clock=%.0fs, mode=%s",
        timestamp, faze_score, opp_score, clock or 0, mode.value,
    )

    return snapshot
