"""Roster table OCR — extracts player stats from top-left and top-right panels.

Parses player name, K/D, streak, and time from roster table ROIs using EasyOCR.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np

from cod_analyst.config import AppConfig
from cod_analyst.game.models import PlayerStatSnapshot
from cod_analyst.ingest.video_loader import crop_roi

logger = logging.getLogger(__name__)

# Lazy-load EasyOCR reader
_reader = None


def _get_reader(languages: list[str] | None = None):
    """Lazily initialize EasyOCR reader."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(languages or ["en"], gpu=True, verbose=False)
    return _reader


def _preprocess_roster_roi(roi: np.ndarray) -> np.ndarray:
    """Preprocess roster ROI for better OCR results.

    Roster tables have white/light text on dark backgrounds.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Threshold to isolate text
    _, binary = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)

    # Scale up for better OCR (EasyOCR works better on larger text)
    h, w = binary.shape
    if h < 200:
        scale = 200 / h
        binary = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return binary


def _parse_kd(text: str) -> tuple[int, int] | None:
    """Parse a K/D string like '12/5' or '12-5' into (kills, deaths)."""
    match = re.search(r"(\d+)\s*[/\-]\s*(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _parse_roster_text(ocr_results: list, team: str) -> list[PlayerStatSnapshot]:
    """Parse OCR results into PlayerStatSnapshot objects.

    CDL roster format (per row): [#] PLAYERNAME  K/D  STREAK  TIME
    """
    snapshots = []
    lines = []

    # Group OCR results by Y position into lines
    for bbox, text, conf in ocr_results:
        if conf < 0.3:  # Skip very low confidence
            continue
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        lines.append((y_center, text.strip(), conf))

    # Sort by vertical position
    lines.sort(key=lambda x: x[0])

    # Try to parse each line as a player entry
    current_line_y = -100
    current_tokens = []

    for y, text, conf in lines:
        # Group tokens that are on the same line (within 10px)
        if abs(y - current_line_y) > 15:
            # Process previous line
            if current_tokens:
                snapshot = _tokens_to_snapshot(current_tokens, team, 0.0)
                if snapshot:
                    snapshots.append(snapshot)
            current_tokens = [(text, conf)]
            current_line_y = y
        else:
            current_tokens.append((text, conf))

    # Process last line
    if current_tokens:
        snapshot = _tokens_to_snapshot(current_tokens, team, 0.0)
        if snapshot:
            snapshots.append(snapshot)

    return snapshots


def _tokens_to_snapshot(
    tokens: list[tuple[str, float]],
    team: str,
    timestamp: float,
) -> PlayerStatSnapshot | None:
    """Convert OCR tokens from a single line into a stat snapshot."""
    full_text = " ".join(t[0] for t in tokens)

    # Try to find player name (typically all caps, 3-8 chars)
    name_match = re.search(r"\b([A-Z][A-Za-z]{2,12})\b", full_text)
    player_name = name_match.group(1) if name_match else None

    if not player_name:
        return None

    # Try to find K/D
    kd = _parse_kd(full_text)
    kills, deaths = kd if kd else (0, 0)

    # Try to find streak
    streak_match = re.search(r"\b(\d{1,2})\b", full_text.split("/")[-1] if "/" in full_text else "")
    streak = int(streak_match.group(1)) if streak_match else 0

    return PlayerStatSnapshot(
        player_name=player_name,
        team=team,
        kills=kills,
        deaths=deaths,
        streak=streak,
        time_on_obj=0.0,
        timestamp=timestamp,
    )


def extract_roster(
    frame: np.ndarray,
    cfg: AppConfig,
    timestamp: float,
) -> tuple[list[PlayerStatSnapshot], list[PlayerStatSnapshot]]:
    """Extract player stats from both roster table ROIs.

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
    tuple[list[PlayerStatSnapshot], list[PlayerStatSnapshot]]
        (faze_players, opponent_players) stat snapshots.
    """
    reader = _get_reader(cfg.hud.ocr_languages)

    # Crop and preprocess both roster regions
    faze_roi = crop_roi(frame, cfg.hud.faze_roster_roi_pct)
    opp_roi = crop_roi(frame, cfg.hud.opponent_roster_roi_pct)

    faze_processed = _preprocess_roster_roi(faze_roi)
    opp_processed = _preprocess_roster_roi(opp_roi)

    # Run OCR
    faze_results = reader.readtext(faze_processed)
    opp_results = reader.readtext(opp_processed)

    # Parse results
    faze_stats = _parse_roster_text(faze_results, "faze")
    opp_stats = _parse_roster_text(opp_results, "opponent")

    # Set timestamps
    for s in faze_stats:
        s.timestamp = timestamp
    for s in opp_stats:
        s.timestamp = timestamp

    logger.debug(
        "Roster OCR at %.1fs: %d faze, %d opponent players",
        timestamp, len(faze_stats), len(opp_stats),
    )

    return faze_stats, opp_stats
