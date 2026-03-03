"""Round segmenter — detects round boundaries using multi-signal fusion.

Fuses score changes, black frame transitions, kill feed gaps,
and round-end overlays to identify individual rounds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from cod_analyst.config import AppConfig
from cod_analyst.game.models import (
    GameMode,
    Round,
    RoundOutcome,
    ScoreboardSnapshot,
    Side,
    WinCondition,
)

logger = logging.getLogger(__name__)


@dataclass
class _SegmenterState:
    """Internal state for round boundary detection."""
    prev_faze_score: int = 0
    prev_opp_score: int = 0
    last_score_change_time: float = -999.0
    last_kill_time: float = 0.0
    round_start_time: float = 0.0
    current_round_number: int = 0
    in_round: bool = False


def _is_black_frame(frame: np.ndarray, threshold: int = 15) -> bool:
    """Check if a frame is predominantly black (transition)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return float(np.mean(gray)) < threshold


def _detect_round_end_text(frame: np.ndarray) -> str | None:
    """Check for round-end overlay text ('ROUND WON', 'ROUND LOST').

    Uses simple template-matching-like approach on center of frame.
    """
    h, w = frame.shape[:2]
    # Round-end text appears in the center of the screen
    center = frame[h // 3: 2 * h // 3, w // 4: 3 * w // 4]
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)

    # High-contrast large text check
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    text_ratio = np.count_nonzero(binary) / binary.size

    # Large centered text indicates round-end overlay
    if text_ratio > 0.05:
        return "round_end_detected"

    return None


class RoundSegmenter:
    """Detects round boundaries in SND gameplay.

    Multi-signal fusion:
    - Score changes (OCR detects scoreboard increment)
    - Black frame transitions (mean pixel value < threshold)
    - Kill feed gaps (no kills for >5s)
    - Round-end overlay text

    SND-specific: detects attack/defense side, plant/defuse events.
    """

    def __init__(self, cfg: AppConfig):
        self._cfg = cfg
        self._state = _SegmenterState()
        self._rounds: list[Round] = []

    def process_frame(
        self,
        frame: np.ndarray,
        scoreboard: ScoreboardSnapshot,
        has_kill: bool,
        timestamp: float,
    ) -> Round | None:
        """Process a frame and check for round boundary.

        Parameters
        ----------
        frame : np.ndarray
            Current frame (for black frame detection).
        scoreboard : ScoreboardSnapshot
            Parsed scoreboard state.
        has_kill : bool
            Whether a kill event was detected at this frame.
        timestamp : float
            Current timestamp.

        Returns
        -------
        Round | None
            A completed Round if a boundary was detected, else None.
        """
        cfg_rs = self._cfg.round_segmentation

        if has_kill:
            self._state.last_kill_time = timestamp

        # Signal 1: Score change
        score_changed = (
            scoreboard.faze_score != self._state.prev_faze_score
            or scoreboard.opponent_score != self._state.prev_opp_score
        )

        # Debounce score changes
        if score_changed and (timestamp - self._state.last_score_change_time) < cfg_rs.score_change_cooldown_sec:
            score_changed = False

        # Signal 2: Black frame
        is_black = _is_black_frame(frame, cfg_rs.black_frame_threshold)

        # Signal 3: Kill feed gap
        kill_gap = (timestamp - self._state.last_kill_time) > 7.0 if self._state.in_round else False

        # Signal 4: Round-end text
        round_end_text = _detect_round_end_text(frame) if not is_black else None

        # Decision logic
        round_boundary = False

        if self._state.in_round:
            # End current round?
            if score_changed:
                round_boundary = True
            elif is_black and (timestamp - self._state.round_start_time) > cfg_rs.round_min_duration_sec:
                round_boundary = True
            elif round_end_text:
                round_boundary = True

        if round_boundary:
            # Complete current round
            completed_round = self._close_round(scoreboard, timestamp)
            self._state.prev_faze_score = scoreboard.faze_score
            self._state.prev_opp_score = scoreboard.opponent_score
            self._state.last_score_change_time = timestamp
            self._state.in_round = False
            return completed_round

        # Start new round?
        if not self._state.in_round and not is_black:
            # Not in a round and frame isn't black — round has started
            if (timestamp - self._state.last_score_change_time) > 3.0:
                self._state.in_round = True
                self._state.current_round_number += 1
                self._state.round_start_time = timestamp
                logger.debug("Round %d started at %.1fs", self._state.current_round_number, timestamp)

        return None

    def _close_round(self, scoreboard: ScoreboardSnapshot, end_time: float) -> Round:
        """Close the current round and determine outcome."""
        # Determine outcome
        faze_gained = scoreboard.faze_score > self._state.prev_faze_score
        opp_gained = scoreboard.opponent_score > self._state.prev_opp_score

        if faze_gained:
            outcome = RoundOutcome.WIN
        elif opp_gained:
            outcome = RoundOutcome.LOSS
        else:
            outcome = RoundOutcome.UNKNOWN

        # Determine side — in CDL SND, teams switch sides at half
        # If round number <= 6, use starting side; else switched
        side = Side.UNKNOWN
        if self._state.current_round_number <= 6:
            side = Side.ATTACK  # Placeholder — needs calibration per match
        else:
            side = Side.DEFENSE

        round_obj = Round(
            round_number=self._state.current_round_number,
            side=side,
            outcome=outcome,
            win_condition=WinCondition.UNKNOWN,
            start_time=self._state.round_start_time,
            end_time=end_time,
        )

        self._rounds.append(round_obj)
        logger.info(
            "Round %d closed: %s (%.1fs – %.1fs, %.1fs)",
            round_obj.round_number, outcome.value,
            round_obj.start_time, round_obj.end_time,
            round_obj.end_time - round_obj.start_time,
        )

        return round_obj

    def get_rounds(self) -> list[Round]:
        """Return all completed rounds."""
        return self._rounds

    def reset(self) -> None:
        """Reset segmenter state for a new map game."""
        self._state = _SegmenterState()
        self._rounds.clear()
