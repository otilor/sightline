"""Kill feed parser — extracts kill events from the mid-left panel.

Combines OCR for player names with template matching for weapon icons.
Processed at high frame rate since entries are transient (~3 seconds).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cod_analyst.config import AppConfig
from cod_analyst.game.models import KillEvent
from cod_analyst.ingest.video_loader import crop_roi

logger = logging.getLogger(__name__)

_reader = None


def _get_reader(languages: list[str] | None = None):
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(languages or ["en"], gpu=True, verbose=False)
    return _reader


@dataclass
class WeaponTemplate:
    """A weapon icon template for matching."""
    name: str
    template: np.ndarray
    w: int
    h: int


class KillfeedParser:
    """Extracts kill events from the kill feed region.

    Each kill feed entry contains: killer name → [weapon icon] → victim name.
    Names are extracted via OCR, weapons via template matching.
    """

    def __init__(self, weapon_templates_dir: str | Path | None = None):
        self._weapon_templates: list[WeaponTemplate] = []
        self._recent_events: list[KillEvent] = []
        self._dedup_window_sec: float = 3.0  # Kill feed entries last ~3 seconds

        if weapon_templates_dir:
            self._load_weapon_templates(weapon_templates_dir)

    def _load_weapon_templates(self, templates_dir: str | Path) -> None:
        """Load weapon icon templates from a directory.

        Each template should be a PNG named after the weapon (e.g., 'mp5.png').
        """
        templates_path = Path(templates_dir)
        if not templates_path.exists():
            logger.warning("Weapon templates directory not found: %s", templates_dir)
            return

        for img_path in sorted(templates_path.glob("*.png")):
            template = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                h, w = template.shape
                self._weapon_templates.append(WeaponTemplate(
                    name=img_path.stem,
                    template=template,
                    w=w,
                    h=h,
                ))

        logger.info("Loaded %d weapon templates", len(self._weapon_templates))

    def _match_weapon(self, region: np.ndarray) -> str:
        """Find the best matching weapon icon in the region.

        Uses normalized cross-correlation template matching.
        """
        if not self._weapon_templates:
            return "unknown"

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region

        best_score = -1.0
        best_weapon = "unknown"

        for wt in self._weapon_templates:
            # Skip if template is larger than region
            if wt.h > gray.shape[0] or wt.w > gray.shape[1]:
                continue

            result = cv2.matchTemplate(gray, wt.template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score and max_val > 0.6:
                best_score = max_val
                best_weapon = wt.name

        return best_weapon

    def _preprocess_killfeed(self, roi: np.ndarray) -> np.ndarray:
        """Preprocess kill feed ROI for better OCR."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 140, 255, cv2.THRESH_BINARY)

        h, w = binary.shape
        if h < 300:
            scale = 300 / h
            binary = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        return binary

    def _is_duplicate(self, event: KillEvent) -> bool:
        """Check if this kill event was already detected recently."""
        for recent in self._recent_events:
            if (abs(recent.timestamp - event.timestamp) < self._dedup_window_sec
                    and recent.killer == event.killer
                    and recent.victim == event.victim):
                return True
        return False

    def parse_killfeed(
        self,
        frame: np.ndarray,
        cfg: AppConfig,
        timestamp: float,
    ) -> list[KillEvent]:
        """Extract kill events from the kill feed region.

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
        list[KillEvent]
            Newly detected kill events (deduplicated).
        """
        reader = _get_reader(cfg.hud.ocr_languages)

        roi = crop_roi(frame, cfg.hud.killfeed_roi_pct)
        processed = self._preprocess_killfeed(roi)

        results = reader.readtext(processed)

        # Parse kill feed entries
        # Each entry is on a separate line: KILLER [weapon_icon] VICTIM
        events: list[KillEvent] = []
        lines: list[list[tuple]] = []

        # Group by Y position
        current_y = -100
        current_line: list[tuple] = []

        sorted_results = sorted(results, key=lambda r: r[0][0][1])

        for bbox, text, conf in sorted_results:
            if conf < 0.3:
                continue
            y_center = (bbox[0][1] + bbox[2][1]) / 2

            if abs(y_center - current_y) > 20:
                if current_line:
                    lines.append(current_line)
                current_line = [(bbox, text.strip(), conf)]
                current_y = y_center
            else:
                current_line.append((bbox, text.strip(), conf))

        if current_line:
            lines.append(current_line)

        # Process each line as a potential kill event
        for line_tokens in lines:
            texts = [t[1] for t in line_tokens]
            full_line = " ".join(texts)

            # Look for name patterns (at least 2 words/names in line)
            names = [t for t in texts if len(t) >= 2 and t[0].isupper()]

            if len(names) >= 2:
                killer = names[0]
                victim = names[-1]

                # Try weapon matching on the original (color) ROI region
                weapon = self._match_weapon(roi)

                event = KillEvent(
                    killer=killer,
                    victim=victim,
                    weapon=weapon,
                    killer_grid_cell="",
                    timestamp=timestamp,
                )

                if not self._is_duplicate(event):
                    events.append(event)

        # Update recent events (keep last N seconds)
        self._recent_events = [
            e for e in self._recent_events
            if timestamp - e.timestamp < self._dedup_window_sec * 2
        ] + events

        if events:
            logger.debug(
                "Kill feed at %.1fs: %d events — %s",
                timestamp, len(events),
                ", ".join(f"{e.killer}>{e.victim}" for e in events),
            )

        return events
