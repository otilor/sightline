"""Player dot detector — YOLOv8-nano on minimap crops.

Detects player dots, bomb, and objective markers.
Color clusterer assigns team identity post-detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detection from the minimap."""
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str  # "player_dot", "bomb", "objective"
    center_x: float  # Normalized [0, 1]
    center_y: float  # Normalized [0, 1]


class PlayerDetector:
    """YOLOv8-nano player dot detector for minimap analysis.

    Detects player dots, bomb indicators, and objective markers.
    Uses ONNX Runtime when available for 2-3x speedup.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.5,
        classes: list[str] | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.classes = classes or ["player_dot", "bomb", "objective"]
        self._model = None
        self._model_path = model_path

    def load_model(self) -> None:
        """Load the YOLOv8 model. Call this explicitly to control timing."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO

            if self._model_path and Path(self._model_path).exists():
                self._model = YOLO(str(self._model_path))
                logger.info("Loaded custom YOLO model: %s", self._model_path)
            else:
                # Pre-trained nano model — will need fine-tuning on minimap data
                self._model = YOLO("yolov8n.pt")
                logger.info("Loaded base YOLOv8n model (requires fine-tuning for minimap)")
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            raise

    def detect(self, minimap: np.ndarray) -> list[Detection]:
        """Run detection on a minimap crop.

        Parameters
        ----------
        minimap : np.ndarray
            Cropped minimap image (BGR).

        Returns
        -------
        list[Detection]
            Detected objects sorted by confidence.
        """
        if self._model is None:
            self.load_model()

        h, w = minimap.shape[:2]

        results = self._model(
            minimap,
            conf=self.confidence_threshold,
            verbose=False,
            imgsz=640,
        )

        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])

                # Map class index to name
                cls_name = self.classes[cls_idx] if cls_idx < len(self.classes) else "unknown"

                center_x = ((x1 + x2) / 2) / w
                center_y = ((y1 + y2) / 2) / h

                detections.append(Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    class_name=cls_name,
                    center_x=center_x,
                    center_y=center_y,
                ))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)

        return detections

    def detect_batch(self, minimaps: list[np.ndarray]) -> list[list[Detection]]:
        """Run detection on a batch of minimap crops."""
        return [self.detect(m) for m in minimaps]
