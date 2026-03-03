"""Game mode classifier — identifies SND, Hardpoint, or Control.

Uses ResNet-18 fine-tuned on mode screenshots with OCR text override.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from cod_analyst.config import AppConfig
from cod_analyst.game.models import GameMode

logger = logging.getLogger(__name__)


class ModeClassifier:
    """Classifies game mode from a full frame.

    Two-stage approach:
    1. OCR override — if scoreboard text contains mode name, use it directly
    2. ResNet-18 — visual classification as fallback
    """

    # Matches the indices of the trained ResNet classes: ['hp', 'menu', 'snd', 'unknown']
    _MODE_LABELS = [
        GameMode.HARDPOINT,
        GameMode.MENU,
        GameMode.SND,
        GameMode.UNKNOWN,
    ]

    _TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.8,
    ):
        self.confidence_threshold = confidence_threshold
        self._model: nn.Module | None = None
        self._model_path = model_path
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> None:
        """Load or create the ResNet-18 mode classifier."""
        if self._model is not None:
            return

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self._MODE_LABELS))

        if self._model_path and Path(self._model_path).exists():
            state = torch.load(self._model_path, map_location=self._device, weights_only=True)
            model.load_state_dict(state)
            logger.info("Loaded mode classifier from %s", self._model_path)
        else:
            logger.warning(
                "No trained mode classifier found at %s — using untrained model. "
                "Run `sightline label --mode` to create training data.",
                self._model_path,
            )

        model = model.to(self._device)
        model.eval()
        self._model = model

    def classify(self, frame: np.ndarray) -> tuple[GameMode, float]:
        """Classify the game mode from a full frame.

        Returns
        -------
        tuple[GameMode, float]
            (predicted_mode, confidence)
        """
        if self._model is None:
            self.load_model()

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Transform and predict
        tensor = self._TRANSFORM(rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = probs.max(dim=1)

        conf = float(confidence.item())
        mode = self._MODE_LABELS[predicted.item()]

        if conf < self.confidence_threshold:
            return GameMode.UNKNOWN, conf

        return mode, conf

    def classify_with_ocr_override(
        self,
        frame: np.ndarray,
        ocr_mode: GameMode | None = None,
    ) -> tuple[GameMode, float]:
        """Classify mode with OCR text taking priority.

        If the scoreboard OCR has already detected the mode text,
        use that directly with confidence 1.0.
        """
        if ocr_mode and ocr_mode != GameMode.UNKNOWN:
            return ocr_mode, 1.0

        return self.classify(frame)

    def save_model(self, path: str | Path) -> None:
        """Save the trained model weights."""
        if self._model is None:
            raise RuntimeError("No model to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
        logger.info("Mode classifier saved to %s", path)
