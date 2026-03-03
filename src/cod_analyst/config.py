"""Configuration loader for Sightline.

Reads config.yaml into typed dataclasses with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    vods_dir: str = "./data/vods"
    annotations_dir: str = "./data/annotations"
    processed_dir: str = "./data/processed"
    models_dir: str = "./models"
    maps_dir: str = "./maps"

    def resolve(self, root: Path) -> None:
        """Resolve all paths relative to *root* and create dirs."""
        for attr in ("vods_dir", "annotations_dir", "processed_dir", "models_dir", "maps_dir"):
            p = Path(getattr(self, attr))
            if not p.is_absolute():
                p = root / p
            p.mkdir(parents=True, exist_ok=True)
            setattr(self, attr, str(p))


@dataclass
class VideoConfig:
    sample_fps_discovery: float = 0.5
    sample_fps_tactical: float = 5.0
    sample_fps_burst: float = 12.0
    burst_window_sec: float = 5.0
    min_resolution: list[int] = field(default_factory=lambda: [1280, 720])
    target_resolution: list[int] = field(default_factory=lambda: [1280, 720])
    scene_change_threshold: float = 30.0
    batch_size: int = 32


@dataclass
class MinimapConfig:
    roi_pct: list[float] = field(default_factory=lambda: [0.0, 0.75, 0.20, 0.25])
    yolo_model: str = "yolov8n"
    confidence_threshold: float = 0.5
    detection_classes: list[str] = field(
        default_factory=lambda: ["player_dot", "bomb", "objective"]
    )
    temporal_vote_window: int = 10


@dataclass
class HUDConfig:
    faze_roster_roi_pct: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.20, 0.15])
    opponent_roster_roi_pct: list[float] = field(default_factory=lambda: [0.80, 0.0, 0.20, 0.15])
    scoreboard_roi_pct: list[float] = field(default_factory=lambda: [0.30, 0.0, 0.40, 0.08])
    killfeed_roi_pct: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.20, 0.40])
    spectated_player_roi_pct: list[float] = field(
        default_factory=lambda: [0.75, 0.80, 0.25, 0.20]
    )
    ocr_languages: list[str] = field(default_factory=lambda: ["en"])
    ocr_confidence_threshold: float = 0.6
    roster_sample_interval_sec: float = 2.5
    scoreboard_sample_interval_sec: float = 1.0


@dataclass
class ModeClassifierConfig:
    model_path: str = "./models/mode_classifier.pt"
    classes: list[str] = field(default_factory=lambda: ["snd", "hardpoint", "control"])
    confidence_threshold: float = 0.8


@dataclass
class RoundSegmentationConfig:
    black_frame_threshold: int = 15
    score_change_cooldown_sec: float = 5.0
    round_min_duration_sec: float = 15.0


@dataclass
class MapGridConfig:
    grid_size: int = 5


@dataclass
class AnalysisConfig:
    llm_provider: str = "openai"
    openai_model: str = "gpt-4o"
    gemini_model: str = "gemini-2.0-flash"
    max_tokens: int = 2000
    temperature: float = 0.7


@dataclass
class TrackerConfig:
    algorithm: str = "bytetrack"
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8


@dataclass
class DatabaseConfig:
    url: str = "sqlite:///./data/sightline.db"


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True


@dataclass
class DownloaderConfig:
    default_resolution: int = 720
    output_format: str = "mp4"


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    minimap: MinimapConfig = field(default_factory=MinimapConfig)
    hud: HUDConfig = field(default_factory=HUDConfig)
    mode_classifier: ModeClassifierConfig = field(default_factory=ModeClassifierConfig)
    round_segmentation: RoundSegmentationConfig = field(
        default_factory=RoundSegmentationConfig
    )
    map_grid: MapGridConfig = field(default_factory=MapGridConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    web: WebConfig = field(default_factory=WebConfig)
    downloader: DownloaderConfig = field(default_factory=DownloaderConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_CONFIG_SEARCH_PATHS = [
    Path("config.yaml"),
    Path("config.yml"),
    Path(os.environ.get("SIGHTLINE_CONFIG", "")),
]

_SECTION_MAP: dict[str, type] = {
    "paths": PathsConfig,
    "video": VideoConfig,
    "minimap": MinimapConfig,
    "hud": HUDConfig,
    "mode_classifier": ModeClassifierConfig,
    "round_segmentation": RoundSegmentationConfig,
    "map_grid": MapGridConfig,
    "analysis": AnalysisConfig,
    "tracker": TrackerConfig,
    "database": DatabaseConfig,
    "web": WebConfig,
    "downloader": DownloaderConfig,
}


def _build_section(cls: type, raw: dict[str, Any] | None) -> Any:
    if raw is None:
        return cls()
    # Only pass fields that exist on the dataclass
    valid = {k: v for k, v in raw.items() if hasattr(cls, k) or k in cls.__dataclass_fields__}
    return cls(**valid)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file.

    Resolution order:
    1. Explicit *path* argument
    2. ``SIGHTLINE_CONFIG`` environment variable
    3. ``config.yaml`` in CWD
    4. Pure defaults
    """
    config_path: Path | None = None
    if path is not None:
        config_path = Path(path)
    else:
        for candidate in _CONFIG_SEARCH_PATHS:
            if candidate and candidate.exists():
                config_path = candidate
                break

    raw: dict[str, Any] = {}
    if config_path and config_path.exists():
        raw = yaml.safe_load(config_path.read_text()) or {}

    cfg = AppConfig(
        **{key: _build_section(cls, raw.get(key)) for key, cls in _SECTION_MAP.items()}
    )

    # Resolve relative paths
    root = config_path.parent.resolve() if config_path else Path.cwd()
    cfg.paths.resolve(root)

    return cfg
