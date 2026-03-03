# Sightline

> AI-powered Call of Duty League VOD analysis platform for competitive team strategy.

Sightline processes tournament VODs to extract player positions, kill events, and game state — then uses ML models and LLM narration to generate actionable scouting reports and strategy suggestions.

## Architecture

```
VOD File → Gameplay Detection → Adaptive Frame Sampling
    → Minimap Extraction → YOLOv8 Player Detection → CIELAB Color Clustering → ByteTrack Tracking
    → HUD OCR (Roster, Scoreboard, Kill Feed)
    → Round Segmentation → Feature Engineering
    → ML Models (LSTM, Transformer, DTW+DBSCAN, GMM, UMAP)
    → Strategy Engine (Profiler, SND Strategist, LLM Narrator)
    → Database Storage → Web Dashboard
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Initialize database
sightline init

# Process a VOD
sightline process match.mp4 --opponent OpTic --event "Major 2"

# Download from YouTube
sightline download "https://youtube.com/playlist?list=..." --keyword faze --limit 20

# Scout an opponent
sightline scout OpTic

# Start web platform
sightline serve
```

## Project Structure

```
src/cod_analyst/
├── config.py              # Config loader
├── pipeline.py            # Main orchestrator
├── cli.py                 # CLI interface
├── ingest/                # Video ingestion
│   ├── video_loader.py    # Video I/O and metadata
│   ├── gameplay_detector.py  # Skip non-gameplay
│   ├── frame_sampler.py   # Adaptive 3-tier sampling
│   └── downloader.py      # YouTube playlist download
├── vision/                # Computer vision
│   ├── minimap_extractor.py  # ROI extraction
│   ├── color_clusterer.py    # CIELAB K-Means team ID
│   ├── player_detector.py    # YOLOv8 dot detection
│   ├── player_tracker.py     # ByteTrack tracking
│   ├── roster_ocr.py         # Player stats OCR
│   ├── scoreboard_ocr.py     # Score/clock OCR
│   ├── killfeed_parser.py    # Kill feed + weapon matching
│   └── mode_classifier.py    # ResNet-18 mode classification
├── game/                  # Game logic
│   ├── models.py          # Domain models & enums
│   ├── map_grid.py        # 5×5 grid with callout aliases
│   └── round_segmenter.py # Multi-signal round boundaries
├── features/              # Feature engineering
│   ├── movement.py        # Speed, heading, zone time
│   ├── formation.py       # Centroid, spread, hull, buddy
│   └── kill_events.py     # First blood, trades, heatmaps
├── ml/                    # Machine learning
│   ├── trajectory_lstm.py # Movement prediction
│   ├── event_transformer.py  # Tipping-point detection
│   ├── route_clustering.py   # DTW + DBSCAN
│   ├── formation_clustering.py  # GMM formations
│   └── playstyle_embedding.py   # UMAP fingerprints
├── analysis/              # Strategy engine
│   ├── profiler.py        # Scouting reports
│   ├── strategist.py      # SND pre-round + loss analysis
│   └── narrator.py        # LLM narration (OpenAI/Gemini)
├── db/                    # Database
│   └── schemas.py         # SQLModel ORM schemas
└── api/                   # Web backend
    └── routes.py          # FastAPI REST endpoints
```

## Configuration

All settings in `config.yaml` — video processing, ROIs, model paths, LLM providers, database URL.

## Requirements

Python 3.11+, PyTorch, OpenCV, Ultralytics, EasyOCR, scikit-learn, FastAPI.
