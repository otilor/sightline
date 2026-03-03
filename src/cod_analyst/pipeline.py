"""Main processing pipeline — orchestrates the full VOD analysis flow.

Ties together: ingestion → vision → game logic → features → ML → strategy → DB.
"""

from __future__ import annotations

import logging
from pathlib import Path

from cod_analyst.config import AppConfig, load_config
from cod_analyst.db.schemas import (
    DBKillEvent,
    DBMapGame,
    DBMatch,
    DBPlayerPosition,
    DBRound,
    DBStatSnapshot,
    VOD,
    get_session,
    init_db,
)
from cod_analyst.game.map_grid import MapGrid
from cod_analyst.game.models import GameMode
from cod_analyst.game.round_segmenter import RoundSegmenter
from cod_analyst.ingest.frame_sampler import sample_gameplay
from cod_analyst.ingest.gameplay_detector import detect_gameplay
from cod_analyst.ingest.video_loader import crop_roi, load_video
from cod_analyst.vision.color_clusterer import ColorClusterer
from cod_analyst.vision.killfeed_parser import KillfeedParser
from cod_analyst.vision.minimap_extractor import extract_minimap
from cod_analyst.vision.mode_classifier import ModeClassifier
from cod_analyst.vision.player_detector import PlayerDetector
from cod_analyst.vision.player_tracker import PlayerTracker
from cod_analyst.vision.roster_ocr import extract_roster
from cod_analyst.vision.scoreboard_ocr import extract_scoreboard

logger = logging.getLogger(__name__)


class SightlinePipeline:
    """Orchestrates the full VOD analysis pipeline.

    Pipeline stages:
    1. Load video → extract metadata
    2. Detect gameplay windows (skip production/intros)
    3. Sample frames adaptively (tactical + burst)
    4. Per-frame extraction:
       a. Minimap → detect dots → cluster colors → track → positions
       b. Roster OCR → player stats
       c. Scoreboard OCR → scores, clock, mode
       d. Kill feed → kill events
    5. Round segmentation
    6. Feature engineering
    7. Database storage
    """

    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or load_config()

        # Vision components (lazy-loaded)
        self._detector: PlayerDetector | None = None
        self._tracker: PlayerTracker | None = None
        self._clusterer: ColorClusterer | None = None
        self._killfeed: KillfeedParser | None = None
        self._mode_clf: ModeClassifier | None = None
        self._grid: MapGrid | None = None
        self._segmenter: RoundSegmenter | None = None

    def _init_components(self) -> None:
        """Initialize all vision and game logic components."""
        if self._detector is not None:
            return

        self._detector = PlayerDetector(
            model_path=self.cfg.minimap.yolo_model,
            confidence_threshold=self.cfg.minimap.confidence_threshold,
            classes=self.cfg.minimap.detection_classes,
        )
        self._tracker = PlayerTracker(
            track_thresh=self.cfg.tracker.track_thresh,
            track_buffer=self.cfg.tracker.track_buffer,
            match_thresh=self.cfg.tracker.match_thresh,
        )
        self._clusterer = ColorClusterer(
            temporal_window=self.cfg.minimap.temporal_vote_window,
        )
        self._killfeed = KillfeedParser(
            weapon_templates_dir=Path(self.cfg.paths.annotations_dir) / "weapons",
        )
        self._mode_clf = ModeClassifier(
            model_path=self.cfg.mode_classifier.model_path,
            confidence_threshold=self.cfg.mode_classifier.confidence_threshold,
        )
        self._grid = MapGrid(
            grid_size=self.cfg.map_grid.grid_size,
            maps_dir=self.cfg.paths.maps_dir,
        )
        self._segmenter = RoundSegmenter(self.cfg)

    def process_vod(
        self,
        vod_path: str | Path,
        opponent: str = "",
        event_name: str = "",
    ) -> int:
        """Process a single VOD through the full pipeline.

        Parameters
        ----------
        vod_path : str | Path
            Path to the VOD file.
        opponent : str
            Opponent team name.
        event_name : str
            Tournament/event name.

        Returns
        -------
        int
            Number of frames processed.
        """
        self._init_components()
        init_db(self.cfg.database.url)
        session = get_session(self.cfg.database.url)

        vod_path = Path(vod_path)
        logger.info("Processing VOD: %s", vod_path.name)

        # ---- Stage 1: Load video ----
        cap, meta = load_video(vod_path)

        # Save VOD record
        vod_record = VOD(
            filename=vod_path.name,
            filepath=str(vod_path),
            duration_sec=meta.duration_sec,
            resolution=f"{meta.width}x{meta.height}",
        )
        session.add(vod_record)
        session.commit()
        session.refresh(vod_record)

        # ---- Stage 2: Detect gameplay ----
        windows = detect_gameplay(cap, meta, self.cfg)
        if not windows:
            logger.warning("No gameplay detected in %s", vod_path.name)
            cap.release()
            return 0

        # ---- Create match record ----
        match_record = DBMatch(
            vod_id=vod_record.id,
            opponent=opponent,
            event_name=event_name,
        )
        session.add(match_record)
        session.commit()
        session.refresh(match_record)

        # ---- Stage 3-4: Sample and extract ----
        frame_count = 0
        current_game: DBMapGame | None = None
        current_round: DBRound | None = None
        detected_mode = GameMode.UNKNOWN
        last_scoreboard_time = -999.0
        last_roster_time = -999.0

        for sf in sample_gameplay(cap, meta, windows, self.cfg):
            frame = sf.image
            ts = sf.timestamp
            frame_count += 1

            # ---- Scoreboard OCR (every ~1s) ----
            if ts - last_scoreboard_time >= self.cfg.hud.scoreboard_sample_interval_sec:
                scoreboard = extract_scoreboard(frame, self.cfg, ts)
                last_scoreboard_time = ts

                # Detect mode
                if detected_mode == GameMode.UNKNOWN and scoreboard.mode != GameMode.UNKNOWN:
                    detected_mode = scoreboard.mode
                    logger.info("Mode detected: %s at %.1fs", detected_mode.value, ts)

                    # Create map game record
                    current_game = DBMapGame(
                        match_id=match_record.id,
                        map_name="",
                        mode=detected_mode.value,
                    )
                    session.add(current_game)
                    session.commit()
                    session.refresh(current_game)

                # Round segmentation
                has_kill = False  # Will be updated below
                result = self._segmenter.process_frame(frame, scoreboard, has_kill, ts)

                if result and current_game:
                    current_round = DBRound(
                        map_game_id=current_game.id,
                        round_number=result.round_number,
                        side=result.side.value,
                        outcome=result.outcome.value,
                        win_condition=result.win_condition.value,
                        start_time=result.start_time,
                        end_time=result.end_time,
                    )
                    session.add(current_round)
                    session.commit()
                    session.refresh(current_round)

            # ---- Roster OCR (every ~2.5s) ----
            if ts - last_roster_time >= self.cfg.hud.roster_sample_interval_sec:
                faze_stats, opp_stats = extract_roster(frame, self.cfg, ts)
                last_roster_time = ts

                if current_round:
                    for stat in faze_stats + opp_stats:
                        db_stat = DBStatSnapshot(
                            round_id=current_round.id,
                            player_name=stat.player_name,
                            team=stat.team,
                            kills=stat.kills,
                            deaths=stat.deaths,
                            streak=stat.streak,
                            time_on_obj=stat.time_on_obj,
                            timestamp=stat.timestamp,
                        )
                        session.add(db_stat)

            # ---- Kill feed (every frame) ----
            kills = self._killfeed.parse_killfeed(frame, self.cfg, ts)
            if kills and current_round:
                for kill in kills:
                    db_kill = DBKillEvent(
                        round_id=current_round.id,
                        killer=kill.killer,
                        victim=kill.victim,
                        weapon=kill.weapon,
                        killer_grid_cell=kill.killer_grid_cell,
                        timestamp=kill.timestamp,
                    )
                    session.add(db_kill)

            # ---- Minimap extraction + detection ----
            minimap = extract_minimap(frame, self.cfg)
            detections = self._detector.detect(minimap)

            if detections:
                # Color clustering for team assignment
                team_labels = {}
                for i, det in enumerate(detections):
                    if det.class_name == "player_dot":
                        lab_color = self._clusterer.extract_dot_color(minimap, det.bbox)
                        team = self._clusterer.classify_with_voting(i, lab_color)
                        team_labels[i] = team

                # Track players
                grid_fn = lambda x, y: self._grid.coord_to_cell(
                    x / minimap.shape[1], y / minimap.shape[0]
                )
                active_tracks = self._tracker.update(detections, team_labels, ts, grid_fn)

                # Store positions
                if current_round:
                    for track in active_tracks:
                        if track.last_position:
                            pos = track.last_position
                            db_pos = DBPlayerPosition(
                                round_id=current_round.id,
                                player_id=pos.player_id,
                                team=pos.team,
                                x=pos.x,
                                y=pos.y,
                                grid_cell=pos.grid_cell,
                                timestamp=pos.timestamp,
                            )
                            session.add(db_pos)

            # Periodic commit
            if frame_count % 100 == 0:
                session.commit()
                logger.info("Processed %d frames (%.1fs)", frame_count, ts)

        # Final commit
        vod_record.processed = True
        session.commit()
        session.close()
        cap.release()

        logger.info("Pipeline complete: %d frames processed from %s", frame_count, vod_path.name)
        return frame_count
