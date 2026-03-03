"""FastAPI backend for Sightline web platform.

Provides REST endpoints for matches, rounds, strategies,
playstyle data, and VOD management.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from cod_analyst.config import load_config
from cod_analyst.db.schemas import (
    DBMatch,
    DBMapGame,
    DBRound,
    DBKillEvent,
    DBPlayerPosition,
    DBStatSnapshot,
    DBStrategySuggestion,
    DBPlaystyleProfile,
    VOD,
    get_engine,
    init_db,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sightline",
    description="AI-powered CoD League VOD analysis platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cfg = load_config()
_engine = get_engine(_cfg.database.url)


def get_db():
    with Session(_engine) as session:
        yield session


@app.on_event("startup")
def on_startup():
    init_db(_cfg.database.url)
    logger.info("Sightline API started")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/api/dashboard")
def dashboard_stats(session: Session = Depends(get_db)):
    """Get dashboard summary statistics."""
    matches = session.exec(select(DBMatch)).all()
    vods = session.exec(select(VOD)).all()
    strategies = session.exec(select(DBStrategySuggestion)).all()

    opponents = set(m.opponent for m in matches)

    recent_activity = []
    if matches:
        m = matches[-1]
        recent_activity.append({"type": "process", "text": f"VOD processed: vs {m.opponent} — {m.event_name}", "time": "Just now", "color": "var(--accent-cyan)"})
    if strategies:
        recent_activity.append({"type": "strategy", "text": f"New strategy generated for {strategies[-1].suggestion_type}", "time": "Just now", "color": "var(--accent-purple)"})

    return {
        "total_matches": len(matches),
        "total_vods": len(vods),
        "processed_vods": sum(1 for v in vods if v.processed),
        "opponents_scouted": len(opponents),
        "strategies_generated": len(strategies),
        "recent_activity": recent_activity,
        "next_match": {"opponent": opponents.pop() if opponents else "Unknown", "event": matches[-1].event_name if matches else "Unknown"}
    }


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------

@app.get("/api/matches")
def list_matches(session: Session = Depends(get_db)):
    """List all matches with basic info."""
    matches = session.exec(select(DBMatch)).all()
    return matches


@app.get("/api/matches/{match_id}")
def get_match(match_id: int, session: Session = Depends(get_db)):
    """Get full match details with map games."""
    match = session.get(DBMatch, match_id)
    if not match:
        raise HTTPException(404, "Match not found")

    games = session.exec(
        select(DBMapGame).where(DBMapGame.match_id == match_id)
    ).all()

    return {"match": match, "map_games": games}


# ---------------------------------------------------------------------------
# Rounds
# ---------------------------------------------------------------------------

@app.get("/api/rounds/{round_id}")
def get_round_detail(round_id: int, session: Session = Depends(get_db)):
    """Get round details with positions, kills, and stats."""
    round_obj = session.get(DBRound, round_id)
    if not round_obj:
        raise HTTPException(404, "Round not found")

    positions = session.exec(
        select(DBPlayerPosition).where(DBPlayerPosition.round_id == round_id)
    ).all()

    kills = session.exec(
        select(DBKillEvent).where(DBKillEvent.round_id == round_id)
    ).all()

    stats = session.exec(
        select(DBStatSnapshot).where(DBStatSnapshot.round_id == round_id)
    ).all()

    return {
        "round": round_obj,
        "positions": positions,
        "kill_events": kills,
        "stat_snapshots": stats,
    }


@app.get("/api/rounds/{round_id}/positions")
def get_round_positions(round_id: int, session: Session = Depends(get_db)):
    """Get position timeline for round replay visualization."""
    positions = session.exec(
        select(DBPlayerPosition)
        .where(DBPlayerPosition.round_id == round_id)
        .order_by(DBPlayerPosition.timestamp)
    ).all()

    return positions


# ---------------------------------------------------------------------------
# Opponents / Scouting
# ---------------------------------------------------------------------------

@app.get("/api/opponents")
def list_opponents(session: Session = Depends(get_db)):
    """List all scouted opponents with match counts."""
    matches = session.exec(select(DBMatch)).all()
    opponent_data: dict[str, dict] = {}

    for m in matches:
        if m.opponent not in opponent_data:
            opponent_data[m.opponent] = {"name": m.opponent, "matches": 0, "latest_event": ""}
        opponent_data[m.opponent]["matches"] += 1
        opponent_data[m.opponent]["latest_event"] = m.event_name

    return list(opponent_data.values())


@app.get("/api/opponents/{team_name}/profile")
def get_opponent_profile(team_name: str, session: Session = Depends(get_db)):
    """Get playstyle profile for an opponent."""
    profiles = session.exec(
        select(DBPlaystyleProfile).where(DBPlaystyleProfile.team_name == team_name)
    ).all()

    return profiles


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@app.get("/api/strategies")
def list_strategies(
    match_id: int | None = None,
    suggestion_type: str | None = None,
    session: Session = Depends(get_db),
):
    """List strategy suggestions with optional filters."""
    query = select(DBStrategySuggestion)

    if match_id is not None:
        query = query.where(DBStrategySuggestion.match_id == match_id)
    if suggestion_type is not None:
        query = query.where(DBStrategySuggestion.suggestion_type == suggestion_type)

    strategies = session.exec(query).all()
    return strategies


@app.get("/api/strategies/{strategy_id}")
def get_strategy(strategy_id: int, session: Session = Depends(get_db)):
    """Get a single strategy suggestion."""
    strategy = session.get(DBStrategySuggestion, strategy_id)
    if not strategy:
        raise HTTPException(404, "Strategy not found")
    return strategy


# ---------------------------------------------------------------------------
# VODs
# ---------------------------------------------------------------------------

@app.get("/api/vods")
def list_vods(session: Session = Depends(get_db)):
    """List all VODs."""
    vods = session.exec(select(VOD)).all()
    return vods


@app.post("/api/vods/upload")
async def upload_vod(file: UploadFile = File(...), session: Session = Depends(get_db)):
    """Upload a VOD file for processing."""
    import shutil
    from pathlib import Path

    upload_dir = Path(_cfg.paths.vods_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    vod = VOD(
        filename=file.filename,
        filepath=str(dest),
    )
    session.add(vod)
    session.commit()
    session.refresh(vod)

    return {"id": vod.id, "filename": vod.filename, "status": "uploaded"}


# ---------------------------------------------------------------------------
# Stats Overview
# ---------------------------------------------------------------------------

@app.get("/api/stats/overview")
def stats_overview(session: Session = Depends(get_db)):
    """Aggregated stats from all processed data."""
    from collections import Counter

    rounds = session.exec(select(DBRound)).all()
    kills = session.exec(select(DBKillEvent)).all()
    snapshots = session.exec(select(DBStatSnapshot)).all()
    games = session.exec(select(DBMapGame)).all()

    # Player K/D from kill events
    kill_counts: Counter = Counter()
    death_counts: Counter = Counter()
    for k in kills:
        if k.killer:
            kill_counts[k.killer] += 1
        if k.victim:
            death_counts[k.victim] += 1

    all_players = sorted(set(list(kill_counts.keys()) + list(death_counts.keys())))
    player_stats = []
    for name in all_players:
        k = kill_counts.get(name, 0)
        d = death_counts.get(name, 0)
        kd = k / max(d, 1)
        player_stats.append({"name": name, "kills": k, "deaths": d, "kd": round(kd, 2)})

    player_stats.sort(key=lambda p: p["kd"], reverse=True)

    # Mode breakdown
    mode_breakdown = Counter(g.mode for g in games)

    # Unique players from snapshots
    unique_players = sorted(set(s.player_name for s in snapshots if s.player_name))

    return {
        "total_rounds": len(rounds),
        "total_kills": len(kills),
        "total_stat_snapshots": len(snapshots),
        "unique_players": unique_players,
        "player_stats": player_stats[:20],
        "mode_breakdown": dict(mode_breakdown),
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}
