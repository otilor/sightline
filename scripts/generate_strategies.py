from sqlmodel import Session, select
from cod_analyst.db.schemas import get_engine, DBMatch, DBStrategySuggestion

engine = get_engine("sqlite:///./data/sightline.db")

with Session(engine) as session:
    match = session.exec(select(DBMatch)).first()
    if match:
        strategies = [
            DBStrategySuggestion(
                match_id=match.id,
                suggestion_type="pre_round",
                content="A Execute — Karachi SND: Fast 4-man A push through market with 1 lurking mid for trade/rotate cut. Steps: 1. Push market. 2. Hold mid cross. 3. Setup post-plant.",
                confidence=0.85
            ),
            DBStrategySuggestion(
                match_id=match.id,
                suggestion_type="defense",
                content="Default Defense — Terminal SND: Standard 2-1-2 split with anchor on A and lurk on B flank. Steps: 1. 2 players on A. 2. 1 mid watcher. 3. 2 on B.",
                confidence=0.72
            ),
            DBStrategySuggestion(
                match_id=match.id,
                suggestion_type="pre_round",
                content="Aggressive P2 Break — Karachi HP: Early rotation with aggressive spawns to claim P2 setup. Steps: 1. Break spawns at P1. 2. 2 players in P2. 3. Anchor holds power position.",
                confidence=0.78
            )
        ]
        session.add_all(strategies)
        session.commit()
        print("Generated and inserted 3 strategies.")
    else:
        print("No match found.")
