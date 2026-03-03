"""SND Strategist — generates pre-round suggestions and loss analysis.

Uses opponent scouting data and current match state to recommend
strategies for upcoming rounds and diagnose what went wrong in losses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cod_analyst.analysis.profiler import ScoutingReport
from cod_analyst.game.models import Round, RoundOutcome, Side, StrategySuggestion

logger = logging.getLogger(__name__)


@dataclass
class MatchContext:
    """Current match state for contextual suggestions."""
    faze_score: int = 0
    opponent_score: int = 0
    current_round: int = 0
    current_side: Side = Side.UNKNOWN
    rounds_played: list[Round] = field(default_factory=list)
    opponent_report: ScoutingReport | None = None


def generate_pre_round_strategy(
    context: MatchContext,
) -> list[StrategySuggestion]:
    """Generate strategy suggestions for the upcoming round.

    Considers:
    - Opponent tendencies from scouting report
    - Score situation (up, down, even)
    - What opponent has done in previous rounds this match
    - Attack vs defense side

    Parameters
    ----------
    context : MatchContext
        Current match state.

    Returns
    -------
    list[StrategySuggestion]
        Ranked strategy suggestions.
    """
    suggestions: list[StrategySuggestion] = []
    report = context.opponent_report

    if not report:
        return [StrategySuggestion(
            suggestion_type="pre_round",
            content="No scouting data available. Play default setups.",
            confidence=0.3,
        )]

    # ---- Counter their default setup ----
    if report.default_setup:
        suggestions.append(StrategySuggestion(
            suggestion_type="pre_round",
            content=f"Opponent typically runs '{report.default_setup}'. "
                    f"Counter with {'aggressive flanks' if 'Stack' in report.default_setup else 'map control pressure'}.",
            confidence=0.7,
            metadata={"based_on": "formation_clusters"},
        ))

    # ---- Exploit preferred routes ----
    if report.preferred_routes:
        top_route = report.preferred_routes[0]
        suggestions.append(StrategySuggestion(
            suggestion_type="pre_round",
            content=f"Most common route: {top_route}. Pre-aim lanes and set up crossfires "
                    f"along this path.",
            confidence=0.8,
            metadata={"based_on": "route_clusters"},
        ))

    # ---- Score context ----
    score_diff = context.faze_score - context.opponent_score
    if score_diff < -2:
        suggestions.append(StrategySuggestion(
            suggestion_type="pre_round",
            content="Down by 2+. Consider aggressive/tempo plays to catch them off guard. "
                    "They may start playing conservatively with the lead.",
            confidence=0.6,
            metadata={"based_on": "score_context"},
        ))
    elif score_diff > 2:
        suggestions.append(StrategySuggestion(
            suggestion_type="pre_round",
            content="Up by 2+. Play fundamentally sound — trade kills, don't overextend.",
            confidence=0.6,
            metadata={"based_on": "score_context"},
        ))

    # ---- Side-specific advice ----
    if context.current_side == Side.ATTACK:
        if report.pace_label == "passive":
            suggestions.append(StrategySuggestion(
                suggestion_type="pre_round",
                content="On attack vs passive defense — use fast executes to beat their setups. "
                        "They take too long to rotate.",
                confidence=0.7,
                metadata={"based_on": "pace_analysis"},
            ))
    elif context.current_side == Side.DEFENSE:
        if report.pace_label == "aggressive":
            suggestions.append(StrategySuggestion(
                suggestion_type="pre_round",
                content="Defending vs aggressive opponent — play anchored positions and "
                        "let them come to you. They rush early and make mistakes.",
                confidence=0.7,
                metadata={"based_on": "pace_analysis"},
            ))

    # ---- Exploit weaknesses ----
    for weakness in report.weaknesses[:2]:
        suggestions.append(StrategySuggestion(
            suggestion_type="pre_round",
            content=f"Weakness: {weakness}",
            confidence=0.65,
            metadata={"based_on": "vulnerability_analysis"},
        ))

    # Sort by confidence
    suggestions.sort(key=lambda s: s.confidence, reverse=True)

    return suggestions


def analyze_round_loss(
    lost_round: Round,
    context: MatchContext,
    tipping_point_event: int | None = None,
) -> StrategySuggestion:
    """Analyze why a round was lost and suggest adjustments.

    Parameters
    ----------
    lost_round : Round
        The lost round to analyze.
    context : MatchContext
        Current match state.
    tipping_point_event : int, optional
        Index from the event transformer's tipping-point analysis.

    Returns
    -------
    StrategySuggestion
        Loss analysis with corrective suggestions.
    """
    analysis_parts = []

    # ---- Kill event analysis ----
    if lost_round.kill_events:
        first_kill = lost_round.kill_events[0]
        if lost_round.kill_events[0].victim and "faze" in str(lost_round.positions[:1]):
            analysis_parts.append(
                f"Lost first blood to {first_kill.killer} at {first_kill.killer_grid_cell or 'unknown position'}."
            )

    # ---- Position analysis ----
    if lost_round.positions:
        from collections import Counter
        death_areas = Counter(
            ke.killer_grid_cell for ke in lost_round.kill_events
            if ke.killer_grid_cell
        )
        if death_areas:
            hot_zone = death_areas.most_common(1)[0]
            analysis_parts.append(f"Most deaths occurred near {hot_zone[0]}.")

    # ---- Tipping point ----
    if tipping_point_event is not None:
        analysis_parts.append(
            f"Model identified event #{tipping_point_event} as the tipping point — "
            f"the round shifted from winnable to lost around this moment."
        )

    # ---- Duration analysis ----
    duration = lost_round.end_time - lost_round.start_time
    if duration < 45:
        analysis_parts.append(
            "Round was very short — opponent likely executed a fast play. "
            "Consider better early-round positioning."
        )
    elif duration > 120:
        analysis_parts.append(
            "Round went to overtime — time management issue. "
            "Need faster decision-making on executes."
        )

    content = " ".join(analysis_parts) if analysis_parts else "No specific loss patterns identified."

    return StrategySuggestion(
        suggestion_type="loss_analysis",
        content=content,
        confidence=0.6,
        source_round_id=lost_round.round_number,
        metadata={"tipping_point": tipping_point_event},
    )
