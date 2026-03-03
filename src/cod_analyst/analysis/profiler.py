"""Playstyle profiler — aggregates ML outputs into readable profiles.

Synthesizes route clusters, formation patterns, engagement stats,
and sequence model outputs into actionable opponent scouting reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cod_analyst.features.kill_events import KillEventFeatures
from cod_analyst.features.movement import MovementFeatures
from cod_analyst.game.models import PlaystyleProfile
from cod_analyst.ml.formation_clustering import FormationProfile
from cod_analyst.ml.route_clustering import RouteCluster

logger = logging.getLogger(__name__)


@dataclass
class ScoutingReport:
    """A comprehensive scouting report for an opponent team."""
    team_name: str
    map_name: str
    mode: str

    # Playstyle summary
    pace_label: str = ""           # "aggressive", "passive", "balanced"
    formation_label: str = ""      # "spread", "stacked", "mixed"
    trading_label: str = ""        # "disciplined", "poor", "average"

    # Key tendencies
    preferred_routes: list[str] = field(default_factory=list)
    default_setup: str = ""
    first_blood_rate: float = 0.0
    favorite_positions: list[str] = field(default_factory=list)
    weapon_preferences: dict[str, float] = field(default_factory=dict)

    # Vulnerabilities
    weaknesses: list[str] = field(default_factory=list)

    # Raw data refs
    profile: PlaystyleProfile | None = None

    def to_text(self) -> str:
        """Convert scouting report to human-readable text."""
        lines = [
            f"## Scouting Report: {self.team_name}",
            f"**Map:** {self.map_name} | **Mode:** {self.mode}",
            f"**Pace:** {self.pace_label} | **Formation:** {self.formation_label} | **Trading:** {self.trading_label}",
            "",
            "### Key Routes",
        ]
        for route in self.preferred_routes:
            lines.append(f"- {route}")

        lines.append(f"\n### Default Setup: {self.default_setup}")
        lines.append(f"### First Blood Rate: {self.first_blood_rate:.0%}")

        if self.favorite_positions:
            lines.append("\n### Favorite Positions")
            for pos in self.favorite_positions:
                lines.append(f"- {pos}")

        if self.weaknesses:
            lines.append("\n### Vulnerabilities")
            for w in self.weaknesses:
                lines.append(f"- {w}")

        return "\n".join(lines)


def build_scouting_report(
    team_name: str,
    map_name: str,
    mode: str,
    route_clusters: list[RouteCluster],
    formation_profile: FormationProfile,
    movement_features: list[MovementFeatures],
    kill_features: list[KillEventFeatures],
) -> ScoutingReport:
    """Build a scouting report from analysis outputs.

    Parameters
    ----------
    team_name : str
    map_name : str
    mode : str
    route_clusters : list[RouteCluster]
    formation_profile : FormationProfile
    movement_features : list[MovementFeatures]
    kill_features : list[KillEventFeatures]

    Returns
    -------
    ScoutingReport
    """
    report = ScoutingReport(team_name=team_name, map_name=map_name, mode=mode)

    # ---- Pace classification ----
    if movement_features:
        avg_speed = sum(f.avg_speed for f in movement_features) / len(movement_features)
        avg_idle = sum(f.idle_fraction for f in movement_features) / len(movement_features)

        if avg_speed > 0.015 and avg_idle < 0.3:
            report.pace_label = "aggressive"
        elif avg_speed < 0.008 or avg_idle > 0.5:
            report.pace_label = "passive"
        else:
            report.pace_label = "balanced"

    # ---- Formation style ----
    if formation_profile.clusters:
        top = formation_profile.clusters[0]
        if top.avg_spread > 0.15:
            report.formation_label = "spread"
        elif top.avg_spread < 0.08:
            report.formation_label = "stacked"
        else:
            report.formation_label = "mixed"
        report.default_setup = top.label

    # ---- Trading quality ----
    if kill_features:
        avg_trade_rate = sum(f.trade_success_rate for f in kill_features) / len(kill_features)
        report.first_blood_rate = sum(1 for f in kill_features if f.got_first_blood) / len(kill_features)

        if avg_trade_rate > 0.6:
            report.trading_label = "disciplined"
        elif avg_trade_rate < 0.3:
            report.trading_label = "poor"
        else:
            report.trading_label = "average"

    # ---- Preferred routes ----
    for rc in route_clusters[:3]:
        route_str = " → ".join(rc.representative[:6])
        report.preferred_routes.append(f"{rc.label} ({rc.frequency:.0%}): {route_str}")

    # ---- Favorite positions ----
    from collections import Counter
    all_zones: Counter = Counter()
    for mf in movement_features:
        for cell, time_frac in mf.time_in_zone.items():
            all_zones[cell] += time_frac

    top_zones = all_zones.most_common(5)
    report.favorite_positions = [f"{cell} ({frac:.0%})" for cell, frac in top_zones]

    # ---- Weapon preferences ----
    weapon_counts: Counter = Counter()
    for kf in kill_features:
        for weapon, count in kf.weapon_distribution.items():
            weapon_counts[weapon] += count
    total_kills = sum(weapon_counts.values())
    if total_kills > 0:
        report.weapon_preferences = {
            w: c / total_kills for w, c in weapon_counts.most_common(5)
        }

    # ---- Weaknesses ----
    if report.trading_label == "poor":
        report.weaknesses.append("Poor trading discipline — exploit isolated picks")
    if report.pace_label == "passive":
        report.weaknesses.append("Passive pace — apply early pressure to force uncomfortable plays")
    if report.first_blood_rate < 0.3:
        report.weaknesses.append("Low first blood rate — they lose opening duels frequently")
    if formation_profile.clusters and formation_profile.clusters[0].frequency > 0.7:
        report.weaknesses.append(
            f"Predictable setups — {formation_profile.clusters[0].label} used {formation_profile.clusters[0].frequency:.0%} of rounds"
        )

    logger.info("Scouting report built: %s on %s", team_name, map_name)
    return report
