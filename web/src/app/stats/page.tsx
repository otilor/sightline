"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface StatsData {
    total_rounds: number;
    total_kills: number;
    total_stat_snapshots: number;
    unique_players: string[];
    player_stats: { name: string; kills: number; deaths: number; kd: number }[];
    mode_breakdown: Record<string, number>;
}

export default function StatsPage() {
    const [stats, setStats] = useState<StatsData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch(`${API_BASE}/api/stats/overview`)
            .then((r) => r.json())
            .then(setStats)
            .catch(() => setStats(null))
            .finally(() => setLoading(false));
    }, []);

    if (loading) {
        return (
            <div>
                <div className="animate-fade-in" style={{ marginBottom: "2rem" }}>
                    <h1 style={{ fontSize: "1.75rem", fontWeight: 700, marginBottom: "0.25rem" }}>
                        Team <span className="gradient-text">Stats</span>
                    </h1>
                </div>
                <div style={{ padding: "2rem", textAlign: "center", color: "var(--text-muted)" }}>Loading stats...</div>
            </div>
        );
    }

    if (!stats) {
        return (
            <div>
                <div className="animate-fade-in" style={{ marginBottom: "2rem" }}>
                    <h1 style={{ fontSize: "1.75rem", fontWeight: 700, marginBottom: "0.25rem" }}>
                        Team <span className="gradient-text">Stats</span>
                    </h1>
                </div>
                <div className="glass-card" style={{ padding: "2rem", textAlign: "center", color: "var(--text-muted)" }}>
                    No stats available yet. Process VODs to generate performance data.
                </div>
            </div>
        );
    }

    return (
        <div>
            <div className="animate-fade-in" style={{ marginBottom: "2rem" }}>
                <h1 style={{ fontSize: "1.75rem", fontWeight: 700, marginBottom: "0.25rem" }}>
                    Team <span className="gradient-text">Stats</span>
                </h1>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>
                    Player and team performance extracted from analyzed VODs
                </p>
            </div>

            {/* Team overview */}
            <div
                className="animate-fade-in-delay-1"
                style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(4, 1fr)",
                    gap: "1rem",
                    marginBottom: "2rem",
                }}
            >
                {[
                    { label: "Rounds Analyzed", value: stats.total_rounds, color: "var(--accent-green)" },
                    { label: "Total Kills Tracked", value: stats.total_kills, color: "var(--accent-purple)" },
                    { label: "Stat Snapshots", value: stats.total_stat_snapshots, color: "var(--accent-cyan)" },
                    { label: "Players Detected", value: stats.unique_players.length, color: "var(--accent-amber)" },
                ].map((stat) => (
                    <div key={stat.label} className="stat-card">
                        <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.5rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                            {stat.label}
                        </div>
                        <div style={{ fontSize: "1.75rem", fontWeight: 700, color: stat.color, fontFamily: "var(--font-mono)" }}>
                            {stat.value.toLocaleString()}
                        </div>
                    </div>
                ))}
            </div>

            {/* Player table */}
            {stats.player_stats.length > 0 && (
                <div className="glass-card animate-fade-in-delay-2" style={{ padding: "1.5rem" }}>
                    <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "1.25rem" }}>Player Performance (from Kill Events)</h2>
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Player</th>
                                <th>Kills</th>
                                <th>Deaths</th>
                                <th>K/D</th>
                            </tr>
                        </thead>
                        <tbody>
                            {stats.player_stats.map((player) => (
                                <tr key={player.name}>
                                    <td style={{ fontWeight: 600, color: "var(--text-primary)" }}>{player.name}</td>
                                    <td style={{ fontFamily: "var(--font-mono)" }}>{player.kills}</td>
                                    <td style={{ fontFamily: "var(--font-mono)" }}>{player.deaths}</td>
                                    <td style={{ fontFamily: "var(--font-mono)", color: player.kd >= 1.2 ? "var(--accent-green)" : player.kd >= 1.0 ? "var(--text-primary)" : "var(--accent-red)" }}>
                                        {player.kd.toFixed(2)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Mode breakdown */}
            {Object.keys(stats.mode_breakdown).length > 0 && (
                <div style={{ display: "grid", gridTemplateColumns: `repeat(${Math.min(Object.keys(stats.mode_breakdown).length, 3)}, 1fr)`, gap: "1rem", marginTop: "1.5rem" }}>
                    {Object.entries(stats.mode_breakdown).map(([mode, count]) => (
                        <div key={mode} className="glass-card animate-fade-in-delay-3" style={{ padding: "1.25rem" }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                                <h3 style={{ fontSize: "0.9375rem", fontWeight: 600, textTransform: "capitalize" }}>{mode}</h3>
                                <span className="badge badge-snd">{count} rounds</span>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
