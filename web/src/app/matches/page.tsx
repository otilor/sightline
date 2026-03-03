"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Match {
    id: number;
    opponent: string;
    event_name: string;
    result: string;
    date_played: string | null;
    created_at: string;
}

interface MapGame {
    id: number;
    match_id: number;
    map_name: string;
    mode: string;
    game_number: number;
    faze_score: number;
    opponent_score: number;
}

const modeColors: Record<string, string> = { hardpoint: "badge-hp", snd: "badge-snd", control: "badge-ctrl" };
const modeBadge: Record<string, string> = { hardpoint: "HP", snd: "SND", control: "CTRL" };

export default function MatchesPage() {
    const [matches, setMatches] = useState<Match[]>([]);
    const [mapGames, setMapGames] = useState<Record<number, MapGame[]>>({});
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch(`${API_BASE}/api/matches`)
            .then((r) => r.json())
            .then(async (data: Match[]) => {
                setMatches(data);
                const games: Record<number, MapGame[]> = {};
                for (const m of data) {
                    try {
                        const res = await fetch(`${API_BASE}/api/matches/${m.id}`);
                        const detail = await res.json();
                        games[m.id] = detail.map_games || [];
                    } catch { games[m.id] = []; }
                }
                setMapGames(games);
            })
            .catch(() => setMatches([]))
            .finally(() => setLoading(false));
    }, []);

    const formatDate = (d: string | null) => {
        if (!d) return "";
        return new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
    };

    return (
        <div>
            <div className="animate-fade-in" style={{ marginBottom: "2rem" }}>
                <h1 style={{ fontSize: "1.75rem", fontWeight: 700, marginBottom: "0.25rem" }}>
                    Match <span className="gradient-text">History</span>
                </h1>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>
                    All analyzed matches with detailed breakdowns
                </p>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                {loading ? (
                    <div style={{ padding: "2rem", textAlign: "center", color: "var(--text-muted)" }}>Loading matches...</div>
                ) : matches.length === 0 ? (
                    <div className="glass-card" style={{ padding: "2rem", textAlign: "center", color: "var(--text-muted)" }}>
                        No matches analyzed yet. Process a VOD to see results here.
                    </div>
                ) : matches.map((match, i) => (
                    <div
                        key={match.id}
                        className={`glass-card animate-fade-in-delay-${Math.min(i, 3)}`}
                        style={{ padding: "1.25rem", cursor: "pointer" }}
                    >
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                            <div>
                                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                                    <h3 style={{ fontSize: "1.0625rem", fontWeight: 600 }}>
                                        FaZe vs <span style={{ color: "var(--accent-cyan)" }}>{match.opponent}</span>
                                    </h3>
                                    {match.result && (
                                        <span className={`badge ${match.result.startsWith("3") ? "badge-win" : "badge-loss"}`}>
                                            {match.result}
                                        </span>
                                    )}
                                </div>
                                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                                    {match.event_name} · {formatDate(match.date_played || match.created_at)}
                                </span>
                            </div>
                            <button className="btn-ghost" style={{ fontSize: "0.75rem" }}>
                                View Details →
                            </button>
                        </div>

                        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                            {(mapGames[match.id] || []).map((game, j) => {
                                const won = game.faze_score > game.opponent_score;
                                return (
                                    <div
                                        key={j}
                                        style={{
                                            display: "flex",
                                            alignItems: "center",
                                            gap: "0.375rem",
                                            padding: "0.375rem 0.625rem",
                                            background: "var(--bg-secondary)",
                                            borderRadius: "6px",
                                            border: "1px solid var(--border)",
                                            fontSize: "0.75rem",
                                        }}
                                    >
                                        <span style={{ color: "var(--text-primary)", fontWeight: 500 }}>{game.map_name || `Map ${game.game_number}`}</span>
                                        <span className={`badge ${modeColors[game.mode] || ""}`}>{modeBadge[game.mode] || game.mode}</span>
                                        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.6875rem" }}>{game.faze_score}-{game.opponent_score}</span>
                                        <span style={{ color: won ? "var(--accent-green)" : "var(--accent-red)", fontWeight: 600 }}>
                                            {won ? "W" : "L"}
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
