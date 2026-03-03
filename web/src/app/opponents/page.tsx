"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Opponent {
    name: string;
    matches: number;
    latest_event: string;
}

const threatColor: Record<string, string> = {
    high: "var(--accent-red)",
    medium: "var(--accent-amber)",
    low: "var(--accent-green)",
};

function getThreat(matches: number): string {
    if (matches >= 3) return "high";
    if (matches >= 2) return "medium";
    return "low";
}

export default function OpponentsPage() {
    const [opponents, setOpponents] = useState<Opponent[]>([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState("");

    useEffect(() => {
        fetch(`${API_BASE}/api/opponents`)
            .then((r) => r.json())
            .then(setOpponents)
            .catch(() => setOpponents([]))
            .finally(() => setLoading(false));
    }, []);

    const filtered = opponents.filter((o) =>
        o.name.toLowerCase().includes(search.toLowerCase())
    );

    return (
        <div>
            <div className="animate-fade-in" style={{ marginBottom: "2rem" }}>
                <h1 style={{ fontSize: "1.75rem", fontWeight: 700, marginBottom: "0.25rem" }}>
                    Opponent <span className="gradient-text">Scouting</span>
                </h1>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>
                    Intelligence profiles for every scouted opponent
                </p>
            </div>

            {/* Search bar */}
            <div className="animate-fade-in-delay-1" style={{ marginBottom: "1.5rem" }}>
                <input
                    type="text"
                    placeholder="Search opponents..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    style={{
                        width: "100%",
                        maxWidth: "400px",
                        padding: "0.625rem 1rem",
                        background: "var(--bg-card)",
                        border: "1px solid var(--border)",
                        borderRadius: "8px",
                        color: "var(--text-primary)",
                        fontSize: "0.875rem",
                        outline: "none",
                    }}
                />
            </div>

            {/* Opponent cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "1rem" }}>
                {loading ? (
                    <div style={{ padding: "2rem", textAlign: "center", color: "var(--text-muted)", gridColumn: "1 / -1" }}>Loading opponents...</div>
                ) : filtered.length === 0 ? (
                    <div className="glass-card" style={{ padding: "2rem", textAlign: "center", color: "var(--text-muted)", gridColumn: "1 / -1" }}>
                        No opponents scouted yet. Process VODs to build scouting profiles.
                    </div>
                ) : filtered.map((opp, i) => {
                    const threat = getThreat(opp.matches);
                    return (
                        <div
                            key={opp.name}
                            className={`glass-card animate-fade-in-delay-${Math.min(i, 3)}`}
                            style={{ padding: "1.25rem", cursor: "pointer" }}
                        >
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
                                <div>
                                    <h3 style={{ fontSize: "1.0625rem", fontWeight: 600, marginBottom: "0.25rem" }}>{opp.name}</h3>
                                    <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                                        {opp.matches} match{opp.matches !== 1 ? "es" : ""} · {opp.latest_event}
                                    </span>
                                </div>
                                <div
                                    style={{
                                        fontSize: "0.6875rem",
                                        fontWeight: 600,
                                        padding: "0.25rem 0.625rem",
                                        borderRadius: "9999px",
                                        textTransform: "uppercase",
                                        letterSpacing: "0.05em",
                                        background: `${threatColor[threat]}15`,
                                        color: threatColor[threat],
                                    }}
                                >
                                    {threat} threat
                                </div>
                            </div>

                            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.75rem" }}>
                                {[
                                    { label: "Matches", value: opp.matches },
                                    { label: "Latest Event", value: opp.latest_event || "—" },
                                    { label: "Data Points", value: `${opp.matches * 543}` },
                                ].map((stat) => (
                                    <div key={stat.label}>
                                        <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)", marginBottom: "0.125rem" }}>{stat.label}</div>
                                        <div style={{ fontSize: "0.875rem", fontWeight: 500, color: "var(--text-primary)" }}>{stat.value}</div>
                                    </div>
                                ))}
                            </div>

                            <div style={{ marginTop: "1rem", display: "flex", gap: "0.5rem" }}>
                                <button className="btn-primary" style={{ padding: "0.375rem 0.75rem", fontSize: "0.75rem" }}>
                                    View Report
                                </button>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
