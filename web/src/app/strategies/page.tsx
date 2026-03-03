"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Strategy {
    id: number;
    match_id: number;
    round_id: number | null;
    suggestion_type: string;
    content: string;
    confidence: number;
    created_at: string;
}

const typeConfig: Record<string, { label: string; color: string; icon: string }> = {
    pre_round: { label: "Pre-Round", color: "var(--accent-purple)", icon: "◆" },
    loss_analysis: { label: "Loss Analysis", color: "var(--accent-red)", icon: "▼" },
    scouting: { label: "Scouting", color: "var(--accent-cyan)", icon: "⬡" },
    defense: { label: "Defense", color: "var(--accent-green)", icon: "⛨" },
};

export default function StrategiesPage() {
    const [strategies, setStrategies] = useState<Strategy[]>([]);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState("All");

    useEffect(() => {
        fetch(`${API_BASE}/api/strategies`)
            .then((r) => r.json())
            .then(setStrategies)
            .catch(() => setStrategies([]))
            .finally(() => setLoading(false));
    }, []);

    const filtered = filter === "All"
        ? strategies
        : strategies.filter((s) => {
            const key = filter.toLowerCase().replace(/-/g, "_").replace(/ /g, "_");
            return s.suggestion_type === key;
        });

    return (
        <div>
            <div className="animate-fade-in" style={{ marginBottom: "2rem" }}>
                <h1 style={{ fontSize: "1.75rem", fontWeight: 700, marginBottom: "0.25rem" }}>
                    AI <span className="gradient-text">Strategies</span>
                </h1>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>
                    Data-driven strategy suggestions powered by ML analysis
                </p>
            </div>

            {/* Filter pills */}
            <div className="animate-fade-in-delay-1" style={{ display: "flex", gap: "0.5rem", marginBottom: "1.5rem" }}>
                {["All", "Pre-Round", "Loss Analysis", "Scouting", "Defense"].map((f) => (
                    <button
                        key={f}
                        className={f === filter ? "btn-primary" : "btn-ghost"}
                        style={{ padding: "0.375rem 0.875rem", fontSize: "0.8125rem" }}
                        onClick={() => setFilter(f)}
                    >
                        {f}
                    </button>
                ))}
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                {loading ? (
                    <div style={{ padding: "2rem", textAlign: "center", color: "var(--text-muted)" }}>Loading strategies...</div>
                ) : filtered.length === 0 ? (
                    <div className="glass-card" style={{ padding: "2rem", textAlign: "center", color: "var(--text-muted)" }}>
                        No strategies generated yet. Process VODs and run the strategy engine.
                    </div>
                ) : filtered.map((strat, i) => {
                    const config = typeConfig[strat.suggestion_type] || typeConfig.pre_round;
                    return (
                        <div
                            key={strat.id}
                            className={`glass-card animate-fade-in-delay-${Math.min(i, 3)}`}
                            style={{ padding: "1.5rem" }}
                        >
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
                                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                                    <span style={{ fontSize: "1.25rem", color: config.color }}>{config.icon}</span>
                                    <div>
                                        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                                            <span
                                                style={{
                                                    fontSize: "0.6875rem",
                                                    fontWeight: 600,
                                                    padding: "0.125rem 0.5rem",
                                                    borderRadius: "4px",
                                                    background: `${config.color}15`,
                                                    color: config.color,
                                                    textTransform: "uppercase",
                                                    letterSpacing: "0.05em",
                                                }}
                                            >
                                                {config.label}
                                            </span>
                                            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                                                Match #{strat.match_id}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                                <div style={{ display: "flex", alignItems: "center", gap: "0.375rem" }}>
                                    <div
                                        style={{
                                            width: "40px",
                                            height: "4px",
                                            background: "var(--bg-elevated)",
                                            borderRadius: "2px",
                                            overflow: "hidden",
                                        }}
                                    >
                                        <div
                                            style={{
                                                width: `${strat.confidence * 100}%`,
                                                height: "100%",
                                                background: strat.confidence > 0.8 ? "var(--accent-green)" : strat.confidence > 0.6 ? "var(--accent-amber)" : "var(--accent-red)",
                                                borderRadius: "2px",
                                            }}
                                        />
                                    </div>
                                    <span style={{ fontSize: "0.75rem", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                                        {(strat.confidence * 100).toFixed(0)}%
                                    </span>
                                </div>
                            </div>

                            <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.6, margin: 0 }}>
                                {strat.content}
                            </p>

                            <div style={{ marginTop: "0.75rem", fontSize: "0.6875rem", color: "var(--text-muted)" }}>
                                Generated: {new Date(strat.created_at).toLocaleDateString()}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
