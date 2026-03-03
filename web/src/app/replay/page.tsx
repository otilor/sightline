"use client";

import { useState, useEffect, useRef } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const GRID_LABELS = "ABCDE";

interface RoundInfo {
    id: number;
    round_number: number;
    side: string;
    outcome: string;
    start_time: number;
    end_time: number;
}

interface Position {
    player_id: number;
    team: string;
    x: number;
    y: number;
    grid_cell: string;
    timestamp: number;
}

interface KillEvent {
    killer: string;
    victim: string;
    weapon: string;
    timestamp: number;
}

export default function ReplayPage() {
    const [rounds, setRounds] = useState<RoundInfo[]>([]);
    const [selectedRound, setSelectedRound] = useState<number | null>(null);
    const [positions, setPositions] = useState<Position[]>([]);
    const [kills, setKills] = useState<KillEvent[]>([]);
    const [time, setTime] = useState(0);
    const [playing, setPlaying] = useState(false);
    const [showTrails, setShowTrails] = useState(true);
    const [loading, setLoading] = useState(true);
    const maxTimeRef = useRef(5);

    // Load rounds list
    useEffect(() => {
        fetch(`${API_BASE}/api/matches`)
            .then((r) => r.json())
            .then(async (matches) => {
                if (matches.length > 0) {
                    const detail = await fetch(`${API_BASE}/api/matches/${matches[0].id}`).then(r => r.json());
                    if (detail.map_games?.length > 0) {
                        // Get rounds for first map game
                        const gameId = detail.map_games[0].id;
                        // Fetch rounds via a simple range check
                        const roundsList: RoundInfo[] = [];
                        for (let i = 1; i <= 20; i++) {
                            try {
                                const r = await fetch(`${API_BASE}/api/rounds/${i}`);
                                if (r.ok) {
                                    const data = await r.json();
                                    roundsList.push(data.round);
                                }
                            } catch { break; }
                        }
                        setRounds(roundsList.slice(0, 10));
                        if (roundsList.length > 0) setSelectedRound(roundsList[0].id);
                    }
                }
            })
            .catch(() => { })
            .finally(() => setLoading(false));
    }, []);

    // Load round detail
    useEffect(() => {
        if (!selectedRound) return;
        fetch(`${API_BASE}/api/rounds/${selectedRound}`)
            .then((r) => r.json())
            .then((data) => {
                setPositions(data.positions || []);
                setKills(data.kill_events || []);
                if (data.round) {
                    const duration = data.round.end_time - data.round.start_time;
                    maxTimeRef.current = Math.max(duration, 5);
                }
                setTime(0);
                setPlaying(false);
            });
    }, [selectedRound]);

    useEffect(() => {
        if (!playing) return;
        const interval = setInterval(() => {
            setTime((t) => {
                if (t >= maxTimeRef.current) { setPlaying(false); return maxTimeRef.current; }
                return Math.min(t + 0.1, maxTimeRef.current);
            });
        }, 50);
        return () => clearInterval(interval);
    }, [playing]);

    // Group positions by player at current time
    const currentPositions = positions.length > 0
        ? Object.values(
            positions
                .filter((p) => p.timestamp <= (rounds.find(r => r.id === selectedRound)?.start_time || 0) + time)
                .reduce((acc, p) => {
                    acc[p.player_id] = p;
                    return acc;
                }, {} as Record<number, Position>)
        )
        : [];

    const currentKills = kills.filter((k) => {
        const startTime = rounds.find(r => r.id === selectedRound)?.start_time || 0;
        return k.timestamp <= startTime + time;
    });

    const roundInfo = rounds.find(r => r.id === selectedRound);

    return (
        <div>
            <div className="animate-fade-in" style={{ marginBottom: "2rem" }}>
                <h1 style={{ fontSize: "1.75rem", fontWeight: 700, marginBottom: "0.25rem" }}>
                    Round <span className="gradient-text">Replay</span>
                </h1>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>
                    Interactive minimap replay with player position data
                </p>
            </div>

            {loading ? (
                <div style={{ padding: "2rem", textAlign: "center", color: "var(--text-muted)" }}>Loading rounds...</div>
            ) : rounds.length === 0 ? (
                <div className="glass-card" style={{ padding: "2rem", textAlign: "center", color: "var(--text-muted)" }}>
                    No rounds available yet. Process VODs to enable replay.
                </div>
            ) : (
                <>
                    {/* Round selector */}
                    <div className="animate-fade-in-delay-1" style={{ display: "flex", gap: "0.5rem", marginBottom: "1.5rem", flexWrap: "wrap" }}>
                        {rounds.map((r) => (
                            <button
                                key={r.id}
                                className={selectedRound === r.id ? "btn-primary" : "btn-ghost"}
                                style={{ padding: "0.375rem 0.75rem", fontSize: "0.75rem" }}
                                onClick={() => setSelectedRound(r.id)}
                            >
                                R{r.round_number}
                            </button>
                        ))}
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: "1.5rem" }}>
                        {/* Minimap */}
                        <div className="animate-fade-in-delay-1">
                            <div className="minimap-canvas" style={{ width: "100%", maxWidth: "600px", aspectRatio: "1" }}>
                                <div className="grid-overlay">
                                    {Array.from({ length: 25 }).map((_, i) => {
                                        const row = Math.floor(i / 5);
                                        const col = i % 5;
                                        return <div key={i} className="grid-cell">{GRID_LABELS[row]}{col + 1}</div>;
                                    })}
                                </div>

                                {/* Player dots */}
                                {currentPositions.map((pos) => (
                                    <div
                                        key={pos.player_id}
                                        className={`minimap-dot ${pos.team}`}
                                        style={{ left: `${pos.x * 100}%`, top: `${pos.y * 100}%` }}
                                        title={`Player ${pos.player_id} (${pos.team})`}
                                    >
                                        <div style={{
                                            position: "absolute", top: "-18px", left: "50%", transform: "translateX(-50%)",
                                            fontSize: "0.6rem", color: "var(--text-secondary)", whiteSpace: "nowrap", fontWeight: 500,
                                        }}>
                                            {pos.grid_cell || `P${pos.player_id}`}
                                        </div>
                                    </div>
                                ))}
                            </div>

                            {/* Timeline */}
                            <div style={{
                                marginTop: "1rem", padding: "1rem", background: "var(--bg-card)",
                                borderRadius: "8px", border: "1px solid var(--border)",
                                display: "flex", alignItems: "center", gap: "1rem",
                            }}>
                                <button
                                    className="btn-primary"
                                    style={{ padding: "0.375rem 0.75rem", fontSize: "0.8125rem" }}
                                    onClick={() => { setPlaying(!playing); if (time >= maxTimeRef.current) setTime(0); }}
                                >
                                    {playing ? "⏸ Pause" : "▸ Play"}
                                </button>
                                <input
                                    type="range" min={0} max={maxTimeRef.current} step={0.1} value={time}
                                    onChange={(e) => setTime(parseFloat(e.target.value))}
                                    style={{ flex: 1, accentColor: "var(--accent-purple)" }}
                                />
                                <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.8125rem", color: "var(--text-secondary)", minWidth: "50px" }}>
                                    {time.toFixed(1)}s
                                </span>
                            </div>
                        </div>

                        {/* Side panel */}
                        <div className="animate-fade-in-delay-2" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                            <div className="glass-card" style={{ padding: "1rem" }}>
                                <h3 style={{ fontSize: "0.875rem", fontWeight: 600, marginBottom: "0.75rem" }}>Round Info</h3>
                                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", fontSize: "0.8125rem" }}>
                                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                                        <span style={{ color: "var(--text-muted)" }}>Round</span>
                                        <span>R{roundInfo?.round_number || "?"} — {roundInfo?.side || "Unknown"}</span>
                                    </div>
                                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                                        <span style={{ color: "var(--text-muted)" }}>Outcome</span>
                                        <span className={`badge ${roundInfo?.outcome === "win" ? "badge-win" : "badge-loss"}`}>
                                            {roundInfo?.outcome || "Unknown"}
                                        </span>
                                    </div>
                                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                                        <span style={{ color: "var(--text-muted)" }}>Positions</span>
                                        <span>{positions.length} tracked</span>
                                    </div>
                                </div>
                            </div>

                            <div className="glass-card" style={{ padding: "1rem" }}>
                                <h3 style={{ fontSize: "0.875rem", fontWeight: 600, marginBottom: "0.75rem" }}>Display</h3>
                                <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.8125rem", cursor: "pointer" }}>
                                    <input type="checkbox" checked={showTrails} onChange={(e) => setShowTrails(e.target.checked)} style={{ accentColor: "var(--accent-purple)" }} />
                                    Show trails
                                </label>
                            </div>

                            <div className="glass-card" style={{ padding: "1rem", flex: 1 }}>
                                <h3 style={{ fontSize: "0.875rem", fontWeight: 600, marginBottom: "0.75rem" }}>Kill Feed</h3>
                                <div style={{ display: "flex", flexDirection: "column", gap: "0.375rem" }}>
                                    {currentKills.length === 0 ? (
                                        <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>No kills at this timestamp</div>
                                    ) : currentKills.map((kill, i) => (
                                        <div
                                            key={i}
                                            style={{
                                                fontSize: "0.75rem", padding: "0.375rem 0.5rem",
                                                background: "var(--bg-secondary)", borderRadius: "4px",
                                                display: "flex", justifyContent: "space-between", alignItems: "center",
                                            }}
                                        >
                                            <span>
                                                <span style={{ color: "var(--accent-purple)" }}>{kill.killer}</span>
                                                {" → "}
                                                <span style={{ color: "var(--accent-red)" }}>{kill.victim}</span>
                                            </span>
                                            <span style={{ color: "var(--text-muted)", fontSize: "0.6875rem" }}>
                                                {kill.weapon}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
