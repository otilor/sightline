"use client";

import { useEffect, useState } from "react";

interface DashboardStats {
  total_matches: number;
  total_vods: number;
  processed_vods: number;
  opponents_scouted: number;
  strategies_generated: number;
  recent_activity: { type: string, text: string, time: string, color: string }[];
  next_match: { opponent: string, event: string };
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/api/dashboard`)
      .then((r) => r.json())
      .then(setStats)
      .catch(() => {
        // Use mock data if API is not running
        setStats({
          total_matches: 24,
          total_vods: 38,
          processed_vods: 31,
          opponents_scouted: 8,
          strategies_generated: 156,
          recent_activity: [
            { type: "process", text: "VOD processed: FaZe vs OpTic — Major 2 Week 3", time: "2h ago", color: "var(--accent-cyan)" }
          ],
          next_match: { opponent: "OpTic Texas", event: "CDL Major 3 — Week 2" }
        });
      })
      .finally(() => setLoading(false));
  }, []);

  const statCards = [
    { label: "Matches Analyzed", value: stats?.total_matches ?? 0, icon: "▣", color: "var(--accent-purple)" },
    { label: "VODs Processed", value: `${stats?.processed_vods ?? 0}/${stats?.total_vods ?? 0}`, icon: "◈", color: "var(--accent-cyan)" },
    { label: "Opponents Scouted", value: stats?.opponents_scouted ?? 0, icon: "⬡", color: "var(--accent-green)" },
    { label: "Strategies Generated", value: stats?.strategies_generated ?? 0, icon: "◆", color: "var(--accent-amber)" },
  ];

  const recentActivity = stats?.recent_activity || [
    { type: "process", text: "VOD processed: FaZe vs OpTic — Major 2 Week 3", time: "2h ago", color: "var(--accent-cyan)" },
  ];

  return (
    <div>
      {/* Header */}
      <div className="animate-fade-in" style={{ marginBottom: "2rem" }}>
        <h1 style={{ fontSize: "1.75rem", fontWeight: 700, marginBottom: "0.25rem" }}>
          Command <span className="gradient-text">Center</span>
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>
          Real-time overview of your competitive intelligence pipeline
        </p>
      </div>

      {/* Stat Cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: "1rem",
          marginBottom: "2rem",
        }}
      >
        {statCards.map((card, i) => (
          <div
            key={card.label}
            className={`stat-card animate-fade-in-delay-${Math.min(i, 3)}`}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.75rem" }}>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                {card.label}
              </span>
              <span style={{ fontSize: "1.25rem", color: card.color }}>{card.icon}</span>
            </div>
            <div style={{ fontSize: "2rem", fontWeight: 700, color: card.color, fontFamily: "var(--font-mono)" }}>
              {loading ? "—" : card.value}
            </div>
          </div>
        ))}
      </div>

      {/* Main content grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
        {/* Recent Activity */}
        <div className="glass-card animate-fade-in-delay-2" style={{ padding: "1.5rem" }}>
          <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "1.25rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ color: "var(--accent-cyan)" }}>◈</span> Recent Activity
          </h2>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            {recentActivity.map((activity, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.75rem",
                  padding: "0.625rem 0.75rem",
                  background: "var(--bg-secondary)",
                  borderRadius: "8px",
                  border: "1px solid var(--border)",
                }}
              >
                <div
                  style={{
                    width: "6px",
                    height: "6px",
                    borderRadius: "50%",
                    background: activity.color,
                    flexShrink: 0,
                  }}
                />
                <span style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", flex: 1 }}>{activity.text}</span>
                <span style={{ fontSize: "0.6875rem", color: "var(--text-muted)", whiteSpace: "nowrap" }}>{activity.time}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="glass-card animate-fade-in-delay-3" style={{ padding: "1.5rem" }}>
          <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "1.25rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ color: "var(--accent-purple)" }}>◆</span> Quick Actions
          </h2>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.625rem" }}>
            {[
              { label: "Upload VOD", desc: "Process a new match recording", icon: "▲" },
              { label: "Generate Strategy", desc: "Get AI-powered pre-round suggestions", icon: "◆" },
              { label: "Scout Opponent", desc: "Pull up scouting report for next match", icon: "⬡" },
              { label: "Review Round", desc: "Replay a round with position data overlay", icon: "▸" },
            ].map((action) => (
              <button
                key={action.label}
                className="btn-ghost"
                style={{
                  width: "100%",
                  justifyContent: "flex-start",
                  padding: "0.75rem 1rem",
                  textAlign: "left",
                }}
              >
                <span style={{ fontSize: "1rem", color: "var(--accent-purple)", width: "24px" }}>{action.icon}</span>
                <div>
                  <div style={{ fontWeight: 500, color: "var(--text-primary)", fontSize: "0.8125rem" }}>{action.label}</div>
                  <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)", marginTop: "1px" }}>{action.desc}</div>
                </div>
              </button>
            ))}
          </div>

          {/* Pipeline Status */}
          <div style={{ marginTop: "1.5rem", padding: "1rem", background: "var(--bg-secondary)", borderRadius: "8px", border: "1px solid var(--border)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
              <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>Pipeline Status</span>
              <span style={{ fontSize: "0.75rem", color: "var(--accent-green)" }}>Idle</span>
            </div>
            <div className="progress-bar">
              <div className="progress-bar-fill" style={{ width: "0%" }} />
            </div>
          </div>
        </div>
      </div>

      {/* Upcoming Match Banner */}
      <div
        className="glass-card animate-fade-in-delay-3"
        style={{
          marginTop: "1.5rem",
          padding: "1.25rem 1.5rem",
          background: "linear-gradient(135deg, rgba(168, 85, 247, 0.08), rgba(6, 182, 212, 0.05))",
          border: "1px solid rgba(168, 85, 247, 0.2)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <div style={{ fontSize: "0.75rem", color: "var(--accent-purple)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.25rem" }}>
            Next Match
          </div>
          <div style={{ fontSize: "1.125rem", fontWeight: 600 }}>
            FaZe vs <span style={{ color: "var(--accent-red)" }}>{stats?.next_match?.opponent || "OpTic Texas"}</span>
          </div>
          <div style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", marginTop: "0.25rem" }}>
            {stats?.next_match?.event || "CDL Major 3 — Week 2"}
          </div>
        </div>
        <button className="btn-primary">
          <span>◆</span> View Scouting Report
        </button>
      </div>
    </div>
  );
}
