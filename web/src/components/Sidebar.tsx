"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
    { href: "/", label: "Dashboard", icon: "◈" },
    { href: "/opponents", label: "Opponents", icon: "⬡" },
    { href: "/matches", label: "Matches", icon: "▣" },
    { href: "/replay", label: "Round Replay", icon: "▸" },
    { href: "/strategies", label: "Strategies", icon: "◆" },
    { href: "/playbook", label: "Playbook", icon: "◫" },
    { href: "/stats", label: "Team Stats", icon: "◩" },
    { href: "/settings", label: "Settings", icon: "⚙" },
];

export default function Sidebar() {
    const pathname = usePathname();

    return (
        <aside
            style={{
                position: "fixed",
                top: 0,
                left: 0,
                width: "240px",
                height: "100vh",
                background: "var(--bg-secondary)",
                borderRight: "1px solid var(--border)",
                display: "flex",
                flexDirection: "column",
                zIndex: 50,
            }}
        >
            {/* Logo */}
            <div
                style={{
                    padding: "1.25rem 1.25rem 1.5rem",
                    borderBottom: "1px solid var(--border)",
                }}
            >
                <Link href="/" style={{ textDecoration: "none" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.625rem" }}>
                        <div
                            style={{
                                width: "32px",
                                height: "32px",
                                borderRadius: "8px",
                                background: "linear-gradient(135deg, var(--accent-purple), var(--accent-cyan))",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                fontSize: "1rem",
                                fontWeight: 700,
                                color: "white",
                            }}
                        >
                            S
                        </div>
                        <div>
                            <div
                                className="gradient-text"
                                style={{ fontSize: "1.125rem", fontWeight: 700, letterSpacing: "-0.02em" }}
                            >
                                Sightline
                            </div>
                            <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: "-2px" }}>
                                VOD ANALYSIS PLATFORM
                            </div>
                        </div>
                    </div>
                </Link>
            </div>

            {/* Navigation */}
            <nav style={{ flex: 1, padding: "1rem 0.75rem", display: "flex", flexDirection: "column", gap: "2px" }}>
                {navItems.map((item) => {
                    const isActive = pathname === item.href;
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={`nav-item ${isActive ? "active" : ""}`}
                            style={{ position: "relative" }}
                        >
                            <span style={{ fontSize: "1rem", width: "20px", textAlign: "center" }}>{item.icon}</span>
                            {item.label}
                        </Link>
                    );
                })}
            </nav>

            {/* Status footer */}
            <div
                style={{
                    padding: "1rem 1.25rem",
                    borderTop: "1px solid var(--border)",
                }}
            >
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
                    <div
                        style={{
                            width: "8px",
                            height: "8px",
                            borderRadius: "50%",
                            background: "var(--accent-green)",
                            animation: "pulseGlow 2s ease-in-out infinite",
                        }}
                    />
                    <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>System Online</span>
                </div>
                <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>v0.1.0 — Sightline</div>
            </div>
        </aside>
    );
}
