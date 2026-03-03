"use client";

import { useState } from "react";

export default function SettingsPage() {
    const [apiUrl, setApiUrl] = useState("http://localhost:8000");
    const [llmProvider, setLlmProvider] = useState("gemini");
    const [autoProcess, setAutoProcess] = useState(true);

    return (
        <div>
            <div className="animate-fade-in" style={{ marginBottom: "2rem" }}>
                <h1 style={{ fontSize: "1.75rem", fontWeight: 700, marginBottom: "0.25rem" }}>
                    <span className="gradient-text">Settings</span>
                </h1>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>
                    Platform configuration and preferences
                </p>
            </div>

            <div style={{ maxWidth: "640px", display: "flex", flexDirection: "column", gap: "1.5rem" }}>
                {/* API Connection */}
                <div className="glass-card animate-fade-in-delay-1" style={{ padding: "1.5rem" }}>
                    <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "1rem" }}>API Connection</h2>
                    <div style={{ marginBottom: "1rem" }}>
                        <label style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", display: "block", marginBottom: "0.375rem" }}>
                            Backend API URL
                        </label>
                        <input
                            type="text"
                            value={apiUrl}
                            onChange={(e) => setApiUrl(e.target.value)}
                            style={{
                                width: "100%",
                                padding: "0.625rem 1rem",
                                background: "var(--bg-secondary)",
                                border: "1px solid var(--border)",
                                borderRadius: "8px",
                                color: "var(--text-primary)",
                                fontSize: "0.875rem",
                                fontFamily: "var(--font-mono)",
                            }}
                        />
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                        <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--accent-green)" }} />
                        <span style={{ fontSize: "0.8125rem", color: "var(--accent-green)" }}>Connected</span>
                    </div>
                </div>

                {/* LLM Settings */}
                <div className="glass-card animate-fade-in-delay-2" style={{ padding: "1.5rem" }}>
                    <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "1rem" }}>LLM Configuration</h2>
                    <div style={{ marginBottom: "1rem" }}>
                        <label style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", display: "block", marginBottom: "0.375rem" }}>
                            Provider
                        </label>
                        <div style={{ display: "flex", gap: "0.5rem" }}>
                            {["openai", "gemini"].map((p) => (
                                <button
                                    key={p}
                                    onClick={() => setLlmProvider(p)}
                                    className={p === llmProvider ? "btn-primary" : "btn-ghost"}
                                    style={{ padding: "0.375rem 1rem", fontSize: "0.8125rem", textTransform: "capitalize" }}
                                >
                                    {p === "openai" ? "OpenAI (GPT-4o)" : "Google Gemini"}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div>
                        <label style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", display: "block", marginBottom: "0.375rem" }}>
                            API Key
                        </label>
                        <input
                            type="password"
                            placeholder="sk-..."
                            style={{
                                width: "100%",
                                padding: "0.625rem 1rem",
                                background: "var(--bg-secondary)",
                                border: "1px solid var(--border)",
                                borderRadius: "8px",
                                color: "var(--text-primary)",
                                fontSize: "0.875rem",
                                fontFamily: "var(--font-mono)",
                            }}
                        />
                    </div>
                </div>

                {/* Pipeline Settings */}
                <div className="glass-card animate-fade-in-delay-3" style={{ padding: "1.5rem" }}>
                    <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "1rem" }}>Pipeline</h2>
                    <label style={{ display: "flex", alignItems: "center", justifyContent: "space-between", cursor: "pointer", marginBottom: "0.75rem" }}>
                        <div>
                            <div style={{ fontSize: "0.875rem", fontWeight: 500 }}>Auto-process uploads</div>
                            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Automatically run the analysis pipeline on new VODs</div>
                        </div>
                        <input type="checkbox" checked={autoProcess} onChange={(e) => setAutoProcess(e.target.checked)} style={{ accentColor: "var(--accent-purple)", width: "16px", height: "16px" }} />
                    </label>
                    <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.5rem" }}>
                        <button className="btn-ghost" style={{ fontSize: "0.8125rem" }}>Rebuild Database</button>
                        <button className="btn-ghost" style={{ fontSize: "0.8125rem", color: "var(--accent-red)" }}>Clear All Data</button>
                    </div>
                </div>

                <button className="btn-primary" style={{ alignSelf: "flex-start" }}>
                    Save Settings
                </button>
            </div>
        </div>
    );
}
