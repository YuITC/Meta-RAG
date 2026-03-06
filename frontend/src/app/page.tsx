"use client";

import { useState, useEffect } from "react";
import QueryForm from "@/components/QueryForm";
import ResultDisplay from "@/components/ResultDisplay";
import DocumentList from "@/components/DocumentList";
import ThemeToggle from "@/components/ThemeToggle";
import { submitQuery, fetchBanditStats, checkHealth } from "@/lib/api";
import type { QueryResponse, BanditStats } from "@/lib/api";

type Status = "idle" | "running" | "done" | "error";

const QUERY_TYPES = ["factual", "comparative", "multi_hop"] as const;

export default function HomePage() {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [lastQuery, setLastQuery] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [banditStats, setBanditStats] = useState<BanditStats[]>([]);
  const [health, setHealth] = useState<{
    qdrant: boolean;
    database: boolean;
  } | null>(null);
  const [docRefreshTicker, setDocRefreshTicker] = useState(0);

  // Health check + bandit stats on mount
  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() => setHealth({ qdrant: false, database: false }));
    fetchBanditStats()
      .then(setBanditStats)
      .catch(() => {});
  }, []);

  const handleQuery = async (query: string) => {
    setStatus("running");
    setError(null);
    setResult(null);
    setLastQuery(query);

    try {
      const res = await submitQuery(query);
      setResult(res);
      setStatus("done");
      // Refresh bandit stats after each query
      fetchBanditStats()
        .then(setBanditStats)
        .catch(() => {});
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setStatus("error");
    }
  };

  return (
    <main className="page">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <span className="title">ara</span>
          <span className="subtitle">autonomous research agent</span>
        </div>
        <div className="header-right">
          {health && (
            <div className="health">
              <span className={`dot ${health.qdrant ? "green" : "red"}`} />
              qdrant
              <span
                className={`dot ${health.database ? "green" : "red"}`}
                style={{ marginLeft: 10 }}
              />
              postgres
            </div>
          )}
          <ThemeToggle />
        </div>
      </header>

      <div className="container">
        {/* Query form */}
        <QueryForm
          onSubmit={handleQuery}
          onSourceUpdated={() => setDocRefreshTicker((prev) => prev + 1)}
          loading={status === "running"}
        />

        {/* Document List */}
        <DocumentList refreshTicker={docRefreshTicker} />

        {/* Running indicator */}
        {status === "running" && (
          <div className="running-indicator">
            <div className="spinner" />
            <span>plan → retrieve → read → write → evaluate</span>
          </div>
        )}

        {/* Error */}
        {status === "error" && error && (
          <div className="error-box">error: {error}</div>
        )}

        {/* Result */}
        {status === "done" && result && (
          <ResultDisplay result={result} query={lastQuery} />
        )}

        {/* Bandit stats */}
        {banditStats.length > 0 && (
          <section className="bandit-section">
            <div className="section-title">
              bandit state — thompson sampling posteriors
            </div>
            <div className="bandit-grid">
              {banditStats.map((bs) => (
                <div key={bs.query_type} className="bandit-card">
                  <div className="bandit-type">{bs.query_type}</div>
                  <table className="bandit-table">
                    <thead>
                      <tr>
                        <th>cfg</th>
                        <th>α</th>
                        <th>β</th>
                        <th>win%</th>
                        <th>n</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(bs.configs).map(([cfg, stats]) => (
                        <tr key={cfg}>
                          <td className="cfg">{cfg}</td>
                          <td>{stats.alpha.toFixed(1)}</td>
                          <td>{stats.beta.toFixed(1)}</td>
                          <td
                            style={{
                              color:
                                stats.win_rate > 0.6
                                  ? "var(--success)"
                                  : "inherit",
                            }}
                          >
                            {(stats.win_rate * 100).toFixed(1)}%
                          </td>
                          <td>{stats.trials}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>

      <style jsx>{`
        .page {
          min-height: 100vh;
          background: var(--background);
        }
        .header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 16px 32px;
          border-bottom: 1px solid var(--border);
        }
        .header-left {
          display: flex;
          align-items: baseline;
          gap: 14px;
        }
        .title {
          font-size: 18px;
          font-weight: 700;
          letter-spacing: -0.02em;
          color: var(--foreground);
        }
        .subtitle {
          font-size: 12px;
          color: var(--muted-fg);
        }
        .header-right {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .health {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 12px;
          color: var(--muted-fg);
        }
        .dot {
          display: inline-block;
          width: 6px;
          height: 6px;
          border-radius: 50%;
        }
        .dot.green {
          background: var(--success);
        }
        .dot.red {
          background: var(--danger);
        }
        .container {
          max-width: 900px;
          margin: 0 auto;
          padding: 32px 24px 64px;
        }
        .running-indicator {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-top: 24px;
          color: var(--muted-fg);
          font-size: 13px;
        }
        .spinner {
          width: 14px;
          height: 14px;
          border: 2px solid var(--border);
          border-top-color: var(--accent);
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
          to {
            transform: rotate(360deg);
          }
        }
        .error-box {
          margin-top: 20px;
          border: 1px solid var(--danger);
          border-radius: 6px;
          padding: 12px 16px;
          color: var(--danger);
          font-size: 13px;
        }
        .bandit-section {
          margin-top: 40px;
          border-top: 1px solid var(--border);
          padding-top: 24px;
        }
        .section-title {
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--muted-fg);
          margin-bottom: 16px;
        }
        .bandit-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
          gap: 12px;
        }
        .bandit-card {
          border: 1px solid var(--border);
          border-radius: 8px;
          overflow: hidden;
        }
        .bandit-type {
          background: var(--card-bg);
          border-bottom: 1px solid var(--border);
          padding: 8px 12px;
          font-size: 11px;
          color: var(--muted-fg);
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }
        .bandit-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 12px;
        }
        .bandit-table th {
          padding: 6px 12px;
          text-align: left;
          color: var(--muted-fg);
          font-weight: 400;
          border-bottom: 1px solid var(--border);
          font-size: 11px;
        }
        .bandit-table td {
          padding: 6px 12px;
          border-bottom: 1px solid var(--border);
          opacity: 0.9;
        }
        .bandit-table tr:last-child td {
          border-bottom: none;
        }
        .cfg {
          color: var(--accent);
          font-weight: 600;
        }
      `}</style>
    </main>
  );
}
