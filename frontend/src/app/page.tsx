"use client";

import { useState } from "react";
import QueryForm from "@/components/QueryForm";
import ResultDisplay from "@/components/ResultDisplay";

export type QueryResult = {
  query: string;
  answer: string;
  citations: { doc_id: string; title: string; text: string; score: number }[];
  query_type: string;
  config_id: string;
  faithfulness: number;
  citation_grounding: number;
  utility: number;
  cost: number;
  latency: number;
  retry_count: number;
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

export default function Home() {
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleQuery(query: string) {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail ?? `HTTP ${res.status}`);
      }
      const data: QueryResult = await res.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 860, margin: "0 auto", padding: "48px 24px" }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 8 }}>
        Autonomous Research Agent
      </h1>
      <p style={{ color: "#888", marginBottom: 32, fontSize: 14 }}>
        Adaptive RAG with multi-armed bandit retrieval optimization
      </p>

      <QueryForm onSubmit={handleQuery} loading={loading} />

      {error && (
        <div style={{ marginTop: 24, color: "#f87171", background: "#1c0a0a", padding: "12px 16px", borderRadius: 8 }}>
          Error: {error}
        </div>
      )}

      {result && <ResultDisplay result={result} />}
    </main>
  );
}
