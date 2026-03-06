"use client";

import type { QueryResult } from "@/app/page";

type Props = { result: QueryResult };

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
      <span style={{ color: "#888", fontSize: 12, minWidth: 120 }}>{label}</span>
      <span style={{ fontWeight: 600, fontSize: 13 }}>{value}</span>
    </div>
  );
}

export default function ResultDisplay({ result }: Props) {
  return (
    <div style={{ marginTop: 32 }}>
      {/* Answer */}
      <div
        style={{
          background: "#161616",
          border: "1px solid #2a2a2a",
          borderRadius: 10,
          padding: "20px 24px",
          lineHeight: 1.7,
          whiteSpace: "pre-wrap",
          fontSize: 15,
        }}
      >
        {result.answer}
      </div>

      {/* Metrics bar */}
      <div
        style={{
          marginTop: 16,
          padding: "14px 20px",
          background: "#111",
          border: "1px solid #222",
          borderRadius: 8,
          display: "flex",
          flexWrap: "wrap",
          gap: "12px 32px",
        }}
      >
        <Metric label="Latency" value={`${result.latency.toFixed(2)}s`} />
        <Metric label="Cost" value={`$${result.cost.toFixed(5)}`} />
        <Metric label="Utility" value={result.utility.toFixed(3)} />
        <Metric label="Faithfulness" value={result.faithfulness.toFixed(2)} />
        <Metric label="Citation grounding" value={result.citation_grounding.toFixed(2)} />
        <Metric label="Config" value={result.config_id} />
        <Metric label="Query type" value={result.query_type} />
        {result.retry_count > 0 && <Metric label="Retries" value={String(result.retry_count)} />}
      </div>

      {/* Citations */}
      {result.citations.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: "#888", marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Citations
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {result.citations.map((c, i) => (
              <div
                key={i}
                style={{
                  padding: "12px 16px",
                  background: "#131313",
                  border: "1px solid #252525",
                  borderRadius: 8,
                  fontSize: 13,
                }}
              >
                <div style={{ color: "#7eb8f7", fontWeight: 600, marginBottom: 4 }}>
                  [{i + 1}] {c.title || c.doc_id}
                </div>
                <div style={{ color: "#aaa", lineHeight: 1.6 }}>{c.text}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
