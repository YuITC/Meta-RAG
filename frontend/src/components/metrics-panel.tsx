"use client";

import { useStore } from "@/lib/store";

interface MetricRowProps {
  label: string;
  value: number | string;
  format?: "pct" | "time" | "raw";
}

function MetricRow({ label, value, format = "pct" }: MetricRowProps) {
  let display: string;
  if (typeof value === "string") {
    display = value;
  } else if (format === "pct") {
    display = (value * 100).toFixed(1) + "%";
  } else if (format === "time") {
    display = value.toFixed(2) + "s";
  } else {
    display = value.toFixed(3);
  }

  return (
    <div className="flex items-center justify-between border-t border-border/60 py-1.5 text-xs first:border-t-0 first:pt-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono font-medium tracking-tight">{display}</span>
    </div>
  );
}

export function MetricsPanel() {
  const result = useStore((s) => s.result);

  if (!result) return null;

  const m = result.metrics;
  const d = m.retrieval_diagnostics;

  return (
    <section className="rounded-xl border bg-card shadow-[0_10px_24px_rgba(37,99,235,0.05)] p-3">
      <div className="mb-3 border-b pb-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Metrics
        </h3>
        <p className="mt-1 text-[11px] text-muted-foreground">
          Reliability, control, and retrieval indicators for the current run.
        </p>
      </div>

      {/* Answer quality */}
      <div className="rounded-lg border border-emerald-200/70 bg-emerald-50/40 p-2.5">
        <p className="mb-1 text-[10px] font-medium uppercase text-muted-foreground tracking-wide">
          Answer Quality
        </p>
        <MetricRow label="Faithfulness" value={m.faithfulness} />
        <MetricRow label="Citation Precision" value={m.citation_precision} />
        <MetricRow label="Answer Completeness" value={m.answer_completeness} />
        <MetricRow label="Unsupported Claims" value={m.unsupported_claim_rate} />
      </div>

      {/* System */}
      <div className="mt-3 rounded-lg border border-blue-200/70 bg-blue-50/35 p-2.5">
        <p className="mb-1 text-[10px] font-medium uppercase text-muted-foreground tracking-wide">
          System
        </p>
        <MetricRow label="Latency" value={m.latency} format="time" />
        <MetricRow label="Utility" value={m.utility} format="raw" />
        <MetricRow label="Config" value={m.config} />
        <MetricRow label="Query Type" value={m.query_type} />
        <MetricRow label="Hops" value={String(m.hops)} />
      </div>

      {/* Retrieval */}
      {d && (
        <div className="mt-3 rounded-lg border border-cyan-200/70 bg-cyan-50/35 p-2.5">
          <p className="mb-1 text-[10px] font-medium uppercase text-muted-foreground tracking-wide">
            Retrieval
          </p>
          <MetricRow label="Query Coverage" value={d.query_coverage} />
          <MetricRow label="Recall Proxy" value={d.estimated_recall_proxy} />
          <MetricRow label="Document Diversity" value={d.document_diversity} />
          <MetricRow label="Redundancy" value={d.retrieval_redundancy} />
        </div>
      )}
    </section>
  );
}
