"use client";

import ReactMarkdown from "react-markdown";
import type { QueryResponse } from "@/lib/api";

interface ResultDisplayProps {
  result: QueryResponse;
  query: string;
}

const CONFIG_LABELS: Record<string, string> = {
  A: "A  top_k=5  chunk=256  rerank=off",
  B: "B  top_k=10 chunk=512  rerank=off",
  C: "C  top_k=10 chunk=512  rerank=on",
};

function MetricBar({
  label,
  value,
  max,
  color,
}: {
  label: string;
  value: number;
  max: number;
  color: string;
}) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="metric-row">
      <span className="metric-label">{label}</span>
      <div className="metric-bar-track">
        <div
          className="metric-bar-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="metric-value">
        {typeof value === "number" ? value.toFixed(3) : value}
      </span>
    </div>
  );
}

function PipelineStage({
  step,
  label,
  detail,
}: {
  step: string;
  label: string;
  detail?: string;
}) {
  return (
    <div className="stage">
      <div className="stage-dot" />
      <div className="stage-body">
        <span className="stage-step">{step}</span>
        <span className="stage-label">{label}</span>
        {detail && <span className="stage-detail">{detail}</span>}
      </div>
    </div>
  );
}

export default function ResultDisplay({ result, query }: ResultDisplayProps) {
  const { answer, citations, metrics } = result;

  const stages = [
    { step: "01", label: "plan", detail: `query_type = ${metrics.query_type}` },
    {
      step: "02",
      label: "retrieve",
      detail: `config ${metrics.config} · ${metrics.hops} hop${metrics.hops !== 1 ? "s" : ""}`,
    },
    {
      step: "03",
      label: "read",
      detail: `${citations.length} citations extracted`,
    },
    { step: "04", label: "write", detail: `answer generated` },
    {
      step: "05",
      label: "evaluate",
      detail: `faithfulness = ${metrics.faithfulness.toFixed(3)}`,
    },
  ];

  return (
    <div className="result">
      {/* Pipeline trace */}
      <section className="panel">
        <div className="panel-header">pipeline trace</div>
        <div className="pipeline">
          {stages.map((s, i) => (
            <PipelineStage key={i} {...s} />
          ))}
        </div>
      </section>

      {/* Metrics */}
      <section className="panel">
        <div className="panel-header">
          metrics
          <span className="config-badge">
            {CONFIG_LABELS[metrics.config] ?? metrics.config}
          </span>
        </div>
        <div className="metrics-grid">
          <MetricBar
            label="faithfulness"
            value={metrics.faithfulness}
            max={1}
            color="var(--success)"
          />
          <MetricBar
            label="utility"
            value={metrics.utility}
            max={1}
            color="var(--accent)"
          />
          <MetricBar
            label="latency"
            value={metrics.latency}
            max={15}
            color="var(--warning)"
          />
          <MetricBar
            label="cost ($)"
            value={metrics.cost}
            max={0.005}
            color="var(--muted-fg)"
          />
        </div>
        <div className="metrics-raw">
          latency {metrics.latency.toFixed(2)}s · cost $
          {metrics.cost.toFixed(6)} · utility {metrics.utility.toFixed(4)}
        </div>
      </section>

      {/* Answer */}
      <section className="panel">
        <div className="panel-header">answer</div>
        <div className="prose answer-body">
          <ReactMarkdown>{answer}</ReactMarkdown>
        </div>
      </section>

      {/* Citations */}
      {citations.length > 0 && (
        <section className="panel">
          <div className="panel-header">citations ({citations.length})</div>
          <div className="citations">
            {citations.map((c) => (
              <div key={c.index} className="citation">
                <span className="citation-index">[{c.index}]</span>
                <div className="citation-body">
                  <div className="citation-source">{c.source}</div>
                  <div className="citation-text">{c.text}</div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <style jsx>{`
        .result {
          display: flex;
          flex-direction: column;
          gap: 16px;
          margin-top: 24px;
        }
        .panel {
          border: 1px solid var(--border);
          border-radius: 8px;
          overflow: hidden;
        }
        .panel-header {
          background: var(--card-bg);
          border-bottom: 1px solid var(--border);
          padding: 8px 16px;
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--muted-fg);
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .config-badge {
          font-size: 11px;
          color: var(--accent);
          text-transform: none;
          letter-spacing: 0;
        }
        /* Pipeline */
        .pipeline {
          display: flex;
          flex-direction: column;
          padding: 12px 16px;
          gap: 0;
        }
        .stage {
          display: flex;
          align-items: flex-start;
          gap: 12px;
          position: relative;
          padding-bottom: 12px;
        }
        .stage:last-child {
          padding-bottom: 0;
        }
        .stage-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--accent);
          margin-top: 5px;
          flex-shrink: 0;
          position: relative;
          z-index: 1;
        }
        .stage:not(:last-child) .stage-dot::after {
          content: "";
          position: absolute;
          left: 3px;
          top: 10px;
          width: 2px;
          height: calc(100% + 4px);
          background: var(--border);
        }
        .stage-body {
          display: flex;
          flex-direction: row;
          gap: 10px;
          align-items: baseline;
        }
        .stage-step {
          color: var(--muted-fg);
          font-size: 11px;
        }
        .stage-label {
          font-size: 13px;
          color: var(--foreground);
        }
        .stage-detail {
          font-size: 12px;
          color: var(--muted-fg);
        }
        /* Metrics */
        .metrics-grid {
          padding: 12px 16px;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .metric-row {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .metric-label {
          width: 90px;
          font-size: 12px;
          color: var(--muted-fg);
          flex-shrink: 0;
        }
        .metric-bar-track {
          flex: 1;
          height: 4px;
          background: var(--border);
          border-radius: 2px;
          overflow: hidden;
        }
        .metric-bar-fill {
          height: 100%;
          border-radius: 2px;
          transition: width 0.6s ease;
        }
        .metric-value {
          width: 60px;
          text-align: right;
          font-size: 12px;
          color: var(--foreground);
        }
        .metrics-raw {
          padding: 6px 16px 12px;
          font-size: 11px;
          color: var(--muted-fg);
        }
        /* Answer */
        .answer-body {
          padding: 16px;
          line-height: 1.7;
        }
        /* Citations */
        .citations {
          display: flex;
          flex-direction: column;
          gap: 0;
        }
        .citation {
          display: flex;
          gap: 12px;
          padding: 10px 16px;
          border-bottom: 1px solid var(--border);
        }
        .citation:last-child {
          border-bottom: none;
        }
        .citation-index {
          color: var(--accent);
          font-size: 12px;
          flex-shrink: 0;
          padding-top: 1px;
        }
        .citation-body {
          display: flex;
          flex-direction: column;
          gap: 3px;
        }
        .citation-source {
          font-size: 11px;
          color: var(--muted-fg);
        }
        .citation-text {
          font-size: 12px;
          color: var(--foreground);
          line-height: 1.5;
        }
      `}</style>
    </div>
  );
}
