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
      <div className="metric-info">
        <span className="metric-label">{label}</span>
        <span className="metric-value">
          {typeof value === "number" ? value.toFixed(3) : value}
        </span>
      </div>
      <div className="metric-bar-container">
        <div className="metric-bar-track">
          <div
            className="metric-bar-fill"
            style={{
              width: `${pct}%`,
              background: color,
              boxShadow: `0 0 10px ${color}44`,
            }}
          />
        </div>
      </div>
      <style jsx>{`
        .metric-row {
          display: flex;
          flex-direction: column;
          gap: 6px;
          padding: 8px 0;
        }
        .metric-info {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
        }
        .metric-label {
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: var(--muted-fg);
          font-weight: 500;
        }
        .metric-value {
          font-family: var(--font-mono);
          font-size: 13px;
          color: var(--foreground);
          font-weight: 600;
        }
        .metric-bar-container {
          width: 100%;
        }
        .metric-bar-track {
          height: 6px;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 3px;
          overflow: hidden;
        }
        .metric-bar-fill {
          height: 100%;
          border-radius: 3px;
          transition: width 1s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
      `}</style>
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
      <div className="stage-visual">
        <div className="stage-dot" />
        <div className="stage-line" />
      </div>
      <div className="stage-body">
        <span className="stage-step">{step}</span>
        <span className="stage-label">{label}</span>
        {detail && <span className="stage-detail">{detail}</span>}
      </div>
      <style jsx>{`
        .stage {
          display: flex;
          gap: 16px;
          min-height: 40px;
        }
        .stage-visual {
          display: flex;
          flex-direction: column;
          align-items: center;
          width: 12px;
          position: relative;
        }
        .stage-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--accent);
          flex-shrink: 0;
          margin-top: 6px;
          z-index: 2;
          box-shadow: 0 0 8px var(--accent);
        }
        .stage-line {
          width: 1px;
          background: var(--border);
          flex-grow: 1;
          margin: 4px 0;
        }
        .stage:last-child .stage-line {
          display: none;
        }
        .stage-body {
          display: flex;
          flex-direction: row;
          align-items: baseline;
          gap: 12px;
          padding-bottom: 20px;
          flex: 1;
        }
        .stage:last-child .stage-body {
          padding-bottom: 0;
        }
        .stage-step {
          font-family: var(--font-mono);
          font-size: 11px;
          color: var(--muted-fg);
          opacity: 0.7;
          min-width: 20px;
        }
        .stage-label {
          font-size: 14px;
          font-weight: 500;
          color: var(--foreground);
          min-width: 80px;
        }
        .stage-detail {
          font-size: 12px;
          color: var(--muted-fg);
          font-family: var(--font-mono);
          background: rgba(255, 255, 255, 0.03);
          padding: 2px 8px;
          border-radius: 4px;
        }
      `}</style>
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
          padding: 16px;
        }
        .stage {
          display: flex;
          gap: 16px;
          min-height: 40px;
        }
        .stage-visual {
          display: flex;
          flex-direction: column;
          align-items: center;
          width: 12px;
          position: relative;
        }
        .stage-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--accent);
          flex-shrink: 0;
          margin-top: 6px;
          z-index: 2;
          box-shadow: 0 0 8px var(--accent);
        }
        .stage-line {
          width: 1px;
          background: var(--border);
          flex-grow: 1;
          margin: 4px 0;
        }
        .stage:last-child .stage-line {
          display: none;
        }
        .stage-body {
          display: flex;
          flex-direction: row;
          align-items: baseline;
          gap: 12px;
          padding-bottom: 20px;
          flex: 1;
        }
        .stage:last-child .stage-body {
          padding-bottom: 0;
        }
        .stage-step {
          font-family: var(--font-mono);
          font-size: 11px;
          color: var(--muted-fg);
          opacity: 0.7;
          min-width: 20px;
        }
        .stage-label {
          font-size: 14px;
          font-weight: 500;
          color: var(--foreground);
          min-width: 80px;
        }
        .stage-detail {
          font-size: 12px;
          color: var(--muted-fg);
          font-family: var(--font-mono);
          background: rgba(255, 255, 255, 0.03);
          padding: 2px 8px;
          border-radius: 4px;
        }
        /* Metrics */
        .metrics-grid {
          padding: 20px 16px;
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 24px;
        }
        .metrics-raw {
          padding: 12px 16px;
          border-top: 1px solid var(--border);
          font-size: 11px;
          color: var(--muted-fg);
          font-family: var(--font-mono);
          display: flex;
          gap: 16px;
          background: rgba(255, 255, 255, 0.01);
        }
        @media (max-width: 640px) {
          .metrics-grid {
            grid-template-columns: 1fr;
            gap: 12px;
          }
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
