"use client";

import { useState } from "react";
import { useStore } from "@/lib/store";
import { Badge } from "@/components/ui/badge";
import { Check, Loader2, Circle, AlertCircle } from "lucide-react";
import type { PipelineStep, PipelineStepStatus } from "@/lib/types";
import { ScrollArea } from "@/components/ui/scroll-area";

function StatusIcon({ status }: { status: PipelineStepStatus }) {
  switch (status) {
    case "done":
      return <Check size={14} className="text-emerald-600" />;
    case "running":
      return <Loader2 size={14} className="animate-spin text-blue-600" />;
    case "error":
      return <AlertCircle size={14} className="text-red-500" />;
    default:
      return <Circle size={14} className="text-gray-300" />;
  }
}

function isQueryField(key: string) {
  return key === "primary_query" || key === "followup_query";
}

function shouldClampValue(value: string, key: string) {
  return isQueryField(key) || value.length > 90;
}

function DetailValue({ fieldKey, value }: { fieldKey: string; value: unknown }) {
  const [expanded, setExpanded] = useState(false);
  const displayValue = typeof value === "number" ? value.toFixed(3) : String(value);
  const clampable = shouldClampValue(displayValue, fieldKey);

  if (isQueryField(fieldKey)) {
    return (
      <div className="space-y-1">
        <div
          className={`rounded-md border bg-muted/40 px-2 py-1.5 font-mono text-[10px] leading-5 text-foreground/85 ${
            clampable && !expanded ? "line-clamp-4" : "whitespace-pre-wrap break-words"
          }`}
        >
          {displayValue}
        </div>
        {clampable && (
          <button
            type="button"
            onClick={() => setExpanded((current) => !current)}
            className="text-[10px] font-medium text-blue-700 transition-colors hover:text-blue-900"
          >
            {expanded ? "collapse" : "expand"}
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <span
        className={`block break-words whitespace-normal text-foreground/80 ${
          clampable && !expanded ? "line-clamp-3" : ""
        }`}
      >
        {displayValue}
      </span>
      {clampable && (
        <button
          type="button"
          onClick={() => setExpanded((current) => !current)}
          className="text-[10px] font-medium text-blue-700 transition-colors hover:text-blue-900"
        >
          {expanded ? "collapse" : "expand"}
        </button>
      )}
    </div>
  );
}

function StepDetails({ details }: { details: Record<string, unknown> | null }) {
  if (!details) return null;
  return (
    <div className="mt-2 space-y-1 text-[11px] text-muted-foreground font-mono whitespace-normal">
      {Object.entries(details).map(([key, value]) => (
        <div
          key={key}
          className="grid gap-1 border-t border-border/60 pt-1 first:border-t-0 first:pt-0"
        >
          <span className="text-muted-foreground/70">{key.replace(/_/g, " ")}</span>
          <DetailValue fieldKey={key} value={value} />
        </div>
      ))}
    </div>
  );
}

function PipelineStepCard({ step, index }: { step: PipelineStep; index: number }) {
  return (
    <div className="relative w-full rounded-lg border bg-background p-3">
      <div className="mb-2 flex items-start gap-2">
        <div className="flex items-center gap-2 pt-0.5">
          <StatusIcon status={step.status} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-2">
            <div>
              <p className="text-[12px] font-medium text-foreground">{step.label}</p>
              <p className="mt-0.5 font-mono text-[10px] text-muted-foreground">
                step {index + 1}
              </p>
            </div>
            {step.status === "running" && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 text-blue-600 border-blue-200">
                active
              </Badge>
            )}
          </div>
        </div>
      </div>

      <StepDetails details={step.details} />
    </div>
  );
}

export function PipelinePanel() {
  const pipeline = useStore((s) => s.pipeline);
  const retryState = useStore((s) => s.retryState);
  const completed = pipeline.filter((step) => step.status === "done").length;

  return (
    <section className="rounded-xl border bg-card shadow-[0_10px_24px_rgba(59,91,219,0.05)] p-3">
      <div className="mb-3 flex items-end justify-between gap-3 border-b pb-2">
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Research Pipeline
          </h2>
          <p className="mt-1 text-[11px] text-muted-foreground">
            Vertical execution trace of the research agent.
          </p>
        </div>
        <span className="font-mono text-[11px] text-muted-foreground">
          {completed}/{pipeline.length}
        </span>
      </div>

      {retryState && (
        <div className="mb-3 rounded-xl border border-amber-200 bg-amber-50/90 px-3 py-2 shadow-[0_8px_18px_rgba(245,158,11,0.08)]">
          <div className="flex items-center justify-between gap-3">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-amber-800">
              Retry #{retryState.retryCount}
            </p>
            <span className="font-mono text-[10px] text-amber-700">
              {retryState.previousConfig} → {retryState.selectedConfig}
            </span>
          </div>
          <p className="mt-1 text-[11px] text-amber-900/80">
            Evaluation fell below threshold and the system selected a new config to run again.
          </p>
          <div className="mt-2 flex gap-4 font-mono text-[10px] text-amber-800/90">
            <span>reason: {retryState.reason}</span>
            <span>faithfulness: {retryState.failedFaithfulness.toFixed(3)}</span>
          </div>
        </div>
      )}

      <ScrollArea className="w-full">
        <div className="space-y-3 pr-1">
          {pipeline.map((step, i) => (
            <div key={step.id} className="flex flex-col items-stretch">
              <PipelineStepCard step={step} index={i} />
              {i < pipeline.length - 1 && (
                <div className="flex justify-center py-1">
                  <div className="h-5 w-px bg-border" />
                </div>
              )}
            </div>
          ))}
        </div>
      </ScrollArea>
    </section>
  );
}
