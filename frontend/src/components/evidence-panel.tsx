"use client";

import { useStore } from "@/lib/store";
import { ScrollArea } from "@/components/ui/scroll-area";
import { FileText, Hash, Layers3, Link2 } from "lucide-react";
import { useState } from "react";

function compactSourceLabel(source: string) {
  return source.split(/[\\/]/).pop() || source;
}

function uniqueSourceCount(sources: string[]) {
  return new Set(sources).size;
}

export function EvidencePanel() {
  const result = useStore((s) => s.result);
  const highlightedCitation = useStore((s) => s.highlightedCitation);
  const setHighlightedCitation = useStore((s) => s.setHighlightedCitation);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  if (!result || result.citations.length === 0) {
    return (
      <section className="flex h-full flex-col rounded-xl border bg-card shadow-[0_10px_24px_rgba(15,118,110,0.05)]">
        <div className="border-b px-3 py-3">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Evidence View
          </h2>
        </div>
        <div className="flex flex-1 items-center justify-center px-4 text-xs text-muted-foreground">
          Retrieved evidence will appear here after running a query.
        </div>
      </section>
    );
  }

  const sources = result.citations.map((citation) => citation.source);

  return (
    <section className="flex h-full flex-col rounded-xl border bg-card shadow-[0_10px_24px_rgba(15,118,110,0.05)]">
      <div className="shrink-0 border-b bg-gradient-to-r from-cyan-50/70 to-background px-3 py-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Evidence View
            </h2>
            <p className="mt-1 text-[11px] text-muted-foreground">
              Retrieved chunks referenced by the generated answer.
            </p>
          </div>
          <div className="text-right font-mono text-[11px] text-muted-foreground">
            <div>{result.citations.length} chunks</div>
            <div>{uniqueSourceCount(sources)} sources</div>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-3 gap-2 text-[11px]">
          <div className="rounded-lg border border-cyan-200/70 bg-cyan-50/50 px-2 py-1.5">
            <div className="flex items-center gap-1 text-muted-foreground">
              <Layers3 size={11} />
              Chunks
            </div>
            <div className="mt-1 font-mono text-foreground">{result.citations.length}</div>
          </div>
          <div className="rounded-lg border border-blue-200/70 bg-blue-50/40 px-2 py-1.5">
            <div className="flex items-center gap-1 text-muted-foreground">
              <FileText size={11} />
              Sources
            </div>
            <div className="mt-1 font-mono text-foreground">{uniqueSourceCount(sources)}</div>
          </div>
          <div className="rounded-lg border border-emerald-200/70 bg-emerald-50/40 px-2 py-1.5">
            <div className="flex items-center gap-1 text-muted-foreground">
              <Link2 size={11} />
              Active
            </div>
            <div className="mt-1 font-mono text-foreground">
              {highlightedCitation ? `#${highlightedCitation}` : "none"}
            </div>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap gap-1">
          {result.citations.map((citation) => {
            const isActive = highlightedCitation === citation.index;
            return (
              <button
                key={citation.index}
                onClick={() => setHighlightedCitation(isActive ? null : citation.index)}
                className={`rounded border px-1.5 py-0.5 text-[10px] font-mono transition-colors ${
                  isActive
                    ? "border-blue-500 bg-blue-600 text-white"
                    : "border-border text-muted-foreground hover:border-blue-300 hover:text-foreground"
                }`}
              >
                [{citation.index}]
              </button>
            );
          })}
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="space-y-2 p-3">
          {result.citations.map((c) => {
            const isHighlighted = highlightedCitation === c.index;
            const isExpanded = expandedIdx === c.index;
            return (
              <button
                key={c.index}
                onClick={() => {
                  setHighlightedCitation(isHighlighted ? null : c.index);
                  setExpandedIdx(isExpanded ? null : c.index);
                }}
                className={`w-full text-left rounded-md border p-2.5 transition-colors ${
                  isHighlighted
                    ? "border-blue-500 bg-blue-50 shadow-[inset_3px_0_0_0_rgb(59,130,246)]"
                    : "hover:border-gray-300 hover:bg-muted/20"
                }`}
              >
                <div className="flex items-start gap-2.5">
                  <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded border bg-muted text-[10px] font-mono font-bold text-gray-700">
                    {c.index}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="mb-1.5 flex items-center justify-between gap-2">
                      <div className="min-w-0">
                        <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
                          <FileText size={10} />
                          <span className="truncate">{compactSourceLabel(c.source)}</span>
                        </div>
                        <div className="mt-0.5 flex items-center gap-2 text-[10px] font-mono text-muted-foreground/80">
                          <span className="inline-flex items-center gap-1">
                            <Hash size={9} />
                            ref {c.index}
                          </span>
                          <span>{c.text.length} chars</span>
                        </div>
                      </div>
                      {isHighlighted && (
                        <span className="rounded bg-blue-600 px-1.5 py-0.5 text-[10px] font-mono text-white">
                          linked
                        </span>
                      )}
                    </div>
                    <p
                      className={`text-[12px] leading-5 text-foreground ${
                        isExpanded ? "" : "line-clamp-4"
                      }`}
                    >
                      {c.text}
                    </p>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </ScrollArea>
    </section>
  );
}
