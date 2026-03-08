"use client";

import { AnswerViewer } from "./answer-viewer";
import { useStore } from "@/lib/store";
import { Badge } from "@/components/ui/badge";

export function ResultPanel() {
  const result = useStore((s) => s.result);

  return (
    <div className="flex flex-col">
      <div className="border-b bg-gradient-to-r from-blue-50/90 via-background to-cyan-50/60 px-5 py-4">
        <div className="mb-2 flex items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold tracking-tight text-foreground">
              Final Synthesis
            </h2>
            <p className="mt-1 text-xs text-muted-foreground">
              Grounded answer generated from retrieved evidence and citation verification.
            </p>
          </div>
          {result && (
            <div className="flex items-center gap-1.5">
              <Badge variant="outline" className="border-blue-200 bg-blue-50 font-mono text-[10px] text-blue-800">
                {result.metrics.query_type}
              </Badge>
              <Badge variant="outline" className="border-cyan-200 bg-cyan-50 font-mono text-[10px] text-cyan-800">
                {result.citations.length} refs
              </Badge>
              <Badge variant="outline" className="border-emerald-200 bg-emerald-50 font-mono text-[10px] text-emerald-800">
                cfg {result.metrics.config}
              </Badge>
            </div>
          )}
        </div>
      </div>

      <div className="px-5 py-5">
        <AnswerViewer />
      </div>
    </div>
  );
}
