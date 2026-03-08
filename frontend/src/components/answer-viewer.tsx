"use client";

import { useStore } from "@/lib/store";
import { useCallback } from "react";

export function AnswerViewer() {
  const result = useStore((s) => s.result);
  const setHighlightedCitation = useStore((s) => s.setHighlightedCitation);
  const highlightedCitation = useStore((s) => s.highlightedCitation);
  const error = useStore((s) => s.error);

  const renderAnswer = useCallback(
    (text: string) => {
      // Split on citation references like [1], [2], etc.
      const parts = text.split(/(\[\d+\])/g);
      return parts.map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (match) {
          const idx = parseInt(match[1], 10);
          return (
            <button
              key={i}
              onClick={() =>
                setHighlightedCitation(highlightedCitation === idx ? null : idx)
              }
              className={`inline-flex items-center justify-center rounded px-1 text-xs font-mono font-medium transition-colors ${
                highlightedCitation === idx
                  ? "bg-blue-600 text-white"
                  : "bg-blue-100 text-blue-700 hover:bg-blue-200"
              }`}
            >
              {part}
            </button>
          );
        }
        return <span key={i}>{part}</span>;
      });
    },
    [highlightedCitation, setHighlightedCitation]
  );

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
        {error}
      </div>
    );
  }

  if (!result) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        Run a query to see the research answer.
      </div>
    );
  }

  if (result.abstained) {
    return (
      <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
        The system abstained from answering — insufficient evidence was found to
        produce a reliable response.
      </div>
    );
  }

  return (
    <div className="max-w-none text-[14px] leading-7 text-foreground">
      {result.answer.split("\n").map((line, i) => (
        <p key={i} className="mb-3 max-w-3xl">
          {renderAnswer(line)}
        </p>
      ))}
    </div>
  );
}
