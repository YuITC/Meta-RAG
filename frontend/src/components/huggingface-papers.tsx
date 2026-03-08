"use client";

import { useEffect } from "react";
import { useStore } from "@/lib/store";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Loader2, ExternalLink, Download } from "lucide-react";

export function HuggingFacePapers() {
  const papers = useStore((s) => s.trendingPapers);
  const selected = useStore((s) => s.selectedPapers);
  const isFetching = useStore((s) => s.isFetchingPapers);
  const isIngesting = useStore((s) => s.isIngestingPapers);
  const fetchPapers = useStore((s) => s.fetchTrendingPapers);
  const toggle = useStore((s) => s.togglePaperSelection);
  const ingest = useStore((s) => s.ingestSelectedPapers);

  useEffect(() => {
    if (papers.length === 0) fetchPapers();
  }, [papers.length, fetchPapers]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          HuggingFace Trending
        </h3>
        <Button
          variant="outline"
          size="sm"
          className="h-7 text-xs"
          onClick={fetchPapers}
          disabled={isFetching}
        >
          {isFetching ? (
            <Loader2 size={12} className="mr-1 animate-spin" />
          ) : null}
          Refresh
        </Button>
      </div>

      {isFetching && papers.length === 0 ? (
        <div className="flex items-center justify-center py-6 text-xs text-muted-foreground">
          <Loader2 size={14} className="mr-2 animate-spin" />
          Loading papers…
        </div>
      ) : papers.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          No papers loaded. Click Refresh.
        </p>
      ) : (
        <>
          <ScrollArea className="h-[320px]">
            <div className="space-y-1 pr-2">
              {papers.map((paper, idx) => {
                const isSelected = selected.has(idx);
                return (
                  <button
                    key={idx}
                    onClick={() => toggle(idx)}
                    className={`w-full text-left rounded-md border p-2 transition-colors ${
                      isSelected
                        ? "border-blue-400 bg-blue-50"
                        : "hover:border-gray-300"
                    }`}
                  >
                    <p className="text-xs font-medium leading-snug line-clamp-2">
                      {paper.title}
                    </p>
                    {paper.author && (
                      <p className="mt-0.5 text-[10px] text-muted-foreground truncate">
                        {paper.author}
                      </p>
                    )}
                    {paper.abstract && (
                      <p className="mt-1 text-[10px] text-muted-foreground line-clamp-2 leading-relaxed">
                        {paper.abstract}
                      </p>
                    )}
                    <div className="mt-1 flex items-center gap-2 text-[10px] text-muted-foreground">
                      {paper.published && <span>{paper.published}</span>}
                      {paper.arxiv_url && (
                        <span className="flex items-center gap-0.5">
                          <ExternalLink size={8} />
                          arXiv
                        </span>
                      )}
                      {paper.github_stars && (
                        <span>★ {paper.github_stars}</span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </ScrollArea>

          {selected.size > 0 && (
            <Button
              size="sm"
              className="w-full text-xs"
              onClick={ingest}
              disabled={isIngesting}
            >
              {isIngesting ? (
                <Loader2 size={12} className="mr-1.5 animate-spin" />
              ) : (
                <Download size={12} className="mr-1.5" />
              )}
              Load {selected.size} Paper{selected.size > 1 ? "s" : ""} into
              Knowledge Base
            </Button>
          )}
        </>
      )}
    </div>
  );
}
