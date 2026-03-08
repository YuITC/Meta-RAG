"use client";

import { useStore } from "@/lib/store";
import { Search, Loader2, PanelRightOpen, PanelRightClose } from "lucide-react";
import { Button } from "@/components/ui/button";

interface QueryInputProps {
  onToggleEvidence?: () => void;
  evidenceOpen?: boolean;
}

export function QueryInput({ onToggleEvidence, evidenceOpen = false }: QueryInputProps) {
  const query = useStore((s) => s.query);
  const setQuery = useStore((s) => s.setQuery);
  const isResearching = useStore((s) => s.isResearching);
  const runResearch = useStore((s) => s.runResearch);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runResearch();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      runResearch();
    }
  };

  return (
    <form onSubmit={handleSubmit} className="border-t bg-card/90 px-4 py-3 backdrop-blur-md">
      <div className="mx-auto flex max-w-5xl items-end gap-2">
        <div className="relative flex-1">
          <Search
            size={16}
            className="absolute left-3 top-3.5 text-muted-foreground"
          />
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a research question…"
            rows={3}
            className="min-h-[88px] w-full resize-none rounded-xl border border-border/80 bg-background/90 py-3 pl-9 pr-3 text-sm leading-6 placeholder:text-muted-foreground shadow-[0_8px_18px_rgba(30,64,175,0.05)] focus:outline-none focus:ring-2 focus:ring-ring"
            disabled={isResearching}
          />
        </div>

        {onToggleEvidence && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onToggleEvidence}
            className="xl:hidden border-border/80 bg-background/80"
          >
            {evidenceOpen ? (
              <PanelRightClose size={14} className="mr-1.5" />
            ) : (
              <PanelRightOpen size={14} className="mr-1.5" />
            )}
            Evidence
          </Button>
        )}

        <Button
          type="submit"
          size="sm"
          disabled={isResearching || !query.trim()}
          className="min-w-[120px] rounded-xl bg-primary text-primary-foreground shadow-[0_10px_24px_rgba(59,91,219,0.22)] hover:bg-primary/90"
        >
          {isResearching ? (
            <>
              <Loader2 size={14} className="mr-1.5 animate-spin" />
              Running…
            </>
          ) : (
            "Run Research"
          )}
        </Button>
      </div>
    </form>
  );
}
