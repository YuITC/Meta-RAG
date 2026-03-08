"use client";

import { useEffect } from "react";
import { useStore } from "@/lib/store";
import { BarChart3, BrainCircuit, Database, FileText, PanelLeft, PanelLeftClose, PanelRight, PanelRightClose, SlidersHorizontal } from "lucide-react";

interface HeaderProps {
  onOpenMetrics: () => void;
  onOpenStrategyLearning: () => void;
  onOpenDocuments: () => void;
  onOpenTrending: () => void;
  onOpenConfigs: () => void;
  leftSidebarOpen: boolean;
  rightSidebarOpen: boolean;
  onToggleLeftSidebar: () => void;
  onToggleRightSidebar: () => void;
}

export function Header({
  onOpenMetrics,
  onOpenStrategyLearning,
  onOpenDocuments,
  onOpenTrending,
  onOpenConfigs,
  leftSidebarOpen,
  rightSidebarOpen,
  onToggleLeftSidebar,
  onToggleRightSidebar,
}: HeaderProps) {
  const health = useStore((s) => s.health);
  const fetchHealth = useStore((s) => s.fetchHealth);

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 30_000);
    return () => clearInterval(interval);
  }, [fetchHealth]);

  const allUp = health?.qdrant && health?.database;

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b bg-card/90 px-4 backdrop-blur-md">
      <div className="flex items-center gap-4">
        <div>
          <h1 className="text-sm font-semibold tracking-tight text-foreground">
          Autonomous Research Agent
          </h1>
        </div>

        <div className="hidden items-center gap-2 md:flex">
          <button
            onClick={onToggleLeftSidebar}
            className="inline-flex items-center gap-1.5 rounded-md border border-border/80 bg-background/70 px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:border-primary/30 hover:text-foreground"
          >
            {leftSidebarOpen ? <PanelLeftClose size={13} /> : <PanelLeft size={13} />}
            Pipeline
          </button>
          <button
            onClick={onToggleRightSidebar}
            className="inline-flex items-center gap-1.5 rounded-md border border-border/80 bg-background/70 px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:border-primary/30 hover:text-foreground"
          >
            {rightSidebarOpen ? <PanelRightClose size={13} /> : <PanelRight size={13} />}
            Evidence
          </button>
          <button
            onClick={onOpenDocuments}
            className="inline-flex items-center gap-1.5 rounded-md border border-border/80 bg-background/70 px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:border-primary/30 hover:text-foreground"
          >
            <FileText size={13} />
            Documents
          </button>
          <button
            onClick={onOpenTrending}
            className="inline-flex items-center gap-1.5 rounded-md border border-border/80 bg-background/70 px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:border-primary/30 hover:text-foreground"
          >
            <Database size={13} />
            HuggingFace Trending
          </button>
          <button
            onClick={onOpenConfigs}
            className="inline-flex items-center gap-1.5 rounded-md border border-blue-200/80 bg-blue-50/70 px-2.5 py-1.5 text-xs text-blue-800 transition-colors hover:border-blue-300 hover:bg-blue-100/70"
          >
            <SlidersHorizontal size={13} />
            Available Configs
          </button>
          <button
            onClick={onOpenMetrics}
            className="inline-flex items-center gap-1.5 rounded-md border border-violet-200/80 bg-violet-50/70 px-2.5 py-1.5 text-xs text-violet-800 transition-colors hover:border-violet-300 hover:bg-violet-100/70"
          >
            <BarChart3 size={13} />
            Metrics
          </button>
          <button
            onClick={onOpenStrategyLearning}
            className="inline-flex items-center gap-1.5 rounded-md border border-fuchsia-200/80 bg-fuchsia-50/70 px-2.5 py-1.5 text-xs text-fuchsia-800 transition-colors hover:border-fuchsia-300 hover:bg-fuchsia-100/70"
          >
            <BrainCircuit size={13} />
            Strategy Learning
          </button>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-xs text-muted-foreground md:hidden">
          <button
            onClick={onOpenMetrics}
            className="inline-flex items-center gap-1 rounded-md border px-2 py-1"
          >
            <BarChart3 size={12} />
            Metrics
          </button>
          <button
            onClick={onOpenStrategyLearning}
            className="inline-flex items-center gap-1 rounded-md border px-2 py-1"
          >
            <BrainCircuit size={12} />
            Strategy
          </button>
          <button
            onClick={onOpenConfigs}
            className="inline-flex items-center gap-1 rounded-md border px-2 py-1"
          >
            <SlidersHorizontal size={12} />
            Configs
          </button>
          <button
            onClick={onToggleLeftSidebar}
            className="inline-flex items-center gap-1 rounded-md border px-2 py-1"
          >
            {leftSidebarOpen ? <PanelLeftClose size={12} /> : <PanelLeft size={12} />}
            Pipeline
          </button>
          <button
            onClick={onOpenDocuments}
            className="inline-flex items-center gap-1 rounded-md border px-2 py-1"
          >
            <FileText size={12} />
            Docs
          </button>
          <button
            onClick={onOpenTrending}
            className="inline-flex items-center gap-1 rounded-md border px-2 py-1"
          >
            <Database size={12} />
            Papers
          </button>
          <button
            onClick={onToggleRightSidebar}
            className="inline-flex items-center gap-1 rounded-md border px-2 py-1"
          >
            {rightSidebarOpen ? <PanelRightClose size={12} /> : <PanelRight size={12} />}
            Evidence
          </button>
        </div>

        <div className="flex items-center gap-2 rounded-full border border-border/70 bg-background/70 px-3 py-1.5 text-xs text-muted-foreground">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              health === null
                ? "bg-gray-400"
                : allUp
                  ? "bg-emerald-500"
                  : "bg-red-500"
            }`}
          />
          {health === null
            ? "Connecting…"
            : allUp
              ? "System Online"
              : "Degraded"}
        </div>
      </div>
    </header>
  );
}
