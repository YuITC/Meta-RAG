"use client";

import { useEffect, useState } from "react";
import { Header } from "@/components/header";
import { QueryInput } from "@/components/query-input";
import { PipelinePanel } from "@/components/pipeline-panel";
import { ResultPanel } from "@/components/result-panel";
import { EvidencePanel } from "@/components/evidence-panel";
import { MetricsPanel } from "@/components/metrics-panel";
import { BanditViewer } from "@/components/bandit-viewer";
import { FileUpload } from "@/components/file-upload";
import { HuggingFacePapers } from "@/components/huggingface-papers";
import { DataSourceModal } from "@/components/data-source-modal";

const CONFIG_SUMMARIES = [
  {
    name: "A",
    description: "top_k 5 · 256 chunks · no rewrite · 1 hop",
  },
  {
    name: "B",
    description: "top_k 10 · template rewrite · 512 chunks",
  },
  {
    name: "C",
    description: "rerank on · 2 hops · 512 chunks",
  },
  {
    name: "D",
    description: "LLM rewrite · 2 hops · 5 variants",
  },
];

export default function Home() {
  const [isMobileViewport, setIsMobileViewport] = useState(false);
  const [pipelineDrawerOpen, setPipelineDrawerOpen] = useState(false);
  const [evidenceDrawerOpen, setEvidenceDrawerOpen] = useState(false);
  const [metricsModalOpen, setMetricsModalOpen] = useState(false);
  const [strategyModalOpen, setStrategyModalOpen] = useState(false);
  const [documentsModalOpen, setDocumentsModalOpen] = useState(false);
  const [papersModalOpen, setPapersModalOpen] = useState(false);
  const [configsModalOpen, setConfigsModalOpen] = useState(false);
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(true);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(true);

  useEffect(() => {
    const handleResize = () => {
      setIsMobileViewport(window.innerWidth < 1024);

      if (window.innerWidth >= 1024) {
        setPipelineDrawerOpen(false);
        setEvidenceDrawerOpen(false);
      }
    };

    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const isDesktopViewport = () =>
    typeof window !== "undefined" && window.innerWidth >= 1024;

  const handleToggleLeftSidebar = () => {
    if (isDesktopViewport()) {
      setLeftSidebarOpen((open) => !open);
      return;
    }

    setPipelineDrawerOpen((open) => !open);
  };

  const handleToggleRightSidebar = () => {
    if (isDesktopViewport()) {
      setRightSidebarOpen((open) => !open);
      return;
    }

    setEvidenceDrawerOpen((open) => !open);
  };

  return (
    <div className="flex h-screen flex-col bg-background">
      {/* Header */}
      <Header
        onOpenMetrics={() => setMetricsModalOpen(true)}
        onOpenStrategyLearning={() => setStrategyModalOpen(true)}
        onOpenDocuments={() => setDocumentsModalOpen(true)}
        onOpenTrending={() => setPapersModalOpen(true)}
        onOpenConfigs={() => setConfigsModalOpen(true)}
        leftSidebarOpen={isMobileViewport ? pipelineDrawerOpen : leftSidebarOpen}
        rightSidebarOpen={isMobileViewport ? evidenceDrawerOpen : rightSidebarOpen}
        onToggleLeftSidebar={handleToggleLeftSidebar}
        onToggleRightSidebar={handleToggleRightSidebar}
      />

      <div className="flex flex-1 overflow-hidden">
        {/* Main content */}
        <main className="flex flex-1 flex-col overflow-hidden">
          {/* Content area: pipeline + metrics left, answer center, evidence right */}
          <div className="flex flex-1 overflow-hidden">
            {/* Left: Research Pipeline */}
            {leftSidebarOpen && (
              <div className="hidden min-w-0 basis-0 overflow-y-auto border-r bg-card/90 p-3 backdrop-blur-sm lg:block lg:flex-[1]">
                <PipelinePanel />
              </div>
            )}

            {/* Center: Answer + Query Bar */}
            <div className="min-w-0 basis-0 flex-[3] overflow-hidden">
              <div className="flex h-full flex-col">
                <div className="min-h-0 flex-1 overflow-y-auto p-3">
                  <div className="mx-auto max-w-[68rem]">
                    <div className="rounded-xl border bg-card shadow-[0_12px_28px_rgba(30,64,175,0.06)]">
                      <ResultPanel />
                    </div>
                  </div>
                </div>

                <QueryInput
                  onToggleEvidence={() => setEvidenceDrawerOpen((open) => !open)}
                  evidenceOpen={evidenceDrawerOpen}
                />
              </div>
            </div>

            {/* Right: Evidence */}
            {rightSidebarOpen && (
              <div className="hidden min-w-0 basis-0 border-l bg-card/90 p-3 backdrop-blur-sm lg:block lg:flex-[1]">
                <EvidencePanel />
              </div>
            )}
          </div>

          {evidenceDrawerOpen && (
            <>
              <div
                className="fixed inset-0 z-30 bg-black/20 xl:hidden"
                onClick={() => setEvidenceDrawerOpen(false)}
              />
              <aside className="fixed inset-y-12 right-0 z-40 w-[min(28rem,92vw)] border-l bg-background p-3 shadow-xl xl:hidden">
                <EvidencePanel />
              </aside>
            </>
          )}

          {pipelineDrawerOpen && (
            <>
              <div
                className="fixed inset-0 z-30 bg-black/20 lg:hidden"
                onClick={() => setPipelineDrawerOpen(false)}
              />
              <aside className="fixed inset-y-12 left-0 z-40 w-[min(28rem,92vw)] border-r bg-background p-3 shadow-xl lg:hidden">
                <PipelinePanel />
              </aside>
            </>
          )}
        </main>
      </div>

      {metricsModalOpen && (
        <DataSourceModal
          title="Metrics"
          description="Quality, retrieval, and system measurements for the current run."
          onClose={() => setMetricsModalOpen(false)}
          size="medium"
        >
          <MetricsPanel />
        </DataSourceModal>
      )}

      {strategyModalOpen && (
        <DataSourceModal
          title="Strategy Learning"
          description="Bandit memory and historical config performance across query types."
          onClose={() => setStrategyModalOpen(false)}
          size="medium"
        >
          <BanditViewer />
        </DataSourceModal>
      )}

      {documentsModalOpen && (
        <DataSourceModal
          title="Documents"
          description="Upload, inspect, and remove indexed source documents."
          onClose={() => setDocumentsModalOpen(false)}
          size="compact"
        >
          <FileUpload />
        </DataSourceModal>
      )}

      {papersModalOpen && (
        <DataSourceModal
          title="HuggingFace Trending"
          description="Select trending research papers and load them into the knowledge base."
          onClose={() => setPapersModalOpen(false)}
          size="wide"
        >
          <HuggingFacePapers />
        </DataSourceModal>
      )}

      {configsModalOpen && (
        <DataSourceModal
          title="Available Configs"
          description="Reference view of the retrieval configurations available to the bandit and planner."
          onClose={() => setConfigsModalOpen(false)}
          size="medium"
        >
          <div className="grid gap-3 md:grid-cols-2">
            {CONFIG_SUMMARIES.map((config) => (
              <div key={config.name} className="rounded-lg border bg-background p-4">
                <div className="flex items-center gap-2">
                  <span className="rounded bg-muted px-2 py-1 font-mono text-xs font-semibold text-foreground">
                    {config.name}
                  </span>
                  <span className="text-xs font-medium text-muted-foreground">
                    Retrieval Config
                  </span>
                </div>
                <p className="mt-3 font-mono text-xs leading-6 text-foreground/90">
                  {config.description}
                </p>
              </div>
            ))}
          </div>
        </DataSourceModal>
      )}
    </div>
  );
}
