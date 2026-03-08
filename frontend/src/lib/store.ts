import { create } from "zustand";
import type {
  QueryResponse,
  QueryStreamEvent,
  DocumentRead,
  PaperInfo,
  BanditStats,
  PipelineStep,
  PipelineStepId,
  PipelineRetryState,
  HealthResponse,
} from "./types";
import * as api from "./api";

const PIPELINE_ORDER: PipelineStepId[] = [
  "planner",
  "query_rewriting",
  "retrieval",
  "reader",
  "research_controller",
  "writer",
  "claim_extraction",
  "citation_verification",
  "evaluation",
];

const RUNNING_DETAILS: Record<PipelineStepId, Record<string, unknown>> = {
  planner: { status: "classifying query", output: "selecting retrieval config" },
  query_rewriting: { status: "generating variants", output: "expanding search intents" },
  retrieval: { status: "retrieving documents", output: "dense + sparse fusion" },
  reader: { status: "reading evidence", output: "extracting relevant spans" },
  research_controller: { status: "evaluating coverage", output: "deciding stop or continue" },
  writer: { status: "drafting answer", output: "grounding synthesis in evidence" },
  claim_extraction: { status: "extracting claims", output: "splitting answer into atomic statements" },
  citation_verification: { status: "verifying citations", output: "checking support for each claim" },
  evaluation: { status: "scoring response", output: "faithfulness and completeness review" },
};

function setRunningStep(
  set: (partial: Partial<AppState> | ((state: AppState) => Partial<AppState>)) => void,
  stepId: PipelineStepId,
  details?: Record<string, unknown>
) {
  set((state) => ({
    pipeline: state.pipeline.map((step) =>
      step.id === stepId
        ? {
            ...step,
            status: "running" as const,
            details: details ?? RUNNING_DETAILS[stepId],
          }
        : step
    ),
  }));
}

function completeStepAndStartNext(
  set: (partial: Partial<AppState> | ((state: AppState) => Partial<AppState>)) => void,
  stepId: PipelineStepId,
  details: Record<string, unknown>
) {
  const currentIndex = PIPELINE_ORDER.indexOf(stepId);
  const nextStep = PIPELINE_ORDER[currentIndex + 1];

  set((state) => ({
    pipeline: state.pipeline.map((step) => {
      if (step.id === stepId) {
        return { ...step, status: "done" as const, details };
      }

      if (nextStep && step.id === nextStep && step.status === "pending") {
        return {
          ...step,
          status: "running" as const,
          details: RUNNING_DETAILS[nextStep],
        };
      }

      return step;
    }),
  }));
}

function startRetryPass(
  set: (partial: Partial<AppState> | ((state: AppState) => Partial<AppState>)) => void,
  retryState: PipelineRetryState
) {
  set((state) => ({
    retryState,
    pipeline: state.pipeline.map((step) => {
      if (step.id === "planner") {
        return step;
      }

      if (step.id === "evaluation") {
        return {
          ...step,
          status: "done" as const,
          details: {
            ...(step.details ?? {}),
            retry_triggered: true,
            failed_faithfulness: retryState.failedFaithfulness,
            retry_reason: retryState.reason,
          },
        };
      }

      if (step.id === "query_rewriting") {
        return {
          ...step,
          status: "running" as const,
          details: {
            status: "retrying with alternate config",
            previous_config: retryState.previousConfig,
            selected_config: retryState.selectedConfig,
          },
        };
      }

      return {
        ...step,
        status: "pending" as const,
        details: null,
      };
    }),
  }));
}

const PIPELINE_STEPS: PipelineStep[] = [
  { id: "planner", label: "Planner", status: "pending", details: null },
  { id: "query_rewriting", label: "Query Rewriting", status: "pending", details: null },
  { id: "retrieval", label: "Retrieval", status: "pending", details: null },
  { id: "reader", label: "Reader", status: "pending", details: null },
  { id: "research_controller", label: "Research Controller", status: "pending", details: null },
  { id: "writer", label: "Writer", status: "pending", details: null },
  { id: "claim_extraction", label: "Claim Extraction", status: "pending", details: null },
  { id: "citation_verification", label: "Citation Verification", status: "pending", details: null },
  { id: "evaluation", label: "Evaluation", status: "pending", details: null },
];

interface AppState {
  // Health
  health: HealthResponse | null;
  fetchHealth: () => Promise<void>;

  // Query
  query: string;
  setQuery: (q: string) => void;
  isResearching: boolean;
  result: QueryResponse | null;
  error: string | null;
  runResearch: () => Promise<void>;

  // Pipeline progress
  pipeline: PipelineStep[];
  advancePipeline: (stepId: PipelineStepId, details?: Record<string, unknown>) => void;
  resetPipeline: () => void;
  retryState: PipelineRetryState | null;

  // Highlighted citation
  highlightedCitation: number | null;
  setHighlightedCitation: (idx: number | null) => void;

  // Documents
  documents: DocumentRead[];
  fetchDocuments: () => Promise<void>;
  uploadFile: (file: File) => Promise<void>;
  removeDocument: (id: number) => Promise<void>;

  // Trending papers
  trendingPapers: PaperInfo[];
  selectedPapers: Set<number>;
  isFetchingPapers: boolean;
  isIngestingPapers: boolean;
  fetchTrendingPapers: () => Promise<void>;
  togglePaperSelection: (idx: number) => void;
  ingestSelectedPapers: () => Promise<void>;

  // Bandit
  banditStats: BanditStats[];
  fetchBanditStats: () => Promise<void>;

  // Sidebar
  sidebarOpen: boolean;
  toggleSidebar: () => void;
}

export const useStore = create<AppState>((set, get) => ({
  // Health
  health: null,
  fetchHealth: async () => {
    try {
      const h = await api.getHealth();
      set({ health: h });
    } catch {
      set({ health: null });
    }
  },

  // Query
  query: "",
  setQuery: (q) => set({ query: q }),
  isResearching: false,
  result: null,
  error: null,

  runResearch: async () => {
    const { query } = get();
    if (!query.trim()) return;

    set({ isResearching: true, result: null, error: null });
    get().resetPipeline();
    setRunningStep(set, "planner");

    try {
      await api.streamQuery(query, (event: QueryStreamEvent) => {
        if (event.type === "step_completed") {
          completeStepAndStartNext(set, event.step, event.details);
          return;
        }

        if (event.type === "retrying") {
          startRetryPass(set, {
            retryCount: event.details.retry_count,
            previousConfig: event.details.previous_config,
            selectedConfig: event.details.selected_config,
            reason: event.details.reason,
            failedFaithfulness: event.details.failed_faithfulness,
          });
          return;
        }

        if (event.type === "query_complete") {
          set({ result: event.response, isResearching: false });
          get().fetchBanditStats();
          return;
        }

        if (event.type === "query_error") {
          throw new Error(event.error);
        }
      });
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : "Unknown error";
      // Mark currently running step as error
      set((s) => ({
        pipeline: s.pipeline.map((p) =>
          p.status === "running" ? { ...p, status: "error" as const } : p
        ),
        error: errMsg,
        isResearching: false,
      }));
    }
  },

  // Pipeline
  pipeline: structuredClone(PIPELINE_STEPS),
  retryState: null,
  advancePipeline: (stepId, details) =>
    set((s) => ({
      pipeline: s.pipeline.map((p) => {
        if (p.id === stepId) return { ...p, status: "running" as const, details: details ?? p.details };
        return p;
      }),
    })),
  resetPipeline: () => set({ pipeline: structuredClone(PIPELINE_STEPS), retryState: null }),

  // Citation highlight
  highlightedCitation: null,
  setHighlightedCitation: (idx) => set({ highlightedCitation: idx }),

  // Documents
  documents: [],
  fetchDocuments: async () => {
    try {
      const docs = await api.getDocuments();
      set({ documents: docs });
    } catch {
      /* ignore */
    }
  },
  uploadFile: async (file) => {
    await api.uploadDocument(file);
    get().fetchDocuments();
  },
  removeDocument: async (id) => {
    await api.deleteDocument(id);
    set((s) => ({ documents: s.documents.filter((d) => d.id !== id) }));
  },

  // Trending papers
  trendingPapers: [],
  selectedPapers: new Set(),
  isFetchingPapers: false,
  isIngestingPapers: false,
  fetchTrendingPapers: async () => {
    set({ isFetchingPapers: true });
    try {
      const papers = await api.getTrendingPapers();
      set({ trendingPapers: papers, isFetchingPapers: false });
    } catch {
      set({ isFetchingPapers: false });
    }
  },
  togglePaperSelection: (idx) =>
    set((s) => {
      const next = new Set(s.selectedPapers);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return { selectedPapers: next };
    }),
  ingestSelectedPapers: async () => {
    const { trendingPapers, selectedPapers } = get();
    const papers = Array.from(selectedPapers).map((i) => trendingPapers[i]);
    if (!papers.length) return;
    set({ isIngestingPapers: true });
    try {
      await api.ingestSelectedPapers(papers);
      set({ isIngestingPapers: false, selectedPapers: new Set() });
      get().fetchDocuments();
    } catch {
      set({ isIngestingPapers: false });
    }
  },

  // Bandit
  banditStats: [],
  fetchBanditStats: async () => {
    try {
      const stats = await api.getAllBanditStats();
      set({ banditStats: stats });
    } catch {
      /* ignore */
    }
  },

  // Sidebar
  sidebarOpen: false,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
}));
