// ── Backend API types ──

export interface PaperInfo {
  title: string;
  url: string;
  arxiv_url: string | null;
  github_url: string | null;
  github_stars: string | null;
  author: string | null;
  published: string | null;
  abstract: string;
}

export interface CitedChunk {
  index: number;
  text: string;
  source: string;
}

export interface RetrievalDiagnostics {
  query_coverage: number;
  document_diversity: number;
  retrieval_redundancy: number;
  estimated_recall_proxy: number;
}

export interface RunMetrics {
  faithfulness: number;
  citation_precision: number;
  unsupported_claim_rate: number;
  answer_completeness: number;
  cost: number;
  latency: number;
  utility: number;
  config: string;
  query_type: string;
  hops: number;
  retrieval_diagnostics: RetrievalDiagnostics;
}

export interface QueryResponse {
  answer: string;
  citations: CitedChunk[];
  metrics: RunMetrics;
  abstained: boolean;
}

export interface QueryStreamStepEvent {
  type: "step_completed";
  step: PipelineStepId;
  details: Record<string, unknown>;
}

export interface QueryStreamRetryEvent {
  type: "retrying";
  details: {
    retry_count: number;
    previous_config: string;
    selected_config: string;
    reason: string;
    failed_faithfulness: number;
  };
}

export interface QueryStreamCompleteEvent {
  type: "query_complete";
  response: QueryResponse;
}

export interface QueryStreamErrorEvent {
  type: "query_error";
  error: string;
}

export type QueryStreamEvent =
  | QueryStreamStepEvent
  | QueryStreamRetryEvent
  | QueryStreamCompleteEvent
  | QueryStreamErrorEvent;

export interface DocumentRead {
  id: number;
  filename: string;
  source: string;
  status: "processing" | "indexed" | "failed";
  chunks_count: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentUploadResponse {
  message: string;
  chunks_indexed: number;
}

export interface ConfigStats {
  alpha: number;
  beta: number;
  win_rate: number;
  trials: number;
}

export interface BanditStats {
  query_type: string;
  configs: Record<string, ConfigStats>;
}

export interface HealthResponse {
  status: string;
  qdrant: boolean;
  database: boolean;
}

// ── Pipeline tracking types ──

export type PipelineStepId =
  | "planner"
  | "query_rewriting"
  | "retrieval"
  | "reader"
  | "research_controller"
  | "writer"
  | "claim_extraction"
  | "citation_verification"
  | "evaluation";

export type PipelineStepStatus = "pending" | "running" | "done" | "error";

export interface PipelineStep {
  id: PipelineStepId;
  label: string;
  status: PipelineStepStatus;
  details: Record<string, unknown> | null;
}

export interface PipelineRetryState {
  retryCount: number;
  previousConfig: string;
  selectedConfig: string;
  reason: string;
  failedFaithfulness: number;
}
