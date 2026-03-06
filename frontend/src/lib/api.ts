const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface CitedChunk {
  index: number;
  text: string;
  source: string;
}

export interface RunMetrics {
  faithfulness: number;
  cost: number;
  latency: number;
  utility: number;
  config: string;
  query_type: string;
  hops: number;
}

export interface QueryResponse {
  answer: string;
  citations: CitedChunk[];
  metrics: RunMetrics;
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

export async function submitQuery(query: string): Promise<QueryResponse> {
  const res = await fetch(`${API_URL}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Query failed");
  }
  return res.json();
}

export async function uploadDocument(
  file: File,
): Promise<{ message: string; chunks_indexed: number }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/api/ingest`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Upload failed");
  }
  return res.json();
}

export async function scrapeHuggingFace(): Promise<{
  message: string;
  chunks_indexed: number;
}> {
  const res = await fetch(`${API_URL}/api/ingest/scrape`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Scrape failed");
  }
  return res.json();
}

export async function fetchBanditStats(): Promise<BanditStats[]> {
  const res = await fetch(`${API_URL}/api/bandit`);
  if (!res.ok) throw new Error("Failed to fetch bandit stats");
  return res.json();
}

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

export interface Paper {
  title: string;
  url: string;
  arxiv_url: string | null;
  github_url: string | null;
  github_stars: string | null;
  author: string | null;
  published: string | null;
  abstract: string;
}

export async function fetchDocuments(): Promise<DocumentRead[]> {
  const res = await fetch(`${API_URL}/api/documents`);
  if (!res.ok) throw new Error("Failed to fetch documents");
  return res.json();
}

export async function deleteDocument(
  docId: number,
): Promise<{ message: string }> {
  const res = await fetch(`${API_URL}/api/documents/${docId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Delete failed");
  }
  return res.json();
}

export async function fetchTrendingPapers(): Promise<Paper[]> {
  const res = await fetch(`${API_URL}/api/ingest/trending`);
  if (!res.ok) throw new Error("Failed to fetch trending papers");
  return res.json();
}

export async function ingestSelectedPapers(
  papers: Paper[],
): Promise<{ message: string; chunks_indexed: number }> {
  const res = await fetch(`${API_URL}/api/ingest/selected`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ papers }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Ingest failed");
  }
  return res.json();
}

export async function checkHealth(): Promise<{
  status: string;
  qdrant: boolean;
  database: boolean;
}> {
  const res = await fetch(`${API_URL}/api/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}
