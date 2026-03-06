const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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

export async function uploadDocument(file: File): Promise<{ message: string; chunks_indexed: number }> {
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

export async function scrapeHuggingFace(): Promise<{ message: string; chunks_indexed: number }> {
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

export async function checkHealth(): Promise<{ status: string; qdrant: boolean; database: boolean }> {
  const res = await fetch(`${API_URL}/api/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}
