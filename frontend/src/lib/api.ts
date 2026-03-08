import type {
  HealthResponse,
  QueryResponse,
  QueryStreamEvent,
  DocumentRead,
  DocumentUploadResponse,
  PaperInfo,
  BanditStats,
} from "./types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

// ── Health ──
export const getHealth = () => request<HealthResponse>("/api/health");

// ── Query ──
export const runQuery = (query: string) =>
  request<QueryResponse>("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

export async function streamQuery(
  query: string,
  onEvent: (event: QueryStreamEvent) => void
) {
  const res = await fetch(`${API}/api/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!res.ok || !res.body) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const dataLine = chunk
        .split("\n")
        .find((line) => line.startsWith("data: "));

      if (!dataLine) continue;

      const payload = dataLine.slice(6);
      onEvent(JSON.parse(payload) as QueryStreamEvent);
    }
  }
}

// ── Documents ──
export const getDocuments = () => request<DocumentRead[]>("/api/documents");

export const uploadDocument = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  return request<DocumentUploadResponse>("/api/ingest", {
    method: "POST",
    body: form,
  });
};

export const deleteDocument = (docId: number) =>
  request<{ message: string }>(`/api/documents/${docId}`, {
    method: "DELETE",
  });

export const wipeDocuments = () =>
  request<{ message: string }>("/api/documents/wipe", { method: "POST" });

// ── Trending papers ──
export const getTrendingPapers = () =>
  request<PaperInfo[]>("/api/ingest/trending");

export const ingestSelectedPapers = (papers: PaperInfo[]) =>
  request<DocumentUploadResponse>("/api/ingest/selected", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ papers }),
  });

// ── Bandit ──
export const getAllBanditStats = () =>
  request<BanditStats[]>("/api/bandit");

export const getBanditStats = (queryType: string) =>
  request<BanditStats>(`/api/bandit/${queryType}`);
