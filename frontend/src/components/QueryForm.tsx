"use client";

import { useState, useRef } from "react";
import { uploadDocument, scrapeHuggingFace } from "@/lib/api";

interface QueryFormProps {
  onSubmit: (query: string) => void;
  onSourceUpdated?: () => void;
  loading: boolean;
}

export default function QueryForm({
  onSubmit,
  onSourceUpdated,
  loading,
}: QueryFormProps) {
  const [query, setQuery] = useState("");
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const [uploadErr, setUploadErr] = useState<string | null>(null);
  const [scraping, setScraping] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) onSubmit(query.trim());
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadMsg(null);
    setUploadErr(null);
    try {
      const result = await uploadDocument(file);
      setUploadMsg(`${result.message}`);
      if (onSourceUpdated) onSourceUpdated();
    } catch (err: unknown) {
      setUploadErr(err instanceof Error ? err.message : "Upload failed");
    }
    if (fileRef.current) fileRef.current.value = "";
  };

  const handleScrape = async () => {
    setScraping(true);
    setUploadMsg(null);
    setUploadErr(null);
    try {
      const result = await scrapeHuggingFace();
      setUploadMsg(result.message);
      if (onSourceUpdated) onSourceUpdated();
    } catch (err: unknown) {
      setUploadErr(err instanceof Error ? err.message : "Scrape failed");
    } finally {
      setScraping(false);
    }
  };

  return (
    <div className="query-form">
      {/* Document sources */}
      <div className="source-bar">
        <span className="label">data sources</span>
        <label
          className="btn-ghost"
          title="Upload a document (PDF, DOCX, HTML, MD, TXT)"
        >
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.docx,.html,.htm,.md,.markdown,.txt"
            onChange={handleFileChange}
            style={{ display: "none" }}
          />
          + upload doc
        </label>
        <button
          className="btn-ghost"
          onClick={handleScrape}
          disabled={scraping}
          title="Scrape top-50 trending papers from HuggingFace"
        >
          {scraping ? "scraping..." : "scrape hf/papers"}
        </button>
        {uploadMsg && <span className="msg-ok">{uploadMsg}</span>}
        {uploadErr && <span className="msg-err">{uploadErr}</span>}
      </div>

      {/* Query input */}
      <form onSubmit={handleSubmit} className="query-row">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a research question..."
          className="query-input"
          disabled={loading}
          autoFocus
        />
        <button
          type="submit"
          className="btn-primary"
          disabled={loading || !query.trim()}
        >
          {loading ? "thinking..." : "run"}
        </button>
      </form>

      <style jsx>{`
        .query-form {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }
        .source-bar {
          display: flex;
          align-items: center;
          gap: 12px;
          flex-wrap: wrap;
        }
        .label {
          color: var(--muted-fg);
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }
        .btn-ghost {
          background: none;
          border: 1px solid var(--border);
          color: var(--muted-fg);
          padding: 4px 10px;
          border-radius: 4px;
          font-size: 12px;
          cursor: pointer;
          font-family: inherit;
          transition:
            border-color 0.15s,
            color 0.15s;
        }
        .btn-ghost:hover:not(:disabled) {
          border-color: var(--accent);
          color: var(--foreground);
        }
        .btn-ghost:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .msg-ok {
          color: var(--success);
          font-size: 12px;
        }
        .msg-err {
          color: var(--danger);
          font-size: 12px;
        }
        .query-row {
          display: flex;
          gap: 8px;
        }
        .query-input {
          flex: 1;
          background: #111113;
          border: 1px solid var(--border);
          color: var(--foreground);
          padding: 10px 14px;
          border-radius: 6px;
          font-size: 14px;
          font-family: inherit;
          outline: none;
          transition: border-color 0.15s;
        }
        .query-input:focus {
          border-color: var(--accent);
        }
        .query-input::placeholder {
          color: var(--muted-fg);
        }
        .btn-primary {
          background: var(--accent);
          border: none;
          color: #fff;
          padding: 10px 20px;
          border-radius: 6px;
          font-size: 13px;
          font-family: inherit;
          cursor: pointer;
          transition: background 0.15s;
          white-space: nowrap;
        }
        .btn-primary:hover:not(:disabled) {
          background: var(--accent-dim);
        }
        .btn-primary:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
}
