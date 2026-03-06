"use client";

import { useEffect, useState } from "react";
import {
  fetchDocuments,
  DocumentRead,
  deleteDocument,
  wipeAllDocuments,
} from "@/lib/api";
import { toast } from "./Toast";

export default function DocumentList({
  refreshTicker,
}: {
  refreshTicker: number;
}) {
  const [documents, setDocuments] = useState<DocumentRead[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [isWiping, setIsWiping] = useState(false);

  const loadDocs = async () => {
    try {
      const data = await fetchDocuments();
      setDocuments(data);
      setError(null);
    } catch (err) {
      setError("Failed to load documents");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (
    e: React.MouseEvent,
    docId: number,
    filename: string,
  ) => {
    e.stopPropagation();
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) return;

    setDeletingId(docId);
    try {
      await deleteDocument(docId);
      setDocuments(documents.filter((d) => d.id !== docId));
      toast.success(`Deleted "${filename}"`);
    } catch (err) {
      toast.error("Failed to delete document");
    } finally {
      setDeletingId(null);
    }
  };

  const handleWipeAll = async () => {
    if (
      !confirm(
        "CRITICAL: This will delete ALL documents from the database and WIPE the entire vector collection. Are you absolutely sure?",
      )
    )
      return;

    setIsWiping(true);
    try {
      await wipeAllDocuments();
      setDocuments([]);
      toast.success("All data has been wiped.");
    } catch (err) {
      toast.error("Failed to wipe documents");
    } finally {
      setIsWiping(false);
    }
  };

  useEffect(() => {
    loadDocs();
    // Poll for status updates if any document is processing
    const interval = setInterval(() => {
      const hasProcessing = documents.some(
        (doc) => doc.status === "processing",
      );
      if (hasProcessing) {
        loadDocs();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [refreshTicker, documents.some((doc) => doc.status === "processing")]);

  if (loading && documents.length === 0)
    return <div className="loading">Loading documents...</div>;

  return (
    <div className="document-list-container">
      <div className="list-header">
        <span className="label">Documents in system</span>
        <div className="header-actions">
          <button
            className="btn-wipe"
            onClick={handleWipeAll}
            disabled={isWiping || documents.length === 0}
            title="Wipe ALL data"
          >
            {isWiping ? "wiping..." : "wipe all"}
          </button>
          <button
            className="btn-refresh"
            onClick={loadDocs}
            title="Refresh list"
          >
            refresh
          </button>
        </div>
      </div>

      {error && <div className="error-small">{error}</div>}

      <div className="document-grid">
        {documents.length === 0 ? (
          <div className="empty-state">
            No documents found. Upload or scrape to start.
          </div>
        ) : (
          documents.map((doc) => (
            <div key={doc.id} className={`document-card ${doc.status}`}>
              <div className="doc-main">
                <span className="doc-name" title={doc.filename}>
                  {doc.filename}
                </span>
                <div className="doc-right">
                  <span className={`status-badge ${doc.status}`}>
                    {doc.status}
                  </span>
                  <button
                    className="btn-delete"
                    onClick={(e) => handleDelete(e, doc.id, doc.filename)}
                    disabled={deletingId === doc.id}
                    title="Delete document"
                  >
                    {deletingId === doc.id ? "..." : "×"}
                  </button>
                </div>
              </div>
              <div className="doc-meta">
                <span>{doc.chunks_count} chunks</span>
                {doc.status === "failed" && doc.error_message && (
                  <span className="error-text" title={doc.error_message}>
                    {" "}
                    — {doc.error_message}
                  </span>
                )}
                <span className="date">
                  {new Date(doc.created_at).toLocaleString()}
                </span>
              </div>
              {doc.status === "processing" && <div className="progress-bar" />}
            </div>
          ))
        )}
      </div>

      <style jsx>{`
        .document-list-container {
          margin-top: 24px;
          border-top: 1px solid var(--border);
          padding-top: 20px;
        }
        .list-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        .header-actions {
          display: flex;
          gap: 12px;
          align-items: center;
        }
        .label {
          color: var(--muted-fg);
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }
        .btn-wipe {
          background: none;
          border: none;
          color: var(--danger);
          font-size: 11px;
          cursor: pointer;
          font-family: inherit;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          opacity: 0.7;
        }
        .btn-wipe:hover:not(:disabled) {
          opacity: 1;
          text-decoration: underline;
        }
        .btn-wipe:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }
        .btn-refresh {
          background: none;
          border: none;
          color: var(--accent);
          font-size: 11px;
          cursor: pointer;
          font-family: inherit;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        .btn-refresh:hover {
          text-decoration: underline;
        }
        .document-grid {
          display: flex;
          flex-direction: column;
          gap: 8px;
          max-height: 400px;
          overflow-y: auto;
          padding-right: 4px;
        }
        .document-card {
          border: 1px solid var(--border);
          border-radius: 6px;
          padding: 8px 12px;
          background: var(--card-bg);
          position: relative;
          overflow: hidden;
        }
        .doc-main {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 4px;
        }
        .doc-name {
          font-size: 13px;
          font-weight: 500;
          color: var(--foreground);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          max-width: 60%;
        }
        .doc-right {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .btn-delete {
          background: none;
          border: none;
          color: var(--muted-fg);
          font-size: 18px;
          cursor: pointer;
          padding: 0 4px;
          line-height: 1;
          transition: color 0.15s;
        }
        .btn-delete:hover:not(:disabled) {
          color: var(--danger);
        }
        .btn-delete:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .status-badge {
          font-size: 10px;
          text-transform: uppercase;
          padding: 2px 6px;
          border-radius: 4px;
          font-weight: 600;
          letter-spacing: 0.02em;
        }
        .status-badge.processing {
          background: #332b00;
          color: #ffcc00;
        }
        .status-badge.indexed {
          background: #002200;
          color: var(--success);
        }
        .status-badge.failed {
          background: #220000;
          color: var(--danger);
        }

        .doc-meta {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          color: var(--muted-fg);
        }
        .error-text {
          color: var(--danger);
          font-style: italic;
        }
        .date {
          opacity: 0.6;
        }

        .progress-bar {
          position: absolute;
          bottom: 0;
          left: 0;
          height: 2px;
          background: var(--accent);
          width: 30%;
          animation: slide 1.5s infinite ease-in-out;
        }
        @keyframes slide {
          0% {
            left: -30%;
          }
          100% {
            left: 100%;
          }
        }

        .empty-state {
          padding: 24px;
          text-align: center;
          color: var(--muted-fg);
          font-size: 13px;
          border: 1px dashed var(--border);
          border-radius: 8px;
        }
        .error-small {
          color: var(--danger);
          font-size: 12px;
          margin-bottom: 10px;
        }
        .loading {
          color: var(--muted-fg);
          font-size: 12px;
          padding: 20px 0;
        }
      `}</style>
    </div>
  );
}
