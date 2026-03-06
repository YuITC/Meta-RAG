"use client";

import { useState } from "react";
import { Paper } from "@/lib/api";

interface PaperSelectionModalProps {
  papers: Paper[];
  onConfirm: (selected: Paper[]) => void;
  onCancel: () => void;
  loading?: boolean;
}

export default function PaperSelectionModal({
  papers,
  onConfirm,
  onCancel,
  loading = false,
}: PaperSelectionModalProps) {
  const [selectedUrls, setSelectedUrls] = useState<Set<string>>(new Set());

  const toggleSelect = (url: string) => {
    const next = new Set(selectedUrls);
    if (next.has(url)) {
      next.delete(url);
    } else {
      next.add(url);
    }
    setSelectedUrls(next);
  };

  const handleSelectAll = () => {
    if (selectedUrls.size === papers.length) {
      setSelectedUrls(new Set());
    } else {
      setSelectedUrls(new Set(papers.map((p) => p.url)));
    }
  };

  const confirm = () => {
    const selected = papers.filter((p) => selectedUrls.has(p.url));
    onConfirm(selected);
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <div className="modal-header">
          <h3>Select Papers to Ingest</h3>
          <p className="subtitle">
            Choose from Top 50 Trending Papers on Hugging Face
          </p>
        </div>

        <div className="modal-actions-top">
          <button className="btn-secondary" onClick={handleSelectAll}>
            {selectedUrls.size === papers.length
              ? "Deselect All"
              : "Select All"}
          </button>
          <span className="count">{selectedUrls.size} selected</span>
        </div>

        <div className="paper-list">
          {papers.map((paper) => (
            <div
              key={paper.url}
              className={`paper-item ${selectedUrls.has(paper.url) ? "selected" : ""}`}
              onClick={() => toggleSelect(paper.url)}
            >
              <input
                type="checkbox"
                checked={selectedUrls.has(paper.url)}
                readOnly
              />
              <div className="paper-info">
                <div className="paper-title">{paper.title}</div>
                <div className="paper-meta">
                  {paper.author && <span>By {paper.author}</span>}
                  {paper.published && <span> • {paper.published}</span>}
                </div>
                <p className="paper-abstract">
                  {paper.abstract.length > 150
                    ? paper.abstract.substring(0, 150) + "..."
                    : paper.abstract}
                </p>
              </div>
            </div>
          ))}
        </div>

        <div className="modal-footer">
          <button className="btn-cancel" onClick={onCancel} disabled={loading}>
            Cancel
          </button>
          <button
            className="btn-confirm"
            onClick={confirm}
            disabled={loading || selectedUrls.size === 0}
          >
            {loading ? "Processing..." : `Import ${selectedUrls.size} Papers`}
          </button>
        </div>
      </div>

      <style jsx>{`
        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.85);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          backdrop-filter: blur(4px);
        }
        .modal-content {
          background: var(--modal-bg);
          border: 1px solid var(--border);
          width: 90%;
          max-width: 800px;
          max-height: 85vh;
          border-radius: 12px;
          display: flex;
          flex-direction: column;
          box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
        }
        .modal-header {
          padding: 20px 24px;
          border-bottom: 1px solid var(--border);
        }
        .modal-header h3 {
          margin: 0;
          font-size: 18px;
          color: var(--foreground);
        }
        .subtitle {
          margin: 4px 0 0;
          font-size: 13px;
          color: var(--muted-fg);
        }
        .modal-actions-top {
          padding: 12px 24px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          background: var(--background);
        }
        .btn-secondary {
          background: none;
          border: 1px solid var(--border);
          color: var(--foreground);
          padding: 4px 12px;
          border-radius: 4px;
          font-size: 12px;
          cursor: pointer;
        }
        .count {
          font-size: 12px;
          color: var(--accent);
          font-weight: 600;
        }
        .paper-list {
          flex: 1;
          overflow-y: auto;
          padding: 10px 0;
        }
        .paper-item {
          display: flex;
          padding: 12px 24px;
          gap: 16px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.05);
          cursor: pointer;
          transition: background 0.15s;
        }
        .paper-item:hover {
          background: rgba(255, 255, 255, 0.03);
        }
        .paper-item.selected {
          background: rgba(var(--accent-rgb), 0.05);
        }
        .paper-item input {
          transform: scale(1.2);
          margin-top: 4px;
        }
        .paper-info {
          flex: 1;
        }
        .paper-title {
          font-size: 14px;
          font-weight: 600;
          color: var(--foreground);
          margin-bottom: 4px;
        }
        .paper-meta {
          font-size: 12px;
          color: var(--muted-fg);
          margin-bottom: 6px;
        }
        .paper-abstract {
          font-size: 12px;
          color: var(--muted-fg);
          line-height: 1.5;
          margin: 0;
          opacity: 0.8;
        }
        .modal-footer {
          padding: 20px 24px;
          border-top: 1px solid var(--border);
          display: flex;
          justify-content: flex-end;
          gap: 12px;
        }
        .btn-cancel {
          background: none;
          border: 1px solid var(--border);
          color: var(--muted-fg);
          padding: 8px 16px;
          border-radius: 6px;
          cursor: pointer;
        }
        .btn-confirm {
          background: var(--accent);
          border: none;
          color: white;
          padding: 8px 20px;
          border-radius: 6px;
          cursor: pointer;
          font-weight: 500;
        }
        .btn-confirm:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
}
