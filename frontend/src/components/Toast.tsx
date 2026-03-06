"use client";

import { useState, useEffect, useCallback } from "react";

export type ToastType = "success" | "error" | "info";

export interface Toast {
  id: string;
  message: string;
  type: ToastType;
}

let toastListener: ((toast: Toast) => void) | null = null;

export const toast = {
  success: (message: string) => notify(message, "success"),
  error: (message: string) => notify(message, "error"),
  info: (message: string) => notify(message, "info"),
};

function notify(message: string, type: ToastType) {
  if (toastListener) {
    toastListener({
      id: Math.random().toString(36).substr(2, 9),
      message,
      type,
    });
  }
}

export default function ToastContainer() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  useEffect(() => {
    toastListener = (newToast: Toast) => {
      setToasts((prev) => [...prev, newToast]);
      setTimeout(() => removeToast(newToast.id), 5000);
    };
    return () => {
      toastListener = null;
    };
  }, [removeToast]);

  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`toast ${t.type}`}
          onClick={() => removeToast(t.id)}
        >
          <div className="toast-content">{t.message}</div>
          <button className="toast-close">×</button>
        </div>
      ))}
      <style jsx>{`
        .toast-container {
          position: fixed;
          bottom: 24px;
          right: 24px;
          display: flex;
          flex-direction: column;
          gap: 10px;
          z-index: 9999;
          max-width: 400px;
        }
        .toast {
          padding: 12px 16px;
          border-radius: 8px;
          background: var(--card-bg);
          border: 1px solid var(--border);
          color: var(--foreground);
          font-size: 13px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          cursor: pointer;
          animation: slide-in 0.3s ease-out;
          transition:
            transform 0.2s,
            opacity 0.2s;
        }
        @keyframes slide-in {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
        .toast:hover {
          transform: translateY(-2px);
        }
        .toast.success {
          border-left: 4px solid var(--success);
        }
        .toast.error {
          border-left: 4px solid var(--danger);
        }
        .toast.info {
          border-left: 4px solid var(--accent);
        }

        .toast-close {
          background: none;
          border: none;
          color: var(--muted-fg);
          font-size: 18px;
          cursor: pointer;
          padding: 0;
          line-height: 1;
        }
        .toast-close:hover {
          color: var(--foreground);
        }
      `}</style>
    </div>
  );
}
