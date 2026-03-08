"use client";

import { ReactNode } from "react";
import { X } from "lucide-react";

type DataSourceModalSize = "compact" | "medium" | "wide";

const SIZE_CLASSES: Record<DataSourceModalSize, string> = {
  compact: "w-[min(34rem,92vw)]",
  medium: "w-[min(44rem,92vw)]",
  wide: "w-[min(56rem,92vw)]",
};

interface DataSourceModalProps {
  title: string;
  description: string;
  onClose: () => void;
  children: ReactNode;
  size?: DataSourceModalSize;
}

export function DataSourceModal({
  title,
  description,
  onClose,
  children,
  size = "medium",
}: DataSourceModalProps) {
  return (
    <>
      <div className="fixed inset-0 z-40 bg-black/30" onClick={onClose} />
      <div
        className={`fixed left-1/2 top-1/2 z-50 flex max-h-[88vh] -translate-x-1/2 -translate-y-1/2 flex-col rounded-xl border bg-background shadow-2xl ${SIZE_CLASSES[size]}`}
      >
        <div className="flex items-start justify-between gap-4 border-b px-5 py-4">
          <div>
            <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
            <p className="mt-1 text-xs text-muted-foreground">{description}</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            aria-label="Close modal"
          >
            <X size={16} />
          </button>
        </div>
        <div className="overflow-y-auto px-5 py-4">{children}</div>
      </div>
    </>
  );
}