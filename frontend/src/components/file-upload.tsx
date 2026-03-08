"use client";

import { useRef, useEffect } from "react";
import { useStore } from "@/lib/store";
import { Upload, X, FileText, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useState } from "react";

export function FileUpload() {
  const documents = useStore((s) => s.documents);
  const uploadFile = useStore((s) => s.uploadFile);
  const removeDocument = useStore((s) => s.removeDocument);
  const fetchDocuments = useStore((s) => s.fetchDocuments);
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      await uploadFile(file);
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const statusColor: Record<string, string> = {
    indexed: "bg-emerald-100 text-emerald-700",
    processing: "bg-amber-100 text-amber-700",
    failed: "bg-red-100 text-red-700",
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Documents
        </h3>
        <Button
          variant="outline"
          size="sm"
          className="h-7 text-xs"
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? (
            <Loader2 size={12} className="mr-1 animate-spin" />
          ) : (
            <Upload size={12} className="mr-1" />
          )}
          Upload
        </Button>
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept=".pdf,.html,.htm,.md,.markdown,.docx,.txt"
          onChange={handleUpload}
        />
      </div>

      {documents.length === 0 ? (
        <p className="text-xs text-muted-foreground">No documents uploaded.</p>
      ) : (
        <div className="space-y-1">
          {documents.map((doc) => (
            <div
              key={doc.id}
              className="flex items-center gap-2 rounded-md border px-2 py-1.5 text-xs"
            >
              <FileText size={12} className="shrink-0 text-muted-foreground" />
              <span className="flex-1 truncate">{doc.filename}</span>
              <Badge
                variant="secondary"
                className={`text-[10px] px-1.5 py-0 ${statusColor[doc.status] ?? ""}`}
              >
                {doc.status}
              </Badge>
              <button
                onClick={() => removeDocument(doc.id)}
                className="text-muted-foreground hover:text-red-500 transition-colors"
              >
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
